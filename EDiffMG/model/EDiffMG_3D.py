import torch
from torch import nn
from torch_scatter import scatter_mean
from tqdm.auto import tqdm

from utils.chem import BOND_TYPES
from .diffusion import get_timestep_embedding, get_beta_schedule
from ..common import MultiLayerPerceptron, assemble_atom_pair_feature, _extend_to_radius_graph
from ..encoder import SchNetEncoder, get_edge_encoder, EGNNSparseNetwork
from ..geometry import get_distance, eq_transform

# from ..gate_MLP.gMLP_V2 import gate_MultiLayerPerceptron
from ..gate_MLP.gMLP_V4 import gate_MultiLayerPerceptron
from ..encoder import caSchNetEncoder


class EDiffMG(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        """
        edge_encoder:  Takes both edge type and edge length as input and outputs a vector
        [Note]: node embedding is done in SchNetEncoder
        """
        self.edge_encoder_global = get_edge_encoder(config)
        self.edge_encoder_local = get_edge_encoder(config)
        # self.hidden_dim = config.hidden_dim
        self.atom_type_input_dim = config.num_atom if 'num_atom' in config else 6
        # contains simple tmb or charge or not qm9:5+1(charge) ZINC: 7+1(charge)
        self.atom_out_dim = config.num_atom if 'num_atom' in config else 6  # contains charge or not
        self.time_emb = True

        '''
        timestep embedding
        '''
        if self.time_emb:
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(config.hidden_dim,
                                config.hidden_dim * 4),
                torch.nn.Linear(config.hidden_dim * 4,
                                config.hidden_dim * 4),
            ])
            self.temb_proj = torch.nn.Linear(config.hidden_dim * 4,
                                             config.hidden_dim)



        """
        Global encoder (SchNet, EGNN, SphereNet)
        """

        self.encoder_global = caSchNetEncoder(
            hidden_channels=config.hidden_dim,
            num_filters=config.hidden_dim,
            num_interactions=config.num_convs,
            caFilter_per_block=config.caFilter_per_block,
            edge_channels=self.edge_encoder_global.out_channels,
            cutoff=10, #config.cutoff
            smooth=config.smooth_conv,
            input_dim=self.atom_type_input_dim,
            time_emb=self.time_emb

        )

        self.encoder_local = caSchNetEncoder(
            hidden_channels=config.hidden_dim,
            num_filters=config.hidden_dim,
            num_interactions=config.num_convs_local,
            caFilter_per_block=config.caFilter_per_block,
            edge_channels=self.edge_encoder_local.out_channels,
            cutoff=2.1, #config.cutoff
            smooth=config.smooth_conv,
            input_dim=self.atom_type_input_dim,
            time_emb=self.time_emb

        )



        """
            `output_gmlp` takes a mixture of two nodewise features and edge features as input and outputs
            gradients w.r.t. edge_length (out_dim = 1) and node type.
        """
        self.grad_global_dist_gmlp = gate_MultiLayerPerceptron(num_tokens=1,
                                                      dim=2*config.hidden_dim,
                                                      depth=2,
                                                      attn_dim=64,
                                                      act=nn.GELU()
                                                      )

        self.grad_local_dist_gmlp = gate_MultiLayerPerceptron(num_tokens=1,
                                                      dim=2*config.hidden_dim,
                                                      depth=2,
                                                      attn_dim=64,
                                                      act=nn.GELU()
                                                      )

        self.grad_global_node_gmlp = gate_MultiLayerPerceptron(num_tokens=self.atom_out_dim,
                                                      dim=config.hidden_dim,
                                                      depth=2,
                                                      attn_dim=64,
                                                      act=nn.GELU()
                                                      )

        self.grad_local_node_gmlp = gate_MultiLayerPerceptron(num_tokens=self.atom_out_dim,
                                                      dim=config.hidden_dim,
                                                      depth=2,
                                                      attn_dim=64,
                                                      act=nn.GELU()
                                                      )


        '''
        Incorporate parameters together
        '''
        self.model_global = nn.ModuleList([self.edge_encoder_global, self.encoder_global, self.grad_global_dist_gmlp])
        self.model_local = nn.ModuleList([self.edge_encoder_local, self.encoder_local, self.grad_local_dist_gmlp])

        self.model_type = config.type  # config.type  # 'diffusion'; 'dsm'

        betas = get_beta_schedule(
            beta_schedule=config.beta_schedule,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            num_diffusion_timesteps=config.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float()
        self.betas = nn.Parameter(betas, requires_grad=False)
        # variances
        alphas = (1. - betas).cumprod(dim=0)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.num_timesteps = self.betas.size(0)

    def net(self, atom_type, pos, bond_index, bond_type, batch, time_step,
            edge_index=None, edge_type=None, edge_length=None, vae_noise=None):
        """
        Args:
            atom_type:  Types of atoms, (N, ).
            pos: atom coordinates
            bond_index: Indices of bonds (not extended, not radius-graph), (2, E).
            bond_type:  Bond types, (E, ).
            batch:      Node index to graph index, (N, ).
        """
        # N = atom_type.size(0)
        if not self.time_emb:
            time_step = time_step / self.num_timesteps
            time_emb = time_step.index_select(0, batch).unsqueeze(1)
            atom_type = torch.cat([atom_type, time_emb], dim=1)


        '''
        Time embedding for node
        '''
        if self.time_emb:
            nonlinearity = nn.ReLU()
            temb = get_timestep_embedding(time_step, self.config.hidden_dim)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
            temb = self.temb_proj(nonlinearity(temb))  # (G, dim)
            atom_type = torch.cat([atom_type, temb.index_select(0, batch)], dim=1)

        if edge_index is None or edge_type is None or edge_length is None:
            bond_type = torch.ones(bond_index.size(1), dtype=torch.long).to(bond_index.device)

            edge_index, edge_type, global_edge_section_mask =\
                _extend_to_radius_graph(
                pos=pos,
                edge_index=bond_index,
                edge_type=bond_type,
                batch=batch,
                cutoff=self.config.cutoff,
            )
            edge_length = get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)
        local_edge_mask = is_radius_edge(edge_type)

        # Emb time_step for edge
        if self.time_emb:
            node2graph = batch
            edge2graph = node2graph.index_select(0, edge_index[0])
            temb_edge = temb.index_select(0, edge2graph)

        # Encoding global
        edge_attr_global = self.edge_encoder_global(
            edge_length=edge_length,
            edge_type=edge_type
        )  # Embed edges

        if self.time_emb:
            edge_attr_global += temb_edge


        # caSchNet
        node_attr_global = self.encoder_global(
            z=atom_type,
            edge_index=edge_index,
            edge_length=edge_length,
            edge_attr=edge_attr_global,
            embed_node=False # default is True
        )

        """
        Assemble pairwise features
        """
        h_pair_global = assemble_atom_pair_feature(
            node_attr=node_attr_global,
            edge_index=edge_index,
            edge_attr=edge_attr_global,
        )  # (E_global, 2H)
        """
        Invariant features of edges (radius graph, global)
        """

        dist_score_global = self.grad_global_dist_gmlp(h_pair_global) # (E_global, 1)

        # print("dist_score_global shape", dist_score_global.shape)
        """
        Encoding local
        """
        edge_attr_local = self.edge_encoder_local(
            edge_length=edge_length,
            edge_type=edge_type
        )  # Embed edges
        if self.time_emb:
            edge_attr_local += temb_edge



        # caSchnet
        node_attr_local = self.encoder_local(
            z=atom_type,
            edge_index=edge_index[:, local_edge_mask],
            edge_length=edge_length[local_edge_mask],
            edge_attr=edge_attr_local[local_edge_mask],
            embed_node=False # default is True
        )

        """
        Assemble pairwise features
        """

        h_pair_local = assemble_atom_pair_feature(
            node_attr=node_attr_local,
            edge_index=edge_index[:, local_edge_mask],
            edge_attr=edge_attr_local[local_edge_mask],
        )  # (E_local, 2H)

        """
        Invariant features of edges (bond graph, local)
        """

        dist_score_local = self.grad_local_dist_gmlp(h_pair_local)

        node_score_global = self.grad_global_node_gmlp(node_attr_global)
        node_score_local = self.grad_local_node_gmlp(node_attr_local)


        return dist_score_global, dist_score_local, node_score_global, node_score_local, \
                   edge_index, edge_type, edge_length, local_edge_mask, global_edge_section_mask

    def forward(self, batch, return_unreduced_loss=False, return_unreduced_edge_loss=False,
                extend_order=True, extend_radius=True, is_sidechain=None):

        atom_type = batch.atom_feat_full.float()
        pos = batch.pos
        bond_index = batch.edge_index
        bond_type = batch.edge_type
        batch = batch.batch

        # N = atom_type.size(0)
        node2graph = batch
        num_graphs = node2graph[-1] + 1

        # Sample time step
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=pos.device)

        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]

        a = self.alphas.index_select(0, time_step)  # (G, )

        a_pos = a.index_select(0, node2graph).unsqueeze(-1)  # (N, 1)

        """
        Independently
        - Perterb pos
        """
        pos_noise = torch.zeros(size=pos.size(), device=pos.device)
        pos_noise.normal_()
        pos_perturbed = pos + center_pos(pos_noise, batch) * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        # pos_perturbed = a_pos.sqrt()*pos+(1.0 - a_pos).sqrt()*center_pos(pos_noise,batch)
        """
        Perterb atom
        """
        atom_noise = torch.zeros(size=atom_type.size(), device=atom_type.device)
        atom_noise.normal_()
        atom_type = torch.cat([atom_type[:, :-1] / 4, atom_type[:, -1:] / 10], dim=1)
        atom_perturbed = a_pos.sqrt() * atom_type + (1.0 - a_pos).sqrt() * atom_noise

        """
        Jointly
        """
        # noise = torch.zeros(size=(pos.size(0), pos.size(1)+atom_type.size(1)), device=pos.device)
        # noise.normal_()
        """
        Perterb pos
        """
        # pos_perturbed = pos + center_pos(noise[:,:3], batch) * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        # pos_perturbed = center_pos(pos_perturbed, batch)
        """
        Perterb atom and normalize
        """
        # atom_type = torch.cat([atom_type[:,:-1]/4,atom_type[:,-1:]/10], dim=1)
        # print(atom_type)
        # atom_perturbed = atom_type + noise[:,3:] * (1.0 - a_pos).sqrt() / a_pos.sqrt()
        # print(atom_perturbed)

        # vae_noise = torch.randn_like(atom_type)
        vae_noise = torch.randn_like(atom_type)  # N(0,1)
        # vae_noise = torch.clamp(torch.randn_like(atom_type), min=-3, max=3) # clip N(0,1)
        # vae_noise = torch.normal(0,3,size=(atom_type.size())).to(atom_type.device) # N(0,3)
        # vae_noise = torch.zeros_like(atom_type).uniform_(-1,+1) # U(-1,1)
        net_out = self.net(
            atom_type=atom_perturbed,
            pos=pos_perturbed,
            bond_index=bond_index,
            bond_type=bond_type,
            batch=batch,
            time_step=time_step,
            vae_noise=vae_noise

        )  # (E_global, 1), (E_local, 1)

        dist_score_global, dist_score_local, node_score_global, node_score_local, \
        edge_index, _, edge_length, local_edge_mask, global_edge_section_mask = net_out


        edge2graph = node2graph.index_select(0, edge_index[0])

        # Compute sigmas_edge
        a_edge = a.index_select(0, edge2graph).unsqueeze(-1)  # (E, 1)

        # Compute original and perturbed distances
        d_gt = get_distance(pos, edge_index).unsqueeze(-1)  # (E, 1)
        d_perturbed = edge_length

        train_edge_mask = is_train_edge(edge_index, is_sidechain)
        d_perturbed = torch.where(train_edge_mask.unsqueeze(-1), d_perturbed, d_gt)

        if self.config.edge_encoder == 'gaussian':
            # Distances must be greater than 0
            d_sgn = torch.sign(d_perturbed)
            d_perturbed = torch.clamp(d_perturbed * d_sgn, min=0.01, max=float('inf'))

        d_target = (d_gt - d_perturbed) / (1.0 - a_edge).sqrt() * a_edge.sqrt()  # (E_global, 1), denoising direction

        global_mask = torch.logical_and(
            torch.logical_or(torch.logical_and(d_perturbed > self.config.cutoff, d_perturbed <= 10),
                             local_edge_mask.unsqueeze(-1)),
            ~local_edge_mask.unsqueeze(-1)
        )
        if global_edge_section_mask.numel() > 0:

            global_mask = global_mask | global_edge_section_mask

        edge_inv_global = torch.where(global_mask, dist_score_global, torch.zeros_like(dist_score_global))
        node_eq_global = eq_transform(edge_inv_global, pos_perturbed, edge_index, edge_length)

        # global pos
        target_d_global = torch.where(global_mask, d_target, torch.zeros_like(d_target))
        target_pos_global = eq_transform(target_d_global, pos_perturbed, edge_index, edge_length)
        loss_global = (node_eq_global - target_pos_global) ** 2
        loss_global = 1 * torch.sum(loss_global, dim=-1, keepdim=True)

        node_eq_local = eq_transform(dist_score_local, pos_perturbed, edge_index[:, local_edge_mask],
                                     edge_length[local_edge_mask])

        # local pos
        target_pos_local = eq_transform(d_target[local_edge_mask], pos_perturbed, edge_index[:, local_edge_mask],
                                        edge_length[local_edge_mask])
        loss_local = (node_eq_local - target_pos_local) ** 2

        loss_local = 1 * torch.sum(loss_local, dim=-1, keepdim=True)

        loss_node_global = (node_score_global - atom_noise) ** 2
        loss_node_global = 1 * torch.sum(loss_node_global, dim=-1, keepdim=True)

        loss_node_local = (node_score_local - atom_noise) ** 2
        loss_node_local = 1 * torch.sum(loss_node_local, dim=-1, keepdim=True)

        # loss for atomic eps regression

        loss = loss_global + loss_local + loss_node_global + loss_node_local

        if return_unreduced_edge_loss:
            pass
        elif return_unreduced_loss:
            return loss, loss_global, loss_local, loss_node_global, loss_node_local
        else:
            return loss

    def langevin_dynamics_sample(self, atom_type, pos_init, bond_index, bond_type, batch, num_graphs,
                                 n_steps=100, step_lr=0.0000010, clip=1000, clip_local=None, clip_pos=None,
                                 global_start_sigma=float('inf'), local_start_sigma=float('inf'), w_global_pos=1,
                                 w_global_node=1, w_local_pos=1, w_local_node=1, **kwargs):

        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)  # .view(-1, 1, 1, 1)
            return a

        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        pos_traj = []
        atom_traj = []

        with torch.no_grad():
            seq = range(self.num_timesteps - n_steps, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])
            pos = center_pos(pos_init, batch)
            # vae_noise = torch.zeros_like(atom_type).uniform_(-1, +1)
            # bond_index = radius_graph(pos, self.config.cutoff, batch=batch, loop=False)
            # bond_type = torch.ones(bond_index.size(1),dtype=torch.long).to(bond_index.device)
            # VAE noise
            vae_noise = torch.zeros_like(atom_type).uniform_(-1, +1)
            # vae_noise = torch.randn_like(atom_type)
            # vae_noise = torch.clamp(torch.randn_like(atom_type), min=-3, max=3)
            # vae_noise = torch.normal(0,3,size=(atom_type.size())).to(atom_type.device)

            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc='sample'):
                t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=pos.device)

                net_out = self.net(
                    atom_type=atom_type,
                    pos=pos,
                    bond_index=bond_index,
                    bond_type=bond_type,
                    batch=batch,
                    time_step=t,
                    vae_noise=vae_noise

                )  # (E_global, 1), (E_local, 1)

                edge_inv_global, edge_inv_local, node_score_global, node_score_local, \
                edge_index, _, edge_length, local_edge_mask, global_edge_section_mask = net_out
                # Local float('inf')
                # import random
                # local_start_sigma = random.uniform(0.1, 1)
                if sigmas[i] < local_start_sigma:
                    node_eq_local = eq_transform(edge_inv_local, pos, edge_index[:, local_edge_mask],
                                                 edge_length[local_edge_mask])
                    if clip_local is not None:
                        node_eq_local = clip_norm(node_eq_local, limit=clip_local)
                else:
                    node_eq_local = 0
                    node_score_local = 0

                # Global
                if sigmas[i] < global_start_sigma:
                    edge_inv_global = edge_inv_global * (1 - local_edge_mask.view(-1, 1).float())
                    node_eq_global = eq_transform(edge_inv_global, pos, edge_index, edge_length)
                    node_eq_global = clip_norm(node_eq_global, limit=clip)
                else:
                    node_eq_global = 0
                    node_score_global = 0
                # Sum
                eps_pos = w_local_pos * node_eq_local + w_global_pos * node_eq_global  # + eps_pos_reg * w_reg
                eps_node = w_local_node * node_score_local + w_global_node * node_score_global

                # Update

                sampling_type = kwargs.get("sampling_type", 'ddpm_noisy')  # types: generalized, ddpm_noisy, ld

                noise = center_pos(torch.randn_like(pos) + torch.randn_like(pos), batch)
                noise_node = torch.randn_like(atom_type) + torch.randn_like(
                    atom_type)  # center_pos(torch.randn_like(pos), batch)
                b = self.betas
                t = t[0]
                next_t = (torch.ones(1) * j).to(pos.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())

                if sampling_type == 'generalized':
                    eta = kwargs.get("eta", 1.)
                    et = -eps_pos

                    c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                    c2 = ((1 - at_next) - c1 ** 2).sqrt()

                    step_size_pos_ld = step_lr * (sigmas[i] / 0.01) ** 2 / sigmas[i]
                    step_size_pos_generalized = 1 * ((1 - at).sqrt() / at.sqrt() - c2 / at_next.sqrt())
                    step_size_pos = step_size_pos_ld if step_size_pos_ld < step_size_pos_generalized \
                        else step_size_pos_generalized

                    step_size_noise_ld = torch.sqrt((step_lr * (sigmas[i] / 0.01) ** 2) * 2)
                    step_size_noise_generalized = 10 * (c1 / at_next.sqrt())
                    step_size_noise = step_size_noise_ld if step_size_noise_ld < step_size_noise_generalized \
                        else step_size_noise_generalized

                    w = 1

                    pos_next = pos - et * step_size_pos + w * noise * step_size_noise
                    atom_next = atom_type - eps_node * step_size_pos + w * noise_node * step_size_noise
                elif sampling_type == 'ddpm_noisy':
                    atm1 = at_next
                    beta_t = 1 - at / atm1
                    e = -eps_pos

                    mean = (pos - beta_t * e) / (1 - beta_t).sqrt()
                    mask = 1 - (t == 0).float()
                    logvar = beta_t.log()
                    pos_next = mean + mask * torch.exp(
                        0.5 * logvar) * noise  # torch.exp(0.5 * logvar) = σ pos_next = μ+z*σ

                    e = eps_node
                    node0_from_e = (1.0 / at).sqrt() * atom_type - (1.0 / at - 1).sqrt() * e
                    mean_eps = (
                                       (atm1.sqrt() * beta_t) * node0_from_e + (
                                       (1 - beta_t).sqrt() * (1 - atm1)) * atom_type
                               ) / (1.0 - at)
                    mean = mean_eps
                    mask = 1 - (t == 0).float()
                    logvar = beta_t.log()
                    atom_next = mean + mask * torch.exp(
                        0.5 * logvar) * noise_node  # torch.exp(0.5 * logvar) = σ pos_next = μ+z*σ
                elif sampling_type == 'ld':
                    step_size = step_lr * (sigmas[i] / 0.01) ** 2
                    pos_next = pos + step_size * eps_pos / sigmas[i] + noise * torch.sqrt(step_size * 2)
                    eps_node = eps_node / (1 - at).sqrt()
                    atom_next = atom_type - step_size * eps_node / sigmas[i] + noise_node * torch.sqrt(step_size * 2)

                pos = pos_next
                atom_type = atom_next
                # atom_type = atom_next

                if torch.isnan(pos).any():
                    print('NaN detected. Please restart.')
                    # print(node_eq_local)
                    # print(node_eq_global)
                    raise FloatingPointError()
                pos = center_pos(pos, batch)
                if clip_pos is not None:
                    pos = torch.clamp(pos, min=-clip_pos, max=clip_pos)
                pos_traj.append(pos.clone().cpu())
                atom_traj.append(atom_type.clone().cpu())
            # atom_f = atom_type[:, :-1] * 4
            # atom_charge =  torch.round(atom_type[:, -1:] * 10).long()
            atom_type = torch.cat([atom_type[:, :-1] * 4, atom_type[:, -1:] * 10], dim=1)
        return pos, pos_traj, atom_type, atom_traj



    def scaffold_genOpt(self, atom_type, pos_init, bond_index, bond_type, batch, scaffold_mask, num_graphs,
                        n_steps=100, step_lr=0.0000010, clip=1000, clip_local=None, clip_pos=None,
                        global_start_sigma=float('inf'), local_start_sigma=float('inf'), w_global_pos=1,
                        w_global_node=1, w_local_pos=1, w_local_node=1, **kwargs):

        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)  # .view(-1, 1, 1, 1)
            return a

        sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        pos_traj = []
        atom_traj = []

        with torch.no_grad():
            skip = self.num_timesteps // n_steps
            # seq = range(self.num_timesteps - n_steps, self.num_timesteps)
            seq = range(0, self.num_timesteps, skip)

            seq_next = [-1] + list(seq[:-1])

            """-----------------"""
            opt_mask = ~scaffold_mask
            print("pos_init", pos_init)
            print("scaffold_mask", scaffold_mask)

            scaffold_pos = pos_init[scaffold_mask, :]

            pos = pos_init
            opt_pos = pos_init[opt_mask, :]

            scaffold_batch = batch[scaffold_mask]


            # pos = center_pos(pos_init, batch)
            pos = center_pos(pos, batch)
            pos[opt_mask] = opt_pos

            original_atom_type = atom_type.clone()


            # VAE noise
            vae_noise = torch.zeros_like(atom_type).uniform_(-1, +1)

            original_pos = pos.clone()

            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc='sample'):
                t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=pos.device)

                b = self.betas
                at = compute_alpha(b, t[0].long())

                pos_noise = torch.randn(size=pos[scaffold_mask, :].size(), device=pos.device)
                atom_noise = torch.randn(size=atom_type[scaffold_mask, :].size(), device=atom_type.device)

                mask = 1 - (t[0] == 0).float()

                scaffold_pos = pos[scaffold_mask, :]
                scaffold_atom_type = atom_type[scaffold_mask, :]


                scaffold_atom_perturbed = at.sqrt() * scaffold_atom_type + (1.0 - at).sqrt() * atom_noise * mask
                atom_type[scaffold_mask] = scaffold_atom_perturbed

                net_out= self.net(
                    atom_type=atom_type,
                    pos=pos,
                    bond_index=bond_index,
                    bond_type=bond_type,
                    batch=batch,
                    time_step=t,
                    vae_noise=vae_noise

                )  # (E_global, 1), (E_local, 1)


                edge_inv_global, edge_inv_local, node_score_global, node_score_local, \
                edge_index, _, edge_length, local_edge_mask, _ = net_out
                # Local float('inf')
                # import random
                # local_start_sigma = random.uniform(0.1, 1)
                if sigmas[i] < local_start_sigma:
                    node_eq_local = eq_transform(edge_inv_local, pos, edge_index[:, local_edge_mask],
                                                 edge_length[local_edge_mask])
                    if clip_local is not None:
                        node_eq_local = clip_norm(node_eq_local, limit=clip_local)
                else:
                    node_eq_local = 0
                    node_score_local = 0

                # Global
                if sigmas[i] < global_start_sigma:
                    edge_inv_global = edge_inv_global * (1 - local_edge_mask.view(-1, 1).float())
                    node_eq_global = eq_transform(edge_inv_global, pos, edge_index, edge_length)
                    node_eq_global = clip_norm(node_eq_global, limit=clip)
                else:
                    node_eq_global = 0
                    node_score_global = 0
                # Sum
                eps_pos = w_local_pos * node_eq_local + w_global_pos * node_eq_global  # + eps_pos_reg * w_reg
                eps_node = w_local_node * node_score_local + w_global_node * node_score_global

                # Update

                sampling_type = kwargs.get("sampling_type", 'ddpm_noisy')  # types: generalized, ddpm_noisy, ld

                noise = center_pos(torch.randn_like(pos) + torch.randn_like(pos), batch)
                noise_node = torch.randn_like(atom_type) + torch.randn_like(
                    atom_type)  # center_pos(torch.randn_like(pos), batch)
                # b = self.betas

                t = t[0]
                next_t = (torch.ones(1) * j).to(pos.device)
                # at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())

                if sampling_type == 'generalized':
                    eta = kwargs.get("eta", 1.)
                    et = -eps_pos

                    c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                    c2 = ((1 - at_next) - c1 ** 2).sqrt()

                    step_size_pos_ld = step_lr * (sigmas[i] / 0.01) ** 2 / sigmas[i]
                    # 这里在PMDM中是3, MDM是1
                    step_size_pos_generalized = 1 * ((1 - at).sqrt() / at.sqrt() - c2 / at_next.sqrt())
                    step_size_pos = step_size_pos_ld if step_size_pos_ld < step_size_pos_generalized \
                        else step_size_pos_generalized

                    step_size_noise_ld = torch.sqrt((step_lr * (sigmas[i] / 0.01) ** 2) * 2)
                    # 这里在PMDM中是5, MDM是10
                    step_size_noise_generalized = 10 * (c1 / at_next.sqrt())
                    step_size_noise = step_size_noise_ld if step_size_noise_ld < step_size_noise_generalized \
                        else step_size_noise_generalized

                    w = 1

                    pos_next = pos - et * step_size_pos + w * noise * step_size_noise
                    atom_next = atom_type - eps_node * step_size_pos + w * noise_node * step_size_noise

                elif sampling_type == 'ddpm_noisy':
                    atm1 = at_next
                    beta_t = 1 - at / atm1
                    e = -eps_pos

                    mean = (pos - beta_t * e) / (1 - beta_t).sqrt()
                    mask = 1 - (t == 0).float()
                    logvar = beta_t.log()
                    pos_next = mean + mask * torch.exp(
                        0.5 * logvar) * noise  # torch.exp(0.5 * logvar) = σ pos_next = μ+z*σ

                    e = eps_node
                    node0_from_e = (1.0 / at).sqrt() * atom_type - (1.0 / at - 1).sqrt() * e
                    mean_eps = (
                                       (atm1.sqrt() * beta_t) * node0_from_e + (
                                       (1 - beta_t).sqrt() * (1 - atm1)) * atom_type
                               ) / (1.0 - at)
                    mean = mean_eps
                    mask = 1 - (t == 0).float()
                    logvar = beta_t.log()
                    atom_next = mean + mask * torch.exp(
                        0.5 * logvar) * noise_node  # torch.exp(0.5 * logvar) = σ pos_next = μ+z*σ
                elif sampling_type == 'ld':
                    step_size = step_lr * (sigmas[i] / 0.01) ** 2
                    pos_next = pos + step_size * eps_pos / sigmas[i] + noise * torch.sqrt(step_size * 2)
                    eps_node = eps_node / (1 - at).sqrt()
                    atom_next = atom_type - step_size * eps_node / sigmas[i] + noise_node * torch.sqrt(step_size * 2)

                pos = pos_next
                atom_type = atom_next
                # atom_type = atom_next

                if torch.isnan(pos).any():
                    print('NaN detected. Please restart.')
                    print(node_eq_local)
                    print(node_eq_global)
                    raise FloatingPointError()
                pos[scaffold_mask] = scaffold_pos
                atom_type[scaffold_mask] = scaffold_atom_type

                pos = center_pos(pos, batch)

                if clip_pos is not None:
                    pos = torch.clamp(pos, min=-clip_pos, max=clip_pos)

                pos_traj.append(pos.clone().cpu())
                atom_traj.append(atom_type.clone().cpu())
            # atom_f = atom_type[:, :-1] * 4
            # atom_charge =  torch.round(atom_type[:, -1:] * 10).long()
            atom_type = torch.cat([atom_type[:, :-1] * 4, atom_type[:, -1:] * 10], dim=1)
        return pos, pos_traj, atom_type, atom_traj


def is_bond(edge_type):
    return torch.logical_and(edge_type < len(BOND_TYPES), edge_type > 0)


def is_angle_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 1 - 1


def is_dihedral_edge(edge_type):
    return edge_type == len(BOND_TYPES) + 2 - 1


def is_radius_edge(edge_type):
    return edge_type == 0


# def is_radius_edge(edge_type):
#     return edge_type == 0

def is_local_edge(edge_type):
    return edge_type > 0
    # return edge_type == 0


def is_train_edge(edge_index, is_sidechain):
    if is_sidechain is None:
        return torch.ones(edge_index.size(1), device=edge_index.device).bool()
    else:
        is_sidechain = is_sidechain.bool()
        return torch.logical_or(is_sidechain[edge_index[0]], is_sidechain[edge_index[1]])


def regularize_bond_length(edge_type, edge_length, rng=5.0):
    mask = is_bond(edge_type).float().reshape(-1, 1)
    d = -torch.clamp(edge_length - rng, min=0.0, max=float('inf')) * mask
    return d


def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center


def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom
