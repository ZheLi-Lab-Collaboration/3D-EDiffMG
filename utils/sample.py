from typing import List

import numpy as np
import torch
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch_geometric.data import Data
from torch_geometric.utils import degree
from tqdm import tqdm


# 后边我们可以设定为参数，让用户自己来去设计 
n_nodes = {31: 1, 32: 1, 33: 1, 34: 1, 35: 1, 36: 1, 37: 1, 38: 1, 39: 1, 40: 1, 41: 1, 42: 1, 43: 1, 44: 1, 45: 1, 46: 1, 47: 1, 48: 1, 49: 1, 50: 1}



def adjacent_matrix(n_particles):
    rows, cols = [], []
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            rows.append(i)
            cols.append(j)
            rows.append(j)
            cols.append(i)
    # print(n_particles)
    rows = torch.LongTensor(rows).unsqueeze(0)
    cols = torch.LongTensor(cols).unsqueeze(0)
    # print(rows.size())
    adj = torch.cat([rows, cols], dim=0)
    return adj


def unbatch(src: Tensor, batch: Tensor, dim: int = 0) -> List[Tensor]:
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)

    :rtype: :class:`List[Tensor]`
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)


class DistributionNodes:
    def __init__(self, histogram):
        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])
        self.n_nodes = torch.tensor(self.n_nodes)
        # prob = np.array(prob)

        prob = np.array(prob, dtype=np.float32)
        # 引入平滑或均匀化处理
        prob += 1e-2  # 添加一个小常数，避免概率为零

        prob = prob / np.sum(prob)

        self.prob = torch.from_numpy(prob).float()






        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs


class DistributionProperty:
    def __init__(self, dataloader, properties, num_bins=1000, normalizer=None):
        self.num_bins = num_bins
        self.distributions = {}
        self.properties = properties
        for prop in properties:
            self.distributions[prop] = {}
            self._create_prob_dist(dataloader.dataset.data['num_atoms'],
                                   dataloader.dataset.data[prop],
                                   self.distributions[prop])

        self.normalizer = normalizer

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _create_prob_dist(self, nodes_arr, values, distribution):
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {'probs': probs, 'params': params}

    def _create_prob_given_nodes(self, values):
        n_bins = self.num_bins  # min(self.num_bins, len(values))
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = torch.zeros(n_bins)
        for val in values:
            i = int((val - prop_min) / prop_range * n_bins)
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if tat happens)
            if i == n_bins:
                i = n_bins - 1
            histogram[i] += 1
        probs = histogram / torch.sum(histogram)
        probs = Categorical(torch.tensor(probs))
        params = [prop_min, prop_max]
        return probs, params

    def normalize_tensor(self, tensor, prop):
        # print(self.normalizer)
        assert self.normalizer is not None
        mean = self.normalizer[prop]['mean']
        mad = self.normalizer[prop]['mad']
        return (tensor - mean) / mad

    def sample(self, n_nodes=19):
        vals = []
        for prop in self.properties:
            dist = self.distributions[prop][n_nodes]
            idx = dist['probs'].sample((1,))
            val = self._idx2value(idx, dist['params'], len(dist['probs'].probs))
            val = self.normalize_tensor(val, prop)
            vals.append(val)
        vals = torch.cat(vals)
        return vals

    def sample_batch(self, nodesxsample):
        vals = []
        for n_nodes in nodesxsample:
            vals.append(self.sample(int(n_nodes)).unsqueeze(0))
        vals = torch.cat(vals, dim=0)
        return vals

    def _idx2value(self, idx, params, n_bins):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left
        return val


# n_nodes = {26: 4711, 31: 3365, 19: 3093, 22: 3344, 32: 3333, 25: 4533,
#            36: 1388, 23: 4375, 33: 2686, 29: 3242, 14: 2469, 28: 4838,
#            41: 630, 9: 1858, 18: 2621, 27: 5417, 10: 2865, 30: 3605,
#            42: 502, 13: 2164, 11: 3051, 21: 4493, 15: 2292, 12: 2900,
#            40: 691, 45: 184, 20: 4883, 24: 3716, 46: 213, 39: 752,
#            17: 2446, 16: 3094, 35: 1879, 38: 915, 44: 691, 43: 360,
#            50: 37, 8: 1041, 7: 655, 34: 2168, 47: 119, 49: 73, 6: 705,
#            37: 928, 51: 21, 4: 45, 48: 187, 5: 111, 52: 42, 54: 93,
#            56: 12, 57: 8, 55: 35, 71: 1, 61: 9, 58: 18, 59: 5, 67: 28,
#            3: 4, 65: 2, 63: 5, 62: 1, 86: 1, 66: 20,  106:2, 53: 3, 77: 1, 68: 1, 98: 1}
#
# qm9_noh_n_nodes = {22: 3393, 17: 13025, 23: 4848, 21: 9970, 19: 13832, 20: 9482, 16: 10644, 13: 3060,
#                    15: 7796, 25: 1506, 18: 13364, 12: 1689, 11: 807, 24: 539, 14: 5136, 26: 48, 7: 16, 10: 362,
#                    8: 49, 9: 124, 27: 266, 4: 4, 29: 25, 6: 9, 5: 5, 3: 1}
#
# MAX_NODES = 29


def construct_dataset(num_sample, batch_size, dataset_info, num_points=None, scaffold=None, ligand_data=None):
    # nodes_dist = DistributionNodes(dataset_info['n_nodes'])
    if scaffold is None:
        nodes_dist = DistributionNodes(dataset_info['n_nodes'])
    else:
        nodes_dist = DistributionNodes(dataset_info['n_nodes'])

    data_list = []

    num_atom = len(dataset_info['atom_decoder']) + 1  # charge
    # num_atom = 20
    nodesxsample_list = []


    num_node_frag = 0
    if scaffold is not None:
        # 这里也就是scaffold的原子数量
        print('scaffold atom number:', len(scaffold['element']))
        num_node_frag = len(scaffold['element'])
    if batch_size > num_sample:
        batch_size = num_sample

    for _ in tqdm(range(int(num_sample / batch_size))):
        datas = []

        if num_points is not None:
            if batch_size >= 100:
                nodesxsample = nodes_dist.sample(batch_size - 1).tolist()
                nodesxsample.append(num_points)
            else:
                nodesxsample = nodes_dist.sample(batch_size).tolist()
        else:
            nodesxsample = nodes_dist.sample(batch_size).tolist()

        nodesxsample_list.append(nodesxsample)
        # atom_type_list = torch.randn(batch_size,MAX_NODES,6)
        # pos_list = torch.randn(batch_size,MAX_NODES,3)
        for n_particles in nodesxsample:
            # n_particles = 19
            # atom_type = torch.randn(n_particles, num_atom)
            # pos = torch.randn(n_particles, 3)


            # 初始化 atom_type
            atom_noise_scale = np.random.uniform(3, 5)  # 在范围内随机选择
            pos_noise_scale = np.random.uniform(4, 6)  # 4-6

            # atom_noise_scale = 4  # 根据需要调整
            atom_type = torch.randn(n_particles, num_atom) * atom_noise_scale

            # 初始化 pos
            # pos_noise_scale = 5  # 根据需要调整
            pos = torch.randn(n_particles, 3) * pos_noise_scale

            # 添加噪声
            atom_type += torch.randn_like(atom_type) * 0.1
            pos += torch.randn_like(pos) * 0.1





            if scaffold is not None:
                print("atom_type", atom_type)
                atom_type_frag = torch.cat([scaffold['linker_atom_type'], atom_type])
                """
                atom_type_frag tensor([[ 0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  1.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  1.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.6825,  0.6159,  1.6541, -1.8866, -0.6502, -0.2590],
                [ 0.4133,  0.2616, -1.2374, -1.2068,  0.0204,  0.3438],
                [-0.1360, -0.6652, -2.4073,  1.1336, -0.3948, -0.9646],
                [ 0.1299, -1.4467,  0.3212, -0.7578, -1.0810, -0.8223]])

                """


                print("atom_type_frag", atom_type_frag)

                scaffold_mask = torch.cat(
                    [torch.ones(num_node_frag, dtype=torch.long), torch.zeros(n_particles, dtype=torch.long)])
                """
                scaffold_mask tensor([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])

                """
                print("scaffold_mask", scaffold_mask)
                # pos = scaffold['pos']
                # print("scaffold['pos']", scaffold['pos'])

                pos = torch.cat([scaffold['pos'], pos])
                print("scaffold['pos']", pos)


                # coors = pos
                adj = adjacent_matrix(n_particles + num_node_frag)
                # num_node = torch.tensor([n_particles + num_node_frag])
                data = Data(x=atom_type_frag, edge_index=adj, pos=pos, scaffold_mask=scaffold_mask)
                datas.append(data)
            else:
                adj = adjacent_matrix(n_particles)
                # num_node = torch.tensor([n_particles + num_node_frag])
                data = Data(x=atom_type, edge_index=adj, pos=pos)
                datas.append(data)


        data_list.append(datas)
    return data_list, nodesxsample_list

# def construct_dataset_pocket(num_sample, batch_size, dataset_info, num_points=None, *protein_information):
#     nodes_dist = DistributionNodes(dataset_info['n_nodes'])
#     data_list = []

#     num_atom = len(dataset_info['atom_decoder']) + 1  # charge
#     # num_atom = 20
#     nodesxsample_list = []
#     protein_atom_feature_full, protein_pos, protein_bond_index = protein_information
#     for n in tqdm(range(int(num_sample / batch_size))):
#         datas = []
#         if num_points is not None:
#             nodesxsample = nodes_dist.sample(batch_size - 1).tolist().append(num_points)
#         else:
#             nodesxsample = nodes_dist.sample(batch_size).tolist()
#         nodesxsample_list.append(nodesxsample)
#         # atom_type_list = torch.randn(batch_size,MAX_NODES,6)
#         # pos_list = torch.randn(batch_size,MAX_NODES,3)
#         for i, n_particles in enumerate(nodesxsample):
#             # n_particles = 19
#             # atom_type = torch.randn(n_particles, num_atom)
#             # pos = torch.randn(n_particles, 3)
#             atom_type = torch.zeros(n_particles, num_atom).uniform_(-3, +3)
#             pos = torch.zeros(n_particles, 3).uniform_(-3, +3)
#             coors = pos
#             adj = adjacent_matrix(n_particles)
#             data = Data(ligand_atom_type=atom_type, ligand_bond_index=adj, ligand_pos=pos,
#                         protein_atom_feature_full=protein_atom_feature_full, protein_pos=protein_pos,
#                         protein_bond_index=protein_bond_index)
#             datas.append(data)
#         data_list.append(datas)
#     return data_list, nodesxsample_list
