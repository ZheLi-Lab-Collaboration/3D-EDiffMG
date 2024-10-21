# -*- coding: utf-8 -*-
# @Time    : 2024/6/28 16:10
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com

from configs.datasets_config import get_dataset_info
from utils.datasets import QM93D, Geom
from utils.transforms import *
from torch_geometric.data import DataLoader
import numpy as np

torch.set_printoptions(threshold=np.inf)
dataset_info = get_dataset_info('qm9', remove_h=False)
transforms = Compose([CountNodesPerGraph(), GetAdj(), AtomFeat(dataset_info['atom_index'])])

val_set = QM93D('valid', pre_transform=transforms)

val_loader = DataLoader(val_set, 1, shuffle=True)
print("val_set", val_loader)
for batch in val_loader:

    atom_type = batch.atom_feat_full.float()
    pos = batch.pos
    bond_index = batch.edge_index
    bond_type = batch.edge_type
    batch1 = batch.batch
    print("----------------------------------------------------")
    print("atom_type", atom_type)
    print("pos", pos)
    print("bond_index", bond_index)
    print("bond_type", bond_type)
    print("batch1", batch1)
    break



