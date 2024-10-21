import logging
import os
import urllib
import urllib.request
from os.path import join as join

import numpy as np

from process import process_xyz_zinc10w, process_xyz_files_zinc



def process_dataset_zinc(datadir, dataname, splits=None):
    """
    Download and prepare the QM9 (GDB9) dataset.
    """
    # Define directory for which data will be output.

    # zinc_dir = join(*[datadir, dataname])
    #
    # # Important to avoid a race condition
    # os.makedirs(zinc_dir, exist_ok=True)


    # If splits are not specified, automatically generate them.
    if splits is None:
        splits = gen_splits_ZINC10w()

    # Process GDB9 dataset, and return dictionary of splits
    zinc_data = {}
    for split, split_idx in splits.items():
        zinc_data[split] = process_xyz_files_zinc(
            datadir, process_xyz_zinc10w, file_idx_list=split_idx, stack=True)

    zinc_dir = join(*[datadir, dataname])

    # Important to avoid a race condition
    os.makedirs(zinc_dir, exist_ok=True)

    print(
        ' processing zinc dataset. Output will be in directory: {}.'.format(zinc_dir))

    # Save processed zinc data into train/validation/test splits
    logging.info('Saving processed data:')
    for split, data in zinc_data.items():
        savedir = join(zinc_dir, split + '.npz')
        np.savez_compressed(savedir, **data)

    print('Processing/saving complete!')



def gen_splits_ZINC10w():
    """
    Generate training/validation/test splits for the ZINC10W dataset.

    The dataset is split into:
    - Training set: 44431 molecules
    - Test set: 5554 molecules
    - Validation set: 5554 molecules

    Parameters
    ----------
    zinc10wdir : str
        Directory where ZINC10W data is located (not used here, but kept for consistency).

    Returns
    -------
    splits : dict
        Dictionary containing 'train', 'valid', and 'test' molecule indices as numpy arrays.
    """
    # Number of molecules in the dataset
    # Nmols = 55539  # 44431 + 5554 + 5554
    Nmols = 92825 # 43904 + 5488 + 5488

    # Define the split sizes
    Ntrain = 74259

    Ntest = 9283

    Nvalid = Nmols - (Ntrain + Ntest)  # This should be 11108

    # Generate a random permutation of molecule indices
    np.random.seed(0)
    data_perm = np.random.permutation(Nmols)

    # Split the data into train, valid, and test sets

    train, valid, test = np.split(data_perm, [Ntrain, Ntrain + Nvalid])

    # Return the splits as numpy arrays
    splits = {
        'train': train,
        'valid': valid,
        'test': test
    }

    return splits


# import os
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit.Chem.Scaffolds import MurckoScaffold
# import numpy as np
# from collections import defaultdict
#
#
# def xyz_to_mol(xyz_file):
#     """
#     将XYZ文件转换为RDKit分子对象。
#
#     参数
#     ----------
#     xyz_file : str
#         XYZ文件的路径。
#
#     返回
#     -------
#     mol : rdkit.Chem.rdchem.Mol or None
#         对应的RDKit分子对象。如果转换失败，返回None。
#     """
#     with open(xyz_file, 'r') as f:
#         lines = f.readlines()
#
#     # XYZ文件的第3行及其之后包含分子结构信息
#     xyz_block = ''.join(lines[2:])
#
#     try:
#         mol = Chem.MolFromXYZBlock(xyz_block)
#         if mol is not None:
#             AllChem.Compute2DCoords(mol)
#         return mol
#     except Exception as e:
#         print(f"无法解析文件 {xyz_file}: {e}")
#         return None
#
#
# def generate_scaffold(smiles, include_chirality=False):
#     """
#     生成给定分子的Bemis-Murcko骨架（Scaffold）。
#
#     参数
#     ----------
#     smiles : str
#         分子的SMILES字符串。
#     include_chirality : bool, optional
#         是否在生成骨架时包括手性信息。
#
#     返回
#     -------
#     scaffold : str
#         分子的骨架SMILES字符串。
#     """
#     mol = Chem.MolFromSmiles(smiles)
#     scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
#     return scaffold
#
#
# def gen_splits_scaffold_ZINC10w(xyz_dir):
#     """
#     使用基于骨架（scaffold）划分的方式生成ZINC10W数据集的训练/验证/测试集。
#
#     数据集划分为：
#     - 训练集：43904个分子
#     - 测试集：5488个分子
#     - 验证集：5488个分子
#
#     参数
#     ----------
#     xyz_dir : str
#         存放ZINC10W数据集的XYZ文件夹路径。
#
#     返回
#     -------
#     splits : dict
#         包含'训练集'（train）、'验证集'（valid）、'测试集'（test）分子索引的字典，格式为numpy数组。
#     """
#     Nmols = 54880  # 数据集的总分子数量
#     Ntrain = 43904  # 训练集大小
#     Ntest = 5488  # 测试集大小
#     Nvalid = Ntest  # 验证集大小
#
#     # 解析XYZ文件夹中的所有分子，并生成每个分子的骨架
#     scaffold_to_indices = defaultdict(list)
#     smiles_list = []
#
#     xyz_files = [f for f in os.listdir(xyz_dir) if f.endswith('.xyz')]
#
#     for i, xyz_file in enumerate(xyz_files):
#         xyz_path = os.path.join(xyz_dir, xyz_file)
#         mol = xyz_to_mol(xyz_path)
#         if mol is not None:
#             smiles = Chem.MolToSmiles(mol)
#             smiles_list.append(smiles)
#             scaffold = generate_scaffold(smiles)
#             scaffold_to_indices[scaffold].append(i)
#
#     # 将所有骨架随机打乱，确保随机性
#     np.random.seed(0)
#     scaffold_sets = list(scaffold_to_indices.values())
#     np.random.shuffle(scaffold_sets)
#
#     # 按照骨架分组划分数据集
#     train_idx, valid_idx, test_idx = [], [], []
#     for scaffold_group in scaffold_sets:
#         if len(train_idx) + len(scaffold_group) <= Ntrain:
#             train_idx.extend(scaffold_group)
#         elif len(valid_idx) + len(scaffold_group) <= Nvalid:
#             valid_idx.extend(scaffold_group)
#         else:
#             test_idx.extend(scaffold_group)
#
#     # 返回的划分格式与原代码保持一致
#     splits = {
#         'train': np.array(train_idx),
#         'valid': np.array(valid_idx),
#         'test': np.array(test_idx)
#     }
#
#     return splits
