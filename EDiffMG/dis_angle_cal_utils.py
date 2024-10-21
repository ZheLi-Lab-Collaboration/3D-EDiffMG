# -*- coding: utf-8 -*-
# @Time    : 2024/9/26 3:17
# @Author  : TkiChus
# @Email   : XU.Chao.TkiChus@gmail.com


import torch
import datetime
import time
from torch_geometric.nn import radius_graph

from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, subgraph
from torch_sparse import coalesce

def compute_bond_lengths(pos, edge_index):
    start_pos = pos[edge_index[0]]
    end_pos = pos[edge_index[1]]
    bond_lengths = torch.norm(start_pos - end_pos, dim=1)
    return bond_lengths

def compute_angle(vec1, vec2):
    """
    计算两个向量之间的夹角
    """

    dot_product = torch.sum(vec1 * vec2, dim=1)
    norm_v1 = torch.norm(vec1, dim=1)
    norm_v2 = torch.norm(vec2, dim=1)
    cos_theta = dot_product / (norm_v1 * norm_v2 + 1e-8)
    angle = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
    return angle


"""1.4版本"""

# def filter_angle_non_covalent_bonds_and_check_cyclopropyl(pos, edge_index, batch, cyclopropyl_tolerance=0.8,
#                                                           bond_length_threshold=1.6):
#     non_covalent_edges = []
#     cyclopropyl_detected = []
#
#     # 筛选出属于同一分子的边
#     batch_start, batch_end = batch[edge_index[0]], batch[edge_index[1]]
#     valid_bonds = batch_start == batch_end
#     merged_edge_index = edge_index[:, valid_bonds]
#
#     # 将 merged_edge_index 转换为稀疏邻接矩阵，加速邻接表查找
#     edge_index_coalesced, _ = coalesce(merged_edge_index, None, pos.size(0), pos.size(0))
#
#     # 向量化批处理
#     num_edges = edge_index_coalesced.size(1)
#
#     for idx in range(num_edges):
#         start_node, end_node = edge_index_coalesced[:, idx].tolist()
#
#         # 确保 start_node 始终小于 end_node，避免重复计算
#         if start_node >= end_node:
#             continue
#
#         # 查找 start_node 和 end_node 的共同邻居
#         neighbors_start = set(edge_index_coalesced[1, edge_index_coalesced[0] == start_node].tolist())
#         neighbors_end = set(edge_index_coalesced[1, edge_index_coalesced[0] == end_node].tolist())
#         common_neighbors = neighbors_start.intersection(neighbors_end)
#
#         if len(common_neighbors) == 0:
#             continue
#
#         # 计算距离
#         vec1 = pos[start_node] - pos[end_node]
#         dist_ab = torch.norm(vec1)
#
#         for neighbor in common_neighbors:
#             # 确保 neighbor > end_node，避免重复计算
#             if neighbor <= end_node:
#                 continue
#
#             # 计算三边向量
#             vec2 = pos[start_node] - pos[neighbor]
#             vec3 = pos[end_node] - pos[neighbor]
#
#             # 计算角度
#             angles = torch.tensor([
#                 compute_angle(vec1.unsqueeze(0), vec2.unsqueeze(0)).squeeze(0),
#                 compute_angle(-vec1.unsqueeze(0), vec3.unsqueeze(0)).squeeze(0),
#                 compute_angle(-vec2.unsqueeze(0), -vec3.unsqueeze(0)).squeeze(0)
#             ]).to(pos.device)
#
#             # 环丙基检测
#             angle_diffs = torch.abs(angles[:-1] - angles[1:])
#             if torch.all(angle_diffs <= cyclopropyl_tolerance):
#                 distances = torch.tensor([dist_ab, torch.norm(vec2), torch.norm(vec3)])
#
#                 if torch.all(distances < bond_length_threshold):
#                     # 添加所有环丙基的双向边
#                     cyclopropyl_detected.extend([
#                         [start_node, end_node, neighbor], [end_node, start_node, neighbor],
#                         [start_node, neighbor, end_node], [neighbor, start_node, end_node],
#                         [neighbor, end_node, start_node], [end_node, neighbor, start_node]
#                     ])
#
#             # 检测非共价键，确保双向边添加
#             max_angle = angles.max()
#             if max_angle == angles[0]:
#                 non_covalent_edges.extend([[neighbor, end_node], [end_node, neighbor]])
#             elif max_angle == angles[1]:
#                 non_covalent_edges.extend([[start_node, neighbor], [neighbor, start_node]])
#             else:
#                 non_covalent_edges.extend([[end_node, start_node], [start_node, end_node]])
#
#     # 转换为张量输出，确保双向边存在
#     non_covalent_edges = torch.tensor(non_covalent_edges, dtype=torch.long,
#                                       device=edge_index.device) if non_covalent_edges else torch.empty((0, 2),
#                                                                                                        dtype=torch.long,
#                                                                                                        device=edge_index.device)
#     cyclopropyl_detected = torch.tensor(cyclopropyl_detected, dtype=torch.long,
#                                         device=pos.device) if cyclopropyl_detected else torch.empty((0, 3),
#                                                                                                     dtype=torch.long,
#                                                                                                     device=pos.device)
#     return non_covalent_edges, cyclopropyl_detected

"""1.5版本"""
# def filter_angle_non_covalent_bonds_and_check_cyclopropyl(pos, edge_index, batch, cyclopropyl_tolerance=0.8,
#                                                           bond_length_threshold=1.6):
#     non_covalent_edges = []
#     cyclopropyl_detected = []
#
#     # 筛选出属于同一分子的边
#     batch_start, batch_end = batch[edge_index[0]], batch[edge_index[1]]
#     valid_bonds = batch_start == batch_end
#     merged_edge_index = edge_index[:, valid_bonds]
#
#     # 将 merged_edge_index 转换为稀疏邻接矩阵，加速邻接表查找
#     edge_index_coalesced, _ = coalesce(merged_edge_index, None, pos.size(0), pos.size(0))
#     num_edges = edge_index_coalesced.size(1)
#
#     # 预先计算所有边上的向量和距离
#     start_pos = pos[edge_index_coalesced[0]]  # 每条边起始点的坐标
#     end_pos = pos[edge_index_coalesced[1]]    # 每条边终止点的坐标
#     vec_ab = start_pos - end_pos              # 起始点到终止点的向量
#     dists_ab = torch.norm(vec_ab, dim=1)      # 每条边的距离，作为一维张量
#
#     # 处理每个边对的共同邻居
#     for idx in range(num_edges):
#         start_node, end_node = edge_index_coalesced[:, idx].tolist()
#
#         # 跳过不符合条件的边
#         if start_node >= end_node:
#             continue
#
#         # 获取 start_node 和 end_node 的共同邻居
#         neighbors_start = edge_index_coalesced[1, edge_index_coalesced[0] == start_node].tolist()
#         neighbors_end = edge_index_coalesced[1, edge_index_coalesced[0] == end_node].tolist()
#         common_neighbors = set(neighbors_start).intersection(neighbors_end)
#
#         if not common_neighbors:
#             continue
#
#         # 使用已经预先计算的边向量 vec_ab 和距离 dists_ab
#         vec1 = vec_ab[idx]   # 起始点到终止点的向量
#         dist_ab = dists_ab[idx]  # 已经计算好的距离
#
#         # 处理每一个共同邻居
#         for neighbor in common_neighbors:
#             if neighbor <= end_node:
#                 continue
#
#             # 计算涉及共同邻居的三边向量
#             vec2 = pos[start_node] - pos[neighbor]
#             vec3 = pos[end_node] - pos[neighbor]
#
#             # 计算角度
#             angles = torch.tensor([
#                 compute_angle(vec1.unsqueeze(0), vec2.unsqueeze(0)).squeeze(0),
#                 compute_angle(-vec1.unsqueeze(0), vec3.unsqueeze(0)).squeeze(0),
#                 compute_angle(-vec2.unsqueeze(0), -vec3.unsqueeze(0)).squeeze(0)
#             ]).to(pos.device)
#
#             # 检测非共价键
#             max_angle = angles.max()
#             if max_angle == angles[0]:
#                 non_covalent_edges.extend([[neighbor, end_node], [end_node, neighbor]])
#             elif max_angle == angles[1]:
#                 non_covalent_edges.extend([[start_node, neighbor], [neighbor, start_node]])
#             else:
#                 non_covalent_edges.extend([[end_node, start_node], [start_node, end_node]])
#
#             # 环丙基检测
#             angle_diffs = torch.abs(angles[:-1] - angles[1:])
#             if torch.all(angle_diffs <= cyclopropyl_tolerance):
#                 distances = torch.tensor([dist_ab, torch.norm(vec2), torch.norm(vec3)], device=pos.device)
#
#                 if torch.all(distances < bond_length_threshold):
#                     cyclopropyl_detected.extend([
#                         [start_node, end_node, neighbor], [end_node, start_node, neighbor],
#                         [start_node, neighbor, end_node], [neighbor, start_node, end_node],
#                         [neighbor, end_node, start_node], [end_node, neighbor, start_node]
#                     ])
#
#     # 转换为张量输出，确保双向边存在
#     non_covalent_edges = torch.tensor(non_covalent_edges, dtype=torch.long,
#                                       device=edge_index.device) if non_covalent_edges else torch.empty((0, 2),
#                                                                                                        dtype=torch.long,
#                                                                                                        device=edge_index.device)
#     cyclopropyl_detected = torch.tensor(cyclopropyl_detected, dtype=torch.long,
#                                         device=pos.device) if cyclopropyl_detected else torch.empty((0, 3),
#                                                                                                     dtype=torch.long,
#                                                                                                     device=pos.device)
#     return non_covalent_edges, cyclopropyl_detected

"""1.6版本"""
# def filter_angle_non_covalent_bonds_and_check_cyclopropyl(pos, edge_index, batch, cyclopropyl_tolerance=0.8,
#                                                           bond_length_threshold=1.6):
#     non_covalent_edges = []
#     cyclopropyl_detected = []
#
#     # 筛选出属于同一分子的边
#     valid_bonds = batch[edge_index[0]] == batch[edge_index[1]]
#     merged_edge_index = edge_index[:, valid_bonds]
#
#     # 将 merged_edge_index 转换为稀疏邻接矩阵，加速邻接表查找
#     edge_index_coalesced, _ = coalesce(merged_edge_index, None, pos.size(0), pos.size(0))
#     num_edges = edge_index_coalesced.size(1)
#
#     # 预先计算所有边上的向量和距离
#     start_pos = pos[edge_index_coalesced[0]]  # 每条边起始点的坐标
#     end_pos = pos[edge_index_coalesced[1]]    # 每条边终止点的坐标
#     vec_ab = start_pos - end_pos              # 起始点到终止点的向量
#     dists_ab = torch.norm(vec_ab, dim=1)      # 每条边的距离，作为一维张量
#
#     # 构建每个节点的邻居索引以避免重复查找，初始化 neighbor_dict
#     neighbor_dict = {}
#     for idx in range(num_edges):
#         start_node, end_node = edge_index_coalesced[:, idx].tolist()
#         if start_node not in neighbor_dict:
#             neighbor_dict[start_node] = set()
#         if end_node not in neighbor_dict:
#             neighbor_dict[end_node] = set()
#         neighbor_dict[start_node].add(end_node)
#         neighbor_dict[end_node].add(start_node)
#
#     # 处理每个边对的共同邻居
#     for idx in range(num_edges):
#         start_node, end_node = edge_index_coalesced[:, idx].tolist()
#
#         # 跳过不符合条件的边
#         if start_node >= end_node:
#             continue
#
#         # 检查字典中是否存在 start_node 和 end_node，并安全获取邻居
#         if start_node not in neighbor_dict or end_node not in neighbor_dict:
#             continue  # 如果某个节点不在邻居字典中，跳过该边
#
#         # 获取 start_node 和 end_node 的共同邻居
#         common_neighbors = neighbor_dict[start_node].intersection(neighbor_dict[end_node])
#         if not common_neighbors:
#             continue
#
#         # 使用已经预先计算的边向量 vec_ab 和距离 dists_ab
#         vec1 = vec_ab[idx]   # 起始点到终止点的向量
#         dist_ab = dists_ab[idx]  # 已经计算好的距离
#
#         # 处理每一个共同邻居
#         for neighbor in common_neighbors:
#             if neighbor <= end_node:
#                 continue
#
#             # 计算涉及共同邻居的三边向量
#             vec2 = pos[start_node] - pos[neighbor]
#             vec3 = pos[end_node] - pos[neighbor]
#
#             # 计算角度并矢量化
#             angles = torch.tensor([
#                 compute_angle(vec1.unsqueeze(0), vec2.unsqueeze(0)).squeeze(0),
#                 compute_angle(-vec1.unsqueeze(0), vec3.unsqueeze(0)).squeeze(0),
#                 compute_angle(-vec2.unsqueeze(0), -vec3.unsqueeze(0)).squeeze(0)
#             ], device=pos.device)
#
#             # 检测非共价键：这里改为判断最小角度（更严谨）
#             max_angle = angles.max()
#             if max_angle == angles[0]:
#                 non_covalent_edges.extend([[neighbor, end_node], [end_node, neighbor]])
#             elif max_angle == angles[1]:
#                 non_covalent_edges.extend([[start_node, neighbor], [neighbor, start_node]])
#             else:
#                 non_covalent_edges.extend([[end_node, start_node], [start_node, end_node]])
#             # 环丙基检测
#             angle_diffs = torch.abs(angles[:-1] - angles[1:])
#             if torch.all(angle_diffs <= cyclopropyl_tolerance):
#                 distances = torch.tensor([dist_ab, torch.norm(vec2), torch.norm(vec3)], device=pos.device)
#                 if torch.all(distances < bond_length_threshold):
#                     cyclopropyl_detected.append([start_node, end_node, neighbor])
#
#     # 转换为张量输出，确保双向边存在
#     non_covalent_edges = torch.tensor(non_covalent_edges, dtype=torch.long,
#                                       device=edge_index.device) if non_covalent_edges else torch.empty((0, 2),
#                                                                                                        dtype=torch.long,
#                                                                                                        device=edge_index.device)
#     cyclopropyl_detected = torch.tensor(cyclopropyl_detected, dtype=torch.long,
#                                         device=pos.device) if cyclopropyl_detected else torch.empty((0, 3),
#                                                                                                     dtype=torch.long,
#                                                                                                     device=pos.device)
#     return non_covalent_edges, cyclopropyl_detected

"""1.7版本"""
# from collections import defaultdict
# def filter_angle_non_covalent_bonds_and_check_cyclopropyl(pos, edge_index, batch, cyclopropyl_tolerance=0.8,
#                                                           bond_length_threshold=1.6):
#     non_covalent_edges = []
#     cyclopropyl_detected = []
#
#     # 筛选出属于同一分子的边
#     valid_bonds = batch[edge_index[0]] == batch[edge_index[1]]
#     merged_edge_index = edge_index[:, valid_bonds]
#
#     # 将 merged_edge_index 转换为稀疏邻接矩阵，加速邻接表查找
#     edge_index_coalesced, _ = coalesce(merged_edge_index, None, pos.size(0), pos.size(0))
#     num_edges = edge_index_coalesced.size(1)
#
#     # 预先计算所有边上的向量和距离
#     start_pos = pos[edge_index_coalesced[0]]  # 每条边起始点的坐标
#     end_pos = pos[edge_index_coalesced[1]]    # 每条边终止点的坐标
#     vec_ab = start_pos - end_pos              # 起始点到终止点的向量
#     dists_ab = torch.norm(vec_ab, dim=1)      # 每条边的距离
#
#     # 构建邻居索引字典，并同时计算 start_node 和 end_node 的共同邻居
#     neighbor_dict = defaultdict(set)
#     for idx in range(num_edges):
#         start_node, end_node = edge_index_coalesced[:, idx].tolist()
#         neighbor_dict[start_node].add(end_node)
#         neighbor_dict[end_node].add(start_node)
#
#     # 处理每个边对的共同邻居
#     for idx in range(num_edges):
#         start_node, end_node = edge_index_coalesced[:, idx].tolist()
#
#         if start_node >= end_node:
#             continue
#
#         common_neighbors = neighbor_dict[start_node].intersection(neighbor_dict[end_node])
#         if not common_neighbors:
#             continue
#
#         vec1 = vec_ab[idx]
#         dist_ab = dists_ab[idx]
#
#         for neighbor in common_neighbors:
#             if neighbor <= end_node:
#                 continue
#
#             vec2 = pos[start_node] - pos[neighbor]
#             vec3 = pos[end_node] - pos[neighbor]
#
#             # 批量计算角度
#             angles = torch.stack([
#                 compute_angle(vec1.unsqueeze(0), vec2.unsqueeze(0)).squeeze(0),
#                 compute_angle(-vec1.unsqueeze(0), vec3.unsqueeze(0)).squeeze(0),
#                 compute_angle(-vec2.unsqueeze(0), -vec3.unsqueeze(0)).squeeze(0)
#             ]).to(pos.device)
#
#             # 根据最大角度判定非共价键
#             max_angle_idx = torch.argmax(angles)
#             if max_angle_idx == 0:
#                 non_covalent_edges.extend([[neighbor, end_node], [end_node, neighbor]])
#             elif max_angle_idx == 1:
#                 non_covalent_edges.extend([[start_node, neighbor], [neighbor, start_node]])
#             else:
#                 non_covalent_edges.extend([[end_node, start_node], [start_node, end_node]])
#
#
#             # 环丙基检测
#             angle_diffs = torch.abs(angles[:-1] - angles[1:])
#             if torch.all(angle_diffs <= cyclopropyl_tolerance):
#                 distances = torch.tensor([dist_ab, torch.norm(vec2), torch.norm(vec3)], device=pos.device)
#                 if torch.all(distances < bond_length_threshold):
#                     cyclopropyl_detected.extend([
#                         [start_node, end_node, neighbor],
#                         [start_node, neighbor, end_node],
#                         [end_node, start_node, neighbor],
#                         [end_node, neighbor, start_node],
#                         [neighbor, start_node, end_node],
#                         [neighbor, end_node, start_node]
#                     ])
#     # 转换为张量输出，确保双向边存在
#     non_covalent_edges = torch.tensor(non_covalent_edges, dtype=torch.long,
#                                       device=edge_index.device) if non_covalent_edges else torch.empty((0, 2),
#                                                                                                        dtype=torch.long,
#                                                                                                        device=edge_index.device)
#     cyclopropyl_detected = torch.tensor(cyclopropyl_detected, dtype=torch.long,
#                                         device=pos.device) if cyclopropyl_detected else torch.empty((0, 3),
#                                                                                                     dtype=torch.long,
#                                                                                                     device=pos.device)
#     return non_covalent_edges, cyclopropyl_detected

"""1.9版本"""
def filter_angle_non_covalent_bonds_and_check_cyclopropyl(pos, edge_index, batch, cyclopropyl_tolerance=0.8,
                                                          bond_length_threshold=1.6):
    # 筛选属于同一分子的边
    valid_bonds = batch[edge_index[0]] == batch[edge_index[1]]
    merged_edge_index = edge_index[:, valid_bonds]

    # 合并边以删除重复项并排序
    edge_index_coalesced, _ = coalesce(merged_edge_index, None, pos.size(0), pos.size(0))


    # 构建邻接列表
    num_nodes = pos.size(0)
    adj_list = [[] for _ in range(num_nodes)]
    edges = edge_index_coalesced.t()
    for u, v in edges.tolist():
        adj_list[u].append(v)
        adj_list[v].append(u)

    # 三角形枚举
    triangles = []
    for u in range(num_nodes):
        neighbors_u = adj_list[u]
        neighbors_u_set = set(neighbors_u)
        for v in neighbors_u:
            if v > u:
                neighbors_v = adj_list[v]
                common_neighbors = neighbors_u_set.intersection(neighbors_v)
                for w in common_neighbors:
                    if w > v:
                        triangles.append([u, v, w])
    if not triangles:
        return torch.empty((0, 2), dtype=torch.long, device=pos.device), torch.empty((0, 3), dtype=torch.long, device=pos.device)


    triangles = torch.tensor(triangles, dtype=torch.long, device=pos.device)
    u = triangles[:, 0]
    v = triangles[:, 1]
    w = triangles[:, 2]

    # 计算向量
    vec_uv = pos[u] - pos[v]
    vec_uw = pos[u] - pos[w]
    vec_vw = pos[v] - pos[w]

    # 计算角度
    angles = torch.stack([
        compute_angle(vec_uv, vec_uw),
        compute_angle(-vec_uv, vec_vw),
        compute_angle(-vec_uw, -vec_vw)
    ], dim=1)  # 形状：[num_triangles, 3]

    # 根据最大角度确定非共价键
    max_angle_idx = torch.argmax(angles, dim=1)

    # 准备非共价键的索引
    non_covalent_edges = []
    idx0 = max_angle_idx == 0
    idx1 = max_angle_idx == 1
    idx2 = max_angle_idx == 2

    if idx0.any():
        edges0 = torch.stack([w[idx0], v[idx0]], dim=1)
        edges0_rev = torch.stack([v[idx0], w[idx0]], dim=1)
        non_covalent_edges.append(edges0)
        non_covalent_edges.append(edges0_rev)

    if idx1.any():
        edges1 = torch.stack([u[idx1], w[idx1]], dim=1)
        edges1_rev = torch.stack([w[idx1], u[idx1]], dim=1)
        non_covalent_edges.append(edges1)
        non_covalent_edges.append(edges1_rev)

    if idx2.any():
        edges2 = torch.stack([v[idx2], u[idx2]], dim=1)
        edges2_rev = torch.stack([u[idx2], v[idx2]], dim=1)
        non_covalent_edges.append(edges2)
        non_covalent_edges.append(edges2_rev)

    if non_covalent_edges:
        non_covalent_edges = torch.cat(non_covalent_edges, dim=0)
    else:
        non_covalent_edges = torch.empty((0, 2), dtype=torch.long, device=edge_index.device)

    # 环丙基检测
    angle_diffs = torch.abs(angles[:, :-1] - angles[:, 1:])
    distances = torch.stack([
        torch.norm(vec_uv, dim=1),
        torch.norm(vec_uw, dim=1),
        torch.norm(vec_vw, dim=1)
    ], dim=1)

    cyclopropyl_mask = torch.all(angle_diffs <= cyclopropyl_tolerance, dim=1) & \
                       torch.all(distances < bond_length_threshold, dim=1)

    cyclopropyl_detected = []
    if cyclopropyl_mask.any():
        u_c = u[cyclopropyl_mask]
        v_c = v[cyclopropyl_mask]
        w_c = w[cyclopropyl_mask]

        # 生成所有排列以确保双向性
        cyclo_nodes = torch.stack([u_c, v_c, w_c], dim=1)
        permutations = torch.tensor([
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0]
        ], device=pos.device)

        cyclo_nodes = cyclo_nodes[:, permutations].reshape(-1, 3)
        cyclopropyl_detected.append(cyclo_nodes)

    if cyclopropyl_detected:
        cyclopropyl_detected = torch.cat(cyclopropyl_detected, dim=0)
    else:
        cyclopropyl_detected = torch.empty((0, 3), dtype=torch.long, device=pos.device)

    return non_covalent_edges, cyclopropyl_detected





# 这个可以不加batch

# def filter_angle_length(filter_length_edge_index, filter_angle_non_covalent_edges_t):
#     # 转置 A 和 B，使其形状为 (N, 2)
#     filter_length_edge_index_t = filter_length_edge_index.t()  # (N, 2)
#
#     # 对 A_t 和 B_t 中的每条边进行排序，使得每条边的第一个元素小于等于第二个元素
#     filter_length_edge_index_sorted, _ = torch.sort(filter_length_edge_index_t, dim=1)
#     filter_angle_non_covalent_edges_sorted, _ = torch.sort(filter_angle_non_covalent_edges_t, dim=1)
#
#     # unique_B_sorted = torch.unique(filter_angle_non_covalent_edges_sorted, dim=0)
#
#     A_expanded = filter_length_edge_index_sorted.unsqueeze(1)  # (N_A, 1, 2)
#     B_expanded = filter_angle_non_covalent_edges_sorted.unsqueeze(0)  # (1, N_B, 2)
#
#     # 使用广播机制比较 A_sorted 中的每条边是否存在于 B_sorted 中
#     # (N_A, 2) 与 (N_B, 2) 比较
#     matches = (A_expanded == B_expanded).all(dim=2)  # (N_A, N_B)
#     # print("match", matches)
#     # 创建掩码，标记 A 中不在 B 中的边
#     mask = ~matches.any(dim=1)  # 结果为 (56,)
#
#     # 使用掩码过滤 A_t，保留不在 B 中的边
#     filter_length_angle = filter_length_edge_index_t[mask]
#
#     return filter_length_angle

"""1.2版本"""
def filter_angle_length(filter_length_edge_index, filter_angle_non_covalent_edges_t):
    # 将边转置为 (N, 2) 形式
    filter_length_edge_index_t = filter_length_edge_index.t()  # (N, 2)

    # 对每条边进行排序，确保双向边的一致性
    filter_length_edge_index_sorted, _ = torch.sort(filter_length_edge_index_t, dim=1)
    filter_angle_non_covalent_edges_sorted, _ = torch.sort(filter_angle_non_covalent_edges_t, dim=1)

    # 使用哈希编码避免逐一比较，确保效率
    max_node_idx = filter_length_edge_index_sorted.max() + 1
    filter_length_edge_hash = filter_length_edge_index_sorted[:, 0] * max_node_idx + filter_length_edge_index_sorted[:, 1]
    filter_angle_non_covalent_edges_hash = filter_angle_non_covalent_edges_sorted[:, 0] * max_node_idx + filter_angle_non_covalent_edges_sorted[:, 1]

    # 过滤不在 filter_angle_non_covalent_edges 中的边
    mask = ~torch.isin(filter_length_edge_hash, filter_angle_non_covalent_edges_hash)

    # 直接返回过滤后的边
    filter_length_angle = filter_length_edge_index_t[mask]

    return filter_length_angle



def filter_real_non_covalent_edges(filter_angle_non_covalent_edges, cyclopropyl_edges):
    # 将两个输入张量按行排序，确保边的顺序一致
    filter_angle_non_covalent_edges_sorted, _ = torch.sort(filter_angle_non_covalent_edges, dim=1)
    cyclopropyl_edges_sorted, _ = torch.sort(cyclopropyl_edges, dim=1)

    # 对边进行哈希编码，确保双向边被视为同一条边
    max_node_idx = max(filter_angle_non_covalent_edges_sorted.max(), cyclopropyl_edges_sorted.max()) + 1
    filter_angle_non_covalent_edges_hash = filter_angle_non_covalent_edges_sorted[:, 0] * max_node_idx + filter_angle_non_covalent_edges_sorted[:, 1]
    cyclopropyl_edges_hash = cyclopropyl_edges_sorted[:, 0] * max_node_idx + cyclopropyl_edges_sorted[:, 1]

    # 使用 torch.isin 来找到不在 cyclopropyl_edges 中的边
    isin_mask = torch.isin(filter_angle_non_covalent_edges_hash, cyclopropyl_edges_hash)

    # 取反，保留不属于 cyclopropyl_edges 的边
    real_non_covalent_edges = filter_angle_non_covalent_edges[~isin_mask]

    return real_non_covalent_edges


#
# def filter_real_non_covalent_edges(filter_angle_non_covalent_edges_t, cyclopropyl_edges_t):
#
#     # 找出 a 中不在 b 中的行（即列）
#     mask = torch.ones(filter_angle_non_covalent_edges_t.size(0), dtype=torch.bool, device=filter_angle_non_covalent_edges_t.device)
#
#     for i in range(cyclopropyl_edges_t.size(0)):
#         mask &= ~((filter_angle_non_covalent_edges_t[:, 0] == cyclopropyl_edges_t[i, 0]) &
#                   (filter_angle_non_covalent_edges_t[:, 1] == cyclopropyl_edges_t[i, 1]))
#     # 取出 a 中没有出现在 b 中的列
#     real_non_covalent_edges = filter_angle_non_covalent_edges_t[mask]
#
#     return real_non_covalent_edges



def filter_global_section_mask(edge_index, non_covalent_edges_t):

    # 转置 edge_index 和 non_covalent_edges
    edge_index_t = edge_index.t()

    # 使用矩阵运算来比较所有边
    matches = (edge_index_t.unsqueeze(1) == non_covalent_edges_t.unsqueeze(0)).all(dim=2)

    # 如果任何一个非共价边匹配，则将 mask 置为 True
    mask = matches.any(dim=1)
    return mask


# def filter_global_section_mask(edge_index, non_covalent_edges):
#     # 转置 edge_index
#     edge_index_t = edge_index.t()
#
#     # 将 edge_index 和 non_covalent_edges 转换为行向量的集合表示
#     edge_index_set = {tuple(edge.tolist()) for edge in edge_index_t}
#     non_covalent_set = {tuple(edge.tolist()) for edge in non_covalent_edges}
#
#     # 直接判断每一条边是否在非共价键集合中
#     mask = torch.tensor([tuple(edge) in non_covalent_set for edge in edge_index_set], device=edge_index.device)
#
#     return mask


def filter_bonds_by_length(bond_lengths, edge_index, min_length=0.9, max_length=2.05):
    valid_bonds = (bond_lengths >= min_length) & (bond_lengths <= max_length)
    new_edge_index = edge_index[:, valid_bonds]
    return new_edge_index

