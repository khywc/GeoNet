from torch.utils.data import Dataset
import numpy as np
import glob
import trimesh
import pandas as pd
import os
import json
import torch
import numpy as np
from io import StringIO

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from sklearn.neighbors import KDTree


def compute_normals(points, k=10):
    """
    计算点云的法向量。

    参数:
    points: ndarray, 点云坐标 (N, 3)
    k: int, 邻域大小，用于计算法向量

    返回:
    normals: ndarray, 每个点的法向量 (N, 3)
    """
    # 创建KDTree，方便查找邻域
    kdtree = KDTree(points)

    # 用于存储法向量
    normals = np.zeros_like(points)

    for i in range(points.shape[0]):
        # 找到当前点的k个邻居（包括自己）
        _, idx = kdtree.query(points[i:i + 1], k=k)
        neighbors = points[idx[0]]  # 邻域点

        # 将邻域点中心化
        neighbors_centered = neighbors - np.mean(neighbors, axis=0)

        # 计算协方差矩阵
        cov_matrix = np.dot(neighbors_centered.T, neighbors_centered)

        # 通过SVD分解获取法向量（最小特征值对应的特征向量）
        _, _, vh = np.linalg.svd(cov_matrix)
        normal = vh[-1]  # 最小特征值对应的特征向量

        # 保存法向量
        normals[i] = normal

    return normals


# 示例: 读取点坐标并计算法向量
def load_points_and_compute_normals(mesh_file):
    # 读取点数据（过滤掉不需要的非数值行）
    points = []
    with open(mesh_file, 'r') as f:
        for line in f:
            if line.startswith('v '):  # 只处理顶点行
                parts = line.split()
                points.append([float(parts[1]), float(parts[2]), float(parts[3])])  # 解析 x, y, z

    points = np.array(points)
    # print(points.shape)

    # 计算法向量
    normals = compute_normals(points)

    return points, normals


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc, centroid, m


class Mesh_Dataset(Dataset):
    def __init__(self, data_list_path, labels_dir, num_classes=16, patch_size=7000):
        self.data_list = pd.read_csv(data_list_path, header=None)
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.labels_dir = labels_dir

    def __len__(self):
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        try:
            if torch.is_tensor(idx):
                idx = idx.tolist()

            mesh_file = self.data_list.iloc[idx][0]
            # print(mesh_file)
            points, normals = load_points_and_compute_normals(mesh_file)

            points_normals = np.concatenate((points, normals), axis=-1)
            rawpoints = points_normals

            points_normals[:, 0:3], c, m = pc_normalize(points_normals[:, 0:3])
            points_normals[:, 3:6], _, _ = pc_normalize(points_normals[:, 3:6])

            label_file = glob.glob(
                os.path.join(self.labels_dir,
                             f'**/{os.path.basename(mesh_file).replace(".obj", ".txt")}')) + glob.glob(
                os.path.join(self.labels_dir, f'{os.path.basename(mesh_file).replace(".obj", ".txt")}'))

            if not label_file:
                raise FileNotFoundError(f"Label file not found for {mesh_file}")

            label_file = label_file[0]

            with open(label_file, 'r', encoding='utf-8') as file:
                data = json.load(file)

            label_values = data.get('labels', [])

            # 根据映射关系替换label_values的键
            replacement_dict = {11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7,
                                21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14}  # 将标签为18和28的设置为牙龈的标签

            # 使用字典替换
            new_label_values = [replacement_dict.get(val, val) for val in label_values]

            labels = np.array(new_label_values).reshape(-1, 1)

            points = points_normals[:, :3]
            # 求质心坐标和标签

            temp_lab = labels[:, 0]
            lab_tooth_dict = {}
            for i in range(15):
                lab_tooth_dict[i] = []
            for i, lab in enumerate(temp_lab):
                lab_tooth_dict[lab].append(list(points[i]))
            barycenter = np.zeros([15, 3])
            for k, v in lab_tooth_dict.items():
                if v == []:
                    continue
                temp = np.array(lab_tooth_dict[k])
                barycenter[k] = temp.mean(axis=0)
            barycenter_label = np.zeros([15, ])
            for i, j in enumerate(barycenter_label):
                barycenter_label[i] = 1
                if barycenter[i][0] == 0 and barycenter[i][1] == 0 and barycenter[i][2] == 0:
                    barycenter_label[i] = 0
            barycenter_label = barycenter_label[1:]
            barycenter = barycenter[1:]
            barycenter_label = barycenter_label.reshape(-1, 1)  # (15, 1)

            X = points_normals.transpose(1, 0)
            Y = labels.transpose(1, 0)
            barycenter_label = barycenter_label.transpose(1, 0)

            sample = {
                'cells': torch.from_numpy(X).float(),
                'labels': torch.from_numpy(Y).long(),
                'mesh_file': mesh_file,
                'c': c,
                'm': m,
                'barycenter': torch.from_numpy(barycenter).float(),
                'barycenter_label': torch.from_numpy(barycenter_label).long(),
                'rawpoints': rawpoints
            }

            return sample

        except Exception as e:
            print(f"Error loading sample at index {idx}: {e}")
            raise e