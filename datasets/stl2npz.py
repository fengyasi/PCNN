# import os
# import trimesh
# import numpy as np
# from tqdm import tqdm
#
# # 定义输入文件夹路径和输出文件夹路径
# input_folder = '/media/zyxie/8b54bbcd-9be7-4c38-8f97-9396ceef7d1b/zyxie/Documents/EVAR/非内漏'
# output_folder = '/media/zyxie/8b54bbcd-9be7-4c38-8f97-9396ceef7d1b/zyxie/Documents/EVAR/非内漏npz'
# # 获取输入文件夹中的所有STL文件
# stl_files = [f for f in os.listdir(input_folder) if f.endswith('.stl')]
# # 遍历每个STL文件进行转换
# for stl_file in tqdm(stl_files):
#     new_name = os.path.splitext(stl_file)[0].split("_")[3] #zhuyuanhao
#     stl_path = os.path.join(input_folder, stl_file)
#     # 使用trimesh加载STL文件
#     mesh = trimesh.load_mesh(stl_path)
#     # 提取顶点信息
#     vertices = mesh.vertices
#     # 将顶点转换为NumPy数组
#     vertices_array = np.array(vertices, dtype=np.float32)
#     # 创建保存NPZ文件的路径
#     npz_file = new_name + '.npz'
#     npz_path = os.path.join(output_folder, npz_file)
#     # 保存顶点的NumPy数组到NPZ文件
#     np.savez(npz_path, points=vertices_array)

import os
import trimesh
import numpy as np
from tqdm import tqdm
# 定义输入文件夹路径和输出文件夹路径
# input_folder = '/media/zyxie/8b54bbcd-9be7-4c38-8f97-9396ceef7d1b/zyxie/Documents/EVAR/EVAR/非内漏'
input_folder = r'E:\桌面\非内漏'
output_folder = r'E:\桌面\非内漏 - 副本'
# 获取输入文件夹中的所有STL文件
stl_files = [f for f in os.listdir(input_folder) if f.endswith('.stl')]
# 遍历每个STL文件进行转换
for stl_file in tqdm(stl_files):
    new_name = os.path.splitext(stl_file)[0].split("_")[3] #zhuyuanhao
    npz_file = new_name + '.npz'
    npz_path = os.path.join(output_folder, npz_file)
    # 检查输出文件夹中是否已存在同名NPZ文件
    if os.path.exists(npz_path):
        continue  # 如果文件已存在，则跳过当前循环
    stl_path = os.path.join(input_folder, stl_file)
    # 使用trimesh加载STL文件
    mesh = trimesh.load_mesh(stl_path)
    # 提取顶点信息
    vertices = mesh.vertices
    # 将顶点转换为NumPy数组
    vertices_array = np.array(vertices, dtype=np.float32)
    # 保存顶点的NumPy数组到NPZ文件
    np.savez(npz_path, points=vertices_array)