import os

import numpy as np
from tqdm import tqdm


input_folder = 'E:/桌面/Old'
output_folder = 'E:/桌面/Old - 副本'
npz_files =  [f for f in os.listdir(input_folder) if f.endswith('.npz')]

for npz_file in tqdm(npz_files):
    npz_path = os.path.join(input_folder, npz_file)
    npz = np.load(npz_path)
    pcd = npz['points']
    print(pcd.shape)

    min_coords = np.ones_like(pcd)*np.min(pcd, axis=0)
    max_coords = np.ones_like(pcd)*np.max(pcd, axis=0)

    coords_range = max_coords - min_coords

    max_coords_range = np.max(coords_range, axis=1)

    pcd_nmlzed = (pcd - min_coords)/max_coords_range[0]

    npz_path = os.path.join(output_folder, npz_file)
    np.savez(npz_path, points=pcd_nmlzed)


    print(" ")