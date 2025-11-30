from datetime import datetime

from models.utils import *
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
from sklearn.decomposition import PCA

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'######################################################################33#####3333

class get_model(nn.Module):
    def __init__(self, num_class=1, use_pcd=True, use_text=False, normal_channel=3 ** 3, feat_channel=0):
        super(get_model, self).__init__()
        in_channel = normal_channel if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.sa4 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa5 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa6 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.sa7 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa8 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa9 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024*3+feat_channel, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)
        # self.fc1 = nn.Linear(1024+feat_channel, 256)
        # self.bn1 = nn.BatchNorm1d(256)
        # self.drop1 = nn.Dropout(0.4)
        # self.fc2 = nn.Linear(256, 64)
        # self.bn2 = nn.BatchNorm1d(64)
        # self.drop2 = nn.Dropout(0.5)
        # self.fc3 = nn.Linear(64, num_class)
        self.use_pcd = use_pcd
        self.use_text = use_text

    def forward(self, xyz, feat=None):
        B, _, _ = xyz.shape
        # print('xyz.shape',xyz.shape)
        if self.normal_channel:
            norm1 = xyz[:, 3:4, :]
            norm2 = xyz[:, 4:5, :]
            norm3 = xyz[:, 5:6, :]
            # print(f'norm1 shape: {norm1.shape}, norm2 shape: {norm2.shape}, norm3 shape: {norm3.shape}')
            norm1 = norm1.repeat(1, 3, 1)
            norm2 = norm2.repeat(1, 3, 1)
            norm3 = norm3.repeat(1, 3, 1)
            xyz = xyz[:, :3, :]
        else:
            norm1 = norm2 = norm3 = None  # 确保在 else 分支中 norm1, norm2, norm3 不会为 None
        # if B == 1:
        #     norm1 = norm1.unsqueeze(0)
        #     norm2 = norm2.unsqueeze(0)
        #     norm3 = norm3.unsqueeze(0)
        #     xyz = xyz.unsqueeze(0)

        # print(f'xyz shape: {xyz.shape}, norm1 shape: {norm1.shape}, norm2 shape: {norm2.shape}, norm3 shape: {norm3.shape}')

        l1_xyz1, l1_points1 = self.sa1(xyz, norm1)
        l2_xyz1, l2_points1 = self.sa2(l1_xyz1, l1_points1)
        l3_xyz1, l3_points1 = self.sa3(l2_xyz1, l2_points1)
        x1 = l3_points1.view(B, 1024)
        l1_xyz2, l1_points2 = self.sa4(xyz, norm2)
        l2_xyz2, l2_points2 = self.sa5(l1_xyz2, l1_points2)
        l3_xyz2, l3_points2 = self.sa6(l2_xyz2, l2_points2)
        x2 = l3_points2.view(B, 1024)
        l1_xyz3, l1_points3 = self.sa7(xyz, norm3)
        l2_xyz3, l2_points3 = self.sa8(l1_xyz3, l1_points3)
        l3_xyz3, l3_points3 = self.sa9(l2_xyz3, l2_points3)
        x3 = l3_points3.view(B, 1024)
        x = torch.cat([x1, x2, x3], dim=1)  # torch.Size([B,3072])
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x



