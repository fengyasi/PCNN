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
        self.fc1 = nn.Linear(1024+feat_channel, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)
        self.use_pcd = use_pcd
        self.use_text = use_text

    def forward(self, xyz, feat=None):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)

        if self.use_pcd and self.use_text:
            feat = self.mlp(feat)
            x = (torch.cat([feat, x], dim=-1))

        if self.use_pcd and (not self.use_text):
            x = x

        if self.use_text and (not self.use_pcd):
            # x = self.mlp(feat)
            x = feat

        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = torch.sigmoid(x)


        return x


