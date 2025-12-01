import torch
import numpy as np
import open3d as o3d
import os.path as osp

import tqdm as tqdm
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import xlrd
import nibabel as nib


def to_one_hot(num, length):
    # example: to_one_hot(10,4)
    # transfer 10 to one-hot vector with length 4
    num = bin(int(num)).replace('0b', '')
    vector = np.zeros(length)
    for i in range(1, len(num) + 1):
        vector[-i] = float(num[-i])
    return vector



complete_nd_list = []#['lusbp', 'ludp', 'rusbp', 'rudp', 'ldsbp', 'lddp', 'rdsbp', 'rddp'] # include ND
complete_blank = []#['hemoglobin', 'WBC', 'PLT', 'IL_6', 'IL_2R', 'ESR', 'CRP', 'Kerr', 'hormone'] # include blank
normalize_list = [] + complete_blank + complete_nd_list
binarized_list = {}#{'age': 8, 'age_diagnosis': 8, 'course': 10, 's2d': 9, }
label_list = []#['vision_disorder', 'cerebral_infarction', 'syncope', 'SIE'] # label

class EVARVessel(Dataset):
    def __init__(self, num_points=8192, phase='train',
                 root='./data/EVARnpz',############################################
                 threshold=None, resample=True, category=2, scale=0.05,
                 trans=True, rot=True, factor=10, imbalance=True, sample_rate=1.0,randomseed = 70,
                 task = 'EL'): # 'EL1' 'EL2'
        self.gaussian_noise = 0.01
        assert phase in ['train0', 'test0', 'train1', 'test1', 'train2', 'test2', 'train3', 'test3', 'train4', 'test4','test5']####5fold
        self.phase = phase
        self.num_points = num_points
        self.root = os.path.join(root)

        self.threshold = threshold
        self.resample = resample  # shi fou chong cai yang
        self.category = category  #
        self.scale = scale  # sui ji fang da
        self.trans = trans  # shi fou ping yi
        self.rot = rot  # shi fou xuan zhuan
        self.factor = factor  # xuan zhuan jiao fan wei
        self.randomseed = randomseed
        self.task = task

        self.init_data()##############lin chuang shu ju

        self.names = np.load('./data/seed{}/{}.npy'.format(self.randomseed, self.phase))
        self.names = self.names.tolist()
        # print(self.names)##################################################
        self.names_copy = []
        for item in self.names:
            item = item.split('.npz')[0]
            self.names_copy.append(item)
        self.names = self.names_copy
        print(len(self.names))
        self.files_copy = {}
        for name in self.files:
            if name in self.names:
                self.files_copy[name] = self.files[name]
        self.files = self.files_copy
        print(len(self.files))

        print('load {} data'.format(len(self.files)) )

        self.imbalance = imbalance
        self.names = list(self.files.keys())
        names = list(self.files.keys())
        if self.imbalance and 'train' in self.phase:
            self.names = []
            self.sample_pro_list = {}
            print('struggle to solve imbalance')
            for name in names:
                label = self.files[name][self.task]
                self.sample_pro_list[name] = label

            pro_0 = 0
            pro_1 = 0
            for k in self.sample_pro_list:
                if self.sample_pro_list[k] < 0.5:
                    pro_0 += 1
                else:
                    pro_1 += 1
            pro_0 = pro_0 / len(self.sample_pro_list)
            pro_1 = pro_1 / len(self.sample_pro_list)

            for k in self.sample_pro_list:
                if self.sample_pro_list[k] < 0.5:
                    self.sample_pro_list[k] = pro_0
                else:
                    self.sample_pro_list[k] = pro_1

            for k in self.sample_pro_list:
                if self.sample_pro_list[k] > min(pro_0, pro_1):
                    self.names.append(k)
                else:
                    for i in range(int(sample_rate * np.around(max(pro_0 // pro_1, pro_1 // pro_0)))):
                        self.names.append(k)

    def init_data(self):
        excel_dir = './data/EVARlabel.xls'
        wb = xlrd.open_workbook(excel_dir)
        sh = wb.sheet_by_name('Sheet1')
        total_list = {}
        for i in tqdm(range(1, sh.nrows)):
            total_list[str(int(sh.cell(i, 0).value))] = {} # 病历号作为key################################################
            total_list[str(int(sh.cell(i, 0).value))]['EL'] = sh.cell(i, 1).value
            total_list[str(int(sh.cell(i, 0).value))]['EL1'] = sh.cell(i, 2).value
            total_list[str(int(sh.cell(i, 0).value))]['EL2'] = sh.cell(i, 3).value

        # normalize and complete data and binarized
        # complete ND with mean
        ND_mean = {}
        for key in complete_nd_list:
            ND_mean[key] = []
            for name in total_list.keys():
                if total_list[name][key] != 'ND':# not ND
                    ND_mean[key].append(total_list[name][key])
        for key in ND_mean.keys():
            ND_mean[key] = np.mean(ND_mean[key])
        for name in total_list.keys():
            for key in total_list[name].keys():
                if key in complete_nd_list:
                    if total_list[name][key] == 'ND':
                        total_list[name][key] = ND_mean[key]

        # complete blank with mean
        Blank_mean = {}
        for key in complete_blank:
            Blank_mean[key] = []
            for name in total_list.keys():
                if total_list[name][key] != '':
                    Blank_mean[key].append(total_list[name][key])
        for key in Blank_mean.keys():
            Blank_mean[key] = np.mean(Blank_mean[key])
        for name in total_list.keys():
            for key in total_list[name].keys():
                if key in complete_blank:
                    if total_list[name][key] == '':
                        total_list[name][key] = Blank_mean[key]

        # normalize
        Norm_mean = {}
        Norm_std = {}
        for key in normalize_list:
            Norm_mean[key] = []
            Norm_std[key] = []
            for name in total_list.keys():
                Norm_mean[key].append(total_list[name][key])
                Norm_std[key].append(total_list[name][key])
        for key in Norm_mean.keys():
            Norm_mean[key] = np.mean(Norm_mean[key])
            Norm_std[key] = np.std(Norm_std[key])
        for name in total_list.keys():
            for key in total_list[name].keys():
                if key in normalize_list:
                    total_list[name][key] = (total_list[name][key] - Norm_mean[key]) / Norm_std[key]

        # binarized
        for name in total_list.keys():
            for key in total_list[name].keys():
                if key in binarized_list:
                    total_list[name][key] = to_one_hot(total_list[name][key], binarized_list[key])

        for _, _, files in os.walk(self.root):
            break

        names = {}
        for item in files:
            name = item.split('.npz')[0]
            names[name] = item

        self.files = {}
        for name in total_list.keys():
            if name in names:
                self.files[name] = total_list[name]
                self.files[name]['root'] = names[name]
            else:
                print(name)

    def __getitem__(self, item):
        name = self.names[item]
        label = self.files[name][self.task]
        # feature = self.files[name]['feature']

        data_root = os.path.join(self.root, self.files[name]['root'])
        data= np.load(data_root)
        # pcd = data['points']

        if True:
            pcd = data['points']
            idx = np.random.randint(0, len(pcd), self.num_points)
            np.random.shuffle(idx)
            pcd = pcd[idx]
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data['points'])
            pcd = pcd.farthest_point_down_sample(self.num_points)
            pcd = np.array(pcd.points)

        pcd = np.concatenate([pcd, np.ones_like(pcd)], axis=-1)###################################check

        return pcd.T, np.array(label).reshape(-1).astype(np.float32)

    def __len__(self):
        return len(self.names)

##################################################################################################################
##################################################################################################################
##################################################################################################################
# 注意_getitem_
class Vessel_PCNNradiomics_4fold(Dataset):
    def __init__(self, feature_list,
                 num_points=8192, phase='train',
                 root='./data_CFD/preprocessed',############################################根目录，数据名称应当与xls中的患者名称保持一致
                 threshold=None, resample=True, category=2,  # 二分类
                 scale=0.05, 
                 trans=True, rot=True, factor=10, imbalance=True, sample_rate=1.0,feature_num = 25,  randomseed = 60,### 这里为什么是30，在CFDPCNN最开始的dataset中，调用时显示这里是25
                 task = 'EL'):  ######################
        self.gaussian_noise = 0.01
        assert phase in ['train0', 'test0', 'train1', 'test1', 'train2', 'test2', 'train3', 'test3', 'train4', 'test4','test5']####5fold
        self.phase = phase
        self.num_points = num_points
        self.root = os.path.join(root)

        self.threshold = threshold
        self.resample = resample  # shi fou chong cai yang
        self.category = category  #
        self.scale = scale  # sui ji fang da
        self.trans = trans  # shi fou ping yi
        self.rot = rot  # shi fou xuan zhuan
        self.factor = factor  # xuan zhuan jiao fan wei
        self.feature_num = feature_num
        self.randomseed = randomseed
        self.task = task#################################

        self.init_data()##############lin chuang shu ju
        self.init_feature(feature_list)
        self.names = np.load('./data_CFD/seed{}/{}.npy'.format(self.randomseed, self.phase))  # 名字
        self.names = self.names.tolist()
        # print(len(self.names))###########################################################################################################
        self.names_copy = []
        for item in self.names:
            item = item.split('.npz')[0]
            self.names_copy.append(item)
        self.names = self.names_copy
        # print(len(self.names))
        self.files_copy = {}
        for name in self.files:
            if name in self.names:
                self.files_copy[name] = self.files[name]
        self.files = self.files_copy
        # print(len(self.files))

        print('load {} data'.format(len(self.files)) )#########################################

        self.imbalance = imbalance
        self.names = list(self.files.keys())
        names = list(self.files.keys())
        if self.imbalance and 'train' in self.phase:
            self.names = []
            self.sample_pro_list = {}
            print('struggle to solve imbalance')
            for name in names:
                label = self.files[name][self.task]
                self.sample_pro_list[name] = label
 
            pro_0 = 0
            pro_1 = 0
            for k in self.sample_pro_list:
                if self.sample_pro_list[k] < 0.5:
                    pro_0 += 1
                else:
                    pro_1 += 1
            pro_0 = pro_0 / len(self.sample_pro_list)
            pro_1 = pro_1 / len(self.sample_pro_list)

            for k in self.sample_pro_list:
                if self.sample_pro_list[k] < 0.5:
                    self.sample_pro_list[k] = pro_0
                else:
                    self.sample_pro_list[k] = pro_1

            for k in self.sample_pro_list:
                if self.sample_pro_list[k] > min(pro_0, pro_1):
                    self.names.append(k)
                else:
                    for i in range(int(sample_rate * np.around(max(pro_0 // pro_1, pro_1 // pro_0)))):
                        self.names.append(k)


    def init_data(self):
        randomseed = int(str(self.randomseed)[:2])
        # print(randomseed)
        excel_dir = './data/radiomics_data/seed{}/radiomics_selected{}.xls'.format(randomseed, self.feature_num)  #####################radiomics feature xls# 选中的影像组学特征有几个
        wb = xlrd.open_workbook(excel_dir)
        sh = wb.sheet_by_name('labeled')
        total_list = {}
        # 获取表格的总行数和总列数
        num_rows = sh.nrows
        num_cols = sh.ncols
        print(f"Total rows: {num_rows}, Total cols: {num_cols}")
        # for i in tqdm(range(1, sh.nrows)):
            # total_list[str(int(sh.cell(i, 0).value))] = {}
            # total_list[str(int(sh.cell(i, 0).value))]['EL'] = sh.cell(i, 1).value  ###########################
            # total_list[str(int(sh.cell(i, 0).value))]['EL1'] = sh.cell(i, 2).value  ###########################这里要改，现在的keys是患者名字不是病例号
            # total_list[str(int(sh.cell(i, 0).value))]['EL2'] = sh.cell(i, 3).value  ###########################
        for i in tqdm(range(1, sh.nrows)):
            total_list[sh.cell(i, 1).value] = {}
            total_list[sh.cell(i, 1).value]['EL'] = int(sh.cell(i, 2).value)  # label一列
            for j in range(self.feature_num):
                feature_index: str = 'feature' + str(j)
                total_list[sh.cell(i, 1).value][feature_index] = sh.cell(i, 3 + j).value
        # print((total_list))###############################################################################
        for _, _, files in os.walk(self.root):
            break

        # print(files)

        names = {}
        for item in files:
            name = item.split('.npz')[0]
            names[name] = item
        

        self.files = {}
        for name in total_list.keys():
            if name in names:
                self.files[name] = total_list[name]
                self.files[name]['root'] = names[name]
            else:
                print(name)###################################
        # print(self.files)
        # normalize and complete data and binarized
        # complete ND with mean
        ND_mean = {}
        for key in complete_nd_list:
            ND_mean[key] = []
            for name in total_list.keys():
                if total_list[name][key] != 'ND':# not ND
                    ND_mean[key].append(total_list[name][key])
        for key in ND_mean.keys():
            ND_mean[key] = np.mean(ND_mean[key])
        for name in total_list.keys():
            for key in total_list[name].keys():
                if key in complete_nd_list:
                    if total_list[name][key] == 'ND':
                        total_list[name][key] = ND_mean[key]

        # complete blank with mean
        Blank_mean = {}
        for key in complete_blank:
            Blank_mean[key] = []
            for name in total_list.keys():
                if total_list[name][key] != '':
                    Blank_mean[key].append(total_list[name][key])
        for key in Blank_mean.keys():
            Blank_mean[key] = np.mean(Blank_mean[key])
        for name in total_list.keys():
            for key in total_list[name].keys():
                if key in complete_blank:
                    if total_list[name][key] == '':
                        total_list[name][key] = Blank_mean[key]

        # normalize
        Norm_mean = {}
        Norm_std = {}
        for key in normalize_list:
            Norm_mean[key] = []
            Norm_std[key] = []
            for name in total_list.keys():
                Norm_mean[key].append(total_list[name][key])
                Norm_std[key].append(total_list[name][key])
        for key in Norm_mean.keys():
            Norm_mean[key] = np.mean(Norm_mean[key])
            Norm_std[key] = np.std(Norm_std[key])
        for name in total_list.keys():
            for key in total_list[name].keys():
                if key in normalize_list:
                    total_list[name][key] = (total_list[name][key] - Norm_mean[key]) / Norm_std[key]

        # binarized
        for name in total_list.keys():
            for key in total_list[name].keys():
                if key in binarized_list:
                    total_list[name][key] = to_one_hot(total_list[name][key], binarized_list[key])

        for _, _, files in os.walk(self.root):
            break

        names = {}
        for item in files:
            name = item.split('.npz')[0]
            names[name] = item

        self.files = {}
        for name in total_list.keys():
            if name in names:
                self.files[name] = total_list[name]
                self.files[name]['root'] = names[name]
            else:
                print(name)#########################################

    def init_feature(self, feature_list):
        for name in self.files:
            self.files[name]['label'] = (self.files[name]['EL'] > 0) + 0.0

            self.files[name]['feature'] = []
            for key in feature_list:
                if type(self.files[name][key]) == np.ndarray:
                    self.files[name]['feature'] += (self.files[name][key]).tolist()
                else:
                    self.files[name]['feature'].append(self.files[name][key])
            self.files[name]['feature'] = np.array(self.files[name]['feature'])
        print('feature length is', len(self.files[name]['feature']))############################################


    def __getitem__(self, item):######################################################################################
        name = self.names[item]
        label = self.files[name]['EL']  # 是否内漏
        feature = self.files[name]['feature']#######################################

        data_root = os.path.join(self.root, self.files[name]['root'])
        data= np.load(data_root)
        print(data)

        if True:
            # pcd = data['points']
            pcd = data['data']
            idx = np.random.randint(0, len(pcd), self.num_points)
            np.random.shuffle(idx)
            pcd = pcd[idx]
        # else:
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(data['points'])
        #     pcd = pcd.farthest_point_down_sample(self.num_points)
        #     pcd = np.array(pcd.points)

        pcd = np.concatenate([pcd, np.ones_like(pcd)], axis=-1)

        return pcd.T, np.array(label).reshape(-1).astype(np.float32), feature.astype(np.float32)###################################### point label feat
#########################################################################################################################################

    def __len__(self):
        return len(self.names)
################################

if __name__ == "__main__":
    feature_list = [f'feature{i}' for i in range(25)]
    dataset = Vessel_PCNNradiomics_4fold(feature_list=feature_list, scale=0.05, rot=True, trans=True, phase='train', category=2, imbalance=True,randomseed=60)
    # dataset = Vessel_PCNNradiomics_4fold(scale=0.05, rot=True, trans=True, phase='train0', category=2, imbalance=True, randomseed=3, task='EL')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        # collate_fn=collate_fn_vessel,
        drop_last=False
    )
    # # for point, label, feats in tqdm(dataloader):
###########################################################
    # for point, label in tqdm(dataloader):
    #     point = point[:,:3,:]
    #     # 将张量转换为numpy数组
    #     point_cloud_np = point.squeeze().numpy()
    #     # 创建Open3D点云数据结构
    #     point_cloud_o3d = o3d.geometry.PointCloud()
    #     point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_np.T)
    #     # 可选：设置点云的颜色
    #     point_cloud_o3d.paint_uniform_color([1, 0, 0])  # 设置为红色
    #     # 可选：设置点云的大小
    #     point_cloud_o3d.estimate_normals()
    #     # 可视化点云
    #     o3d.visualization.draw_geometries([point_cloud_o3d])
    #     # 创建一个绘图窗口
    #     vis = o3d.visualization.Visualizer()
    #     # 将点云添加到绘图窗口
    #     vis.create_window()
    #     # 添加点云到窗口
    #     vis.add_geometry(point_cloud_o3d)
    #     # 渲染可视化
    #     vis.update_geometry(point_cloud_o3d)
    #     # vis.poll_events()
    #     vis.update_renderer()
    #     vis.destroy_window()

    #     point_cloud_o3d.paint_uniform_color([1,0.706,0])
##########################################################################
        # 获取渲染结果并显示在Notebook中
        # image = vis.capture_screen_float_buffer()
        # plt.imshow(image)
        # plt.axis('off')
        # plt.show()