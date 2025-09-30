import argparse

import torch
import numpy as np
import open3d as o3d
import os.path as osp
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import xlrd
import nibabel as nib
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--randomseed',type=int,default=3,help='randomseed in split')
    opt = parser.parse_args()
    randomseed =opt.randomseed
    # randomseed =5
    # print(randomseed)

    fold = os.path.join('./data','seed'+str(randomseed))
    if not os.path.exists(fold):
        os.makedirs(fold)
        # print('success mkdir')

    data_root = './data/EVARnpz'
    excel_dir = './data/EVARlabel.xls'
    wb = xlrd.open_workbook(excel_dir)
    sh = wb.sheet_by_name('Sheet1')
    total_list = {}
    for i in tqdm(range(1, sh.nrows)):
        total_list[str(int(sh.cell(i, 0).value))] = {}
        total_list[str(int(sh.cell(i, 0).value))]['EL'] = sh.cell(i, 1).value

    names = {}
    for _, _, files in os.walk(data_root):
        break
    for item in files:
        name = item.split('.npz')[0]

        if name in total_list:
            names[name] = item
        else:
            print(name)
    # print(len(names))

    total_set = []
    for name in names:
        total_set.append(names[name])

    np.random.seed(randomseed)# 3 4
    # np.random.seed(4)
    np.random.shuffle(total_set)

    # seperate test ##################################################################################################
    frametest = int(len(total_set)*0.25)

    # 5 fold
    frame = (len(total_set) - frametest)//5
    set0 = total_set[:frame]
    set1 = total_set[frame:2*frame]
    set2 = total_set[2*frame:3*frame]
    set3 = total_set[3*frame:4*frame]
    set4 = total_set[4*frame:5*frame]
    # seperate test ##################################################################################################
    set5 = total_set[5*frame:len(total_set)]

    # seperate test ##################################################################################################
    test5 = set5

    train0 = set1 + set2 + set3 + set4
    test0 = set0

    train1 = set0 + set2 + set3 + set4
    test1 = set1

    train2 = set0 + set1 + set3 + set4
    test2 = set2

    train3 = set0 + set1 + set2 + set4
    test3 = set3

    train4 = set0 + set1 + set2 + set3
    test4 = set4



    total = []

    trainfold = os.path.join(fold,'train0.npy')
    testfold = os.path.join(fold, 'test0.npy')
    # print(trainfold)
    # print(testfold)
    np.save(trainfold, np.array(train0))
    np.save(testfold, np.array(test0))
    label_train0 = []
    label_test0 = []
    for item in train0:
        for name in names:
            if names[name] == item:
                break
        label_train0.append(total_list[name]['EL'])
    # print(np.mean(label_train0))
    total.append(np.mean(label_train0))
    for item in test0:
        for name in names:
            if names[name] == item:
                break
        label_test0.append(total_list[name]['EL'])
    # print(np.mean(label_test0))
    total.append(np.mean(label_test0))


    trainfold = os.path.join(fold,'train1.npy')
    testfold = os.path.join(fold, 'test1.npy')
    np.save(trainfold, np.array(train1))
    np.save(testfold, np.array(test1))
    label_train1 = []
    label_test1 = []
    for item in train1:
        for name in names:
            if names[name] == item:
                break
        label_train1.append(total_list[name]['EL'])
    # print(np.mean(label_train1))
    total.append(np.mean(label_train1))

    for item in test1:
        for name in names:
            if names[name] == item:
                break
        label_test1.append(total_list[name]['EL'])
    # print(np.mean(label_test1))
    total.append(np.mean(label_test1))


    trainfold = os.path.join(fold,'train2.npy')
    testfold = os.path.join(fold, 'test2.npy')
    np.save(trainfold, np.array(train2))
    np.save(testfold, np.array(test2))
    label_train2 = []
    label_test2 = []
    for item in train2:
        for name in names:
            if names[name] == item:
                break
        label_train2.append(total_list[name]['EL'])
    # print(np.mean(label_train2))
    total.append(np.mean(label_train2))

    for item in test2:
        for name in names:
            if names[name] == item:
                break
        label_test2.append(total_list[name]['EL'])
    # print(np.mean(label_test2))
    total.append(np.mean(label_test2))


    trainfold = os.path.join(fold,'train3.npy')
    testfold = os.path.join(fold, 'test3.npy')
    np.save(trainfold, np.array(train3))
    np.save(testfold, np.array(test3))
    label_train3 = []
    label_test3 = []
    for item in train3:
        for name in names:
            if names[name] == item:
                break
        label_train3.append(total_list[name]['EL'])
    # print(np.mean(label_train3))
    total.append(np.mean(label_train3))

    for item in test3:
        for name in names:
            if names[name] == item:
                break
        label_test3.append(total_list[name]['EL'])
    # print(np.mean(label_test3))
    total.append(np.mean(label_test3))


    trainfold = os.path.join(fold,'train4.npy')
    testfold = os.path.join(fold, 'test4.npy')
    np.save(trainfold, np.array(train4))
    np.save(testfold, np.array(test4))
    label_train4 = []
    label_test4 = []
    for item in train4:
        for name in names:
            if names[name] == item:
                break
        label_train4.append(total_list[name]['EL'])
    # print(np.mean(label_train4))
    total.append(np.mean(label_train4))
    for item in test4:
        for name in names:
            if names[name] == item:
                break
        label_test4.append(total_list[name]['EL'])
    # print(np.mean(label_test4))
    total.append(np.mean(label_test4))




    testfold = os.path.join(fold, 'test5.npy')
    np.save(testfold, np.array(test5))
    label_test5 = []
    for item in test5:
        for name in names:
            if names[name] == item:
                break
        label_test5.append(total_list[name]['EL'])
    # print(np.mean(label_test5))
    total.append(np.mean(label_test5))



