import numpy as np
import torch
import os

def print_json(data):
    for item in data:
        image_name = item[:13]
        for _, _, names in os.walk('/home/xiaoyu/PycharmProjects/pythonProject8/dataset/imagesTr'):
            break
        image_names = []
        for name in names:
            if image_name in name:
                image_names.append(name)

        if len(image_names) > 1:
            print('error',image_names)
            break


        print('{')
        print('"'+'image'+'"'+': '+ '"' + 'imagesTr/' + image_names[0] + '",')
        print('"'+'label'+'"'+': '+ '"' + 'labelsTr/' + item + '"')
        print('},')

if __name__ == '__main__':
    train0 = np.load('/home/xiaoyu/PycharmProjects/TAK1007/data/train0.npy')
    test0 = np.load('/home/xiaoyu/PycharmProjects/TAK1007/data/test0.npy')
    train1 = np.load('/home/xiaoyu/PycharmProjects/TAK1007/data/train1.npy')
    test1 = np.load('/home/xiaoyu/PycharmProjects/TAK1007/data/test1.npy')
    train2 = np.load('/home/xiaoyu/PycharmProjects/TAK1007/data/train2.npy')
    test2 = np.load('/home/xiaoyu/PycharmProjects/TAK1007/data/test2.npy')



    print_json(train0)
    print(len(test0))
    #
    # {
    #     "image": "imagesTr/TA006LHZ20090122-bstest.niitest1.nii",
    #     "label": "labelsTr/TA006LHZ20090122-bslabel.nii"
    # },