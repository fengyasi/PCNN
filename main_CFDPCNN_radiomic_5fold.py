import os

import argparse
import torch.optim as optim
from wandb.cli.cli import disabled, offline

from datasets.dataset_add_feat_five_fold_cfd import Vessel_PCNNradiomics_4fold  ## 五折 + radiomics
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from models.pointnet_CFD_radiomics2 import get_model  ## 注意这里
from data import *
from torch.nn import BCELoss
from tensorboardX import SummaryWriter
import time
import datetime
from tqdm import tqdm
import warnings
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import ExponentialLR
import wandb
from matplotlib import pyplot as plt

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# ======================== 参数 ========================

parser = argparse.ArgumentParser()
parser.add_argument('--randomseed', type=int, default=60, help='randomseed in split')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--batch_size_val', type=int, default=1)
parser.add_argument('--num_epoch', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--use_pcd', type=bool, default=True)
parser.add_argument('--use_text', type=bool, default=True)
parser.add_argument('--num_worker', type=int, default=2)
parser.add_argument('--predthreshold', type=float, default=0.5)
parser.add_argument('--setting', type=str, default='CFDPCNN_radiomics')

opt = parser.parse_args()
print(opt, flush=True)

randomseed = opt.randomseed
batch_size = opt.batch_size
batch_size_val = opt.batch_size_val
num_epoch = opt.num_epoch
learning_rate = opt.learning_rate
use_pcd = opt.use_pcd
use_text = opt.use_text
num_worker = opt.num_worker
predthreshold = opt.predthreshold
setting = opt.setting

config = dict(
    randomseed=randomseed,
    batch_size=batch_size,
    batch_size_val=batch_size_val,
    num_epoch=num_epoch,
    learning_rate=learning_rate,
    use_pcd=use_pcd,
    use_text=use_text,
    num_worker=num_worker,
    predthreshold=predthreshold,
    setting=setting
)

checkpoint = './checkpoint/seed{}/{}'.format(randomseed, setting)
category = 2

if not os.path.exists(checkpoint):
    os.makedirs(checkpoint)


def _create_lr_scheduler(optimizer):
    lr = ExponentialLR(optimizer, gamma=0.99)
    return lr


# ======================== 训练 / 验证 / 测试 ========================

def train(model, opt, train_loader, val_loader, bceloss, fold, board, lr_scheduler, trainF, path):
    print("training fold {}".format(fold))
    best_acc = 0.0
    best_auc = 0.0

    acc_bestauc = 0.0
    auc_bestauc = 0.0
    recall_bestauc = 0.0
    precision_bestauc = 0.0
    f1_score_bestauc = 0.0

    # 这几个用于画该折的 ROC
    fpr_bestauc, tpr_bestauc, thresholds_bestauc = None, None, None

    for epoch in range(num_epoch):
        train_loss, train_acc, train_auc = train_one_epoch(model, train_loader, opt, bceloss, epoch)

        board.add_scalar('train_loss', train_loss, epoch)
        board.add_scalar('train_acc', train_acc, epoch)
        board.add_scalar('train_auc', train_auc, epoch)

        val_loss, val_acc, val_auc, val_labels, val_preds, val_preds_post = val_one_epoch(
            model, val_loader, bceloss, lr_scheduler
        )

        lr_scheduler.step(val_loss)

        board.add_scalar('val_loss', val_loss, epoch)
        board.add_scalar('val_acc', val_acc, epoch)
        board.add_scalar('val_auc', val_auc, epoch)

        print('fold {} epoch {}:'.format(fold, epoch))
        print('  val_loss    : {}'.format(val_loss))
        print('  val_acc     : {}'.format(val_acc))
        print('  val_auc     : {}'.format(val_auc))

        trainF.write('fold{},epoch,{}\n'.format(fold, epoch))
        trainF.write('fold{},val_acc,{}\n'.format(fold, val_acc))
        trainF.write('fold{},val_auc,{}\n'.format(fold, val_auc))
        trainF.write('####################################################\n')

        # 保存 val_auc 最优的模型
        if val_auc > best_auc:
            fold_dir = os.path.join(path, 'exp_fold{}'.format(fold))
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)
            torch.save(model.state_dict(), os.path.join(fold_dir, 'best_auc.pth'))

            best_auc = val_auc
            acc_bestauc = val_acc
            auc_bestauc = val_auc

            fpr_bestauc, tpr_bestauc, thresholds_bestauc = roc_curve(
                val_labels, val_preds, drop_intermediate=False
            )
            recall_bestauc = recall_score(val_labels, val_preds_post, average='binary')
            precision_bestauc = precision_score(val_labels, val_preds_post, average='binary')
            f1_score_bestauc = f1_score(val_labels, val_preds_post, average='binary')

        if val_acc > best_acc:
            best_acc = val_acc

    metric = {
        'acc': best_acc,
        'auc': best_auc,
        'acc_bestauc': acc_bestauc,
        'auc_bestauc': auc_bestauc,
        'recall_bestauc': recall_bestauc,
        'precision_bestauc': precision_bestauc,
        'f1_score_bestauc': f1_score_bestauc
    }

    return metric, fpr_bestauc, tpr_bestauc, thresholds_bestauc


def train_one_epoch(model, train_loader, opt, bceloss, epoch):
    since = time.time()
    model.train()

    total_loss = 0.0
    total_batch = 0

    preds = []
    labels = []
    pred_posts = []

    for point, label, feat in tqdm(train_loader):
        batch_size = len(label)
        total_batch += batch_size

        opt.zero_grad()

        point = point.cuda()
        feat = feat.cuda()
        label = label.cuda().reshape(batch_size, -1)

        pred = model(point, feat)
        loss = bceloss(pred, label)
        loss.backward()
        opt.step()

        pred_post = (pred > predthreshold).float()

        total_loss += loss.item() * batch_size

        labels.append(label.detach().cpu().squeeze(0).numpy())
        preds.append(pred.detach().cpu().squeeze(0).numpy())
        pred_posts.append(pred_post.detach().cpu().squeeze(0).numpy())

        a = np.concatenate(labels, axis=0)
        b = np.concatenate(preds, axis=0)
        c = np.concatenate(pred_posts, axis=0)

    time_one_epoch = time.time() - since
    print('Training epoch {:0d} time: {:0f}m {:0f}s'.format(
        epoch, time_one_epoch // 60, time_one_epoch % 60))

    return (total_loss / total_batch,
            accuracy_score(a, c),
            roc_auc_score(a, b))


def val_one_epoch(model, val_loader, bceloss, lr_scheduler):
    model.eval()
    total_loss = 0.0
    total_batch = 0

    preds = []
    labels = []
    pred_posts = []

    for point, label, feat in tqdm(val_loader):
        batch_size_val = len(label)
        total_batch += batch_size_val

        point = point.cuda()
        feat = feat.cuda()
        label = label.cuda().reshape(batch_size_val, -1)

        with torch.no_grad():
            pred = model(point, feat)
            loss = bceloss(pred, label)

        pred_post = (pred > predthreshold).float()

        total_loss += loss.item() * batch_size_val

        labels.append(label.detach().cpu().squeeze(0).numpy())
        preds.append(pred.detach().cpu().squeeze(0).numpy())
        pred_posts.append(pred_post.detach().cpu().squeeze(0).numpy())

        a = np.concatenate(labels, axis=0)
        b = np.concatenate(preds, axis=0)
        c = np.concatenate(pred_posts, axis=0)

    return (total_loss / total_batch,
            accuracy_score(a, c),
            roc_auc_score(a, b),
            a, b, c)


def test(model, test_loader):
    model.eval()
    total_batch = 0

    preds = []
    labels = []
    pred_posts = []

    for point, label, feat in tqdm(test_loader):
        batch_size = len(label)
        total_batch += batch_size

        point = point.cuda()
        feat = feat.cuda()
        label = label.cuda().reshape(batch_size, -1)

        with torch.no_grad():
            pred = model(point, feat)

        pred_post = (pred > predthreshold).float()

        labels.append(label.detach().cpu().squeeze(0).numpy())
        preds.append(pred.detach().cpu().squeeze(0).numpy())
        pred_posts.append(pred_post.detach().cpu().squeeze(0).numpy())

        a = np.concatenate(labels, axis=0)
        b = np.concatenate(preds, axis=0)
        c = np.concatenate(pred_posts, axis=0)

    metric = {
        'acc_test': accuracy_score(a, c),
        'auc_test': roc_auc_score(a, b),
        'recall_test': recall_score(a, c, average='binary'),
        'precision_test': precision_score(a, c, average='binary'),
        'f1_score_test': f1_score(a, c, average='binary')
    }
    fpr_test, tpr_test, thresholds_test = roc_curve(a, b, drop_intermediate=False)

    return metric, fpr_test, tpr_test, thresholds_test


# ======================== 主程序：5 折 + 测试 ========================

if __name__ == '__main__':

    now = datetime.datetime.now()
    time_name = now.strftime("%Y-%m-%d-%H:%M")
    time_now = now.strftime("%Y-%m-%d.%H.%M.%S")

    print('#########################################################################################')
    path = os.path.join(checkpoint)

    # 确保每一折目录存在
    for k in range(5):
        os.makedirs(os.path.join(path, f'exp_fold{k}'), exist_ok=True)

    # TensorBoard & csv
    checkpoint_name0 = os.path.join(path, 'exp_fold0', time_now)
    board0 = SummaryWriter(log_dir=checkpoint_name0)
    trainF0 = open(os.path.join(path, 'exp_fold0', '{}.csv'.format(time_name)), 'w')

    checkpoint_name1 = os.path.join(path, 'exp_fold1', time_now)
    board1 = SummaryWriter(log_dir=checkpoint_name1)
    trainF1 = open(os.path.join(path, 'exp_fold1', '{}.csv'.format(time_name)), 'w')

    checkpoint_name2 = os.path.join(path, 'exp_fold2', time_now)
    board2 = SummaryWriter(log_dir=checkpoint_name2)
    trainF2 = open(os.path.join(path, 'exp_fold2', '{}.csv'.format(time_name)), 'w')

    checkpoint_name3 = os.path.join(path, 'exp_fold3', time_now)
    board3 = SummaryWriter(log_dir=checkpoint_name3)
    trainF3 = open(os.path.join(path, 'exp_fold3', '{}.csv'.format(time_name)), 'w')

    checkpoint_name4 = os.path.join(path, 'exp_fold4', time_now)
    board4 = SummaryWriter(log_dir=checkpoint_name4)
    trainF4 = open(os.path.join(path, 'exp_fold4', '{}.csv'.format(time_name)), 'w')

    # 总体的 csv（5 折平均 + test）
    trainF = open(os.path.join(path, '{}.csv'.format(time_name)), 'w')

    # ====== 构造 5 折数据集 ======
    feature_list = [f'feature{i}' for i in range(25)]

    trainset0 = Vessel_PCNNradiomics_4fold(feature_list=feature_list, phase='train0', category=category, randomseed=randomseed)
    valset0 = Vessel_PCNNradiomics_4fold(feature_list=feature_list, phase='test0', category=category, randomseed=randomseed)

    trainset1 = Vessel_PCNNradiomics_4fold(feature_list=feature_list, phase='train1', category=category, randomseed=randomseed)
    valset1 = Vessel_PCNNradiomics_4fold(feature_list=feature_list, phase='test1', category=category, randomseed=randomseed)

    trainset2 = Vessel_PCNNradiomics_4fold(feature_list=feature_list, phase='train2', category=category, randomseed=randomseed)
    valset2 = Vessel_PCNNradiomics_4fold(feature_list=feature_list, phase='test2', category=category, randomseed=randomseed)

    trainset3 = Vessel_PCNNradiomics_4fold(feature_list=feature_list, phase='train3', category=category, randomseed=randomseed)
    valset3 = Vessel_PCNNradiomics_4fold(feature_list=feature_list, phase='test3', category=category, randomseed=randomseed)

    trainset4 = Vessel_PCNNradiomics_4fold(feature_list=feature_list, phase='train4', category=category, randomseed=randomseed)
    valset4 = Vessel_PCNNradiomics_4fold(feature_list=feature_list, phase='test4', category=category, randomseed=randomseed)

    # 独立测试集
    testset = Vessel_PCNNradiomics_4fold(feature_list=feature_list, phase='test5', category=category, randomseed=randomseed)

    # ====== DataLoader ======
    train_loader0 = torch.utils.data.DataLoader(
        trainset0, batch_size=batch_size, shuffle=False,
        num_workers=num_worker, drop_last=True
    )
    val_loader0 = torch.utils.data.DataLoader(
        valset0, batch_size=batch_size_val, shuffle=False,
        num_workers=num_worker, drop_last=False
    )

    train_loader1 = torch.utils.data.DataLoader(
        trainset1, batch_size=batch_size, shuffle=False,
        num_workers=num_worker, drop_last=True
    )
    val_loader1 = torch.utils.data.DataLoader(
        valset1, batch_size=batch_size_val, shuffle=False,
        num_workers=num_worker, drop_last=False
    )

    train_loader2 = torch.utils.data.DataLoader(
        trainset2, batch_size=batch_size, shuffle=False,
        num_workers=num_worker, drop_last=True
    )
    val_loader2 = torch.utils.data.DataLoader(
        valset2, batch_size=batch_size_val, shuffle=False,
        num_workers=num_worker, drop_last=False
    )

    train_loader3 = torch.utils.data.DataLoader(
        trainset3, batch_size=batch_size, shuffle=False,
        num_workers=num_worker, drop_last=True
    )
    val_loader3 = torch.utils.data.DataLoader(
        valset3, batch_size=batch_size_val, shuffle=False,
        num_workers=num_worker, drop_last=False
    )

    train_loader4 = torch.utils.data.DataLoader(
        trainset4, batch_size=batch_size, shuffle=False,
        num_workers=num_worker, drop_last=True
    )
    val_loader4 = torch.utils.data.DataLoader(
        valset4, batch_size=batch_size_val, shuffle=False,
        num_workers=num_worker, drop_last=False
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_val, shuffle=False,
        num_workers=num_worker, drop_last=False
    )

    # 确认一下数据能取到
    for name in trainset0.files:
        break

    # ====== 五个模型 ======
    model0 = get_model(num_class=1, use_pcd=use_pcd, use_text=use_text, normal_channel=3, feat_channel=25).cuda()
    model1 = get_model(num_class=1, use_pcd=use_pcd, use_text=use_text, normal_channel=3, feat_channel=25).cuda()
    model2 = get_model(num_class=1, use_pcd=use_pcd, use_text=use_text, normal_channel=3, feat_channel=25).cuda()
    model3 = get_model(num_class=1, use_pcd=use_pcd, use_text=use_text, normal_channel=3, feat_channel=25).cuda()
    model4 = get_model(num_class=1, use_pcd=use_pcd, use_text=use_text, normal_channel=3, feat_channel=25).cuda()

    bceloss = BCELoss()
    opt0 = optim.Adam(model0.parameters(), lr=learning_rate)
    opt1 = optim.Adam(model1.parameters(), lr=learning_rate)
    opt2 = optim.Adam(model2.parameters(), lr=learning_rate)
    opt3 = optim.Adam(model3.parameters(), lr=learning_rate)
    opt4 = optim.Adam(model4.parameters(), lr=learning_rate)

    lr_scheduler0 = _create_lr_scheduler(opt0)
    lr_scheduler1 = _create_lr_scheduler(opt1)
    lr_scheduler2 = _create_lr_scheduler(opt2)
    lr_scheduler3 = _create_lr_scheduler(opt3)
    lr_scheduler4 = _create_lr_scheduler(opt4)

    # ====== 训练 5 折 ======
    metric0, fpr_val0, tpr_val0, thresholds_val0 = train(
        model0, opt0, train_loader0, val_loader0, bceloss, 0, board0, lr_scheduler0, trainF0, path)
    print('#########################################################################################')

    metric1, fpr_val1, tpr_val1, thresholds_val1 = train(
        model1, opt1, train_loader1, val_loader1, bceloss, 1, board1, lr_scheduler1, trainF1, path)
    print('#########################################################################################')

    metric2, fpr_val2, tpr_val2, thresholds_val2 = train(
        model2, opt2, train_loader2, val_loader2, bceloss, 2, board2, lr_scheduler2, trainF2, path)
    print('#########################################################################################')

    metric3, fpr_val3, tpr_val3, thresholds_val3 = train(
        model3, opt3, train_loader3, val_loader3, bceloss, 3, board3, lr_scheduler3, trainF3, path)
    print('#########################################################################################')

    metric4, fpr_val4, tpr_val4, thresholds_val4 = train(
        model4, opt4, train_loader4, val_loader4, bceloss, 4, board4, lr_scheduler4, trainF4, path)
    print('#########################################################################################')

    # ====== 选出 AUC 最好的折，用 test5 做测试 ======
    auc_values = [
        metric0['auc_bestauc'],
        metric1['auc_bestauc'],
        metric2['auc_bestauc'],
        metric3['auc_bestauc'],
        metric4['auc_bestauc']
    ]

    max_auc = max(auc_values)
    best_fold = auc_values.index(max_auc)
    print(f"Best model is from fold {best_fold} with AUC: {max_auc}")

    model_best = get_model(num_class=1, use_pcd=use_pcd, use_text=use_text, normal_channel=3, feat_channel=25).cuda()
    checkpoint_best = os.path.join(path, 'exp_fold{}'.format(best_fold), 'best_auc.pth')
    checkpoint_best = torch.load(checkpoint_best)
    model_best.load_state_dict(checkpoint_best)

    metric_test, fpr_test, tpr_test, thresholds_test = test(model_best, test_loader)
    print('#########################################################################################')

    # ====== 5 折平均指标 ======
    acc = metric0['acc'] + metric1['acc'] + metric2['acc'] + metric3['acc'] + metric4['acc']
    auc = metric0['auc'] + metric1['auc'] + metric2['auc'] + metric3['auc'] + metric4['auc']

    acc_bestauc = metric0['acc_bestauc'] + metric1['acc_bestauc'] + metric2['acc_bestauc'] + metric3['acc_bestauc'] + metric4['acc_bestauc']
    auc_bestauc = metric0['auc_bestauc'] + metric1['auc_bestauc'] + metric2['auc_bestauc'] + metric3['auc_bestauc'] + metric4['auc_bestauc']
    recall_bestauc = metric0['recall_bestauc'] + metric1['recall_bestauc'] + metric2['recall_bestauc'] + metric3['recall_bestauc'] + metric4['recall_bestauc']
    precision_bestauc = metric0['precision_bestauc'] + metric1['precision_bestauc'] + metric2['precision_bestauc'] + metric3['precision_bestauc'] + metric4['precision_bestauc']
    f1_score_bestauc = metric0['f1_score_bestauc'] + metric1['f1_score_bestauc'] + metric2['f1_score_bestauc'] + metric3['f1_score_bestauc'] + metric4['f1_score_bestauc']

    print(
        'best acc for fold0: {}, fold1: {}, fold2: {}, fold3: {}, fold4: {}.'.format(
            metric0['acc'], metric1['acc'], metric2['acc'], metric3['acc'], metric4['acc']
        ))
    print('best acc under 5-fold :', acc / 5)

    print(
        'best auc for fold0: {}, fold1: {}, fold2: {}, fold3: {}, fold4: {}.'.format(
            metric0['auc'], metric1['auc'], metric2['auc'], metric3['auc'], metric4['auc']
        ))
    print('best auc under 5-fold :', auc / 5)

    print(
        'acc_bestauc for fold0: {}, fold1: {}, fold2: {}, fold3: {}, fold4: {}.'.format(
            metric0['acc_bestauc'], metric1['acc_bestauc'], metric2['acc_bestauc'], metric3['acc_bestauc'], metric4['acc_bestauc']
        ))
    print('acc_bestauc under 5-fold :', acc_bestauc / 5)

    print(
        'auc_bestauc for fold0: {}, fold1: {}, fold2: {}, fold3: {}, fold4: {}.'.format(
            metric0['auc_bestauc'], metric1['auc_bestauc'], metric2['auc_bestauc'], metric3['auc_bestauc'], metric4['auc_bestauc']
        ))
    print('auc_bestauc under 5-fold :', auc_bestauc / 5)

    print(
        'recall_bestauc for fold0: {}, fold1: {}, fold2: {}, fold3: {}, fold4: {}.'.format(
            metric0['recall_bestauc'], metric1['recall_bestauc'], metric2['recall_bestauc'], metric3['recall_bestauc'], metric4['recall_bestauc']
        ))
    print('recall_bestauc under 5-fold :', recall_bestauc / 5)

    print(
        'precision_bestauc for fold0: {}, fold1: {}, fold2: {}, fold3: {}, fold4: {}.'.format(
            metric0['precision_bestauc'], metric1['precision_bestauc'], metric2['precision_bestauc'], metric3['precision_bestauc'], metric4['precision_bestauc']
        ))
    print('precision_bestauc under 5-fold :', precision_bestauc / 5)

    print(
        'f1_score_bestauc for fold0: {}, fold1: {}, fold2: {}, fold3: {}, fold4: {}.'.format(
            metric0['f1_score_bestauc'], metric1['f1_score_bestauc'], metric2['f1_score_bestauc'], metric3['f1_score_bestauc'], metric4['f1_score_bestauc']
        ))
    print('f1_score_bestauc under 5-fold :', f1_score_bestauc / 5)

    print(f"Best model is from fold {best_fold} with AUC: {max_auc}")
    print('acc_test      : {}'.format(metric_test['acc_test']))
    print('auc_test      : {}'.format(metric_test['auc_test']))
    print('recall_test   : {}'.format(metric_test['recall_test']))
    print('precision_test: {}'.format(metric_test['precision_test']))
    print('f1_score_test : {}'.format(metric_test['f1_score_test']))
    print('')

    # ========= 各折 csv =========
    trainF0.write('fold:,fold0\n')
    trainF0.write('batch_size,{}\n'.format(batch_size))
    trainF0.write('batch_size_val,{}\n'.format(batch_size_val))
    trainF0.write('num_epoch,{}\n'.format(num_epoch))
    trainF0.write('learning_rate,{}\n'.format(learning_rate))
    trainF0.write('use_pcd,{}\n'.format(use_pcd))
    trainF0.write('use_text,{}\n'.format(use_text))
    trainF0.write('num_worker,{}\n'.format(num_worker))
    trainF0.write('*****************************************************\n')
    trainF0.write('acc,{}\n'.format(metric0['acc']))
    trainF0.write('auc,{}\n'.format(metric0['auc']))
    trainF0.write('acc_bestauc,{}\n'.format(metric0['acc_bestauc']))
    trainF0.write('auc_bestauc,{}\n'.format(metric0['auc_bestauc']))
    trainF0.write('recall_bestauc,{}\n'.format(metric0['recall_bestauc']))
    trainF0.write('precision_bestauc,{}\n'.format(metric0['precision_bestauc']))
    trainF0.write('f1_score_bestauc,{}\n'.format(metric0['f1_score_bestauc']))
    trainF0.write('####################################################\n')

    trainF1.write('fold:,fold1\n')
    trainF1.write('acc,{}\n'.format(metric1['acc']))
    trainF1.write('auc,{}\n'.format(metric1['auc']))
    trainF1.write('acc_bestauc,{}\n'.format(metric1['acc_bestauc']))
    trainF1.write('auc_bestauc,{}\n'.format(metric1['auc_bestauc']))
    trainF1.write('recall_bestauc,{}\n'.format(metric1['recall_bestauc']))
    trainF1.write('precision_bestauc,{}\n'.format(metric1['precision_bestauc']))
    trainF1.write('f1_score_bestauc,{}\n'.format(metric1['f1_score_bestauc']))
    trainF1.write('####################################################\n')

    trainF2.write('fold:,fold2\n')
    trainF2.write('acc,{}\n'.format(metric2['acc']))
    trainF2.write('auc,{}\n'.format(metric2['auc']))
    trainF2.write('acc_bestauc,{}\n'.format(metric2['acc_bestauc']))
    trainF2.write('auc_bestauc,{}\n'.format(metric2['auc_bestauc']))
    trainF2.write('recall_bestauc,{}\n'.format(metric2['recall_bestauc']))
    trainF2.write('precision_bestauc,{}\n'.format(metric2['precision_bestauc']))
    trainF2.write('f1_score_bestauc,{}\n'.format(metric2['f1_score_bestauc']))
    trainF2.write('####################################################\n')

    trainF3.write('fold:,fold3\n')
    trainF3.write('acc,{}\n'.format(metric3['acc']))
    trainF3.write('auc,{}\n'.format(metric3['auc']))
    trainF3.write('acc_bestauc,{}\n'.format(metric3['acc_bestauc']))
    trainF3.write('auc_bestauc,{}\n'.format(metric3['auc_bestauc']))
    trainF3.write('recall_bestauc,{}\n'.format(metric3['recall_bestauc']))
    trainF3.write('precision_bestauc,{}\n'.format(metric3['precision_bestauc']))
    trainF3.write('f1_score_bestauc,{}\n'.format(metric3['f1_score_bestauc']))
    trainF3.write('####################################################\n')

    trainF4.write('fold:,fold4\n')
    trainF4.write('acc,{}\n'.format(metric4['acc']))
    trainF4.write('auc,{}\n'.format(metric4['auc']))
    trainF4.write('acc_bestauc,{}\n'.format(metric4['acc_bestauc']))
    trainF4.write('auc_bestauc,{}\n'.format(metric4['auc_bestauc']))
    trainF4.write('recall_bestauc,{}\n'.format(metric4['recall_bestauc']))
    trainF4.write('precision_bestauc,{}\n'.format(metric4['precision_bestauc']))
    trainF4.write('f1_score_bestauc,{}\n'.format(metric4['f1_score_bestauc']))
    trainF4.write('####################################################\n')

    # ========= 5 折平均 + test =========
    trainF.write('fold:average 5 fold\n')
    trainF.write('batchsize,{}\n'.format(batch_size))
    trainF.write('batchsize_val,{}\n'.format(batch_size_val))
    trainF.write('num_epoch,{}\n'.format(num_epoch))
    trainF.write('learning_rate,{}\n'.format(learning_rate))
    trainF.write('predthreshold,{}\n'.format(predthreshold))

    trainF.write('acc,{}\n'.format(acc / 5))
    trainF.write('auc,{}\n'.format(auc / 5))
    trainF.write('acc var,{}\n'.format(np.var([metric0['acc'], metric1['acc'], metric2['acc'], metric3['acc'], metric4['acc']])))
    trainF.write('auc var,{}\n'.format(np.var([metric0['auc'], metric1['auc'], metric2['auc'], metric3['auc'], metric4['auc']])))
    trainF.write('acc_bestauc,{}\n'.format(acc_bestauc / 5))
    trainF.write('auc_bestauc,{}\n'.format(auc_bestauc / 5))
    trainF.write('acc_bestauc var,{}\n'.format(np.var(
        [metric0['acc_bestauc'], metric1['acc_bestauc'], metric2['acc_bestauc'], metric3['acc_bestauc'], metric4['acc_bestauc']])))
    trainF.write('auc_bestauc var,{}\n'.format(np.var(
        [metric0['auc_bestauc'], metric1['auc_bestauc'], metric2['auc_bestauc'], metric3['auc_bestauc'], metric4['auc_bestauc']])))

    trainF.write(f"Best model is from fold {best_fold} with AUC: {max_auc}\n")
    trainF.write('acc_test,{}\n'.format(metric_test['acc_test']))
    trainF.write('auc_test,{}\n'.format(metric_test['auc_test']))
    trainF.write('recall_test,{}\n'.format(metric_test['recall_test']))
    trainF.write('precision_test,{}\n'.format(metric_test['precision_test']))
    trainF.write('f1_score_test,{}\n'.format(metric_test['f1_score_test']))

    # ========= ROC 图 =========
    # 验证集
    fig_val, ax_val = plt.subplots()
    ax_val.plot([0, 1], [0, 1], color='r', linestyle='--')

    ax_val.plot(fpr_val0, tpr_val0, label='Fold1 auc={}'.format(format(metric0['auc_bestauc'], '.4f')))
    ax_val.plot(fpr_val1, tpr_val1, label='Fold2 auc={}'.format(format(metric1['auc_bestauc'], '.4f')))
    ax_val.plot(fpr_val2, tpr_val2, label='Fold3 auc={}'.format(format(metric2['auc_bestauc'], '.4f')))
    ax_val.plot(fpr_val3, tpr_val3, label='Fold4 auc={}'.format(format(metric3['auc_bestauc'], '.4f')))
    ax_val.plot(fpr_val4, tpr_val4, label='Fold5 auc={}'.format(format(metric4['auc_bestauc'], '.4f')))
    ax_val.set_xlabel('False Positive Rate')
    ax_val.set_ylabel('True Positive Rate')
    ax_val.set_title('ROC Curve on Validation Sets')
    ax_val.legend(loc='lower right')

    # 测试集
    fig_test, ax_test = plt.subplots()
    ax_test.plot([0, 1], [0, 1], color='r', linestyle='--')
    ax_test.plot(fpr_test, tpr_test, label='auc={}'.format(format(metric_test['auc_test'], '.4f')))
    ax_test.set_xlabel('False Positive Rate')
    ax_test.set_ylabel('True Positive Rate')
    ax_test.set_title('ROC Curve on Test Set')
    ax_test.legend(loc='lower right')

    fig_test.savefig(os.path.join(checkpoint, 'roc_test.png'), dpi=300, transparent=True)
    fig_val.savefig(os.path.join(checkpoint, 'roc_valid.png'), dpi=300, transparent=True)
    # plt.show()
    print('#########################################################################################')