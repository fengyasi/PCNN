import os

import argparse
import torch.optim as optim
from matplotlib import pyplot as plt
from wandb.cli.cli import disabled, offline

from datasets.dataset import EVARVessel
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from models.pointnet import get_model
from data import *
from torch.nn import BCELoss
from tensorboardX import SummaryWriter
import time
import datetime
from tqdm import tqdm
import warnings
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import ExponentialLR
import wandb
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
warnings.filterwarnings("ignore")
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# wandb offline

parser = argparse.ArgumentParser()
parser.add_argument('--randomseed',type=int,default=2,help='randomseed in split')
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--batch_size_val',type=int,default=1)
parser.add_argument('--num_epoch',type=int,default=50)
parser.add_argument('--learning_rate',type=float,default=0.0001)
parser.add_argument('--use_pcd',type=bool,default=True)
parser.add_argument('--use_text',type=bool,default=False)
parser.add_argument('--num_worker',type=int,default=32)

parser.add_argument('--predthreshold',type=float,default=0.5)
parser.add_argument('--task', type=str, default="EL", help='task')
# parser.add_argument('--task', type=str, default="EL1", help='task')
# parser.add_argument('--task', type=str, default="EL2", help='task')

opt = parser.parse_args()
print(opt)

randomseed = opt.randomseed
batch_size = opt.batch_size
batch_size_val = opt.batch_size_val
num_epoch = opt.num_epoch
learning_rate = opt.learning_rate
use_pcd = opt.use_pcd
use_text = opt.use_text
num_worker = opt.num_worker
predthreshold = opt.predthreshold
task = opt.task


config = dict(randomseed = randomseed,
              batch_size = batch_size,
            batch_size_val = batch_size_val,
            num_epoch = num_epoch,
            learning_rate = learning_rate,
            use_pcd = use_pcd, # xueguan
            use_text = use_text,#True # linchuangxinxi
            num_worker = num_worker,
            predthreshold = predthreshold,
            task = task)


# wandb.init(project="TVT-"+task, entity="zyx_fdu",config=config)



checkpoint = './checkpoint/seed{}/{}'.format(randomseed,task)
category = 2

if not os.path.exists(checkpoint):
    os.makedirs(checkpoint)

def _create_lr_scheduler(optimizer):
    lr = ExponentialLR(optimizer, gamma=0.99)
    return lr


def train(model, opt, train_loader, val_loader, bceloss, fold, board, lr_scheduler, trainF):
    print("training")
    best_acc = 0
    best_auc = 0
    acc_bestauc = 0
    auc_bestauc = 0

    for epoch in range(num_epoch):
        train_loss, train_acc, train_auc = train_one_epoch(model,train_loader,opt,bceloss, epoch)

        board.add_scalar('train_loss', train_loss, epoch)
        board.add_scalar('train_acc', train_acc, epoch)
        board.add_scalar('train_auc', train_auc, epoch)

        val_loss, val_acc, val_auc, val_labels, val_preds, val_preds_post= val_one_epoch(model,val_loader,bceloss,lr_scheduler)

        lr_scheduler.step(val_loss)

        board.add_scalar('val_loss', val_loss, epoch)
        board.add_scalar('val_acc', val_acc, epoch)
        board.add_scalar('val_auc', val_auc, epoch)


        print('epoch {}:'.format(epoch))
        print('val_loss {}:'.format(val_loss))
        print('val_acc {}:'.format(val_acc))
        print('val_auc {}:'.format(val_auc))
        trainF.write('{},{}\n'.format("epoch", epoch))
        trainF.write('{},{}\n'.format("val_acc", val_acc))
        trainF.write('{},{}\n'.format("val_auc", val_auc))
        trainF.write('####################################################\n')

        # torch.save(model.state_dict(), os.path.join(path, 'epoch_{}.pth'.format(epoch))) # save every model

        if val_auc > best_auc:
            torch.save(model.state_dict(), os.path.join(path, 'exp_fold{}'.format(fold),'best_auc.pth'))
            best_auc = val_auc
            acc_bestauc = val_acc
            auc_bestauc = val_auc
            fpr_bestauc, tpr_bestauc, thresholds_bestauc = roc_curve(val_labels, val_preds, drop_intermediate=False)
            recall_bestauc = recall_score(val_labels, val_preds_post, average='binary')
            precision_bestauc = precision_score(val_labels, val_preds_post, average='binary')
            f1_score_bestauc = f1_score(val_labels, val_preds_post, average='binary')
        if val_acc > best_acc:
            best_acc = val_acc

    metric = {}
    metric['acc'] = best_acc
    metric['auc'] = best_auc

    metric['acc_bestauc'] = acc_bestauc
    metric['auc_bestauc'] = auc_bestauc
    metric['recall_bestauc'] =recall_bestauc
    metric['precision_bestauc'] =precision_bestauc
    metric['f1_score_bestauc'] =f1_score_bestauc

    return metric, fpr_bestauc, tpr_bestauc, thresholds_bestauc


def train_one_epoch(model, train_loader, opt, bceloss, epoch):
    since = time.time()
    model.train()

    total_loss = 0
    total_acc = 0
    total_recall = 0
    total_batch = 0
    total_precise = 0
    preds = []
    labels = []
    pred_posts = []

    for point, label in tqdm(train_loader):
        batch_size = len(label)
        total_batch += batch_size
        opt.zero_grad()

        point = point.cuda()

        label = label.cuda().reshape(batch_size, -1)
        pred = model(point)
        loss = bceloss(pred, label)
        loss.backward()
        opt.step()

        pred_post = (pred > predthreshold).float()#################
        acc = (pred_post == label).float().mean()
        precise = (torch.mul(label, pred_post)).float().mean()
        recall = (torch.mul(label, pred_post)).float().mean()

        total_loss += loss.item() * batch_size
        total_acc += acc.item() * batch_size
        total_precise += precise.item() * batch_size
        total_recall += recall.item() * batch_size

        labels.append(label.detach().cpu().squeeze(0).numpy())
        preds.append(pred.detach().cpu().squeeze(0).numpy())
        pred_posts.append(pred_post.detach().cpu().squeeze(0).numpy())

        a = np.concatenate(labels, axis=0)
        b = np.concatenate(preds, axis=0)
        c = np.concatenate(pred_posts, axis=0)

    cm = confusion_matrix(a, c)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    NPV = tn/(tn+fn)

    time_one_epoch = time.time() - since
    print('Training one epoch{:0d} time in {:0f}m {:0f}s'.format(epoch, time_one_epoch // 60, time_one_epoch % 60))

    return total_loss / total_batch, \
           accuracy_score(a, c), \
           roc_auc_score(a, b)



def val_one_epoch(model, val_loader, bceloss, lr_scheduler):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_batch = 0
    total_recall = 0
    total_precise = 0
    preds = []
    labels = []
    pred_posts = []

    for point, label in tqdm(val_loader):
        batch_size_val = len(label)
        total_batch += batch_size_val

        point = point.cuda()
        label = label.cuda().reshape(batch_size_val, -1)

        pred = model(point)
        loss = bceloss(pred, label)

        pred_post = (pred > predthreshold).float()###########
        acc = (pred_post == label).float().mean()
        precise = (torch.mul(label, pred_post)).float().mean()
        recall = (torch.mul(label, pred_post)).float().mean()

        total_loss += loss.item() * batch_size_val
        total_acc += acc.item() * batch_size_val
        total_recall += recall.item() * batch_size_val
        total_precise += precise.item() * batch_size_val

        labels.append(label.detach().cpu().squeeze(0).numpy())
        preds.append(pred.detach().cpu().squeeze(0).numpy())
        pred_posts.append(pred_post.detach().cpu().squeeze(0).numpy())

        a = np.concatenate(labels, axis=0)
        b = np.concatenate(preds, axis=0)
        c = np.concatenate(pred_posts, axis=0)

    cm = confusion_matrix(a, c)
    tn, fp, fn, tp = cm.ravel()

    return total_loss / total_batch, \
           accuracy_score(a, c), \
           roc_auc_score(a, b),a,b,c

def test(model, test_loader):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_batch = 0
    total_recall = 0
    total_precise = 0
    preds = []
    labels = []
    pred_posts = []

    for point, label in tqdm(test_loader):
        batch_size = len(label)
        total_batch += batch_size

        point = point.cuda()
        label = label.cuda().reshape(batch_size, -1)
        # feat = feat.cuda()

        pred = model(point)

        pred_post = (pred > predthreshold).float()  ###########
        acc = (pred_post == label).float().mean()
        precise = (torch.mul(label, pred_post)).float().mean()
        recall = (torch.mul(label, pred_post)).float().mean()

        total_acc += acc.item() * batch_size
        total_recall += recall.item() * batch_size
        total_precise += precise.item() * batch_size

        labels.append(label.detach().cpu().squeeze(0).numpy())
        preds.append(pred.detach().cpu().squeeze(0).numpy())
        pred_posts.append(pred_post.detach().cpu().squeeze(0).numpy())

        a = np.concatenate(labels, axis=0)
        b = np.concatenate(preds, axis=0)
        c = np.concatenate(pred_posts, axis=0)


    metric = {}
    metric['acc_test'] = accuracy_score(a, c)
    metric['auc_test'] = roc_auc_score(a, b)
    metric['recall_test'] = recall_score(a, c, average='binary')
    metric['precision_test'] = precision_score(a, c, average='binary')
    metric['f1_score_test'] = f1_score(a, c, average='binary')
    fpr_test, tpr_test, thresholds_test = roc_curve(a, b, drop_intermediate=False)

    return metric, fpr_test, tpr_test, thresholds_test

if __name__ == '__main__':

    now = datetime.datetime.now()
    time_name = now.strftime("%Y-%m-%d-%H:%M")
    time_now = time_now = now.strftime("%Y-%m-%d.%H.%M.%S")
    # feature = 'Baseline'

    print('#########################################################################################')
    # path = os.path.join(checkpoint, feature, 'kener_size=1')
    path = os.path.join(checkpoint)
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

    trainF = open(os.path.join(path, '{}.csv'.format(time_name)), 'w')

    trainset0 = EVARVessel( phase='train0', category=category, randomseed=randomseed,task = task)
    valset0 = EVARVessel( phase='test0', category=category, randomseed=randomseed,task = task)
    trainset1 = EVARVessel( phase='train1', category=category, randomseed=randomseed,task = task)
    valset1 = EVARVessel( phase='test1', category=category, randomseed=randomseed,task = task)
    trainset2 = EVARVessel( phase='train2', category=category, randomseed=randomseed,task = task)
    valset2 = EVARVessel( phase='test2', category=category, randomseed=randomseed,task = task)
    trainset3 = EVARVessel( phase='train3', category=category, randomseed=randomseed,task = task)
    valset3 = EVARVessel( phase='test3', category=category, randomseed=randomseed,task = task)
    trainset4 = EVARVessel( phase='train4', category=category, randomseed=randomseed,task = task)
    valset4 = EVARVessel( phase='test4', category=category, randomseed=randomseed,task = task)
    testset = EVARVessel( phase='test5', category=category, randomseed=randomseed,task = task)

    # dandu test loader
    train_loader0 = torch.utils.data.DataLoader(
        trainset0,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        drop_last=True
    )
    val_loader0 = torch.utils.data.DataLoader(
        valset0,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_worker,
        drop_last=False
    )
    train_loader1 = torch.utils.data.DataLoader(
        trainset1,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        drop_last=True
    )
    val_loader1 = torch.utils.data.DataLoader(
        valset1,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_worker,
        drop_last=False
    )
    train_loader2 = torch.utils.data.DataLoader(
        trainset2,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        drop_last=True
    )
    val_loader2 = torch.utils.data.DataLoader(
        valset2,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_worker,
        drop_last=False
    )
    train_loader3 = torch.utils.data.DataLoader(
        trainset3,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        drop_last=True
    )
    val_loader3 = torch.utils.data.DataLoader(
        valset3,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_worker,
        drop_last=False
    )
    train_loader4 = torch.utils.data.DataLoader(
        trainset4,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        drop_last=True#########################ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 512])
    )
    val_loader4 = torch.utils.data.DataLoader(
        valset4,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_worker,
        drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_worker,
        drop_last=False
    )

    for name in trainset0.files:
        break
    model0 = get_model(num_class=1, use_pcd=use_pcd, use_text=use_text, normal_channel=3, feat_channel=0).cuda()
    model1 = get_model(num_class=1, use_pcd=use_pcd, use_text=use_text, normal_channel=3, feat_channel=0).cuda()
    model2 = get_model(num_class=1, use_pcd=use_pcd, use_text=use_text, normal_channel=3, feat_channel=0).cuda()
    model3 = get_model(num_class=1, use_pcd=use_pcd, use_text=use_text, normal_channel=3, feat_channel=0).cuda()
    model4 = get_model(num_class=1, use_pcd=use_pcd, use_text=use_text, normal_channel=3, feat_channel=0).cuda()

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

    metric0, fpr_val0, tpr_val0, thresholds_val0 = train(model0, opt0, train_loader0, val_loader0, bceloss, 0, board0, lr_scheduler0, trainF0)
    print('#########################################################################################')
    metric1, fpr_val1, tpr_val1, thresholds_val1 = train(model1, opt1, train_loader1, val_loader1, bceloss, 1, board1, lr_scheduler1, trainF1)
    print('#########################################################################################')
    metric2, fpr_val2, tpr_val2, thresholds_val2 = train(model2, opt2, train_loader2, val_loader2, bceloss, 2, board2, lr_scheduler2, trainF2)
    print('#########################################################################################')
    metric3, fpr_val3, tpr_val3, thresholds_val3 = train(model3, opt3, train_loader3, val_loader3, bceloss, 3, board3, lr_scheduler3, trainF3)
    print('#########################################################################################')
    metric4, fpr_val4, tpr_val4, thresholds_val4 = train(model4, opt4, train_loader4, val_loader4, bceloss, 4, board4, lr_scheduler4, trainF4)
    print('#########################################################################################')

    # test
    # 存储每一折的 AUC 值
    auc_values = [
        metric0['auc_bestauc'],
        metric1['auc_bestauc'],
        metric2['auc_bestauc'],
        metric3['auc_bestauc'],
        metric4['auc_bestauc']
    ]
    # 找出 AUC 最高的模型
    max_auc = max(auc_values)
    best_fold = auc_values.index(max_auc)
    print(f"Best model is from fold {best_fold} with AUC: {max_auc}")

    model_best = get_model(num_class=1, use_pcd=use_pcd, use_text=use_text, normal_channel=3, feat_channel=0).cuda()
    checkpoint_best = os.path.join(path, 'exp_fold{}'.format(best_fold),'best_auc.pth') # best_auc.pth
    checkpoint_best = torch.load(checkpoint_best)
    model_best.load_state_dict(checkpoint_best)

    metric_test, fpr_test, tpr_test, thresholds_test = test(model_best, test_loader)
    print('#########################################################################################')


    # ge zi zui gao
    acc = metric0['acc'] + metric1['acc'] + metric2['acc'] + metric3['acc'] + metric4['acc']
    auc = metric0['auc'] + metric1['auc'] + metric2['auc'] + metric3['auc'] + metric4['auc']

    # zui jia mo xing
    acc_bestauc = metric0['acc_bestauc'] + metric1['acc_bestauc'] + metric2['acc_bestauc'] + metric3['acc_bestauc'] + metric4['acc_bestauc']
    auc_bestauc = metric0['auc_bestauc'] + metric1['auc_bestauc'] + metric2['auc_bestauc'] + metric3['auc_bestauc'] + metric4['auc_bestauc']
    recall_bestauc = metric0['recall_bestauc'] + metric1['recall_bestauc'] + metric2['recall_bestauc'] + metric3['recall_bestauc'] + \
                  metric4['recall_bestauc']
    precision_bestauc = metric0['precision_bestauc'] + metric1['precision_bestauc'] + metric2['precision_bestauc'] + metric3['precision_bestauc'] + \
                  metric4['precision_bestauc']
    f1_score_bestauc = metric0['f1_score_bestauc'] + metric1['f1_score_bestauc'] + metric2['f1_score_bestauc'] + metric3['f1_score_bestauc'] + \
                  metric4['f1_score_bestauc']


    print(
        'best acc for fold0 is: %s, fold1 is: %s, fold2 is: %s, fold3 is: %s, fold4 is: %s.' % (
        metric0['acc'], metric1['acc'], metric2['acc'], metric3['acc'], metric4['acc']))
    print('best acc under 5-fold is :', acc / 5)
    print(
        'best auc for fold0 is: %s, fold1 is: %s, fold2 is: %s, fold3 is: %s, fold4 is: %s.' % (
            metric0['auc'], metric1['auc'], metric2['auc'], metric3['auc'], metric4['auc']))
    print('best auc under 5-fold is :', auc / 5)



    print(
        'acc_bestauc for fold0 is: %s, fold1 is: %s, fold2 is: %s, fold3 is: %s, fold4 is: %s.' % (
        metric0['acc_bestauc'], metric1['acc_bestauc'], metric2['acc_bestauc'], metric3['acc_bestauc'], metric4['acc_bestauc']))
    print('acc_bestauc under 5-fold is :', acc_bestauc / 5)

    print(
        'auc_bestauc for fold0 is: %s, fold1 is: %s, fold2 is: %s, fold3 is: %s, fold4 is: %s.' % (
            metric0['auc_bestauc'], metric1['auc_bestauc'], metric2['auc_bestauc'], metric3['auc_bestauc'], metric4['auc_bestauc']))
    print('auc_bestauc under 5-fold is :', auc_bestauc / 5)
    print(
        'recall_bestauc for fold0 is: %s, fold1 is: %s, fold2 is: %s, fold3 is: %s, fold4 is: %s.' % (
            metric0['recall_bestauc'], metric1['recall_bestauc'], metric2['recall_bestauc'], metric3['recall_bestauc'], metric4['recall_bestauc']))
    print('recall_bestauc under 5-fold is :', recall_bestauc / 5)
    print(
        'precision_bestauc for fold0 is: %s, fold1 is: %s, fold2 is: %s, fold3 is: %s, fold4 is: %s.' % (
            metric0['precision_bestauc'], metric1['precision_bestauc'], metric2['precision_bestauc'], metric3['precision_bestauc'], metric4['precision_bestauc']))
    print('precision_bestauc under 5-fold is :', precision_bestauc / 5)
    print(
        'f1_score_bestauc for fold0 is: %s, fold1 is: %s, fold2 is: %s, fold3 is: %s, fold4 is: %s.' % (
            metric0['f1_score_bestauc'], metric1['f1_score_bestauc'], metric2['f1_score_bestauc'], metric3['f1_score_bestauc'], metric4['f1_score_bestauc']))
    print('f1_score_bestauc under 5-fold is :', f1_score_bestauc / 5)

    print(f"Best model is from fold {best_fold} with AUC: {max_auc}")
    print('acc_test is: %s' % (metric_test['acc_test']))
    print('auc_test is: %s' % (metric_test['auc_test']))
    print('recall_test is: %s' % (metric_test['recall_test']))
    print('precision_test is: %s' % (metric_test['precision_test']))
    print('f1_score_test is: %s' % (metric_test['f1_score_test']))

    print('')

    trainF0.write('fold:,fold0\n')
    trainF0.write('{},{}\n'.format("batch_size", batch_size))
    trainF0.write('{},{}\n'.format("batch_size_val", batch_size_val))
    trainF0.write('{},{}\n'.format("num_epoch", num_epoch))
    trainF0.write('{},{}\n'.format("learning_rate", learning_rate))
    trainF0.write('{},{}\n'.format("use_pcd", use_pcd))
    trainF0.write('{},{}\n'.format("use_text", use_text))
    trainF0.write('{},{}\n'.format("num_worker", num_worker))
    # trainF0.write('{},{}\n'.format("kernel_size", kernel_size))
    trainF0.write('*****************************************************\n')
    # trainF0.write('{},{}\n'.format("feature", feature_list))
    trainF0.write('{},{}\n'.format("acc", metric0['acc']))
    trainF0.write('{},{}\n'.format("auc", metric0['auc']))
    trainF0.write('{},{}\n'.format("acc_bestauc", metric0['acc_bestauc']))
    trainF0.write('{},{}\n'.format("auc_bestauc", metric0['auc_bestauc']))
    trainF0.write('{},{}\n'.format("recall_bestauc", metric0['recall_bestauc']))
    trainF0.write('{},{}\n'.format("precision_bestauc", metric0['precision_bestauc']))
    trainF0.write('{},{}\n'.format("f1_score_bestauc", metric0['f1_score_bestauc']))
    trainF0.write('####################################################\n')

    trainF1.write('fold:,fold1\n')
    # trainF1.write('{},{}\n'.format("feature", feature_list))
    trainF1.write('{},{}\n'.format("acc", metric1['acc']))
    trainF1.write('{},{}\n'.format("auc", metric1['auc']))
    trainF1.write('{},{}\n'.format("acc_bestauc", metric1['acc_bestauc']))
    trainF1.write('{},{}\n'.format("auc_bestauc", metric1['auc_bestauc']))
    trainF1.write('{},{}\n'.format("recall_bestauc", metric1['recall_bestauc']))
    trainF1.write('{},{}\n'.format("precision_bestauc", metric1['precision_bestauc']))
    trainF1.write('{},{}\n'.format("f1_score_bestauc", metric1['f1_score_bestauc']))
    trainF1.write('####################################################\n')

    trainF2.write('fold:,fold2\n')
    # trainF2.write('{},{}\n'.format("feature", feature_list))
    trainF2.write('{},{}\n'.format("acc", metric2['acc']))
    trainF2.write('{},{}\n'.format("auc", metric2['auc']))
    trainF2.write('{},{}\n'.format("acc_bestauc", metric2['acc_bestauc']))
    trainF2.write('{},{}\n'.format("auc_bestauc", metric2['auc_bestauc']))
    trainF2.write('{},{}\n'.format("recall_bestauc", metric2['recall_bestauc']))
    trainF2.write('{},{}\n'.format("precision_bestauc", metric2['precision_bestauc']))
    trainF2.write('{},{}\n'.format("f1_score_bestauc", metric2['f1_score_bestauc']))
    trainF2.write('####################################################\n')

    trainF3.write('fold:,fold2\n')
    # trainF3.write('{},{}\n'.format("feature", feature_list))
    trainF3.write('{},{}\n'.format("acc", metric3['acc']))
    trainF3.write('{},{}\n'.format("auc", metric3['auc']))
    trainF3.write('{},{}\n'.format("acc_bestauc", metric3['acc_bestauc']))
    trainF3.write('{},{}\n'.format("auc_bestauc", metric3['auc_bestauc']))
    trainF3.write('{},{}\n'.format("recall_bestauc", metric3['recall_bestauc']))
    trainF3.write('{},{}\n'.format("precision_bestauc", metric3['precision_bestauc']))
    trainF3.write('{},{}\n'.format("f1_score_bestauc", metric3['f1_score_bestauc']))
    trainF3.write('####################################################\n')

    trainF4.write('fold:,fold2\n')
    # trainF4.write('{},{}\n'.format("feature", feature_list))
    trainF4.write('{},{}\n'.format("acc", metric4['acc']))
    trainF4.write('{},{}\n'.format("auc", metric4['auc']))
    trainF4.write('{},{}\n'.format("acc_bestauc", metric4['acc_bestauc']))
    trainF4.write('{},{}\n'.format("auc_bestauc", metric4['auc_bestauc']))
    trainF4.write('{},{}\n'.format("recall_bestauc", metric4['recall_bestauc']))
    trainF4.write('{},{}\n'.format("precision_bestauc", metric4['precision_bestauc']))
    trainF4.write('{},{}\n'.format("f1_score_bestauc", metric4['f1_score_bestauc']))
    trainF4.write('####################################################\n')

    trainF.write('fold:average 5 fold\n')
    trainF.write('{}{}\n'.format("batchsize",batch_size))
    trainF.write('{}{}\n'.format("batchsize_Val", batch_size_val))
    trainF.write('{}{}\n'.format("num_epoch", num_epoch))
    trainF.write('{}{}\n'.format("learning_rate",learning_rate))
    trainF.write('{}{}\n'.format("predthreshold", predthreshold))

    # trainF.write('{},{}\n'.format("feature", feature_list))
    trainF.write('{},{}\n'.format("acc", acc / 5))
    trainF.write('{},{}\n'.format("auc", auc / 5))
    trainF.write('{},{}\n'.format("acc var", np.var([metric0['acc'], metric1['acc'], metric2['acc'], metric3['acc'], metric4['acc']])))
    trainF.write('{},{}\n'.format("auc var", np.var([metric0['auc'], metric1['auc'], metric2['auc'], metric3['auc'], metric4['auc']])))
    trainF.write('{},{}\n'.format("acc_bestauc", acc_bestauc / 5))
    trainF.write('{},{}\n'.format("auc_bestauc", auc_bestauc / 5))
    trainF.write('{},{}\n'.format("acc_bestauc var", np.var([metric0['acc_bestauc'], metric1['acc_bestauc'], metric2['acc_bestauc'], metric3['acc_bestauc'], metric4['acc_bestauc']])))
    trainF.write('{},{}\n'.format("auc_bestauc var", np.var([metric0['auc_bestauc'], metric1['auc_bestauc'], metric2['auc_bestauc'], metric3['auc_bestauc'], metric4['auc_bestauc']])))

    trainF.write(f"Best model is from fold {best_fold} with AUC: {max_auc}")
    trainF.write('{},{}\n'.format("acc_test",metric_test['acc_test']))
    trainF.write('{},{}\n'.format("auc_test",metric_test['auc_test']))
    trainF.write('{},{}\n'.format("recall_test",metric_test['recall_test']))
    trainF.write('{},{}\n'.format("precision_test",metric_test['precision_test']))
    trainF.write('{},{}\n'.format("f1_score_test",metric_test['f1_score_test']))



    # valid ROC curve
    fig_val, ax_val = plt.subplots()
    ax_val.plot([0, 1], [0, 1], color='r', linestyle='--')

    ax_val.plot(fpr_val0, tpr_val0, label='Fold1 auc={}'.format(format(metric0['auc_bestauc'], '.4f')))
    ax_val.plot(fpr_val1, tpr_val1, label='Fold2 auc={}'.format(format(metric1['auc_bestauc'], '.4f')))
    ax_val.plot(fpr_val2, tpr_val2, label='Fold3 auc={}'.format(format(metric2['auc_bestauc'], '.4f')))
    ax_val.plot(fpr_val3, tpr_val3, label='Fold3 auc={}'.format(format(metric3['auc_bestauc'], '.4f')))
    ax_val.plot(fpr_val4, tpr_val4, label='Fold4 auc={}'.format(format(metric4['auc_bestauc'], '.4f')))
    ax_val.set_xlabel('False Positive Rate')
    ax_val.set_ylabel('True Positive Rate')
    ax_val.set_title('ROC Curve on Validation Sets')
    ax_val.legend(loc='lower right')

    # test ROC curve
    fig_test, ax_test = plt.subplots()
    ax_test.plot([0, 1], [0, 1], color='r', linestyle='--')
    ax_test.plot(fpr_test, tpr_test, label='auc={}'.format(format(metric_test['auc_test'], '.4f')))
    ax_test.set_xlabel('False Positive Rate')
    ax_test.set_ylabel('True Positive Rate')
    ax_test.set_title('ROC Curve on Test Set')
    ax_test.legend(loc='lower right')

    # save fig
    fig_test.savefig(os.path.join(checkpoint, 'roc_test.png'), dpi=300, transparent=True)
    fig_val.savefig(os.path.join(checkpoint, 'roc_valid.png'), dpi=300, transparent=True)

    plt.show()
