from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("GPU not available, using the CPU instead.")

import wandb
wandb.login()



parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=2500, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'
#wandb

# 定义一个包含三个值的列表
values = ['voxel_grid','random','fps']

# 使用两层嵌套循环来分配不同的值
for value1 in values:
    for value2 in values:
        my_string1 = value1
        my_string2 = value2
        print("String 1:", my_string1, "String 2:", my_string2)
        trainAndVal = f"{my_string1}_Train_And_{my_string2}_Val"
        wandb.init(
            project="Point_cloud_nettwork",
            name=trainAndVal,
            config={
                "epochs": opt.nepoch,
                "method_for_training": my_string1,
                "method_for_val": my_string1,
            })
        config = wandb.config

        opt.manualSeed = random.randint(1, 10000)  # fix seed
        print("Random Seed: ", opt.manualSeed)
        random.seed(opt.manualSeed)
        torch.manual_seed(opt.manualSeed)

        if opt.dataset_type == 'shapenet':
            dataset = ShapeNetDataset(
                root=opt.dataset,
                classification=True,
                npoints=opt.num_points,
                sampling_method=config.method_for_training)

            test_dataset = ShapeNetDataset(
                root=opt.dataset,
                classification=True,
                split='test',
                npoints=opt.num_points,
                data_augmentation=False,
                sampling_method='random')

            val_dataset = ShapeNetDataset(
                root=opt.dataset,
                classification=True,
                split='val',
                npoints=opt.num_points,
                data_augmentation=False,
                sampling_method=config.method_for_val)

        elif opt.dataset_type == 'modelnet40':
            dataset = ModelNetDataset(
                root=opt.dataset,
                npoints=opt.num_points,
                split='trainval')

            test_dataset = ModelNetDataset(
                root=opt.dataset,
                split='test',
                npoints=opt.num_points,
                data_augmentation=False)
        else:
            exit('wrong dataset type')

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers))

        testdataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers))

        valdataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers))

        print(len(dataset), len(test_dataset))
        num_classes = len(dataset.classes)
        print('classes', num_classes)

        try:
            os.makedirs(opt.outf)
        except OSError:
            pass

        classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

        if opt.model != '':
            classifier.load_state_dict(torch.load(opt.model))

        optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        classifier.cuda()

        num_batch = len(dataset) / opt.batchSize
        for epoch in range(config.epochs):
            scheduler.step()
            for i, data in enumerate(dataloader, 0):
                points, target = data
                target = target[:, 0]
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                optimizer.zero_grad()
                classifier = classifier.train()
                pred, trans, trans_feat = classifier(points)
                loss = F.nll_loss(pred, target)
                if opt.feature_transform:
                    loss += feature_transform_regularizer(trans_feat) * 0.001
                loss.backward()
                optimizer.step()
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] train loss: %f accuracy: %f' % (
                epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

                metrics = {
                    "accuracy": correct.item() / float(opt.batchSize),
                    "loss": 1-(correct.item() / float(opt.batchSize)),
                    "train_epoch": epoch,
                }
                wandb.log(metrics)

                if i % 10 == 0:
                    j, data = next(enumerate(valdataloader, 0))
                    points, target = data
                    target = target[:, 0]
                    points = points.transpose(2, 1)
                    points, target = points.cuda(), target.cuda()
                    classifier = classifier.eval()
                    pred, _, _ = classifier(points)
                    loss = F.nll_loss(pred, target)
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(target.data).cpu().sum()
                    print('[%d: %d/%d] %s loss: %f accuracy: %f' % (
                    epoch, i, num_batch, blue('test'), loss.item(), correct.item() / float(opt.batchSize)))
                    val_metrics = {
                        "val_accuracy": correct.item() / float(opt.batchSize),
                        "val_loss": 1 - (correct.item() / float(opt.batchSize)),
                        "train_epoch": epoch,
                    }
                    wandb.log(val_metrics)

            file_name = '%s/%s_Train_And_%s_Val_cls_model_%d.pth' % (opt.outf, my_string1, my_string2, epoch)
            torch.save(classifier.state_dict(), file_name)

        total_correct = 0
        total_testset = 0
        for i, data in tqdm(enumerate(testdataloader, 0)):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct += correct.item()
            total_testset += points.size()[0]

        print("final accuracy {}".format(total_correct / float(total_testset)))
        wandb.finish()


