from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets import PartDataset
from pointnet import PointNetCls
import torch.nn.functional as F
import matplotlib.pyplot as plt


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = 'cls/cls_model_24.pth',  help='model path')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)

opt = parser.parse_args()
print (opt)

test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0' , train = False, classification = True,  npoints = opt.num_points)

testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle = True,num_workers=int(opt.workers))


classifier = PointNetCls(k = len(test_dataset.classes))
#classifier = PointNetCls(k = len(test_dataset.classes), num_points = opt.num_points)
classifier = nn.DataParallel(classifier, device_ids=[0, 1])
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model,map_location=lambda storage, loc: storage))
classifier=classifier.module
print('model is loaded successfully!!!')
classifier.eval()

accuracy_sum=0
for i, data in enumerate(testdataloader, 0):
    points, target = data
    points, target = Variable(points), Variable(target[:, 0])
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    pred, _ = classifier(points)
    loss = F.nll_loss(pred, target)

    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    accuracy=float(correct)/float(32)
    accuracy_sum+=accuracy
    print('i:%d  loss: %f accuracy: %f' %(i, loss.item(), accuracy))
    #print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))
print('mean accuracy_sum:',float(accuracy_sum)/float(i+1))
