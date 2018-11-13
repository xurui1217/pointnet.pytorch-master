from __future__ import print_function
from show3d_balls import *
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
from pointnet import PointNetDenseCls
import torch.nn.functional as F
import matplotlib.pyplot as plt



#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = 'seg/seg_model_24.pth',  help='model path')
parser.add_argument('--idx', type=int, default = 1,   help='model index')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)

opt = parser.parse_args()
print (opt)

d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Car'], train = False)

idx = opt.idx

print("model %d/%d" %( idx, len(d)))

point, seg = d[idx]
print(point.size(), seg.size())

point_np = point.numpy()



cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:,:3]
gt = cmap[seg.numpy() - 1, :]

classifier = PointNetDenseCls(k = 4)
classifier = nn.DataParallel(classifier, device_ids=[0, 1])
classifier.load_state_dict(torch.load(opt.model,map_location='cpu'))
classifier=classifier.module
print('model is loaded successfully!!!')
classifier.eval()


point = point.transpose(1,0).contiguous()

point = Variable(point.view(1, point.size()[0], point.size()[1]))
pred, _ = classifier(point)
pred_choice = pred.data.max(2)[1]
print(pred_choice)

#print(pred_choice.size())
pred_color = cmap[pred_choice.numpy()[0], :]

#print(pred_color.shape)
showpoints(point_np, gt, pred_color)

