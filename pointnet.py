from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):#32,3,2500
        batchsize = x.size()[0]#32
        x = F.relu(self.bn1(self.conv1(x)))#32,64,2500
        #print(x.size())
        x = F.relu(self.bn2(self.conv2(x)))#32,128,2500
        x = F.relu(self.bn3(self.conv3(x)))#32,1024,2500
        x = torch.max(x, 2, keepdim=True)[0]#32,1024,1  value?????why
        x = x.view(-1, 1024)#32,1024   
        x = F.relu(self.bn4(self.fc1(x)))#32,512
        #print(x.size())
        x = F.relu(self.bn5(self.fc2(x)))#32,256
        x = self.fc3(x)#32,9

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        #print(iden)#32,9
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        #print(x)
        x = x.view(-1, 3, 3)#32,3,3
        #print(x)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
    def forward(self, x):#32,3,2500
        batchsize = x.size()[0]#32
        n_pts = x.size()[2]#2500
        trans = self.stn(x)#32,3,3
        #print(trans.size())
        x = x.transpose(2,1)#32,2500,3
        #print(x.size())
        x = torch.bmm(x, trans)#32,2500,3
        #print(x.size())
        x = x.transpose(2,1)#32,3,2500
        #print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))#32,64,2500
        #print(x.size())t
        pointfeat = x#32,64,2500
        x = F.relu(self.bn2(self.conv2(x)))#32,128,2500
        x = self.bn3(self.conv3(x))#32,1024,2500
        #x = torch.max(x, 2, keepdim=True)[0]#32,1024,1 #use max function!
        x = torch.sum(x, 2, keepdim=True)#use sum fuction! #32,1024,1
        x = x.view(-1, 1024)#32,1024
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)#32,1024,2500
            return torch.cat([x, pointfeat], 1), trans#32,(1024+64)=1088,2500 1024(global) 64(unique point)

class PointNetCls(nn.Module):
    def __init__(self, k = 2):
        super(PointNetCls, self).__init__()
        self.feat = PointNetfeat(global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    def forward(self, x):#32,3,2500
        x, trans = self.feat(x)#32,1024    32,3,3
        x = F.relu(self.bn1(self.fc1(x)))#32,512
        x = F.relu(self.bn2(self.fc2(x)))#32,256
        x = self.fc3(x)#32,16
        pred=F.log_softmax(x, dim=0)
        #print(pred.size())
        return F.log_softmax(x, dim=0), trans

class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feat = PointNetfeat(global_feat=False)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):#32,3,2500
        batchsize = x.size()[0]#32
        #print(batchsize)
        n_pts = x.size()[2]#2500
        x, trans = self.feat(x)#32,1088,2500   32,3,3
        #print(x.size(),trans.size())
        x = F.relu(self.bn1(self.conv1(x)))#32,512,2500
        #print(x.size())
        x = F.relu(self.bn2(self.conv2(x)))#32,256,2500
        #print(x.size())
        x = F.relu(self.bn3(self.conv3(x)))#32,128,2500
        #print(x.size())
        x = self.conv4(x)#32,num_classes(seg)=k,2500
        #print(x.size())
        x = x.transpose(2,1).contiguous()#32,2500,k=4   x.view(-1,self.k)#32*2500,k
        x = F.log_softmax(x.view(-1,self.k), dim=-1)#32*2500=80000,k
        #print(x.size())
        x = x.view(batchsize, n_pts, self.k)#32,2500,k=4
        #print(x.size())
        return x, trans


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())

    pointfeat = PointNetfeat(global_feat=True)
    out, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _ = seg(sim_data)
    print('seg', out.size())
