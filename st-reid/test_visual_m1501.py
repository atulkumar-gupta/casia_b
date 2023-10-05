# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import sys
sys.path.append("/home/nirbhay/tharun/dg-net/base_reid")
from teacher_model import ft_net, ft_net_dense, PCB, PCB_test, RandomErasing

from collections import OrderedDict
from types import SimpleNamespace
######################################################################

parse = {
    'which_epoch' : 'last',
    'test_dir':'/home/nirbhay/tharun/dataset/M1501_rename/',
    'name': 'ft_ResNet50_PCB_m1501',
    'batchsize':8,
    'use_dense': False,
    'PCB':True
}

opt = SimpleNamespace(**parse)

# str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

# gpu_ids = []
# for str_id in str_ids:
#     id = int(str_id)
#     if id >=0:
#         gpu_ids.append(id)

# set gpu ids
# if len(gpu_ids)>0:
#     torch.cuda.set_device(gpu_ids[0])
# torch.cuda.set_device(0)
# device = torch.device('cuda:0')
# print(device)

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
        transforms.Resize((288,144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
############### Ten Crop        
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.ToTensor()(crop) 
          #      for crop in crops]
           # )),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
          #       for crop in crops]
          # ))
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])


data_dir = test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query']}

class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('/home/nirbhay/tharun/dg-net/st-reid/model',name,name+'net_%s.pth'%opt.which_epoch)

    #chaning data parallelly saved to normal saved 
    # original saved file with DataParallel
    state_dict = torch.load(save_path)
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        nym = k[7:] # remove `module.`
        new_state_dict[nym] = v
    # load params
    network.load_state_dict(new_state_dict)
    # network.load_state_dict(torch.load(save_path))
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        if opt.use_dense:
            ff = torch.FloatTensor(n,1024).zero_()
        else:
            ff = torch.FloatTensor(n,2048).zero_()
        if opt.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_() # we have four parts
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            # input_img = Variable(img.to(device))
            outputs = model(input_img) 
            f = outputs.data.cpu()
            ff = ff+f
        # norm feature
        if opt.PCB:
            # feature size (n,2048,4)
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    frames = []
    for path, v in img_path:
        filename = path.split('/')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        # frame = filename[9:16]
        frame = filename.split('_')[2][1:]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
        frames.append(int(frame))
    return camera_id, labels, frames

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label, gallery_frames = get_id(gallery_path)
query_cam,query_label, query_frames = get_id(query_path)

######################################################################
# Load Collected data Trained model
class_num=751
print('-------test-----------')


if opt.PCB:
    print('PCB')
    model_structure = PCB(class_num)
elif opt.use_dense:
    print('densenet')
    model_structure = ft_net_dense(class_num)
else:
    print('ft_net')
    model_structure = ft_net(class_num)

# model_structure= torch.nn.DataParallel(model_structure)
# model_structure.cuda()
model = load_network(model_structure)
print(model)

# Remove the final fc layer and classifier layer
if not opt.PCB:
    print('non-pcb test')
    model.model.fc = nn.Sequential()
    model.classifier = nn.Sequential()
else:
    print('pcb test')
    model = PCB_test(model)

# Change to test mode
# model.to(device)

model= torch.nn.DataParallel(model)
model.cuda()
model = model.eval()

# Extract feature
gallery_feature = extract_feature(model,dataloaders['gallery'])
query_feature = extract_feature(model,dataloaders['query'])

# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'gallery_frames':gallery_frames,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam,'query_frames':query_frames}
scipy.io.savemat('/home/nirbhay/tharun/dg-net/st-reid/model/'+name+'/'+'pytorch_result.mat',result)
