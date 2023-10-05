
%matplotlib inline
import torch
from torch import optim,nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms
import torchvision
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
import re
import glob
from PIL import Image
import time

import copy
plt.ion()

from loess import Loess
import pickle
import cv2

from tqdm import tqdm 
Pretrained resnet model for feature vect
num_classes = 125

#pretrained model
model_conv = torchvision.models.resnet18(pretrained=True)

#change last layer
num_ftrs = model_conv.fc.in_features
print(num_ftrs)
model_conv.fc = nn.Linear(num_ftrs, num_classes)

#load model
PATH = "/home/nirbhay/tharun/casia_b/rs18_nm14_ft_fe.pth"
model_conv.load_state_dict(torch.load(PATH,map_location='cpu'))

# Use the model object to select the desired layer
layer = model_conv._modules.get('avgpool')
# Set model to evaluation mode
model_conv.eval()

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()
def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)    
    img = img.convert('RGB')
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))    
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)    
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data[0,:,0,0])    
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)    
    # 6. Run the model on our transformed image
    model_conv(t_img)   
    # 7. Detach our copy function from the layer
    h.remove()   
    # 8. Return the feature vector
    return my_embedding
Pretrained model to map latent vectors to person id
class latent_vect(nn.Module):
    def __init__(self,in_sz,out_sz):
        super(latent_vect, self).__init__()
        self.fc = nn.Linear(in_sz,out_sz)
    def forward(self,x):
        out = self.fc(x)
        return out

model_lat = latent_vect(512,125)
PATH = "/home/nirbhay/tharun/casia_b/lt_fe_ft.pth"
model_lat.load_state_dict(torch.load(PATH))
model_lat.eval()
glob_corr = 0
glob_pics = 0

rank = [0 for i in range(11)]

def find_miss(y,app):
        y = np.array(y)
        x = [i for i in range(y.shape[0])]
        x = np.array(x)
        
        #split data into trianing and test or missing data
        fx,vx,fy,vy = train_test_split(x,y,test_size=0.2, random_state=4)#try shuffle on off
# #         print(f"train {fx.shape} {fy.shape} val {vx.shape} {vy.shape}")
        
        fx, fy = zip(*sorted(zip(fx, fy)))
        vx, vy = zip(*sorted(zip(vx, vy)))
        
        fx = np.array(fx)
        vx = np.array(vx)
        fy = np.array(fy)
        vy = np.array(vy)
        
#         pred_y = np.zeros(vy.shape)
        print(f"complete cyc {y.shape} non-occ {fy.shape}")
        gait = np.zeros(512)
        
        pred_y = np.zeros(vy.shape)
        
        for i,ind in enumerate(fx):
            gait+=fy[i]
        
        for i in range(fy.shape[1]):
            loess = Loess(fx, fy[:,i])
            for j,gx in enumerate(vx):
                pred_y[j,i] = loess.estimate(gx, window=5)
        
#         print("pred_y shape",pred_y.shape)
        
        for i,ind in enumerate(vx):
            gait+=pred_y[i]

        gei = gait/y.shape[0]
    
        #debugging
#         gei = y.mean(0)
#         print("latent gei vector ",gei.shape)
        
        
        tensor_gei = torch.from_numpy(gei).float()
        tensor_gei = torch.unsqueeze(tensor_gei, 0)
#         print("latent gei tensor ",tensor_gei.size())
        with torch.set_grad_enabled(False):
            ot = model_lat(tensor_gei)
            _, preds = torch.max(ot, 1)
        
        probs = ot.numpy()
        ids = np.argsort(-probs,axis=1)
        
        for i in range(10):
            if ids[0][i] == int(app):
                rank[i+1] +=1
                break;
        
        
        print(f"prediction {preds} label {app}")
        print(preds==int(app))
        global glob_corr
        global glob_pics
        if preds==int(app):
            glob_corr +=1
        glob_pics += 1
        print("^"*10)
#find gait nm-05 nm-06
with open("indices_gait.txt", "rb") as fl:
    ind = pickle.load(fl)
    
#gait energy --helper function
def find_gait(path,app,nm):
    files = glob.glob(path+"*.png")
    files.sort()
    
    num_gait=0
    for j in range(len(ind[int(app)][nm-1])-2):
        if j is None:
            continue
        c=0
        #all images in gait cycle
        y = []
        for i in range(ind[int(app)][nm-1][j],ind[int(app)][nm-1][j+2]+1):
            img = cv2.imread(files[i],0)
            y.append(get_vector(files[i]).numpy())
        #predict missing latent vector
        find_miss(y,app)
        num_gait+=1
    print(f"num_gaits {num_gait}")
    print('-'*10)
#gait energy images
for i in tqdm(range(1,125)):
    if i<10:
        app = "00"+str(i)
    elif i<100:
        app = "0"+str(i)
    else :
        app = str(i)
    for j in range(5,7):
        path = "/SSD/Pratik/Gait_Data/Casia_data_preprocessed/GaitDatasetB-silh_PerfectlyAlingedFullPossibleCyclesImages/"+app+"/nm-0"+str(j)+"/"
        print(f"person {app} nm {j}")
        find_gait(path,app,j)
    print("*"*20)

print(f"accuracy {glob_corr/glob_pics}")
for i in range(1,11):
    rank[i]+=rank[i-1]
for i in range(1,11):
    rank[i]/=glob_pics
