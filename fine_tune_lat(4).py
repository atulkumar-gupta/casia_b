
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
model_conv.eval()

# Use the model object to select the desired layer
layer = model_conv._modules.get('avgpool')
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
x = []
y = []

with open("indices_gait.txt", "rb") as fl:
    ind = pickle.load(fl)
#gait energy --helper function
def find_gait(path,app,nm):
    files = glob.glob(path+"*.png")
    files.sort()
    
    path = "/DATA/nirbhay/tharun/gei/"+app
    if os.path.isdir(path) == False:
            os.mkdir(path)
    
    num_gait=0
    for j in range(len(ind[int(app)][nm-1])-2):
        if j is None:
            continue
        c=0
        #all images in gait cycle
        gei_img = np.zeros((150,75))
        vec = []
        for i in range(ind[int(app)][nm-1][j],ind[int(app)][nm-1][j+2]+1):
            vec.append(get_vector(files[i]).numpy())
            #predict missing latent vector
        num_gait+=1
        vec = np.array(vec)
        v = vec.mean(0)
        x.append(v)
        y.append(int(app))
    print(f"num_gaits {num_gait}")
    print('-'*10)
#gait energy images
for i in range(1,125):
    if i<10:
        app = "00"+str(i)
    elif i<100:
        app = "0"+str(i)
    else :
        app = str(i)
    for j in range(1,5):
        path = "/DATA/nirbhay/tharun/dataset_CASIA/"+app+"/nm-0"+str(j)+"/"
        print(f"person {app} nm {j}")
        find_gait(path,app,j)
    print("*"*20)
import pickle
with open("feature_vects_lat.txt", "wb") as fp:   #Pickling
    pickle.dump(x, fp)
    
with open("feature_vects_lat.txt", "rb") as fp:   # Unpickling
    nx = pickle.load(fp)

with open("feature_vects_lab.txt", "wb") as fp:   #Pickling
    pickle.dump(y, fp)
    
with open("feature_vects_lab.txt", "rb") as fp:   # Unpickling
    ny = pickle.load(fp)
y[0]
x_train, x_val, y_train, y_val = train_test_split(x,y, test_size=0.2, random_state=101, stratify=y)
dfs = {'train':(x_train,y_train),'val':(x_val,y_val)}
len(x_train), len(y_train), len(x_val), len(y_val)
#custom dataset
class casia(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        lat_vec = torch.tensor(self.x[index])
        label = torch.tensor(self.y[index])
        return lat_vec,label
datasets = {
    x : casia(dfs[x][0],dfs[x][1])
    for x in ['train','val']
}

dataloaders = {x: DataLoader(datasets[x], batch_size=8,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']
              }
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
num_classes = 125
class_names = [i for i in range(1,125)]
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
num_classes = 125
model_res = torchvision.models.resnet18(pretrained=True)

# # Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_res.fc.in_features
model_res.fc = nn.Linear(num_ftrs, num_classes)

PATH = "/home/nirbhay/tharun/casia_b/rs18_nm14_ft_fe.pth"
model_res.load_state_dict(torch.load(PATH))
model_res = model_res.eval()

class latent_vect(nn.Module):
    def __init__(self,in_sz,out_sz):
        super(latent_vect, self).__init__()
        self.fc = nn.Linear(in_sz,out_sz)
    def forward(self,x):
        out = self.fc(x)
        return out

model_lat = latent_vect(512,125)

#copying weights from resnet for better convergence
params1 = model_res.named_parameters()
params2 = model_lat.named_parameters()

dict_params2 = dict(params2)

for name1, param1 in params1:
    if name1 in dict_params2:
        dict_params2[name1].data.copy_(param1.data)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model_lat = model_lat.to(device)

criterion = nn.CrossEntropyLoss()

#all params
optimizer_conv = optim.SGD(model_lat.parameters(), lr=0.001, momentum=0.9)

# Decay LR 
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=15, gamma=0.1)
# exp_lr_scheduler = None
#40
model_lat = train_model(model_lat, criterion, optimizer_conv,exp_lr_scheduler, num_epochs=5)
PATH = "/home/nirbhay/tharun/casia_b/lt_fe_ft.pth"
torch.save(model_lat.state_dict(),PATH)
model_lat.load_state_dict(torch.load(PATH))
 
