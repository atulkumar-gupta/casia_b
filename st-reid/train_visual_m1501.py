import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import sys
sys.path.append("/home/nirbhay/tharun/dg-net/base_reid")
from teacher_model import ft_net, ft_net_dense, PCB, RandomErasing
import json

from types import SimpleNamespace

parse = {
	'name' : 'ft_ResNet50_PCB_m1501',
	'PCB': True,
	'use_dense':False,
	'train_all':False,
	'data_dir':'/home/nirbhay/tharun/dataset/M1501_rename/',
	'color_jitter': False,
	'batchsize': 32,
	'erasing_p': 0.5, #go up to 0.5
} 

opt = SimpleNamespace(**parse)

name = opt.name
data_dir = opt.data_dir
use_gpu = torch.cuda.is_available()

if not os.path.exists('./model/'):
	os.makedirs('./model/')

transform_train_list = [
		#transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
		transforms.Resize((288,144), interpolation=3),
		transforms.RandomCrop((256,128)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]

transform_val_list = [
		transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]

if opt.PCB:
	print('PCB transforms')
	transform_train_list = [
		transforms.Resize((384,192), interpolation=3),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]
	transform_val_list = [
		transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]
else:
	print('Plain transforms')

if opt.erasing_p>0:
	print('RandomErasing ',opt.erasing_p)
	transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
	print('Color jitter')
	transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

data_transforms = {
	'train' : transforms.Compose(transform_train_list),
	'val' : transforms.Compose(transform_val_list),
}

train_all = ''

if opt.train_all:
	train_all = '_all'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
										  data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
										  data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
											 shuffle=True, num_workers=16)
			  for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

inputs, classes = next(iter(dataloaders['train']))


y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
	since = time.time()

	best_model_wts = model.state_dict()
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				scheduler.step()
				model.train(True)  # Set model to training mode
			else:
				model.train(False)  # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0
			# Iterate over data.
			for data in dataloaders[phase]:
				# get the inputs
				inputs, labels = data
				#print(inputs.shape)
				# wrap them in Variable
				if use_gpu:
					inputs = Variable(inputs.cuda())
					labels = Variable(labels.cuda())
				else:
					inputs, labels = Variable(inputs), Variable(labels)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				outputs = model(inputs)
				if not opt.PCB:
					_, preds = torch.max(outputs.data, 1)
					loss = criterion(outputs, labels)
				else:
					part = {}
					sm = nn.Softmax(dim=1)
					num_part = 6
					for i in range(num_part):
						part[i] = outputs[i]

					score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5])
					_, preds = torch.max(score.data, 1)

					loss = criterion(part[0], labels)
					for i in range(num_part-1):
						loss += criterion(part[i+1], labels)

				# backward + optimize only if in training phase
				if phase == 'train':
					loss.backward()
					optimizer.step()

				# statistics
				running_loss += loss.item()
				running_corrects += torch.sum(preds == labels.data)
				# print(running_corrects)

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = (running_corrects.item()) / dataset_sizes[phase]
			# print(running_corrects.item())
			# print(dataset_sizes[phase])
			# print(epoch_acc)
			
			print('{} Loss: {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc))
			
			y_loss[phase].append(epoch_loss)
			y_err[phase].append(1.0-epoch_acc) 

			# deep copy the model
			if phase == 'val':
				if epoch_acc > best_acc:
					print('saving ', epoch_acc)
					best_acc = epoch_acc
					torch.save(model.state_dict(),'/home/nirbhay/tharun/dg-net/st-reid/model/'+name+'net_last.pth')
			#     last_model_wts = model.state_dict()
			#     if epoch%10 == 9:
			#         save_network(model, epoch)
			#     draw_curve(epoch)


		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	#print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	# model.load_state_dict(last_model_wts)
	# save_network(model, 'last')
	return model


if opt.PCB:
	print('PCB')
	model = PCB(len(class_names))
	
elif opt.use_dense:
	print('ft_net_dense')
	model = ft_net_dense(len(class_names))
else:
	print('ft_net')
	model = ft_net(len(class_names))

# print(model)

# criterion = nn.CrossEntropyLoss()

if not opt.PCB:
	ignored_params = list(map(id, model.model.fc.parameters() )) + list(map(id, model.classifier.parameters() ))
	base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
	optimizer_ft = optim.SGD([
			 {'params': base_params, 'lr': 0.01},
			 {'params': model.model.fc.parameters(), 'lr': 0.1},
			 {'params': model.classifier.parameters(), 'lr': 0.1}
		 ], weight_decay=5e-4, momentum=0.9, nesterov=True)
else:
	ignored_params = list(map(id, model.model.fc.parameters() ))
	ignored_params += (list(map(id, model.classifier0.parameters() )) 
					 +list(map(id, model.classifier1.parameters() ))
					 +list(map(id, model.classifier2.parameters() ))
					 +list(map(id, model.classifier3.parameters() ))
					 +list(map(id, model.classifier4.parameters() ))
					 +list(map(id, model.classifier5.parameters() ))
					 #+list(map(id, model.classifier6.parameters() ))
					 #+list(map(id, model.classifier7.parameters() ))
					  )
	base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
	optimizer_ft = optim.SGD([
			 {'params': base_params, 'lr': 0.01},
			 {'params': model.model.fc.parameters(), 'lr': 0.1},
			 {'params': model.classifier0.parameters(), 'lr': 0.1},
			 {'params': model.classifier1.parameters(), 'lr': 0.1},
			 {'params': model.classifier2.parameters(), 'lr': 0.1},
			 {'params': model.classifier3.parameters(), 'lr': 0.1},
			 {'params': model.classifier4.parameters(), 'lr': 0.1},
			 {'params': model.classifier5.parameters(), 'lr': 0.1},
			 #{'params': model.classifier6.parameters(), 'lr': 0.01},
			 #{'params': model.classifier7.parameters(), 'lr': 0.01}
		 ], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#
dir_name = os.path.join('/home/nirbhay/tharun/dg-net/st-reid/model',name)
if not os.path.isdir(dir_name):
	os.mkdir(dir_name)

# save opts
with open('%s/opts.json'%dir_name,'w') as fp:
	json.dump(vars(opt), fp, indent=1)

model= torch.nn.DataParallel(model)

if use_gpu:
	model = model.cuda()
	criterion = nn.CrossEntropyLoss().cuda()

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
					   num_epochs=60)
