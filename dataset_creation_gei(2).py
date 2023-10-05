
import pickle
import os 
from PIL import Image
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
with open("indices_gait.txt", "rb") as fl:
    ind = pickle.load(fl)
path = "/DATA/nirbhay/tharun/gei"
if os.path.isdir(path) == False:
        os.mkdir(path)
#gait energy image image printing --helper function
def print_img(path,app,nm):
    files = glob.glob(path+"*.png")
    files.sort()
    
    path = "/DATA/nirbhay/tharun/gei/"+app
    if os.path.isdir(path) == False:
            os.mkdir(path)
    
    for j in range(len(ind[int(app)][nm-1])-2):
        if j is None:
            continue
        gei = np.zeros((150,75))
        c=0
        print(app)
        for i in range(ind[int(app)][nm-1][j],ind[int(app)][nm-1][j+2]+1):
            img = cv2.imread(files[i],0)
            gei +=img
            c+=1
        gei/=c
        im = Image.fromarray(gei)
        im = im.convert("L")
        img_path = path+"/"+app+"_"+str(nm)+"_"+str(j)+'.png'
        im.save(img_path)
        print(img_path)
#gait energy images
#images from nm01-04 only
for i in range(1,125):
    if i<10:
        app = "00"+str(i)
    elif i<100:
        app = "0"+str(i)
    else :
        app = str(i)
    for j in range(1,5):
        path = "/DATA/nirbhay/tharun/dataset_CASIA/"+app+"/nm-0"+str(j)+"/"
        print_img(path,app,j)
    print("*"*20)
for i in range(1,125):
    if i<10:
        app = "00"+str(i)
    elif i<100:
        app = "0"+str(i)
    else :
        app = str(i)
        
    path = "/DATA/nirbhay/tharun/gei/"+app+"/"
    files = glob.glob(path+"*.png")
    files.sort()
    
    for f in files:
        print(f)
        img = cv2.imread(f,0)
        plt.imshow(img)
        plt.show()
 
