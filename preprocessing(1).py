
%matplotlib inline
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from PIL import Image, ImageOps
import pickle

from loess import Loess
files = glob.glob('/SSD/Pratik/Gait_Data/GaitDatasetB-silh/001/001/nm-01/090/*.png')
folder = []
for i in range(1,125):
    if i<10:
        app = "00"+str(i)
    elif i<100:
        app = "0"+str(i)
    else :
        app = str(i)
    folder.append("/SSD/Pratik/Gait_Data/GaitDatasetB-silh/"+app+"/"+app)
files
# cropping the person out --helper function
def preprocessing(file):
    img = cv2.imread(file)
    im_bw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    #gradient and contour
    (thresh, im_bw) = cv2.threshold(im_bw, 127, 255, 0)
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.boundingRect(contours[0])
    x,y,w,h = rect
                            
    #cropped image
    new_img = im_bw[y:y+h,x:x+w]
    
    #centering
    mean = 0
    for i in range(new_img.shape[0]):
        count = 0
        value = 0
        for j in range(new_img.shape[1]):
            if new_img[i][j]>0:
                value += j
                count += 1
        if count!=0:
            mean += value/count
    mean = int(mean/new_img.shape[0])
    if mean < new_img.shape[1]/2:
        add = new_img.shape[1] - 2*mean
        val = np.zeros((new_img.shape[0],add))

        cent = np.c_[val,new_img]
    else :
        add = (2*mean - new_img.shape[1])//3
        val = np.zeros((new_img.shape[0],add))

        cent = np.c_[new_img,val]
    pil_img = Image.fromarray(cent)
    pil_img = pil_img.resize((75,150))
    print(pil_img.size)
    plt.imshow(pil_img)
    plt.show()
    
    return pil_img
#cropping all the images from dataset and creates new dataset to save the cropped

if os.path.isdir("dataset_silh") == False:
    os.mkdir("dataset_silh")
    
files = glob.glob('/SSD/Pratik/Gait_Data/GaitDatasetB-silh/001/001/nm-01/090/*.png')
folder = []
for i in range(1,125):
    if i<10:
        app = "00"+str(i)
    elif i<100:
        app = "0"+str(i)
    else :
        app = str(i)
    dst_folder = "/SSD/Pratik/Gait_Data/GaitDatasetB-silh/"+app+"/"+app+"/"
    sv_folder = "dataset_silh/"+app
    
    if os.path.isdir(sv_folder) == False:
        os.mkdir(sv_folder)
    
    for j in range(1,7):
        dst_subfolder = dst_folder+"nm-0"+str(j)+"/090/"
        sv_subfolder = sv_folder+"/nm-0"+str(j)
        
        if os.path.isdir(sv_subfolder) == False:
            os.mkdir(sv_subfolder)
        
        files  = glob.glob(dst_subfolder+"*.png")
        for file in files:
            label = file.replace(dst_subfolder,'')
            
            image = preprocessing(file)
            image_gry = ImageOps.grayscale(image)
            image_gry.save(sv_subfolder+"/"+label)
    
im_bw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
(thresh, im_bw) = cv2.threshold(im_bw, 127, 255, 0)
contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
rect = cv2.boundingRect(contours[0])
x,y,w,h = rect
new_img = im_bw[y:y+h,x:x+w]
plt.imshow(new_img)
plt.show()
#centering the images 
mean = 0
for i in range(new_img.shape[0]):
    count = 0
    value = 0
    for j in range(new_img.shape[1]):
        if new_img[i][j]>0:
            value += j
            count += 1
    if count!=0:
        mean += value/count
mean = int(mean/new_img.shape[0])
if mean < new_img.shape[1]/2:
    add = new_img.shape[1] - 2*mean
    val = np.zeros((new_img.shape[0],add))

    cent = np.c_[val,new_img]
else :
    add = (2*mean - new_img.shape[1])/3
    val = np.zeros((new_img.shape[0],add))
    
    cent = np.c_[new_img,val]
plt.imshow(cent)
plt.show()
double_support = [[[None] for i in range(1,7)]  for j in range(1,126)]
ind = [[[None] for i in range(1,7)]  for j in range(1,126)]
# double_support_y = [[[None] for i in range(1,6)]  for j in range(1,126)]

missing = []

#gait cycle extraction --helper function
def find_grp(path,app,nm):
    
    x = []
    y = []
    frames = []
    k=0
    whts=[]
    files = glob.glob(path+"*.png")
    files.sort()
    if len(files)== 0:
        missing.append(app)
        return 
    for index,file in enumerate(files):
#         print(file)
        frames.append(k)
        k+=1
        img = cv2.imread(file,0)
        white=0
        inf_x=0
        inf_y=0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j]>0:
                    white+=1
                    inf_x+=j
                    inf_y+=i
        x.append(inf_x/white)
        y.append(inf_y/white)
        img = cv2.circle(img, (int(x[-1]),int(y[-1])), 1, (0,0,255), 1)
#         plt.imshow(img)
#         plt.show()
#         whts.append(white)
#         print(f"com x-{x[-1]}  y-{y[-1]}")
#         print(f"white {white} index {index} ")
    y_avg=0
#     for val in y:
#         y_avg += val
#     y_avg = y_avg/len(y)
#     epsilon = 2
    
#     print(f"average y pos {y_avg} epsl {epsilon}")
    
#     for i,f in enumerate(files):
#         if y[i]<y_avg-epsilon:
#             print(f,y[i])
#             img = cv2.imread(f,0)
#             plt.imshow(img)
#             plt.show()

    #smoothing
    frames = np.array(frames)
    y = np.array(y)
    loess = Loess(frames, y)
    y_smooth = np.zeros(y.shape[0])
    for i in range(frames.shape[0]):
        y_smooth[i] = loess.estimate(frames[i], window=5)
        y_avg+=y_smooth[i]
        
    y_avg/=y_smooth.shape
        
    #find peaks in graph    
    for i in range(1,frames.shape[0]-1):
        if y_smooth[i]<y_smooth[i+1] and y_smooth[i]<y_smooth[i-1] and y_smooth[i]<y_avg:
            print(files[i],y_smooth[i])
#             img = cv2.imread(files[i],0)
#             plt.imshow(img)
#             plt.show()
            if double_support[int(app)][nm-1][0] is None:
                double_support[int(app)][nm-1].pop(0)
                ind[int(app)][nm-1].pop(0)
            double_support[int(app)][nm-1].append(files[i])
            ind[int(app)][nm-1].append(i)
    
    #plotting 
#     plt.figure()
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(16,6))
    fig.suptitle(app+"---"+str(nm))
    #before
    ax1.plot(frames, y, marker="+" ,linewidth=0.5)
#     ax1.show()
    #after
    ax2.plot(frames, y_smooth,marker="*",linewidth=0.5)
    plt.show()
    
    print("num frames =",len(frames))
    print("double supports =",len(double_support[int(app)][nm-1]))
    print("indices ",ind[int(app)][nm-1])
#gait cycle plot creation
for i in range(1,11):
    if i<10:
        app = "00"+str(i)
    elif i<100:
        app = "0"+str(i)
    else :
        app = str(i)
    for j in range(1,7):
        path = "/SSD/Pratik/Gait_Data/Casia_data_preprocessed/GaitDatasetB-silh_PerfectlyAlingedFullPossibleCyclesImages/"+app+"/nm-0"+str(j)+"/"
        print(app,j)   
        find_grp(path,app,j)
    print("*"*20)
img[:,:,0]==img[:,:,1]
#gait cycle image printing --helper function

def print_img(path,app,nm):
    files = glob.glob(path+"*.png")
    files.sort()
    
    for j in range(len(ind[int(app)][nm-1])-2):
        if j is None:
            continue
        for i in range(ind[int(app)][nm-1][j],ind[int(app)][nm-1][j+2]+1):
            print(files[i])
            img = cv2.imread(files[i],0)
            plt.imshow(img)
            plt.show()
        print('Gait cycle complete')
        print('%%'*10)
        
#gait cycle image printing
for i in range(2,3):
    if i<10:
        app = "00"+str(i)
    elif i<100:
        app = "0"+str(i)
    else :
        app = str(i)
    for j in range(1,7):
        path = "/SSD/Pratik/Gait_Data/Casia_data_preprocessed/GaitDatasetB-silh_PerfectlyAlingedFullPossibleCyclesImages/"+app+"/nm-0"+str(j)+"/"
        print(app,j)
        print_img(path,app,j)
    print("*"*20)
#save ids

with open("indices_ntm_1_10.txt", "wb") as fp:
    pickle.dump(ind,fp)
with open("indices_preprocessed.txt", "rb") as fp:
    new_id = pickle.load(fp)
ind = new_id
ind = np.array(ind)
ind.shape
ind[1]
path = "/SSD/Pratik/Gait_Data/Casia_data_preprocessed/GaitDatasetB-silh_PerfectlyAlingedFullPossibleCyclesImages/006/nm-01/"
files = glob.glob(path+"*.png")
for f in files:
    img = cv2.imread(f)
    plt.imshow(img)
    plt.show
 
