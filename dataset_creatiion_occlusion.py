
import pickle
import os 
from PIL import Image
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
with open("indices_gait.txt", "rb") as fl:
    ind = pickle.load(fl)
path = "/DATA/nirbhay/tharun/with_occ"
if os.path.isdir(path) == False:
        os.mkdir(path)
#gait energy image image printing --helper function
def drop_frms(path,app,nm,degree):
    files = glob.glob(path+"*.png")
    files.sort()
    
    path = "/DATA/nirbhay/tharun/with_occ/"+app+"/"+str(degree)
    print(path)
#     if os.path.isdir(path) == False:
#             os.mkdir(path)
    
    for j in range(len(ind[int(app)][nm-1])-2):
        if j is None:
            continue
        imgs = []
        for i in range(ind[int(app)][nm-1][j],ind[int(app)][nm-1][j+2]+1):
            img = cv2.imread(files[i],0)
            imgs.append(img)        
        
        frame_num = [i for i in range(len(imgs))]
        
        non_occ, occ, non_f, occ_f = train_test_split(imgs,frame_num,test_size=degree,random_state=4)
        
        print(f"non_occ {len(non_occ)==len(non_f)}")
        
        save_path = path+"/"+str(j)
        #save non occluded images
        for i in range(len(non_f)):
            print("frame number ",i)
            im = Image.fromarray(non_occ[i])
            im = im.convert("L")

#             if os.path.isdir(img_path) == False:
#                 os.mkdir(img_path)
            img_path = save_path+"/"+str(non_f[i])+".png"
#             im.save(img_path)
            print(img_path)
    
        #save indices of non occ images 
        print(non_f)
        ids_path = save_path+"/non_occ_frms.txt"
        print(ids_path)
#         with open(ids_path, "wb") as fp:
#             pickle.dump(occ_f,fp)
    
        #save indices of occuled images 
        print(occ_f)
        ids_path = save_path+"/occ_frms.txt"
        print(ids_path)
#         with open(ids_path, "wb") as fp:
#             pickle.dump(occ_f,fp)
            
#gait cycle image printing
for i in range(1,6):
    if i<10:
        app = "00"+str(i)
    elif i<100:
        app = "0"+str(i)
    else :
        app = str(i)
    for j in range(1,7):
        path = "/DATA/nirbhay/tharun/dataset_CASIA/"+app+"/nm-0"+str(j)+"/"
        drop_frms(path,app,j,0.5)
    print("*"*20)
/DATA/nirbhay/tharun/with_occ/001/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/001/0.5/0/0.png
frame number  1
/DATA/nirbhay/tharun/with_occ/001/0.5/0/1.png
frame number  2
/DATA/nirbhay/tharun/with_occ/001/0.5/0/5.png
frame number  3
/DATA/nirbhay/tharun/with_occ/001/0.5/0/7.png
frame number  4
/DATA/nirbhay/tharun/with_occ/001/0.5/0/10.png
[0, 1, 5, 7, 10]
/DATA/nirbhay/tharun/with_occ/001/0.5/0/non_occ_frms.txt
[3, 8, 4, 9, 2, 6]
/DATA/nirbhay/tharun/with_occ/001/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/001/0.5/1/13.png
frame number  1
/DATA/nirbhay/tharun/with_occ/001/0.5/1/9.png
frame number  2
/DATA/nirbhay/tharun/with_occ/001/0.5/1/2.png
frame number  3
/DATA/nirbhay/tharun/with_occ/001/0.5/1/17.png
frame number  4
/DATA/nirbhay/tharun/with_occ/001/0.5/1/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/001/0.5/1/8.png
frame number  6
/DATA/nirbhay/tharun/with_occ/001/0.5/1/1.png
frame number  7
/DATA/nirbhay/tharun/with_occ/001/0.5/1/5.png
frame number  8
/DATA/nirbhay/tharun/with_occ/001/0.5/1/14.png
[13, 9, 2, 17, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/001/0.5/1/non_occ_frms.txt
[6, 3, 16, 11, 4, 0, 15, 12, 10]
/DATA/nirbhay/tharun/with_occ/001/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/001/0.5/2/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/001/0.5/2/10.png
frame number  2
/DATA/nirbhay/tharun/with_occ/001/0.5/2/9.png
frame number  3
/DATA/nirbhay/tharun/with_occ/001/0.5/2/2.png
frame number  4
/DATA/nirbhay/tharun/with_occ/001/0.5/2/16.png
frame number  5
/DATA/nirbhay/tharun/with_occ/001/0.5/2/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/001/0.5/2/8.png
frame number  7
/DATA/nirbhay/tharun/with_occ/001/0.5/2/1.png
frame number  8
/DATA/nirbhay/tharun/with_occ/001/0.5/2/5.png
frame number  9
/DATA/nirbhay/tharun/with_occ/001/0.5/2/14.png
[15, 10, 9, 2, 16, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/001/0.5/2/non_occ_frms.txt
[19, 3, 18, 6, 13, 4, 0, 17, 12, 11]
/DATA/nirbhay/tharun/with_occ/001/0.5/2/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/001/0.5/3/13.png
frame number  1
/DATA/nirbhay/tharun/with_occ/001/0.5/3/9.png
frame number  2
/DATA/nirbhay/tharun/with_occ/001/0.5/3/2.png
frame number  3
/DATA/nirbhay/tharun/with_occ/001/0.5/3/17.png
frame number  4
/DATA/nirbhay/tharun/with_occ/001/0.5/3/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/001/0.5/3/8.png
frame number  6
/DATA/nirbhay/tharun/with_occ/001/0.5/3/1.png
frame number  7
/DATA/nirbhay/tharun/with_occ/001/0.5/3/5.png
frame number  8
/DATA/nirbhay/tharun/with_occ/001/0.5/3/14.png
[13, 9, 2, 17, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/001/0.5/3/non_occ_frms.txt
[6, 3, 16, 11, 4, 0, 15, 12, 10]
/DATA/nirbhay/tharun/with_occ/001/0.5/3/occ_frms.txt
/DATA/nirbhay/tharun/with_occ/001/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/001/0.5/0/0.png
frame number  1
/DATA/nirbhay/tharun/with_occ/001/0.5/0/12.png
frame number  2
/DATA/nirbhay/tharun/with_occ/001/0.5/0/8.png
frame number  3
/DATA/nirbhay/tharun/with_occ/001/0.5/0/1.png
frame number  4
/DATA/nirbhay/tharun/with_occ/001/0.5/0/5.png
frame number  5
/DATA/nirbhay/tharun/with_occ/001/0.5/0/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/001/0.5/0/10.png
[0, 12, 8, 1, 5, 7, 10]
/DATA/nirbhay/tharun/with_occ/001/0.5/0/non_occ_frms.txt
[4, 3, 11, 9, 6, 13, 2]
/DATA/nirbhay/tharun/with_occ/001/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/001/0.5/1/2.png
frame number  1
/DATA/nirbhay/tharun/with_occ/001/0.5/1/12.png
frame number  2
/DATA/nirbhay/tharun/with_occ/001/0.5/1/15.png
frame number  3
/DATA/nirbhay/tharun/with_occ/001/0.5/1/8.png
frame number  4
/DATA/nirbhay/tharun/with_occ/001/0.5/1/1.png
frame number  5
/DATA/nirbhay/tharun/with_occ/001/0.5/1/5.png
frame number  6
/DATA/nirbhay/tharun/with_occ/001/0.5/1/7.png
frame number  7
/DATA/nirbhay/tharun/with_occ/001/0.5/1/14.png
[2, 12, 15, 8, 1, 5, 7, 14]
/DATA/nirbhay/tharun/with_occ/001/0.5/1/non_occ_frms.txt
[10, 3, 9, 4, 0, 13, 6, 16, 11]
/DATA/nirbhay/tharun/with_occ/001/0.5/1/occ_frms.txt
/DATA/nirbhay/tharun/with_occ/001/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/001/0.5/0/2.png
frame number  1
/DATA/nirbhay/tharun/with_occ/001/0.5/0/12.png
frame number  2
/DATA/nirbhay/tharun/with_occ/001/0.5/0/15.png
frame number  3
/DATA/nirbhay/tharun/with_occ/001/0.5/0/8.png
frame number  4
/DATA/nirbhay/tharun/with_occ/001/0.5/0/1.png
frame number  5
/DATA/nirbhay/tharun/with_occ/001/0.5/0/5.png
frame number  6
/DATA/nirbhay/tharun/with_occ/001/0.5/0/7.png
frame number  7
/DATA/nirbhay/tharun/with_occ/001/0.5/0/14.png
[2, 12, 15, 8, 1, 5, 7, 14]
/DATA/nirbhay/tharun/with_occ/001/0.5/0/non_occ_frms.txt
[10, 3, 9, 4, 0, 13, 6, 16, 11]
/DATA/nirbhay/tharun/with_occ/001/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/001/0.5/1/2.png
frame number  1
/DATA/nirbhay/tharun/with_occ/001/0.5/1/4.png
frame number  2
/DATA/nirbhay/tharun/with_occ/001/0.5/1/22.png
frame number  3
/DATA/nirbhay/tharun/with_occ/001/0.5/1/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/001/0.5/1/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/001/0.5/1/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/001/0.5/1/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/001/0.5/1/24.png
frame number  8
/DATA/nirbhay/tharun/with_occ/001/0.5/1/27.png
frame number  9
/DATA/nirbhay/tharun/with_occ/001/0.5/1/8.png
frame number  10
/DATA/nirbhay/tharun/with_occ/001/0.5/1/1.png
frame number  11
/DATA/nirbhay/tharun/with_occ/001/0.5/1/5.png
frame number  12
/DATA/nirbhay/tharun/with_occ/001/0.5/1/23.png
frame number  13
/DATA/nirbhay/tharun/with_occ/001/0.5/1/14.png
frame number  14
/DATA/nirbhay/tharun/with_occ/001/0.5/1/26.png
[2, 4, 22, 13, 7, 9, 18, 24, 27, 8, 1, 5, 23, 14, 26]
/DATA/nirbhay/tharun/with_occ/001/0.5/1/non_occ_frms.txt
[11, 21, 28, 15, 20, 25, 17, 29, 19, 0, 3, 16, 10, 6, 12]
/DATA/nirbhay/tharun/with_occ/001/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/001/0.5/2/2.png
frame number  1
/DATA/nirbhay/tharun/with_occ/001/0.5/2/4.png
frame number  2
/DATA/nirbhay/tharun/with_occ/001/0.5/2/21.png
frame number  3
/DATA/nirbhay/tharun/with_occ/001/0.5/2/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/001/0.5/2/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/001/0.5/2/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/001/0.5/2/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/001/0.5/2/22.png
frame number  8
/DATA/nirbhay/tharun/with_occ/001/0.5/2/8.png
frame number  9
/DATA/nirbhay/tharun/with_occ/001/0.5/2/1.png
frame number  10
/DATA/nirbhay/tharun/with_occ/001/0.5/2/5.png
frame number  11
/DATA/nirbhay/tharun/with_occ/001/0.5/2/23.png
frame number  12
/DATA/nirbhay/tharun/with_occ/001/0.5/2/14.png
frame number  13
/DATA/nirbhay/tharun/with_occ/001/0.5/2/26.png
[2, 4, 21, 13, 7, 9, 18, 22, 8, 1, 5, 23, 14, 26]
/DATA/nirbhay/tharun/with_occ/001/0.5/2/non_occ_frms.txt
[19, 16, 11, 27, 25, 17, 24, 20, 0, 3, 15, 10, 6, 12]
/DATA/nirbhay/tharun/with_occ/001/0.5/2/occ_frms.txt
/DATA/nirbhay/tharun/with_occ/001/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/001/0.5/0/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/001/0.5/0/17.png
frame number  2
/DATA/nirbhay/tharun/with_occ/001/0.5/0/25.png
frame number  3
/DATA/nirbhay/tharun/with_occ/001/0.5/0/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/001/0.5/0/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/001/0.5/0/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/001/0.5/0/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/001/0.5/0/21.png
frame number  8
/DATA/nirbhay/tharun/with_occ/001/0.5/0/8.png
frame number  9
/DATA/nirbhay/tharun/with_occ/001/0.5/0/1.png
frame number  10
/DATA/nirbhay/tharun/with_occ/001/0.5/0/5.png
frame number  11
/DATA/nirbhay/tharun/with_occ/001/0.5/0/23.png
frame number  12
/DATA/nirbhay/tharun/with_occ/001/0.5/0/14.png
[15, 17, 25, 13, 7, 9, 18, 21, 8, 1, 5, 23, 14]
/DATA/nirbhay/tharun/with_occ/001/0.5/0/non_occ_frms.txt
[24, 16, 22, 0, 3, 10, 12, 11, 6, 2, 4, 20, 19]
/DATA/nirbhay/tharun/with_occ/001/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/001/0.5/1/14.png
frame number  1
/DATA/nirbhay/tharun/with_occ/001/0.5/1/13.png
frame number  2
/DATA/nirbhay/tharun/with_occ/001/0.5/1/8.png
frame number  3
/DATA/nirbhay/tharun/with_occ/001/0.5/1/1.png
frame number  4
/DATA/nirbhay/tharun/with_occ/001/0.5/1/5.png
frame number  5
/DATA/nirbhay/tharun/with_occ/001/0.5/1/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/001/0.5/1/10.png
[14, 13, 8, 1, 5, 7, 10]
/DATA/nirbhay/tharun/with_occ/001/0.5/1/non_occ_frms.txt
[12, 0, 6, 3, 4, 9, 11, 2]
/DATA/nirbhay/tharun/with_occ/001/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/001/0.5/2/10.png
frame number  1
/DATA/nirbhay/tharun/with_occ/001/0.5/2/9.png
frame number  2
/DATA/nirbhay/tharun/with_occ/001/0.5/2/2.png
frame number  3
/DATA/nirbhay/tharun/with_occ/001/0.5/2/15.png
frame number  4
/DATA/nirbhay/tharun/with_occ/001/0.5/2/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/001/0.5/2/8.png
frame number  6
/DATA/nirbhay/tharun/with_occ/001/0.5/2/1.png
frame number  7
/DATA/nirbhay/tharun/with_occ/001/0.5/2/5.png
frame number  8
/DATA/nirbhay/tharun/with_occ/001/0.5/2/14.png
[10, 9, 2, 15, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/001/0.5/2/non_occ_frms.txt
[6, 3, 17, 12, 4, 0, 16, 13, 11, 18]
/DATA/nirbhay/tharun/with_occ/001/0.5/2/occ_frms.txt
/DATA/nirbhay/tharun/with_occ/001/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/001/0.5/0/11.png
frame number  1
/DATA/nirbhay/tharun/with_occ/001/0.5/0/8.png
frame number  2
/DATA/nirbhay/tharun/with_occ/001/0.5/0/1.png
frame number  3
/DATA/nirbhay/tharun/with_occ/001/0.5/0/5.png
frame number  4
/DATA/nirbhay/tharun/with_occ/001/0.5/0/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/001/0.5/0/10.png
[11, 8, 1, 5, 7, 10]
/DATA/nirbhay/tharun/with_occ/001/0.5/0/non_occ_frms.txt
[3, 4, 6, 12, 9, 2, 0]
/DATA/nirbhay/tharun/with_occ/001/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/001/0.5/1/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/001/0.5/1/17.png
frame number  2
/DATA/nirbhay/tharun/with_occ/001/0.5/1/25.png
frame number  3
/DATA/nirbhay/tharun/with_occ/001/0.5/1/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/001/0.5/1/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/001/0.5/1/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/001/0.5/1/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/001/0.5/1/21.png
frame number  8
/DATA/nirbhay/tharun/with_occ/001/0.5/1/8.png
frame number  9
/DATA/nirbhay/tharun/with_occ/001/0.5/1/1.png
frame number  10
/DATA/nirbhay/tharun/with_occ/001/0.5/1/5.png
frame number  11
/DATA/nirbhay/tharun/with_occ/001/0.5/1/23.png
frame number  12
/DATA/nirbhay/tharun/with_occ/001/0.5/1/14.png
[15, 17, 25, 13, 7, 9, 18, 21, 8, 1, 5, 23, 14]
/DATA/nirbhay/tharun/with_occ/001/0.5/1/non_occ_frms.txt
[24, 16, 22, 0, 3, 10, 12, 11, 6, 2, 4, 20, 19]
/DATA/nirbhay/tharun/with_occ/001/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/001/0.5/2/16.png
frame number  1
/DATA/nirbhay/tharun/with_occ/001/0.5/2/24.png
frame number  2
/DATA/nirbhay/tharun/with_occ/001/0.5/2/13.png
frame number  3
/DATA/nirbhay/tharun/with_occ/001/0.5/2/7.png
frame number  4
/DATA/nirbhay/tharun/with_occ/001/0.5/2/9.png
frame number  5
/DATA/nirbhay/tharun/with_occ/001/0.5/2/18.png
frame number  6
/DATA/nirbhay/tharun/with_occ/001/0.5/2/20.png
frame number  7
/DATA/nirbhay/tharun/with_occ/001/0.5/2/8.png
frame number  8
/DATA/nirbhay/tharun/with_occ/001/0.5/2/1.png
frame number  9
/DATA/nirbhay/tharun/with_occ/001/0.5/2/5.png
frame number  10
/DATA/nirbhay/tharun/with_occ/001/0.5/2/23.png
frame number  11
/DATA/nirbhay/tharun/with_occ/001/0.5/2/14.png
[16, 24, 13, 7, 9, 18, 20, 8, 1, 5, 23, 14]
/DATA/nirbhay/tharun/with_occ/001/0.5/2/non_occ_frms.txt
[22, 21, 0, 3, 12, 10, 6, 11, 2, 4, 19, 17, 15]
/DATA/nirbhay/tharun/with_occ/001/0.5/2/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/001/0.5/3/13.png
frame number  1
/DATA/nirbhay/tharun/with_occ/001/0.5/3/9.png
frame number  2
/DATA/nirbhay/tharun/with_occ/001/0.5/3/2.png
frame number  3
/DATA/nirbhay/tharun/with_occ/001/0.5/3/17.png
frame number  4
/DATA/nirbhay/tharun/with_occ/001/0.5/3/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/001/0.5/3/8.png
frame number  6
/DATA/nirbhay/tharun/with_occ/001/0.5/3/1.png
frame number  7
/DATA/nirbhay/tharun/with_occ/001/0.5/3/5.png
frame number  8
/DATA/nirbhay/tharun/with_occ/001/0.5/3/14.png
[13, 9, 2, 17, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/001/0.5/3/non_occ_frms.txt
[6, 3, 16, 11, 4, 0, 15, 12, 10]
/DATA/nirbhay/tharun/with_occ/001/0.5/3/occ_frms.txt
/DATA/nirbhay/tharun/with_occ/001/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/001/0.5/0/13.png
frame number  1
/DATA/nirbhay/tharun/with_occ/001/0.5/0/9.png
frame number  2
/DATA/nirbhay/tharun/with_occ/001/0.5/0/2.png
frame number  3
/DATA/nirbhay/tharun/with_occ/001/0.5/0/17.png
frame number  4
/DATA/nirbhay/tharun/with_occ/001/0.5/0/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/001/0.5/0/8.png
frame number  6
/DATA/nirbhay/tharun/with_occ/001/0.5/0/1.png
frame number  7
/DATA/nirbhay/tharun/with_occ/001/0.5/0/5.png
frame number  8
/DATA/nirbhay/tharun/with_occ/001/0.5/0/14.png
[13, 9, 2, 17, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/001/0.5/0/non_occ_frms.txt
[6, 3, 16, 11, 4, 0, 15, 12, 10]
/DATA/nirbhay/tharun/with_occ/001/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/001/0.5/1/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/001/0.5/1/17.png
frame number  2
/DATA/nirbhay/tharun/with_occ/001/0.5/1/25.png
frame number  3
/DATA/nirbhay/tharun/with_occ/001/0.5/1/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/001/0.5/1/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/001/0.5/1/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/001/0.5/1/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/001/0.5/1/21.png
frame number  8
/DATA/nirbhay/tharun/with_occ/001/0.5/1/8.png
frame number  9
/DATA/nirbhay/tharun/with_occ/001/0.5/1/1.png
frame number  10
/DATA/nirbhay/tharun/with_occ/001/0.5/1/5.png
frame number  11
/DATA/nirbhay/tharun/with_occ/001/0.5/1/23.png
frame number  12
/DATA/nirbhay/tharun/with_occ/001/0.5/1/14.png
[15, 17, 25, 13, 7, 9, 18, 21, 8, 1, 5, 23, 14]
/DATA/nirbhay/tharun/with_occ/001/0.5/1/non_occ_frms.txt
[24, 16, 22, 0, 3, 10, 12, 11, 6, 2, 4, 20, 19]
/DATA/nirbhay/tharun/with_occ/001/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/001/0.5/2/17.png
frame number  1
/DATA/nirbhay/tharun/with_occ/001/0.5/2/25.png
frame number  2
/DATA/nirbhay/tharun/with_occ/001/0.5/2/13.png
frame number  3
/DATA/nirbhay/tharun/with_occ/001/0.5/2/7.png
frame number  4
/DATA/nirbhay/tharun/with_occ/001/0.5/2/9.png
frame number  5
/DATA/nirbhay/tharun/with_occ/001/0.5/2/18.png
frame number  6
/DATA/nirbhay/tharun/with_occ/001/0.5/2/21.png
frame number  7
/DATA/nirbhay/tharun/with_occ/001/0.5/2/8.png
frame number  8
/DATA/nirbhay/tharun/with_occ/001/0.5/2/1.png
frame number  9
/DATA/nirbhay/tharun/with_occ/001/0.5/2/5.png
frame number  10
/DATA/nirbhay/tharun/with_occ/001/0.5/2/23.png
frame number  11
/DATA/nirbhay/tharun/with_occ/001/0.5/2/14.png
frame number  12
/DATA/nirbhay/tharun/with_occ/001/0.5/2/26.png
[17, 25, 13, 7, 9, 18, 21, 8, 1, 5, 23, 14, 26]
/DATA/nirbhay/tharun/with_occ/001/0.5/2/non_occ_frms.txt
[24, 16, 22, 0, 3, 10, 12, 11, 6, 2, 4, 20, 19, 15]
/DATA/nirbhay/tharun/with_occ/001/0.5/2/occ_frms.txt
********************
/DATA/nirbhay/tharun/with_occ/002/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/0/1.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/0/5.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/0/6.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/0/2.png
[1, 5, 6, 2]
/DATA/nirbhay/tharun/with_occ/002/0.5/0/non_occ_frms.txt
[4, 7, 3, 0]
/DATA/nirbhay/tharun/with_occ/002/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/1/1.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/1/5.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/1/2.png
[1, 5, 2]
/DATA/nirbhay/tharun/with_occ/002/0.5/1/non_occ_frms.txt
[4, 6, 3, 0]
/DATA/nirbhay/tharun/with_occ/002/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/2/0.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/2/1.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/2/5.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/2/7.png
[0, 1, 5, 7]
/DATA/nirbhay/tharun/with_occ/002/0.5/2/non_occ_frms.txt
[3, 4, 6, 2, 8]
/DATA/nirbhay/tharun/with_occ/002/0.5/2/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/3/2.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/3/12.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/3/15.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/3/8.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/3/1.png
frame number  5
/DATA/nirbhay/tharun/with_occ/002/0.5/3/5.png
frame number  6
/DATA/nirbhay/tharun/with_occ/002/0.5/3/7.png
frame number  7
/DATA/nirbhay/tharun/with_occ/002/0.5/3/14.png
[2, 12, 15, 8, 1, 5, 7, 14]
/DATA/nirbhay/tharun/with_occ/002/0.5/3/non_occ_frms.txt
[10, 3, 9, 4, 0, 13, 6, 16, 11]
/DATA/nirbhay/tharun/with_occ/002/0.5/3/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/4/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/4/10.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/4/9.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/4/2.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/4/16.png
frame number  5
/DATA/nirbhay/tharun/with_occ/002/0.5/4/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/002/0.5/4/8.png
frame number  7
/DATA/nirbhay/tharun/with_occ/002/0.5/4/1.png
frame number  8
/DATA/nirbhay/tharun/with_occ/002/0.5/4/5.png
frame number  9
/DATA/nirbhay/tharun/with_occ/002/0.5/4/14.png
[15, 10, 9, 2, 16, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/002/0.5/4/non_occ_frms.txt
[19, 3, 18, 6, 13, 4, 0, 17, 12, 11]
/DATA/nirbhay/tharun/with_occ/002/0.5/4/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/5/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/5/10.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/5/9.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/5/2.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/5/16.png
frame number  5
/DATA/nirbhay/tharun/with_occ/002/0.5/5/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/002/0.5/5/8.png
frame number  7
/DATA/nirbhay/tharun/with_occ/002/0.5/5/1.png
frame number  8
/DATA/nirbhay/tharun/with_occ/002/0.5/5/5.png
frame number  9
/DATA/nirbhay/tharun/with_occ/002/0.5/5/14.png
[15, 10, 9, 2, 16, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/002/0.5/5/non_occ_frms.txt
[19, 3, 18, 6, 13, 4, 0, 17, 12, 11]
/DATA/nirbhay/tharun/with_occ/002/0.5/5/occ_frms.txt
/DATA/nirbhay/tharun/with_occ/002/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/0/2.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/0/12.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/0/15.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/0/8.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/0/1.png
frame number  5
/DATA/nirbhay/tharun/with_occ/002/0.5/0/5.png
frame number  6
/DATA/nirbhay/tharun/with_occ/002/0.5/0/7.png
frame number  7
/DATA/nirbhay/tharun/with_occ/002/0.5/0/14.png
[2, 12, 15, 8, 1, 5, 7, 14]
/DATA/nirbhay/tharun/with_occ/002/0.5/0/non_occ_frms.txt
[10, 3, 9, 4, 0, 13, 6, 16, 11]
/DATA/nirbhay/tharun/with_occ/002/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/1/13.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/1/9.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/1/2.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/1/17.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/1/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/002/0.5/1/8.png
frame number  6
/DATA/nirbhay/tharun/with_occ/002/0.5/1/1.png
frame number  7
/DATA/nirbhay/tharun/with_occ/002/0.5/1/5.png
frame number  8
/DATA/nirbhay/tharun/with_occ/002/0.5/1/14.png
[13, 9, 2, 17, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/002/0.5/1/non_occ_frms.txt
[6, 3, 16, 11, 4, 0, 15, 12, 10]
/DATA/nirbhay/tharun/with_occ/002/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/2/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/2/13.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/2/8.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/2/1.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/2/5.png
frame number  5
/DATA/nirbhay/tharun/with_occ/002/0.5/2/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/002/0.5/2/14.png
frame number  7
/DATA/nirbhay/tharun/with_occ/002/0.5/2/10.png
[15, 13, 8, 1, 5, 7, 14, 10]
/DATA/nirbhay/tharun/with_occ/002/0.5/2/non_occ_frms.txt
[12, 0, 6, 3, 4, 9, 11, 2]
/DATA/nirbhay/tharun/with_occ/002/0.5/2/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/3/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/3/16.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/3/23.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/3/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/3/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/002/0.5/3/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/002/0.5/3/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/002/0.5/3/20.png
frame number  8
/DATA/nirbhay/tharun/with_occ/002/0.5/3/8.png
frame number  9
/DATA/nirbhay/tharun/with_occ/002/0.5/3/1.png
frame number  10
/DATA/nirbhay/tharun/with_occ/002/0.5/3/5.png
frame number  11
/DATA/nirbhay/tharun/with_occ/002/0.5/3/14.png
[15, 16, 23, 13, 7, 9, 18, 20, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/002/0.5/3/non_occ_frms.txt
[22, 21, 0, 3, 12, 10, 6, 11, 2, 4, 19, 17]
/DATA/nirbhay/tharun/with_occ/002/0.5/3/occ_frms.txt
/DATA/nirbhay/tharun/with_occ/002/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/0/2.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/0/12.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/0/15.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/0/8.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/0/1.png
frame number  5
/DATA/nirbhay/tharun/with_occ/002/0.5/0/5.png
frame number  6
/DATA/nirbhay/tharun/with_occ/002/0.5/0/7.png
frame number  7
/DATA/nirbhay/tharun/with_occ/002/0.5/0/14.png
[2, 12, 15, 8, 1, 5, 7, 14]
/DATA/nirbhay/tharun/with_occ/002/0.5/0/non_occ_frms.txt
[10, 3, 9, 4, 0, 13, 6, 16, 11]
/DATA/nirbhay/tharun/with_occ/002/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/1/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/1/17.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/1/25.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/1/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/1/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/002/0.5/1/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/002/0.5/1/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/002/0.5/1/21.png
frame number  8
/DATA/nirbhay/tharun/with_occ/002/0.5/1/8.png
frame number  9
/DATA/nirbhay/tharun/with_occ/002/0.5/1/1.png
frame number  10
/DATA/nirbhay/tharun/with_occ/002/0.5/1/5.png
frame number  11
/DATA/nirbhay/tharun/with_occ/002/0.5/1/23.png
frame number  12
/DATA/nirbhay/tharun/with_occ/002/0.5/1/14.png
[15, 17, 25, 13, 7, 9, 18, 21, 8, 1, 5, 23, 14]
/DATA/nirbhay/tharun/with_occ/002/0.5/1/non_occ_frms.txt
[24, 16, 22, 0, 3, 10, 12, 11, 6, 2, 4, 20, 19]
/DATA/nirbhay/tharun/with_occ/002/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/2/21.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/2/13.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/2/7.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/2/12.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/2/10.png
frame number  5
/DATA/nirbhay/tharun/with_occ/002/0.5/2/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/002/0.5/2/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/002/0.5/2/8.png
frame number  8
/DATA/nirbhay/tharun/with_occ/002/0.5/2/1.png
frame number  9
/DATA/nirbhay/tharun/with_occ/002/0.5/2/5.png
frame number  10
/DATA/nirbhay/tharun/with_occ/002/0.5/2/14.png
[21, 13, 7, 12, 10, 9, 18, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/002/0.5/2/non_occ_frms.txt
[20, 19, 0, 3, 6, 11, 15, 2, 4, 17, 16]
/DATA/nirbhay/tharun/with_occ/002/0.5/2/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/3/0.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/3/1.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/3/5.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/3/7.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/3/10.png
[0, 1, 5, 7, 10]
/DATA/nirbhay/tharun/with_occ/002/0.5/3/non_occ_frms.txt
[3, 8, 4, 9, 2, 6]
/DATA/nirbhay/tharun/with_occ/002/0.5/3/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/4/11.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/4/8.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/4/1.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/4/5.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/4/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/002/0.5/4/10.png
[11, 8, 1, 5, 7, 10]
/DATA/nirbhay/tharun/with_occ/002/0.5/4/non_occ_frms.txt
[3, 4, 6, 12, 9, 2, 0]
/DATA/nirbhay/tharun/with_occ/002/0.5/4/occ_frms.txt
/DATA/nirbhay/tharun/with_occ/002/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/0/10.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/0/6.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/0/12.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/0/31.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/0/4.png
frame number  5
/DATA/nirbhay/tharun/with_occ/002/0.5/0/32.png
frame number  6
/DATA/nirbhay/tharun/with_occ/002/0.5/0/25.png
frame number  7
/DATA/nirbhay/tharun/with_occ/002/0.5/0/33.png
frame number  8
/DATA/nirbhay/tharun/with_occ/002/0.5/0/13.png
frame number  9
/DATA/nirbhay/tharun/with_occ/002/0.5/0/7.png
frame number  10
/DATA/nirbhay/tharun/with_occ/002/0.5/0/28.png
frame number  11
/DATA/nirbhay/tharun/with_occ/002/0.5/0/26.png
frame number  12
/DATA/nirbhay/tharun/with_occ/002/0.5/0/9.png
frame number  13
/DATA/nirbhay/tharun/with_occ/002/0.5/0/18.png
frame number  14
/DATA/nirbhay/tharun/with_occ/002/0.5/0/8.png
frame number  15
/DATA/nirbhay/tharun/with_occ/002/0.5/0/23.png
frame number  16
/DATA/nirbhay/tharun/with_occ/002/0.5/0/1.png
frame number  17
/DATA/nirbhay/tharun/with_occ/002/0.5/0/5.png
[10, 6, 12, 31, 4, 32, 25, 33, 13, 7, 28, 26, 9, 18, 8, 23, 1, 5]
/DATA/nirbhay/tharun/with_occ/002/0.5/0/non_occ_frms.txt
[15, 22, 17, 20, 34, 16, 29, 21, 2, 24, 11, 19, 30, 14, 35, 27, 0, 3]
/DATA/nirbhay/tharun/with_occ/002/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/1/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/1/16.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/1/22.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/1/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/1/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/002/0.5/1/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/002/0.5/1/19.png
frame number  7
/DATA/nirbhay/tharun/with_occ/002/0.5/1/8.png
frame number  8
/DATA/nirbhay/tharun/with_occ/002/0.5/1/1.png
frame number  9
/DATA/nirbhay/tharun/with_occ/002/0.5/1/5.png
frame number  10
/DATA/nirbhay/tharun/with_occ/002/0.5/1/14.png
[15, 16, 22, 13, 7, 9, 19, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/002/0.5/1/non_occ_frms.txt
[21, 20, 0, 3, 12, 10, 6, 11, 2, 4, 18, 17]
/DATA/nirbhay/tharun/with_occ/002/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/2/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/2/10.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/2/9.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/2/2.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/2/16.png
frame number  5
/DATA/nirbhay/tharun/with_occ/002/0.5/2/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/002/0.5/2/8.png
frame number  7
/DATA/nirbhay/tharun/with_occ/002/0.5/2/1.png
frame number  8
/DATA/nirbhay/tharun/with_occ/002/0.5/2/5.png
frame number  9
/DATA/nirbhay/tharun/with_occ/002/0.5/2/14.png
[15, 10, 9, 2, 16, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/002/0.5/2/non_occ_frms.txt
[19, 3, 18, 6, 13, 4, 0, 17, 12, 11]
/DATA/nirbhay/tharun/with_occ/002/0.5/2/occ_frms.txt
/DATA/nirbhay/tharun/with_occ/002/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/0/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/0/16.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/0/23.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/0/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/0/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/002/0.5/0/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/002/0.5/0/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/002/0.5/0/20.png
frame number  8
/DATA/nirbhay/tharun/with_occ/002/0.5/0/8.png
frame number  9
/DATA/nirbhay/tharun/with_occ/002/0.5/0/1.png
frame number  10
/DATA/nirbhay/tharun/with_occ/002/0.5/0/5.png
frame number  11
/DATA/nirbhay/tharun/with_occ/002/0.5/0/14.png
[15, 16, 23, 13, 7, 9, 18, 20, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/002/0.5/0/non_occ_frms.txt
[22, 21, 0, 3, 12, 10, 6, 11, 2, 4, 19, 17]
/DATA/nirbhay/tharun/with_occ/002/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/1/13.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/1/9.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/1/2.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/1/17.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/1/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/002/0.5/1/8.png
frame number  6
/DATA/nirbhay/tharun/with_occ/002/0.5/1/1.png
frame number  7
/DATA/nirbhay/tharun/with_occ/002/0.5/1/5.png
frame number  8
/DATA/nirbhay/tharun/with_occ/002/0.5/1/14.png
[13, 9, 2, 17, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/002/0.5/1/non_occ_frms.txt
[6, 3, 16, 11, 4, 0, 15, 12, 10]
/DATA/nirbhay/tharun/with_occ/002/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/2/0.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/2/12.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/2/8.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/2/1.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/2/5.png
frame number  5
/DATA/nirbhay/tharun/with_occ/002/0.5/2/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/002/0.5/2/10.png
[0, 12, 8, 1, 5, 7, 10]
/DATA/nirbhay/tharun/with_occ/002/0.5/2/non_occ_frms.txt
[4, 3, 11, 9, 6, 13, 2]
/DATA/nirbhay/tharun/with_occ/002/0.5/2/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/3/0.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/3/1.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/3/5.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/3/7.png
[0, 1, 5, 7]
/DATA/nirbhay/tharun/with_occ/002/0.5/3/non_occ_frms.txt
[3, 4, 6, 2, 8]
/DATA/nirbhay/tharun/with_occ/002/0.5/3/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/4/6.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/4/0.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/4/1.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/4/5.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/4/7.png
[6, 0, 1, 5, 7]
/DATA/nirbhay/tharun/with_occ/002/0.5/4/non_occ_frms.txt
[3, 8, 4, 9, 2]
/DATA/nirbhay/tharun/with_occ/002/0.5/4/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/5/0.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/5/1.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/5/5.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/5/7.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/5/10.png
[0, 1, 5, 7, 10]
/DATA/nirbhay/tharun/with_occ/002/0.5/5/non_occ_frms.txt
[3, 8, 4, 9, 2, 6]
/DATA/nirbhay/tharun/with_occ/002/0.5/5/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/6/14.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/6/13.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/6/8.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/6/1.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/6/5.png
frame number  5
/DATA/nirbhay/tharun/with_occ/002/0.5/6/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/002/0.5/6/10.png
[14, 13, 8, 1, 5, 7, 10]
/DATA/nirbhay/tharun/with_occ/002/0.5/6/non_occ_frms.txt
[12, 0, 6, 3, 4, 9, 11, 2]
/DATA/nirbhay/tharun/with_occ/002/0.5/6/occ_frms.txt
/DATA/nirbhay/tharun/with_occ/002/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/0/17.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/0/25.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/0/13.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/0/7.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/0/9.png
frame number  5
/DATA/nirbhay/tharun/with_occ/002/0.5/0/18.png
frame number  6
/DATA/nirbhay/tharun/with_occ/002/0.5/0/21.png
frame number  7
/DATA/nirbhay/tharun/with_occ/002/0.5/0/8.png
frame number  8
/DATA/nirbhay/tharun/with_occ/002/0.5/0/1.png
frame number  9
/DATA/nirbhay/tharun/with_occ/002/0.5/0/5.png
frame number  10
/DATA/nirbhay/tharun/with_occ/002/0.5/0/23.png
frame number  11
/DATA/nirbhay/tharun/with_occ/002/0.5/0/14.png
frame number  12
/DATA/nirbhay/tharun/with_occ/002/0.5/0/26.png
[17, 25, 13, 7, 9, 18, 21, 8, 1, 5, 23, 14, 26]
/DATA/nirbhay/tharun/with_occ/002/0.5/0/non_occ_frms.txt
[24, 16, 22, 0, 3, 10, 12, 11, 6, 2, 4, 20, 19, 15]
/DATA/nirbhay/tharun/with_occ/002/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/1/17.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/1/25.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/1/13.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/1/7.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/1/9.png
frame number  5
/DATA/nirbhay/tharun/with_occ/002/0.5/1/18.png
frame number  6
/DATA/nirbhay/tharun/with_occ/002/0.5/1/21.png
frame number  7
/DATA/nirbhay/tharun/with_occ/002/0.5/1/8.png
frame number  8
/DATA/nirbhay/tharun/with_occ/002/0.5/1/1.png
frame number  9
/DATA/nirbhay/tharun/with_occ/002/0.5/1/5.png
frame number  10
/DATA/nirbhay/tharun/with_occ/002/0.5/1/23.png
frame number  11
/DATA/nirbhay/tharun/with_occ/002/0.5/1/14.png
frame number  12
/DATA/nirbhay/tharun/with_occ/002/0.5/1/26.png
[17, 25, 13, 7, 9, 18, 21, 8, 1, 5, 23, 14, 26]
/DATA/nirbhay/tharun/with_occ/002/0.5/1/non_occ_frms.txt
[24, 16, 22, 0, 3, 10, 12, 11, 6, 2, 4, 20, 19, 15]
/DATA/nirbhay/tharun/with_occ/002/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/002/0.5/2/7.png
frame number  1
/DATA/nirbhay/tharun/with_occ/002/0.5/2/12.png
frame number  2
/DATA/nirbhay/tharun/with_occ/002/0.5/2/10.png
frame number  3
/DATA/nirbhay/tharun/with_occ/002/0.5/2/9.png
frame number  4
/DATA/nirbhay/tharun/with_occ/002/0.5/2/2.png
frame number  5
/DATA/nirbhay/tharun/with_occ/002/0.5/2/17.png
frame number  6
/DATA/nirbhay/tharun/with_occ/002/0.5/2/8.png
frame number  7
/DATA/nirbhay/tharun/with_occ/002/0.5/2/1.png
frame number  8
/DATA/nirbhay/tharun/with_occ/002/0.5/2/5.png
frame number  9
/DATA/nirbhay/tharun/with_occ/002/0.5/2/14.png
[7, 12, 10, 9, 2, 17, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/002/0.5/2/non_occ_frms.txt
[19, 18, 3, 0, 6, 20, 15, 4, 16, 13, 11]
/DATA/nirbhay/tharun/with_occ/002/0.5/2/occ_frms.txt
********************
/DATA/nirbhay/tharun/with_occ/003/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/003/0.5/0/21.png
frame number  1
/DATA/nirbhay/tharun/with_occ/003/0.5/0/13.png
frame number  2
/DATA/nirbhay/tharun/with_occ/003/0.5/0/7.png
frame number  3
/DATA/nirbhay/tharun/with_occ/003/0.5/0/12.png
frame number  4
/DATA/nirbhay/tharun/with_occ/003/0.5/0/10.png
frame number  5
/DATA/nirbhay/tharun/with_occ/003/0.5/0/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/003/0.5/0/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/003/0.5/0/8.png
frame number  8
/DATA/nirbhay/tharun/with_occ/003/0.5/0/1.png
frame number  9
/DATA/nirbhay/tharun/with_occ/003/0.5/0/5.png
frame number  10
/DATA/nirbhay/tharun/with_occ/003/0.5/0/14.png
[21, 13, 7, 12, 10, 9, 18, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/003/0.5/0/non_occ_frms.txt
[20, 19, 0, 3, 6, 11, 15, 2, 4, 17, 16]
/DATA/nirbhay/tharun/with_occ/003/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/003/0.5/1/21.png
frame number  1
/DATA/nirbhay/tharun/with_occ/003/0.5/1/13.png
frame number  2
/DATA/nirbhay/tharun/with_occ/003/0.5/1/7.png
frame number  3
/DATA/nirbhay/tharun/with_occ/003/0.5/1/12.png
frame number  4
/DATA/nirbhay/tharun/with_occ/003/0.5/1/10.png
frame number  5
/DATA/nirbhay/tharun/with_occ/003/0.5/1/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/003/0.5/1/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/003/0.5/1/8.png
frame number  8
/DATA/nirbhay/tharun/with_occ/003/0.5/1/1.png
frame number  9
/DATA/nirbhay/tharun/with_occ/003/0.5/1/5.png
frame number  10
/DATA/nirbhay/tharun/with_occ/003/0.5/1/14.png
[21, 13, 7, 12, 10, 9, 18, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/003/0.5/1/non_occ_frms.txt
[20, 19, 0, 3, 6, 11, 15, 2, 4, 17, 16]
/DATA/nirbhay/tharun/with_occ/003/0.5/1/occ_frms.txt
/DATA/nirbhay/tharun/with_occ/003/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/003/0.5/0/13.png
frame number  1
/DATA/nirbhay/tharun/with_occ/003/0.5/0/9.png
frame number  2
/DATA/nirbhay/tharun/with_occ/003/0.5/0/2.png
frame number  3
/DATA/nirbhay/tharun/with_occ/003/0.5/0/17.png
frame number  4
/DATA/nirbhay/tharun/with_occ/003/0.5/0/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/003/0.5/0/8.png
frame number  6
/DATA/nirbhay/tharun/with_occ/003/0.5/0/1.png
frame number  7
/DATA/nirbhay/tharun/with_occ/003/0.5/0/5.png
frame number  8
/DATA/nirbhay/tharun/with_occ/003/0.5/0/14.png
[13, 9, 2, 17, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/003/0.5/0/non_occ_frms.txt
[6, 3, 16, 11, 4, 0, 15, 12, 10]
/DATA/nirbhay/tharun/with_occ/003/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/003/0.5/1/2.png
frame number  1
/DATA/nirbhay/tharun/with_occ/003/0.5/1/4.png
frame number  2
/DATA/nirbhay/tharun/with_occ/003/0.5/1/22.png
frame number  3
/DATA/nirbhay/tharun/with_occ/003/0.5/1/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/003/0.5/1/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/003/0.5/1/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/003/0.5/1/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/003/0.5/1/28.png
frame number  8
/DATA/nirbhay/tharun/with_occ/003/0.5/1/8.png
frame number  9
/DATA/nirbhay/tharun/with_occ/003/0.5/1/1.png
frame number  10
/DATA/nirbhay/tharun/with_occ/003/0.5/1/5.png
frame number  11
/DATA/nirbhay/tharun/with_occ/003/0.5/1/23.png
frame number  12
/DATA/nirbhay/tharun/with_occ/003/0.5/1/14.png
frame number  13
/DATA/nirbhay/tharun/with_occ/003/0.5/1/26.png
[2, 4, 22, 13, 7, 9, 18, 28, 8, 1, 5, 23, 14, 26]
/DATA/nirbhay/tharun/with_occ/003/0.5/1/non_occ_frms.txt
[11, 21, 27, 15, 20, 24, 17, 25, 19, 0, 3, 16, 10, 6, 12]
/DATA/nirbhay/tharun/with_occ/003/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/003/0.5/2/2.png
frame number  1
/DATA/nirbhay/tharun/with_occ/003/0.5/2/4.png
frame number  2
/DATA/nirbhay/tharun/with_occ/003/0.5/2/21.png
frame number  3
/DATA/nirbhay/tharun/with_occ/003/0.5/2/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/003/0.5/2/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/003/0.5/2/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/003/0.5/2/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/003/0.5/2/22.png
frame number  8
/DATA/nirbhay/tharun/with_occ/003/0.5/2/8.png
frame number  9
/DATA/nirbhay/tharun/with_occ/003/0.5/2/1.png
frame number  10
/DATA/nirbhay/tharun/with_occ/003/0.5/2/5.png
frame number  11
/DATA/nirbhay/tharun/with_occ/003/0.5/2/23.png
frame number  12
/DATA/nirbhay/tharun/with_occ/003/0.5/2/14.png
frame number  13
/DATA/nirbhay/tharun/with_occ/003/0.5/2/26.png
[2, 4, 21, 13, 7, 9, 18, 22, 8, 1, 5, 23, 14, 26]
/DATA/nirbhay/tharun/with_occ/003/0.5/2/non_occ_frms.txt
[19, 16, 11, 27, 25, 17, 24, 20, 0, 3, 15, 10, 6, 12]
/DATA/nirbhay/tharun/with_occ/003/0.5/2/occ_frms.txt
/DATA/nirbhay/tharun/with_occ/003/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/003/0.5/0/10.png
frame number  1
/DATA/nirbhay/tharun/with_occ/003/0.5/0/9.png
frame number  2
/DATA/nirbhay/tharun/with_occ/003/0.5/0/2.png
frame number  3
/DATA/nirbhay/tharun/with_occ/003/0.5/0/15.png
frame number  4
/DATA/nirbhay/tharun/with_occ/003/0.5/0/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/003/0.5/0/8.png
frame number  6
/DATA/nirbhay/tharun/with_occ/003/0.5/0/1.png
frame number  7
/DATA/nirbhay/tharun/with_occ/003/0.5/0/5.png
frame number  8
/DATA/nirbhay/tharun/with_occ/003/0.5/0/14.png
[10, 9, 2, 15, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/003/0.5/0/non_occ_frms.txt
[6, 3, 17, 12, 4, 0, 16, 13, 11, 18]
/DATA/nirbhay/tharun/with_occ/003/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/003/0.5/1/21.png
frame number  1
/DATA/nirbhay/tharun/with_occ/003/0.5/1/13.png
frame number  2
/DATA/nirbhay/tharun/with_occ/003/0.5/1/7.png
frame number  3
/DATA/nirbhay/tharun/with_occ/003/0.5/1/12.png
frame number  4
/DATA/nirbhay/tharun/with_occ/003/0.5/1/10.png
frame number  5
/DATA/nirbhay/tharun/with_occ/003/0.5/1/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/003/0.5/1/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/003/0.5/1/8.png
frame number  8
/DATA/nirbhay/tharun/with_occ/003/0.5/1/1.png
frame number  9
/DATA/nirbhay/tharun/with_occ/003/0.5/1/5.png
frame number  10
/DATA/nirbhay/tharun/with_occ/003/0.5/1/14.png
[21, 13, 7, 12, 10, 9, 18, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/003/0.5/1/non_occ_frms.txt
[20, 19, 0, 3, 6, 11, 15, 2, 4, 17, 16]
/DATA/nirbhay/tharun/with_occ/003/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/003/0.5/2/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/003/0.5/2/16.png
frame number  2
/DATA/nirbhay/tharun/with_occ/003/0.5/2/22.png
frame number  3
/DATA/nirbhay/tharun/with_occ/003/0.5/2/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/003/0.5/2/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/003/0.5/2/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/003/0.5/2/19.png
frame number  7
/DATA/nirbhay/tharun/with_occ/003/0.5/2/8.png
frame number  8
/DATA/nirbhay/tharun/with_occ/003/0.5/2/1.png
frame number  9
/DATA/nirbhay/tharun/with_occ/003/0.5/2/5.png
frame number  10
/DATA/nirbhay/tharun/with_occ/003/0.5/2/14.png
[15, 16, 22, 13, 7, 9, 19, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/003/0.5/2/non_occ_frms.txt
[21, 20, 0, 3, 12, 10, 6, 11, 2, 4, 18, 17]
/DATA/nirbhay/tharun/with_occ/003/0.5/2/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/003/0.5/3/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/003/0.5/3/10.png
frame number  2
/DATA/nirbhay/tharun/with_occ/003/0.5/3/9.png
frame number  3
/DATA/nirbhay/tharun/with_occ/003/0.5/3/2.png
frame number  4
/DATA/nirbhay/tharun/with_occ/003/0.5/3/16.png
frame number  5
/DATA/nirbhay/tharun/with_occ/003/0.5/3/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/003/0.5/3/8.png
frame number  7
/DATA/nirbhay/tharun/with_occ/003/0.5/3/1.png
frame number  8
/DATA/nirbhay/tharun/with_occ/003/0.5/3/5.png
frame number  9
/DATA/nirbhay/tharun/with_occ/003/0.5/3/14.png
[15, 10, 9, 2, 16, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/003/0.5/3/non_occ_frms.txt
[19, 3, 18, 6, 13, 4, 0, 17, 12, 11]
/DATA/nirbhay/tharun/with_occ/003/0.5/3/occ_frms.txt
/DATA/nirbhay/tharun/with_occ/003/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/003/0.5/0/11.png
frame number  1
/DATA/nirbhay/tharun/with_occ/003/0.5/0/8.png
frame number  2
/DATA/nirbhay/tharun/with_occ/003/0.5/0/1.png
frame number  3
/DATA/nirbhay/tharun/with_occ/003/0.5/0/5.png
frame number  4
/DATA/nirbhay/tharun/with_occ/003/0.5/0/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/003/0.5/0/10.png
[11, 8, 1, 5, 7, 10]
/DATA/nirbhay/tharun/with_occ/003/0.5/0/non_occ_frms.txt
[3, 4, 6, 12, 9, 2, 0]
/DATA/nirbhay/tharun/with_occ/003/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/003/0.5/1/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/003/0.5/1/13.png
frame number  2
/DATA/nirbhay/tharun/with_occ/003/0.5/1/8.png
frame number  3
/DATA/nirbhay/tharun/with_occ/003/0.5/1/1.png
frame number  4
/DATA/nirbhay/tharun/with_occ/003/0.5/1/5.png
frame number  5
/DATA/nirbhay/tharun/with_occ/003/0.5/1/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/003/0.5/1/14.png
frame number  7
/DATA/nirbhay/tharun/with_occ/003/0.5/1/10.png
[15, 13, 8, 1, 5, 7, 14, 10]
/DATA/nirbhay/tharun/with_occ/003/0.5/1/non_occ_frms.txt
[12, 0, 6, 3, 4, 9, 11, 2]
/DATA/nirbhay/tharun/with_occ/003/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/003/0.5/2/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/003/0.5/2/17.png
frame number  2
/DATA/nirbhay/tharun/with_occ/003/0.5/2/25.png
frame number  3
/DATA/nirbhay/tharun/with_occ/003/0.5/2/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/003/0.5/2/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/003/0.5/2/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/003/0.5/2/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/003/0.5/2/21.png
frame number  8
/DATA/nirbhay/tharun/with_occ/003/0.5/2/8.png
frame number  9
/DATA/nirbhay/tharun/with_occ/003/0.5/2/1.png
frame number  10
/DATA/nirbhay/tharun/with_occ/003/0.5/2/5.png
frame number  11
/DATA/nirbhay/tharun/with_occ/003/0.5/2/23.png
frame number  12
/DATA/nirbhay/tharun/with_occ/003/0.5/2/14.png
[15, 17, 25, 13, 7, 9, 18, 21, 8, 1, 5, 23, 14]
/DATA/nirbhay/tharun/with_occ/003/0.5/2/non_occ_frms.txt
[24, 16, 22, 0, 3, 10, 12, 11, 6, 2, 4, 20, 19]
/DATA/nirbhay/tharun/with_occ/003/0.5/2/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/003/0.5/3/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/003/0.5/3/16.png
frame number  2
/DATA/nirbhay/tharun/with_occ/003/0.5/3/22.png
frame number  3
/DATA/nirbhay/tharun/with_occ/003/0.5/3/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/003/0.5/3/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/003/0.5/3/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/003/0.5/3/19.png
frame number  7
/DATA/nirbhay/tharun/with_occ/003/0.5/3/8.png
frame number  8
/DATA/nirbhay/tharun/with_occ/003/0.5/3/1.png
frame number  9
/DATA/nirbhay/tharun/with_occ/003/0.5/3/5.png
frame number  10
/DATA/nirbhay/tharun/with_occ/003/0.5/3/14.png
[15, 16, 22, 13, 7, 9, 19, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/003/0.5/3/non_occ_frms.txt
[21, 20, 0, 3, 12, 10, 6, 11, 2, 4, 18, 17]
/DATA/nirbhay/tharun/with_occ/003/0.5/3/occ_frms.txt
/DATA/nirbhay/tharun/with_occ/003/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/003/0.5/0/13.png
frame number  1
/DATA/nirbhay/tharun/with_occ/003/0.5/0/9.png
frame number  2
/DATA/nirbhay/tharun/with_occ/003/0.5/0/2.png
frame number  3
/DATA/nirbhay/tharun/with_occ/003/0.5/0/17.png
frame number  4
/DATA/nirbhay/tharun/with_occ/003/0.5/0/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/003/0.5/0/8.png
frame number  6
/DATA/nirbhay/tharun/with_occ/003/0.5/0/1.png
frame number  7
/DATA/nirbhay/tharun/with_occ/003/0.5/0/5.png
frame number  8
/DATA/nirbhay/tharun/with_occ/003/0.5/0/14.png
[13, 9, 2, 17, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/003/0.5/0/non_occ_frms.txt
[6, 3, 16, 11, 4, 0, 15, 12, 10]
/DATA/nirbhay/tharun/with_occ/003/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/003/0.5/1/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/003/0.5/1/16.png
frame number  2
/DATA/nirbhay/tharun/with_occ/003/0.5/1/22.png
frame number  3
/DATA/nirbhay/tharun/with_occ/003/0.5/1/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/003/0.5/1/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/003/0.5/1/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/003/0.5/1/19.png
frame number  7
/DATA/nirbhay/tharun/with_occ/003/0.5/1/8.png
frame number  8
/DATA/nirbhay/tharun/with_occ/003/0.5/1/1.png
frame number  9
/DATA/nirbhay/tharun/with_occ/003/0.5/1/5.png
frame number  10
/DATA/nirbhay/tharun/with_occ/003/0.5/1/14.png
[15, 16, 22, 13, 7, 9, 19, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/003/0.5/1/non_occ_frms.txt
[21, 20, 0, 3, 12, 10, 6, 11, 2, 4, 18, 17]
/DATA/nirbhay/tharun/with_occ/003/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/003/0.5/2/2.png
frame number  1
/DATA/nirbhay/tharun/with_occ/003/0.5/2/4.png
frame number  2
/DATA/nirbhay/tharun/with_occ/003/0.5/2/22.png
frame number  3
/DATA/nirbhay/tharun/with_occ/003/0.5/2/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/003/0.5/2/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/003/0.5/2/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/003/0.5/2/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/003/0.5/2/28.png
frame number  8
/DATA/nirbhay/tharun/with_occ/003/0.5/2/8.png
frame number  9
/DATA/nirbhay/tharun/with_occ/003/0.5/2/1.png
frame number  10
/DATA/nirbhay/tharun/with_occ/003/0.5/2/5.png
frame number  11
/DATA/nirbhay/tharun/with_occ/003/0.5/2/23.png
frame number  12
/DATA/nirbhay/tharun/with_occ/003/0.5/2/14.png
frame number  13
/DATA/nirbhay/tharun/with_occ/003/0.5/2/26.png
[2, 4, 22, 13, 7, 9, 18, 28, 8, 1, 5, 23, 14, 26]
/DATA/nirbhay/tharun/with_occ/003/0.5/2/non_occ_frms.txt
[11, 21, 27, 15, 20, 24, 17, 25, 19, 0, 3, 16, 10, 6, 12]
/DATA/nirbhay/tharun/with_occ/003/0.5/2/occ_frms.txt
/DATA/nirbhay/tharun/with_occ/003/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/003/0.5/0/11.png
frame number  1
/DATA/nirbhay/tharun/with_occ/003/0.5/0/8.png
frame number  2
/DATA/nirbhay/tharun/with_occ/003/0.5/0/1.png
frame number  3
/DATA/nirbhay/tharun/with_occ/003/0.5/0/5.png
frame number  4
/DATA/nirbhay/tharun/with_occ/003/0.5/0/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/003/0.5/0/10.png
[11, 8, 1, 5, 7, 10]
/DATA/nirbhay/tharun/with_occ/003/0.5/0/non_occ_frms.txt
[3, 4, 6, 12, 9, 2, 0]
/DATA/nirbhay/tharun/with_occ/003/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/003/0.5/1/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/003/0.5/1/10.png
frame number  2
/DATA/nirbhay/tharun/with_occ/003/0.5/1/9.png
frame number  3
/DATA/nirbhay/tharun/with_occ/003/0.5/1/2.png
frame number  4
/DATA/nirbhay/tharun/with_occ/003/0.5/1/16.png
frame number  5
/DATA/nirbhay/tharun/with_occ/003/0.5/1/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/003/0.5/1/8.png
frame number  7
/DATA/nirbhay/tharun/with_occ/003/0.5/1/1.png
frame number  8
/DATA/nirbhay/tharun/with_occ/003/0.5/1/5.png
frame number  9
/DATA/nirbhay/tharun/with_occ/003/0.5/1/14.png
[15, 10, 9, 2, 16, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/003/0.5/1/non_occ_frms.txt
[19, 3, 18, 6, 13, 4, 0, 17, 12, 11]
/DATA/nirbhay/tharun/with_occ/003/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/003/0.5/2/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/003/0.5/2/16.png
frame number  2
/DATA/nirbhay/tharun/with_occ/003/0.5/2/22.png
frame number  3
/DATA/nirbhay/tharun/with_occ/003/0.5/2/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/003/0.5/2/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/003/0.5/2/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/003/0.5/2/19.png
frame number  7
/DATA/nirbhay/tharun/with_occ/003/0.5/2/8.png
frame number  8
/DATA/nirbhay/tharun/with_occ/003/0.5/2/1.png
frame number  9
/DATA/nirbhay/tharun/with_occ/003/0.5/2/5.png
frame number  10
/DATA/nirbhay/tharun/with_occ/003/0.5/2/14.png
[15, 16, 22, 13, 7, 9, 19, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/003/0.5/2/non_occ_frms.txt
[21, 20, 0, 3, 12, 10, 6, 11, 2, 4, 18, 17]
/DATA/nirbhay/tharun/with_occ/003/0.5/2/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/003/0.5/3/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/003/0.5/3/17.png
frame number  2
/DATA/nirbhay/tharun/with_occ/003/0.5/3/25.png
frame number  3
/DATA/nirbhay/tharun/with_occ/003/0.5/3/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/003/0.5/3/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/003/0.5/3/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/003/0.5/3/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/003/0.5/3/21.png
frame number  8
/DATA/nirbhay/tharun/with_occ/003/0.5/3/8.png
frame number  9
/DATA/nirbhay/tharun/with_occ/003/0.5/3/1.png
frame number  10
/DATA/nirbhay/tharun/with_occ/003/0.5/3/5.png
frame number  11
/DATA/nirbhay/tharun/with_occ/003/0.5/3/23.png
frame number  12
/DATA/nirbhay/tharun/with_occ/003/0.5/3/14.png
[15, 17, 25, 13, 7, 9, 18, 21, 8, 1, 5, 23, 14]
/DATA/nirbhay/tharun/with_occ/003/0.5/3/non_occ_frms.txt
[24, 16, 22, 0, 3, 10, 12, 11, 6, 2, 4, 20, 19]
/DATA/nirbhay/tharun/with_occ/003/0.5/3/occ_frms.txt
********************
/DATA/nirbhay/tharun/with_occ/004/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/0/14.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/0/13.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/0/8.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/0/1.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/0/5.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/0/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/0/10.png
[14, 13, 8, 1, 5, 7, 10]
/DATA/nirbhay/tharun/with_occ/004/0.5/0/non_occ_frms.txt
[12, 0, 6, 3, 4, 9, 11, 2]
/DATA/nirbhay/tharun/with_occ/004/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/1/10.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/1/9.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/1/2.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/1/15.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/1/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/1/8.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/1/1.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/1/5.png
frame number  8
/DATA/nirbhay/tharun/with_occ/004/0.5/1/14.png
[10, 9, 2, 15, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/004/0.5/1/non_occ_frms.txt
[6, 3, 17, 12, 4, 0, 16, 13, 11, 18]
/DATA/nirbhay/tharun/with_occ/004/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/2/14.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/2/13.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/2/8.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/2/1.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/2/5.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/2/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/2/10.png
[14, 13, 8, 1, 5, 7, 10]
/DATA/nirbhay/tharun/with_occ/004/0.5/2/non_occ_frms.txt
[12, 0, 6, 3, 4, 9, 11, 2]
/DATA/nirbhay/tharun/with_occ/004/0.5/2/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/3/2.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/3/12.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/3/15.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/3/8.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/3/1.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/3/5.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/3/7.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/3/14.png
[2, 12, 15, 8, 1, 5, 7, 14]
/DATA/nirbhay/tharun/with_occ/004/0.5/3/non_occ_frms.txt
[10, 3, 9, 4, 0, 13, 6, 16, 11]
/DATA/nirbhay/tharun/with_occ/004/0.5/3/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/4/21.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/4/13.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/4/7.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/4/12.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/4/10.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/4/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/4/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/4/8.png
frame number  8
/DATA/nirbhay/tharun/with_occ/004/0.5/4/1.png
frame number  9
/DATA/nirbhay/tharun/with_occ/004/0.5/4/5.png
frame number  10
/DATA/nirbhay/tharun/with_occ/004/0.5/4/14.png
[21, 13, 7, 12, 10, 9, 18, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/004/0.5/4/non_occ_frms.txt
[20, 19, 0, 3, 6, 11, 15, 2, 4, 17, 16]
/DATA/nirbhay/tharun/with_occ/004/0.5/4/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/5/2.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/5/12.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/5/15.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/5/8.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/5/1.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/5/5.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/5/7.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/5/14.png
[2, 12, 15, 8, 1, 5, 7, 14]
/DATA/nirbhay/tharun/with_occ/004/0.5/5/non_occ_frms.txt
[10, 3, 9, 4, 0, 13, 6, 16, 11]
/DATA/nirbhay/tharun/with_occ/004/0.5/5/occ_frms.txt
/DATA/nirbhay/tharun/with_occ/004/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/0/17.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/0/25.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/0/13.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/0/7.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/0/9.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/0/18.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/0/21.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/0/8.png
frame number  8
/DATA/nirbhay/tharun/with_occ/004/0.5/0/1.png
frame number  9
/DATA/nirbhay/tharun/with_occ/004/0.5/0/5.png
frame number  10
/DATA/nirbhay/tharun/with_occ/004/0.5/0/23.png
frame number  11
/DATA/nirbhay/tharun/with_occ/004/0.5/0/14.png
frame number  12
/DATA/nirbhay/tharun/with_occ/004/0.5/0/26.png
[17, 25, 13, 7, 9, 18, 21, 8, 1, 5, 23, 14, 26]
/DATA/nirbhay/tharun/with_occ/004/0.5/0/non_occ_frms.txt
[24, 16, 22, 0, 3, 10, 12, 11, 6, 2, 4, 20, 19, 15]
/DATA/nirbhay/tharun/with_occ/004/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/1/21.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/1/13.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/1/7.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/1/12.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/1/10.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/1/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/1/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/1/8.png
frame number  8
/DATA/nirbhay/tharun/with_occ/004/0.5/1/1.png
frame number  9
/DATA/nirbhay/tharun/with_occ/004/0.5/1/5.png
frame number  10
/DATA/nirbhay/tharun/with_occ/004/0.5/1/14.png
[21, 13, 7, 12, 10, 9, 18, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/004/0.5/1/non_occ_frms.txt
[20, 19, 0, 3, 6, 11, 15, 2, 4, 17, 16]
/DATA/nirbhay/tharun/with_occ/004/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/2/7.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/2/12.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/2/10.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/2/9.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/2/2.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/2/17.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/2/8.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/2/1.png
frame number  8
/DATA/nirbhay/tharun/with_occ/004/0.5/2/5.png
frame number  9
/DATA/nirbhay/tharun/with_occ/004/0.5/2/14.png
[7, 12, 10, 9, 2, 17, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/004/0.5/2/non_occ_frms.txt
[19, 18, 3, 0, 6, 20, 15, 4, 16, 13, 11]
/DATA/nirbhay/tharun/with_occ/004/0.5/2/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/3/14.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/3/13.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/3/8.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/3/1.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/3/5.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/3/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/3/10.png
[14, 13, 8, 1, 5, 7, 10]
/DATA/nirbhay/tharun/with_occ/004/0.5/3/non_occ_frms.txt
[12, 0, 6, 3, 4, 9, 11, 2]
/DATA/nirbhay/tharun/with_occ/004/0.5/3/occ_frms.txt
/DATA/nirbhay/tharun/with_occ/004/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/0/11.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/0/8.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/0/1.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/0/5.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/0/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/0/10.png
[11, 8, 1, 5, 7, 10]
/DATA/nirbhay/tharun/with_occ/004/0.5/0/non_occ_frms.txt
[3, 4, 6, 12, 9, 2, 0]
/DATA/nirbhay/tharun/with_occ/004/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/1/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/1/13.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/1/8.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/1/1.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/1/5.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/1/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/1/14.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/1/10.png
[15, 13, 8, 1, 5, 7, 14, 10]
/DATA/nirbhay/tharun/with_occ/004/0.5/1/non_occ_frms.txt
[12, 0, 6, 3, 4, 9, 11, 2]
/DATA/nirbhay/tharun/with_occ/004/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/2/7.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/2/12.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/2/10.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/2/9.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/2/2.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/2/17.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/2/8.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/2/1.png
frame number  8
/DATA/nirbhay/tharun/with_occ/004/0.5/2/5.png
frame number  9
/DATA/nirbhay/tharun/with_occ/004/0.5/2/14.png
[7, 12, 10, 9, 2, 17, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/004/0.5/2/non_occ_frms.txt
[19, 18, 3, 0, 6, 20, 15, 4, 16, 13, 11]
/DATA/nirbhay/tharun/with_occ/004/0.5/2/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/3/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/3/10.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/3/9.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/3/2.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/3/16.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/3/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/3/8.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/3/1.png
frame number  8
/DATA/nirbhay/tharun/with_occ/004/0.5/3/5.png
frame number  9
/DATA/nirbhay/tharun/with_occ/004/0.5/3/14.png
[15, 10, 9, 2, 16, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/004/0.5/3/non_occ_frms.txt
[19, 3, 18, 6, 13, 4, 0, 17, 12, 11]
/DATA/nirbhay/tharun/with_occ/004/0.5/3/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/4/10.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/4/9.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/4/2.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/4/15.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/4/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/4/8.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/4/1.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/4/5.png
frame number  8
/DATA/nirbhay/tharun/with_occ/004/0.5/4/14.png
[10, 9, 2, 15, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/004/0.5/4/non_occ_frms.txt
[6, 3, 17, 12, 4, 0, 16, 13, 11, 18]
/DATA/nirbhay/tharun/with_occ/004/0.5/4/occ_frms.txt
/DATA/nirbhay/tharun/with_occ/004/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/0/13.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/0/9.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/0/2.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/0/17.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/0/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/0/8.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/0/1.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/0/5.png
frame number  8
/DATA/nirbhay/tharun/with_occ/004/0.5/0/14.png
[13, 9, 2, 17, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/004/0.5/0/non_occ_frms.txt
[6, 3, 16, 11, 4, 0, 15, 12, 10]
/DATA/nirbhay/tharun/with_occ/004/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/1/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/1/16.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/1/22.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/1/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/1/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/1/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/1/19.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/1/8.png
frame number  8
/DATA/nirbhay/tharun/with_occ/004/0.5/1/1.png
frame number  9
/DATA/nirbhay/tharun/with_occ/004/0.5/1/5.png
frame number  10
/DATA/nirbhay/tharun/with_occ/004/0.5/1/14.png
[15, 16, 22, 13, 7, 9, 19, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/004/0.5/1/non_occ_frms.txt
[21, 20, 0, 3, 12, 10, 6, 11, 2, 4, 18, 17]
/DATA/nirbhay/tharun/with_occ/004/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/2/2.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/2/12.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/2/15.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/2/8.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/2/1.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/2/5.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/2/7.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/2/14.png
[2, 12, 15, 8, 1, 5, 7, 14]
/DATA/nirbhay/tharun/with_occ/004/0.5/2/non_occ_frms.txt
[10, 3, 9, 4, 0, 13, 6, 16, 11]
/DATA/nirbhay/tharun/with_occ/004/0.5/2/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/3/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/3/13.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/3/8.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/3/1.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/3/5.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/3/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/3/14.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/3/10.png
[15, 13, 8, 1, 5, 7, 14, 10]
/DATA/nirbhay/tharun/with_occ/004/0.5/3/non_occ_frms.txt
[12, 0, 6, 3, 4, 9, 11, 2]
/DATA/nirbhay/tharun/with_occ/004/0.5/3/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/4/7.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/4/12.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/4/10.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/4/9.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/4/2.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/4/17.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/4/8.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/4/1.png
frame number  8
/DATA/nirbhay/tharun/with_occ/004/0.5/4/5.png
frame number  9
/DATA/nirbhay/tharun/with_occ/004/0.5/4/14.png
[7, 12, 10, 9, 2, 17, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/004/0.5/4/non_occ_frms.txt
[19, 18, 3, 0, 6, 20, 15, 4, 16, 13, 11]
/DATA/nirbhay/tharun/with_occ/004/0.5/4/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/5/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/5/10.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/5/9.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/5/2.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/5/16.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/5/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/5/8.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/5/1.png
frame number  8
/DATA/nirbhay/tharun/with_occ/004/0.5/5/5.png
frame number  9
/DATA/nirbhay/tharun/with_occ/004/0.5/5/14.png
[15, 10, 9, 2, 16, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/004/0.5/5/non_occ_frms.txt
[19, 3, 18, 6, 13, 4, 0, 17, 12, 11]
/DATA/nirbhay/tharun/with_occ/004/0.5/5/occ_frms.txt
/DATA/nirbhay/tharun/with_occ/004/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/0/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/0/13.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/0/8.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/0/1.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/0/5.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/0/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/0/14.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/0/10.png
[15, 13, 8, 1, 5, 7, 14, 10]
/DATA/nirbhay/tharun/with_occ/004/0.5/0/non_occ_frms.txt
[12, 0, 6, 3, 4, 9, 11, 2]
/DATA/nirbhay/tharun/with_occ/004/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/1/2.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/1/4.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/1/21.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/1/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/1/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/1/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/1/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/1/22.png
frame number  8
/DATA/nirbhay/tharun/with_occ/004/0.5/1/8.png
frame number  9
/DATA/nirbhay/tharun/with_occ/004/0.5/1/1.png
frame number  10
/DATA/nirbhay/tharun/with_occ/004/0.5/1/5.png
frame number  11
/DATA/nirbhay/tharun/with_occ/004/0.5/1/23.png
frame number  12
/DATA/nirbhay/tharun/with_occ/004/0.5/1/14.png
frame number  13
/DATA/nirbhay/tharun/with_occ/004/0.5/1/26.png
[2, 4, 21, 13, 7, 9, 18, 22, 8, 1, 5, 23, 14, 26]
/DATA/nirbhay/tharun/with_occ/004/0.5/1/non_occ_frms.txt
[19, 16, 11, 27, 25, 17, 24, 20, 0, 3, 15, 10, 6, 12]
/DATA/nirbhay/tharun/with_occ/004/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/2/2.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/2/4.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/2/22.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/2/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/2/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/2/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/2/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/2/24.png
frame number  8
/DATA/nirbhay/tharun/with_occ/004/0.5/2/27.png
frame number  9
/DATA/nirbhay/tharun/with_occ/004/0.5/2/8.png
frame number  10
/DATA/nirbhay/tharun/with_occ/004/0.5/2/1.png
frame number  11
/DATA/nirbhay/tharun/with_occ/004/0.5/2/5.png
frame number  12
/DATA/nirbhay/tharun/with_occ/004/0.5/2/23.png
frame number  13
/DATA/nirbhay/tharun/with_occ/004/0.5/2/14.png
frame number  14
/DATA/nirbhay/tharun/with_occ/004/0.5/2/26.png
[2, 4, 22, 13, 7, 9, 18, 24, 27, 8, 1, 5, 23, 14, 26]
/DATA/nirbhay/tharun/with_occ/004/0.5/2/non_occ_frms.txt
[11, 21, 28, 15, 20, 25, 17, 29, 19, 0, 3, 16, 10, 6, 12]
/DATA/nirbhay/tharun/with_occ/004/0.5/2/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/3/2.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/3/12.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/3/15.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/3/8.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/3/1.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/3/5.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/3/7.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/3/14.png
[2, 12, 15, 8, 1, 5, 7, 14]
/DATA/nirbhay/tharun/with_occ/004/0.5/3/non_occ_frms.txt
[10, 3, 9, 4, 0, 13, 6, 16, 11]
/DATA/nirbhay/tharun/with_occ/004/0.5/3/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/4/13.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/4/9.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/4/2.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/4/17.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/4/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/4/8.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/4/1.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/4/5.png
frame number  8
/DATA/nirbhay/tharun/with_occ/004/0.5/4/14.png
[13, 9, 2, 17, 7, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/004/0.5/4/non_occ_frms.txt
[6, 3, 16, 11, 4, 0, 15, 12, 10]
/DATA/nirbhay/tharun/with_occ/004/0.5/4/occ_frms.txt
/DATA/nirbhay/tharun/with_occ/004/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/0/0.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/0/12.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/0/8.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/0/1.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/0/5.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/0/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/0/10.png
[0, 12, 8, 1, 5, 7, 10]
/DATA/nirbhay/tharun/with_occ/004/0.5/0/non_occ_frms.txt
[4, 3, 11, 9, 6, 13, 2]
/DATA/nirbhay/tharun/with_occ/004/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/1/7.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/1/12.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/1/10.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/1/9.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/1/2.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/1/17.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/1/8.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/1/1.png
frame number  8
/DATA/nirbhay/tharun/with_occ/004/0.5/1/5.png
frame number  9
/DATA/nirbhay/tharun/with_occ/004/0.5/1/14.png
[7, 12, 10, 9, 2, 17, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/004/0.5/1/non_occ_frms.txt
[19, 18, 3, 0, 6, 20, 15, 4, 16, 13, 11]
/DATA/nirbhay/tharun/with_occ/004/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/2/2.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/2/4.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/2/22.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/2/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/2/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/2/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/2/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/2/24.png
frame number  8
/DATA/nirbhay/tharun/with_occ/004/0.5/2/27.png
frame number  9
/DATA/nirbhay/tharun/with_occ/004/0.5/2/8.png
frame number  10
/DATA/nirbhay/tharun/with_occ/004/0.5/2/1.png
frame number  11
/DATA/nirbhay/tharun/with_occ/004/0.5/2/5.png
frame number  12
/DATA/nirbhay/tharun/with_occ/004/0.5/2/23.png
frame number  13
/DATA/nirbhay/tharun/with_occ/004/0.5/2/14.png
frame number  14
/DATA/nirbhay/tharun/with_occ/004/0.5/2/26.png
[2, 4, 22, 13, 7, 9, 18, 24, 27, 8, 1, 5, 23, 14, 26]
/DATA/nirbhay/tharun/with_occ/004/0.5/2/non_occ_frms.txt
[11, 21, 28, 15, 20, 25, 17, 29, 19, 0, 3, 16, 10, 6, 12]
/DATA/nirbhay/tharun/with_occ/004/0.5/2/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/3/15.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/3/17.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/3/25.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/3/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/3/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/004/0.5/3/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/004/0.5/3/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/004/0.5/3/21.png
frame number  8
/DATA/nirbhay/tharun/with_occ/004/0.5/3/8.png
frame number  9
/DATA/nirbhay/tharun/with_occ/004/0.5/3/1.png
frame number  10
/DATA/nirbhay/tharun/with_occ/004/0.5/3/5.png
frame number  11
/DATA/nirbhay/tharun/with_occ/004/0.5/3/23.png
frame number  12
/DATA/nirbhay/tharun/with_occ/004/0.5/3/14.png
[15, 17, 25, 13, 7, 9, 18, 21, 8, 1, 5, 23, 14]
/DATA/nirbhay/tharun/with_occ/004/0.5/3/non_occ_frms.txt
[24, 16, 22, 0, 3, 10, 12, 11, 6, 2, 4, 20, 19]
/DATA/nirbhay/tharun/with_occ/004/0.5/3/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/4/0.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/4/1.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/4/5.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/4/7.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/4/10.png
[0, 1, 5, 7, 10]
/DATA/nirbhay/tharun/with_occ/004/0.5/4/non_occ_frms.txt
[3, 8, 4, 9, 2, 6]
/DATA/nirbhay/tharun/with_occ/004/0.5/4/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/004/0.5/5/6.png
frame number  1
/DATA/nirbhay/tharun/with_occ/004/0.5/5/0.png
frame number  2
/DATA/nirbhay/tharun/with_occ/004/0.5/5/1.png
frame number  3
/DATA/nirbhay/tharun/with_occ/004/0.5/5/5.png
frame number  4
/DATA/nirbhay/tharun/with_occ/004/0.5/5/7.png
[6, 0, 1, 5, 7]
/DATA/nirbhay/tharun/with_occ/004/0.5/5/non_occ_frms.txt
[3, 8, 4, 9, 2]
/DATA/nirbhay/tharun/with_occ/004/0.5/5/occ_frms.txt
********************
/DATA/nirbhay/tharun/with_occ/005/0.5
/DATA/nirbhay/tharun/with_occ/005/0.5
/DATA/nirbhay/tharun/with_occ/005/0.5
/DATA/nirbhay/tharun/with_occ/005/0.5
/DATA/nirbhay/tharun/with_occ/005/0.5
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/005/0.5/0/0.png
frame number  1
/DATA/nirbhay/tharun/with_occ/005/0.5/0/12.png
frame number  2
/DATA/nirbhay/tharun/with_occ/005/0.5/0/8.png
frame number  3
/DATA/nirbhay/tharun/with_occ/005/0.5/0/1.png
frame number  4
/DATA/nirbhay/tharun/with_occ/005/0.5/0/5.png
frame number  5
/DATA/nirbhay/tharun/with_occ/005/0.5/0/7.png
frame number  6
/DATA/nirbhay/tharun/with_occ/005/0.5/0/10.png
[0, 12, 8, 1, 5, 7, 10]
/DATA/nirbhay/tharun/with_occ/005/0.5/0/non_occ_frms.txt
[4, 3, 11, 9, 6, 13, 2]
/DATA/nirbhay/tharun/with_occ/005/0.5/0/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/005/0.5/1/21.png
frame number  1
/DATA/nirbhay/tharun/with_occ/005/0.5/1/13.png
frame number  2
/DATA/nirbhay/tharun/with_occ/005/0.5/1/7.png
frame number  3
/DATA/nirbhay/tharun/with_occ/005/0.5/1/12.png
frame number  4
/DATA/nirbhay/tharun/with_occ/005/0.5/1/10.png
frame number  5
/DATA/nirbhay/tharun/with_occ/005/0.5/1/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/005/0.5/1/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/005/0.5/1/8.png
frame number  8
/DATA/nirbhay/tharun/with_occ/005/0.5/1/1.png
frame number  9
/DATA/nirbhay/tharun/with_occ/005/0.5/1/5.png
frame number  10
/DATA/nirbhay/tharun/with_occ/005/0.5/1/14.png
[21, 13, 7, 12, 10, 9, 18, 8, 1, 5, 14]
/DATA/nirbhay/tharun/with_occ/005/0.5/1/non_occ_frms.txt
[20, 19, 0, 3, 6, 11, 15, 2, 4, 17, 16]
/DATA/nirbhay/tharun/with_occ/005/0.5/1/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/005/0.5/2/2.png
frame number  1
/DATA/nirbhay/tharun/with_occ/005/0.5/2/4.png
frame number  2
/DATA/nirbhay/tharun/with_occ/005/0.5/2/21.png
frame number  3
/DATA/nirbhay/tharun/with_occ/005/0.5/2/13.png
frame number  4
/DATA/nirbhay/tharun/with_occ/005/0.5/2/7.png
frame number  5
/DATA/nirbhay/tharun/with_occ/005/0.5/2/9.png
frame number  6
/DATA/nirbhay/tharun/with_occ/005/0.5/2/18.png
frame number  7
/DATA/nirbhay/tharun/with_occ/005/0.5/2/22.png
frame number  8
/DATA/nirbhay/tharun/with_occ/005/0.5/2/8.png
frame number  9
/DATA/nirbhay/tharun/with_occ/005/0.5/2/1.png
frame number  10
/DATA/nirbhay/tharun/with_occ/005/0.5/2/5.png
frame number  11
/DATA/nirbhay/tharun/with_occ/005/0.5/2/23.png
frame number  12
/DATA/nirbhay/tharun/with_occ/005/0.5/2/14.png
frame number  13
/DATA/nirbhay/tharun/with_occ/005/0.5/2/26.png
[2, 4, 21, 13, 7, 9, 18, 22, 8, 1, 5, 23, 14, 26]
/DATA/nirbhay/tharun/with_occ/005/0.5/2/non_occ_frms.txt
[19, 16, 11, 27, 25, 17, 24, 20, 0, 3, 15, 10, 6, 12]
/DATA/nirbhay/tharun/with_occ/005/0.5/2/occ_frms.txt
non_occ True
frame number  0
/DATA/nirbhay/tharun/with_occ/005/0.5/3/17.png
frame number  1
/DATA/nirbhay/tharun/with_occ/005/0.5/3/25.png
frame number  2
/DATA/nirbhay/tharun/with_occ/005/0.5/3/13.png
frame number  3
/DATA/nirbhay/tharun/with_occ/005/0.5/3/7.png
frame number  4
/DATA/nirbhay/tharun/with_occ/005/0.5/3/9.png
frame number  5
/DATA/nirbhay/tharun/with_occ/005/0.5/3/18.png
frame number  6
/DATA/nirbhay/tharun/with_occ/005/0.5/3/21.png
frame number  7
/DATA/nirbhay/tharun/with_occ/005/0.5/3/8.png
frame number  8
/DATA/nirbhay/tharun/with_occ/005/0.5/3/1.png
frame number  9
/DATA/nirbhay/tharun/with_occ/005/0.5/3/5.png
frame number  10
/DATA/nirbhay/tharun/with_occ/005/0.5/3/23.png
frame number  11
/DATA/nirbhay/tharun/with_occ/005/0.5/3/14.png
frame number  12
/DATA/nirbhay/tharun/with_occ/005/0.5/3/26.png
[17, 25, 13, 7, 9, 18, 21, 8, 1, 5, 23, 14, 26]
/DATA/nirbhay/tharun/with_occ/005/0.5/3/non_occ_frms.txt
[24, 16, 22, 0, 3, 10, 12, 11, 6, 2, 4, 20, 19, 15]
/DATA/nirbhay/tharun/with_occ/005/0.5/3/occ_frms.txt
/DATA/nirbhay/tharun/with_occ/005/0.5
********************
 
