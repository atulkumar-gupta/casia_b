import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


def preprocessing(file):
	
	#load image as gray scale
    img = cv2.imread(file)
    im_bw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    #gradient and contour
    (thresh, im_bw) = cv2.threshold(im_bw, 127, 255, 0)
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours)==0:
    	return -1
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
    #position of head's center
    mean = int(mean/new_img.shape[0])

    #check shift
    #if its towards left add left
    if mean < new_img.shape[1]/2:
        add = new_img.shape[1] - 2*mean
        val = np.zeros((new_img.shape[0],add))

        cent = np.c_[val,new_img]
    #right add right
    else :
        add = (2*mean - new_img.shape[1])//3
        val = np.zeros((new_img.shape[0],add))

        cent = np.c_[new_img,val]

    #resize image using pil
    pil_img = Image.fromarray(cent)
    pil_img = pil_img.resize((75,150))
    print(file)
    
    return pil_img


if __name__ == "__main__":
	#create folder if not present to store new images
	root = "/DATA/nirbhay/tharun/dataset_CASIA"
	if os.path.isdir(root) == False:
	    os.mkdir(root)

	#example list of paths of images
	files = glob.glob('/SSD/Pratik/Gait_Data/GaitDatasetB-silh/001/001/nm-01/090/*.png')

	#iterate overal all ppl folders
	for i in range(30,125):
	    if i<10:
	        app = "00"+str(i)
	    elif i<100:
	        app = "0"+str(i)
	    else :
	        app = str(i)
	    #from image directory
	    dst_folder = "/SSD/Pratik/Gait_Data/GaitDatasetB-silh/"+app+"/"+app+"/"
	    #save directory
	    sv_folder = root+"/"+app
	    
	    #create folder for each person
	    if os.path.isdir(sv_folder) == False:
	        os.mkdir(sv_folder)
	    
	    #nms
	    for j in range(1,7):
	        dst_subfolder = dst_folder+"nm-0"+str(j)+"/090/"
	        sv_subfolder = sv_folder+"/nm-0"+str(j)
	        
	        #create nm folder
	        if os.path.isdir(sv_subfolder) == False:
	            os.mkdir(sv_subfolder)
	        
	        #all image files in source img folder
	        files  = glob.glob(dst_subfolder+"*.png")
	        #loop over all images
	        for file in files:
	        	#find image name
	            label = file.replace(dst_subfolder,'')
	            #preprocess
	            image = preprocessing(file)
	            if image == -1:
	            	continue
	            #convert to gray, must as it was converted to pil image
	            image_gry = ImageOps.grayscale(image)
	            #save image
	            name = sv_subfolder+"/"+label
	            image_gry.save(name)
	            print(name)
