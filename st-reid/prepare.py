import os
from shutil import copyfile
import shutil

#market1501 personid_cameraid_timestap_
def prepare():
	download_path = "/home/nirbhay/tharun/dataset/Market-1501-v15.09.15"

	if not os.path.isdir(download_path):
		print('no such folder ', download_path)

	save_path = "/home/nirbhay/tharun/dataset/M1501_prepare/"

	if not os.path.exists(save_path):
		os.makedirs(save_path)

	#---- query ---------
	query_path =  download_path+'/query'
	if not os.path.isdir(query_path):
		print('no such folder ', query_path)

	query_save_path = save_path + '/query'
	if not os.path.exists(query_save_path):
		os.makedirs(query_save_path)

	for roots, dirs, files in os.walk(query_path, topdown=True):
		for name in files:
			if not name[-3:] == 'jpg':
				continue
			ID = name.split('_')
			src_path = query_path + '/' + name
			dst_path = query_save_path + '/' + ID[0]
			if not os.path.isdir(dst_path):
				os.makedirs(dst_path)
			copyfile(src_path,dst_path+'/'+name)

	#---- gallery -------
	gallery_path = download_path + '/bounding_box_test'
	gallery_save_path = save_path + '/gallery'
	if not os.path.exists(gallery_save_path):
		os.makedirs(gallery_save_path)

	for root, dirs, files in os.walk(gallery_path, topdown=True):
		for name in files:
			if not name[-3:] == 'jpg':
				continue
			ID = name.split('_')
			src_path = gallery_path + '/' + name
			dst_path = gallery_save_path + '/' + ID[0]
			if not os.path.isdir(dst_path):
				os.mkdir(dst_path)
			copyfile(src_path, dst_path + '/' + name)

	#----- train_all--------
	train_path = download_path + '/bounding_box_train'
	train_save_path = save_path + '/train_all'
	if not os.path.exists(train_save_path):
		os.makedirs(train_save_path)

	for root, dirs, files in os.walk(train_path, topdown=True):
		for name in files:
			if not name[-3:] == 'jpg':
				continue
			ID = name.split('_')
			src_path = train_path + '/' + name
			dst_path = train_save_path + '/' + ID[0]
			if not os.path.isdir(dst_path):
				os.mkdir(dst_path)
			copyfile(src_path, dst_path + '/' + name)

	# ----- train&val ------------
	train_path = download_path + '/bounding_box_train'
	train_save_path = save_path + '/train'
	val_save_path = save_path + '/val'
	if not os.path.exists(train_save_path):
		os.makedirs(train_save_path)
		os.makedirs(val_save_path)

	for root, dirs, files in os.walk(train_path, topdown=True):
		for name in files:
			if not name[-3:] == 'jpg':
				continue
			ID = name.split('_')
			src_path = train_path + '/' + name
			dst_path = train_save_path + '/' + ID[0]
			if not os.path.isdir(dst_path):
				os.mkdir(dst_path)
				dst_path = val_save_path + '/' + ID[0]  # first image is used as val image
				os.mkdir(dst_path)
			copyfile(src_path, dst_path + '/' + name)


#-------- market rename -------

rename = '/home/nirbhay/tharun/dataset/M1501_rename'

def parse_frame(imgname, dict_cam_seq_max={}):
	
	dict_cam_seq_max = {
		11: 72681, 12: 74546, 13: 74881, 14: 74661, 15: 74891, 16: 54346, 17: 0, 18: 0,
		21: 163691, 22: 164677, 23: 98102, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0,
		31: 161708, 32: 161769, 33: 104469, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0,
		41: 72107, 42: 72373, 43: 74810, 44: 74541, 45: 74910, 46: 50616, 47: 0, 48: 0,
		51: 161095, 52: 161724, 53: 103487, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0,
		61: 87551, 62: 131268, 63: 95817, 64: 30952, 65: 0, 66: 0, 67: 0, 68: 0}

	#imgname = '1490_c6s3_085667_00.jpg'
	fid = imgname.strip().split("_")[0] #1409
	cam = int(imgname.strip().split("_")[1][1]) #6
	seq = int(imgname.strip().split("_")[1][3]) #3
	frame = int(imgname.strip().split("_")[2]) #085667
	count = imgname.strip().split("_")[-1] #00.jpg

	re = 0
	for i in range(1,seq):
		re = re + dict_cam_seq_max[int(str(cam)+str(i))]
	re += frame
	new_name = str(fid) + "_c" + str(cam) + "_f" + "{:0>7}".format(str(re)) + "_"+ count

	return new_name


def gen_train_all_rename():
	path = "/home/nirbhay/tharun/dataset/M1501_prepare/train_all/"
	folderName = []
	for root, dirs, files in os.walk(path):
		folderName = dirs
		break
	# print(len(folderName))

	for fname in folderName:
		# print(fname)

		if not os.path.exists("/home/nirbhay/tharun/dataset/M1501_rename/train_all/" + fname):
			os.makedirs("/home/nirbhay/tharun/dataset/M1501_rename/train_all/" + fname)

		img_names = []
		for root, dirs, files in os.walk(path + fname):
			img_names = files
			break
		# print(img_names)
		# print(len(img_names))
		for imgname in img_names:
			newname = parse_frame(imgname)
			# print(newname)
			srcfile = path + fname + "/" + imgname
			dstfile = "/home/nirbhay/tharun/dataset/M1501_rename/train_all/" + fname + "/" + newname
			shutil.copyfile(srcfile, dstfile)

def gen_train_rename():

	path = "/home/nirbhay/tharun/dataset/M1501_prepare/train_all/"

	folderName = []

	for root, dirs, files in os.walk(path):
		folderName = dirs
		break
	# print(len(folderName))

	for fname in folderName:
		# print(fname)

		if not os.path.exists("/home/nirbhay/tharun/dataset/M1501_rename/train/" + fname):
			os.makedirs("/home/nirbhay/tharun/dataset/M1501_rename/train/" + fname)

		img_names = []
		for root, dirs, files in os.walk(path + fname):
			img_names = files
			break
		
		for imgname in img_names:
			newname = parse_frame(imgname)
			# print(newname)
			srcfile = path + fname + "/" + imgname
			dstfile = "/home/nirbhay/tharun/dataset/M1501_rename/train/" + fname + "/" + newname
			shutil.copyfile(srcfile, dstfile)
			


def gen_val_rename():
	path = "/home/nirbhay/tharun/dataset/M1501_prepare/val/"
	folderName = []
	for root, dirs, files in os.walk(path):
		folderName = dirs
		break
	# print(len(folderName))

	for fname in folderName:
		# print(fname)

		if not os.path.exists("/home/nirbhay/tharun/dataset/M1501_rename/val/" + fname):
			os.makedirs("/home/nirbhay/tharun/dataset/M1501_rename/val/" + fname)

		img_names = []
		for root, dirs, files in os.walk(path + fname):
			img_names = files
			break
		# print(img_names)
		# print(len(img_names))
		for imgname in img_names:
			newname = parse_frame(imgname)
			# print(newname)
			srcfile = path + fname + "/" + imgname
			dstfile = "/home/nirbhay/tharun/dataset/M1501_rename/val/" + fname + "/" + newname
			shutil.copyfile(srcfile, dstfile)


def gen_query_rename():
	path = "/home/nirbhay/tharun/dataset/M1501_prepare/query/"
	folderName = []
	for root, dirs, files in os.walk(path):
		folderName = dirs
		break
	# print(len(folderName))

	for fname in folderName:
		# print(fname)

		if not os.path.exists("/home/nirbhay/tharun/dataset/M1501_rename/query/" + fname):
			os.makedirs("/home/nirbhay/tharun/dataset/M1501_rename/query/" + fname)

		img_names = []
		for root, dirs, files in os.walk(path + fname):
			img_names = files
			break
		# print(img_names)
		# print(len(img_names))
		for imgname in img_names:
			newname = parse_frame(imgname)
			# print(newname)
			srcfile = path + fname + "/" + imgname
			dstfile = "/home/nirbhay/tharun/dataset/M1501_rename/query/" + fname + "/" + newname
			shutil.copyfile(srcfile, dstfile)


def gen_gallery_rename():
	path = "/home/nirbhay/tharun/dataset/M1501_prepare/gallery/"
	folderName = []
	for root, dirs, files in os.walk(path):
		folderName = dirs
		break
	# print(len(folderName))

	for fname in folderName:
		# print(fname)

		if not os.path.exists("/home/nirbhay/tharun/dataset/M1501_rename/gallery/" + fname):
			os.makedirs("/home/nirbhay/tharun/dataset/M1501_rename/gallery/" + fname)

		img_names = []
		for root, dirs, files in os.walk(path + fname):
			img_names = files
			break
		# print(img_names)
		# print(len(img_names))
		for imgname in img_names:
			newname = parse_frame(imgname)
			# print(newname)
			srcfile = path + fname + "/" + imgname
			dstfile = "/home/nirbhay/tharun/dataset/M1501_rename/gallery/" + fname + "/" + newname
			shutil.copyfile(srcfile, dstfile)


if __name__ == "__main__":
	prepare() #reid_baseline model training
	gen_train_all_rename() #st
	gen_train_rename() #st
	gen_val_rename() #st
	gen_query_rename() #st
	gen_gallery_rename() #st
