
import torch.utils.data as data
import PIL
import os
import os.path
import random
import numpy as np
from PIL import Image
import torch
import sys
import pickle
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import matplotlib.pyplot as plt



def visualize_bbox(img, bbox, class_name, color=(220,0,0), thickness=2):
	"""Visualizes a single bounding box on the image"""
	x_min, y_min, w, h = bbox
	x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

	cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
	
	return img
	
def visualize(image, bboxes, category_ids,fname):
	img = image.copy()
	for bbox, category_id in zip(bboxes, category_ids):
		img = visualize_bbox(img, bbox, category_id)
	
	
	plt.figure(figsize=(12, 12))
	plt.axis('off')
	plt.imsave(fname+'_vis.jpg',img)


		
def get_train_transforms():
	return A.Compose([A.OneOf([A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, val_shift_limit=0.2, p=0.9),
							   
					  A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9)],p=0.9),
					  
					  A.ToGray(p=0.01),
					  
					  A.HorizontalFlip(p=0.5),
					  
					  A.VerticalFlip(p=0.5),
					  
					  A.Resize(height=64, width=64, p=1),
					  
					  A.Cutout(num_holes=8, max_h_size=4, max_w_size=4, fill_value=0, p=0.5),
					  
					  ToTensorV2(p=1.0)],
					  
					  p=1.0,
					 
					  bbox_params=A.BboxParams(format='coco',min_area=10, min_visibility=0,label_fields=['labels'])
					  )

def get_valid_transforms():
	return A.Compose([A.Resize(height=64, width=64, p=1.0),
					  ToTensorV2(p=1.0)], 
					  p=1.0, 
					  bbox_params=A.BboxParams(format='coco',min_area=10, min_visibility=0,label_fields=['labels'])
					  )




class MMnistDataset(data.Dataset):
	

	def __init__(self, data_dir, split='train', transform=None):

		self.transforms= transform
		self.data = []
		self.data_dir = data_dir
		self.split_dir = os.path.join(data_dir, split, "normal")
		self.img_dir = self.split_dir + "/imgs/"
		self.filenames = self.load_filenames()
		self.bboxes = self.load_bboxes()
		self.labels = self.load_labels()


	def load_bboxes(self):
		bbox_path = os.path.join(self.split_dir, 'bboxes.pickle')
		with open(bbox_path, "rb") as f:
			bboxes = pickle.load(f,encoding='latin1').astype(np.double)
		return bboxes

	def load_labels(self):
		label_path = os.path.join(self.split_dir, 'labels.pickle')
		with open(label_path, "rb") as f:
			labels = pickle.load(f,encoding='latin1')
			labels = np.argmax(np.array(labels,dtype=np.uint8),axis=-1)
		return labels

	def load_filenames(self):
		filepath = os.path.join(self.split_dir, 'filenames.pickle')
		with open(filepath, 'rb') as f:
			filenames = pickle.load(f,encoding='latin1')
		return filenames

	def __getitem__(self, index):
		# load image
		key = self.filenames[index]
		key = key.split("/")[-1]
		img_id = index
		img_name = self.split_dir + "/imgs/" + key

		image = cv2.imread(img_name,-1).astype(np.uint8)
		image = np.stack([image,image,image],axis=-1)
		image = image.astype(np.float32)
		image /= 255.0

		h,w,c = image.shape
		
		bboxes = self.bboxes[index]
		
		

		#converting bboxes to coco format
		bboxes[:,0]=bboxes[:,0]*w
		bboxes[:,1]=bboxes[:,1]*h
		bboxes[:,2]=bboxes[:,2]*w
		bboxes[:,3]=bboxes[:,3]*h
		
		

		area = bboxes[:,2]*bboxes[:,3]
		area = torch.as_tensor(area, dtype=torch.float32)


		bboxes = bboxes.astype(np.int32).tolist()
		labels = self.labels[index].tolist()



		sample = {
				'image': image,
				'bboxes': bboxes,
				'labels': labels
			}

		sample = self.transforms(**sample)
		
		image = sample['image']
		bboxes = sample['bboxes']
		labels = sample['labels']
		

		_,h,w = image.shape
		bboxes = A.augmentations.bbox_utils.normalize_bboxes(sample['bboxes'],rows=h,cols=w)
		
		target = {}
		target['boxes'] = torch.as_tensor(bboxes,dtype=torch.float32)
		target['labels'] = torch.as_tensor(labels,dtype=torch.long)
		target['image_id'] = torch.tensor([index])
		target['area'] = area

		return image, target, img_id

	def __len__(self):
		return len(self.filenames)


def build(split,args):
	data_dir   = os.path.join(args.dataset_path,args.dataset_file)
	if split == 'train':
		dataset    = MMnistDataset(data_dir=data_dir,split=split,transform=get_train_transforms())
	elif split == 'test':
		dataset    = MMnistDataset(data_dir=data_dir,split=split,transform=get_valid_transforms())
	return dataset