import numpy
import tensorflow as tf
import sys
import os
import matplotlib.pyplot as plt
from skimage import data, io, filters,transform ,feature ,morphology,measure
from test_deep import *
from image_preprocess import *
import math




if __name__=='__main__':
  #imgFileScale('digit_data/train/','digit_data/train_rescale/','1d07b8c5')
	#imgFileScale('digit_data/train/','digit_data/train_rescale/','6a7b8edc')
	#imgFileScale('digit_data/train/','digit_data/train_rescale/','814231a5')
	#imgFileScale('digit_data/train/','digit_data/train_rescale/','9607c4c1')
	#imgFileScale('digit_data/train/','digit_data/train_rescale/','ab3fb229')

	#exit(0)
  img=getImgData('digit_data/train/','1d07b8c5');
  # img = [ 1.0 if i>0.95 else 0.0 for i in img]

  #img=[ 1.0 if i!=0.0 else 0.0 for i in img]

  matrix=numpy.asarray(
    [[math.cos(math.pi*0.25),-math.sin(math.pi*0.25),7],
    [math.sin(math.pi*0.25),math.cos(math.pi*0.25),7],
     [0,0,1 ]])
  img=numpy.asarray(img).reshape(28,28)
  tform = transform.ProjectiveTransform(matrix)

  print(tform)
  
  
  io.imshow(img)
  img=transform.warp(img, tform)
  io.imshow(img)
  io.show()


