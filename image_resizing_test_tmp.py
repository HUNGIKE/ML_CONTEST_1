import numpy
import tensorflow as tf
import sys
import os
import matplotlib.pyplot as plt
from skimage import data, io, filters,transform ,feature ,morphology,measure
from test_deep import *
from image_preprocess import *

def showNumpyImg(img):
  for i in img:
    for j in i:
      sys.stdout.write("X" if j==0.0 else " ")
    sys.stdout.write('\n')


def imgScale(img):
  img=img.reshape(28,28)
  t=-1;l=100
  w=0;h=0
  for x,rowv in enumerate(img):
    for y,v in enumerate(rowv):
      if v!=0.0:
        if t==-1:t=x
        if y<l:l=y
        if x>h:h=x
        if y>w:w=y
  h=h-t+1.0;w=w-l+1.0

  for x,rowv in enumerate(img):
    for y,v in enumerate(rowv):
      nx=int((x-t));ny=int((y-l))
      if nx>=0 and ny>=0 and nx<28 and ny<28:
        img[nx,ny]=v;
      img[x,y]=0.0
  img=img[:int(h),:int(w)]
  img=transform.resize(img,[28,28])
  return img.reshape(784)

def imgsScale(imgs):
  for idx,img in enumerate(imgs):
    imgs[idx]=imgScale(img)
    io.imshow(imgs[idx].reshape(28,28))
    io.show()
  return imgs



def imgFileScale(inpath,outpath,filename):
  img=getImgData(inpath,filename);
  img=numpy.asarray(img).reshape(28,28)

  plt.imsave(os.path.join(outpath,filename),arr=img,cmap='Greys_r')
  img=imgScale(img)
  img=img.reshape(28,28)
  plt.imsave(os.path.join(outpath,filename+'_rescale'),arr=img,cmap='Greys_r')

if __name__=='__main__':
  #imgFileScale('digit_data/train/','digit_data/train_rescale/','1d07b8c5')
	#imgFileScale('digit_data/train/','digit_data/train_rescale/','6a7b8edc')
	#imgFileScale('digit_data/train/','digit_data/train_rescale/','814231a5')
	#imgFileScale('digit_data/train/','digit_data/train_rescale/','9607c4c1')
	#imgFileScale('digit_data/train/','digit_data/train_rescale/','ab3fb229')

	#exit(0)
  img=getImgData('digit_data/train/','1d07b8c5');


  #img=[ 1.0 if i!=0.0 else 0.0 for i in img]

  img=numpy.asarray(img).reshape(28,28)
  img=transform.rotate(img,60)
  io.imshow(img)
  io.show()
