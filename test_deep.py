import numpy
import tensorflow as tf
import sys
import csv
import os
import skimage.feature
from image_preprocess import *
from image_resizing_test_tmp import *
import time

#==================checking data =====================
def showImage(img):
    for i in range(28*28):
      if (i%28)==0:
        sys.stdout.write('\n')
      sys.stdout.write("X" if img[i]==0.0 else " ")
    sys.stdout.write('\n')


def checkData(trainSet,num):
  for idx,i in enumerate(trainSet[0][:num]):
    showImage(i)
    print(trainSet[1][idx])

def evalResult(y_conv,validSet,x):
  R=tf.argmax(y_conv,1)
  for idx,img in enumerate(validSet[0]):
    a=R.eval(feed_dict={x:preprocess(validSet[0][idx:idx+1])} )
    b=validSet[1][idx]
    print(validSet[2][idx],argmax(b),a.item(0),argmax(b)==a.item(0))


#======================================================

#==================getting training data===============

def one_hot_array(idx,len):
    i=[0.0]*len
    i[idx]=1.0
    return i;

def argmax(array):
  v=max(array)
  for idx,value in enumerate(array):
    if value==v:return idx
  return -1

def getImgData(binaryDir,sha1):
  filepath=binaryDir+sha1;
  with open(filepath,'rb') as f:
    return [ int(c.encode('hex'), 16)/255.0 for c in f.read(784) ]


def getTrainSet(binaryDir,labelFile):
    sha1s=[];
    labels=[];
    imgs=[];
    readNum=0;
    with open(binaryDir+'/'+labelFile,'rb') as f:
      csvReader= csv.reader(f,delimiter=',',quotechar='|')
      for row in csvReader:
        sha1=row[0];label=int(row[1]);
        label_one_hot=one_hot_array(label,10)
        img=getImgData(binaryDir,sha1);
        labels+=label_one_hot;
        imgs+=img;
        readNum+=1
        sha1s+=[sha1]
        if readNum%1000==0:print('read '+str(readNum)+' files.')

    sha1s=numpy.asarray(sha1s)
    imgs=numpy.asarray(imgs).reshape(readNum,784);
    labels=numpy.asarray(labels).reshape(readNum,10);
    return [imgs,labels,sha1s];


def getTestSet(binaryDir):
  sha1s=[];
  imgs=[];
  for sha1 in os.listdir(binaryDir):
    sha1s+=[sha1]
    imgs+=getImgData(binaryDir,sha1);
    if len(sha1s)%1000==0:print('read '+str(len(sha1s))+' files.')
    #if len(sha1s)>=3000:break;
  imgs=numpy.asarray(imgs).reshape(len(sha1s),784);
  return [imgs,sha1s];



#======================================================

#==================tensorflow deep learning============

def preprocess1(imgs):
    imgs2=[]
    for img in imgs:
	img2=img.reshape(28,28)
	img2=[ sum([ 1 if k >0.1 else 0  for k in  img2[j,:] ]) for j in range(img2.shape[0]) ] + \
	     [ sum([ 1 if k >0.1 else 0  for k in  img2[:,j] ]) for j in range(img2.shape[1]) ] 
	img2=[ i/28.0 for i in img2 ]
	# img2=img2+img.tolist()
        img2=numpy.array(img2)
        img2=img2.reshape(56)
        imgs2.append(img2)
    imgs2=numpy.array(imgs2)
    imgs2.reshape(imgs.shape[0],56)
    return imgs2

def preprocess2(imgs):
  for idx,img in enumerate(imgs):
      for idx,point in enumerate(img):
        if point>0.8:
          img[idx]=1
        else:
          img[idx]=0

  return imgs



def dummypreprocess(imgs):
  return imgs



def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


#======================================================
preprocess= dummypreprocess


def main():

  featureMapNum=[32,64]


  x = tf.placeholder(tf.float32, shape=[None,784*1])

  #  x_preprocess1 = tf.placeholder(tf.float32, shape=[None,56*1])

  x_image = tf.reshape(x, [-1,28,28,1])
  # x_hog = tf.placeholder(tf.float32, shape=[None,81])

  featureMap_in=x_image;
  in_NUM=1
  img_w_h=28
  fileter_size=5


  for out_NUM in featureMapNum:
    #if img_w_h==7:
    #  fileter_size=3

    W_conv = weight_variable([fileter_size, fileter_size, in_NUM, out_NUM])
    b_conv = bias_variable([out_NUM])
    h_conv = tf.nn.relu(conv2d(featureMap_in, W_conv) + b_conv)
    if img_w_h>7:
      h_conv = max_pool_2x2(h_conv)
      img_w_h/=2
    featureMap_in=h_conv
    in_NUM=out_NUM


  W_fc1 = weight_variable([img_w_h*img_w_h*out_NUM,512])
  b_fc1 = bias_variable([512])
  h_pool_flat = tf.reshape(featureMap_in, [-1,img_w_h*img_w_h*out_NUM])
  h_fc1 = tf.nn.relu(  tf.matmul(h_pool_flat , W_fc1)  + b_fc1)

  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
  # h_fc1 = tf.concat(1,[h_fc1,x_preprocess1])

  W_fc2 = weight_variable([512, 10])
  b_fc2 = bias_variable([10])
  y_conv=tf.nn.softmax( tf.matmul(h_fc1, W_fc2)+ b_fc2)





  y = tf.placeholder(tf.float32, shape=[None,10])
  cross_entropy = tf.reduce_mean(  -tf.reduce_sum(y*tf.log(y_conv), reduction_indices=[1])  )  
  # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y_conv,y)


  learning_rate = tf.train.exponential_decay(
      0.01,                # Base learning rate.
      50 * 8000,  # Current index into the dataset.
      9000,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)
  train_step = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(cross_entropy)



  sess = tf.InteractiveSession()
  sess.run(tf.initialize_all_variables())


  #==========================================================================

  print("reading train data..")
  data=getTrainSet('digit_data/train/','train.csv');
  print("read finished")

   #checkData(trainSet,100);

  trainSetNum=9000
  trainSet=[data[0][:trainSetNum] ,data[1][:trainSetNum],data[2][:trainSetNum]]
  validSet=[data[0][trainSetNum:]  ,data[1][trainSetNum:],data[2][trainSetNum:]]


  RunNum=8000
  batchNum=50
  p=0
  for i in range(RunNum):
    if i%50==0:print('p='+str(p)+',i='+str(i))

    batch=trainSet[0][p:p+batchNum],trainSet[1][p:p+batchNum]
    train_step.run(feed_dict={x: preprocess( batch[0] ),keep_prob:0.5 ,y: batch[1]})

    if i%500==0:
      correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      acc_val=accuracy.eval(feed_dict={x: preprocess( validSet[0] ),keep_prob:1.0,y: validSet[1]})
      print( 'accuracy: '+str(acc_val) )
    p=(p+batchNum)%trainSetNum;

  testData=getTestSet('digit_data/test/');
  imgs=testData[0]
  sha1s=testData[1]


  with open('./submit_'+str(int(time.time()))+'.csv','w') as output_f:
    for idx,sha1 in enumerate(sha1s):
      img=imgs[idx:idx+1];
      result=y_conv.eval( feed_dict={ x: preprocess(img),keep_prob:1.0 } );
      result=[ str(round(r, 2)) for r in result[0] ];
      output_line=','.join([str(sha1)]+result)
      print(output_line)
      output_f.write(output_line+'\n')

    #showImage(img[0])


if __name__=='__main__':
  main();
