import numpy
import skimage.feature


def hog(imgs):
  imgs2=numpy.asarray([]).reshape(0,81)
  for idx,img in enumerate(imgs):
    imgs2=numpy.insert( imgs2 ,imgs2.shape[0], skimage.feature.hog(img.reshape(28,28)),0);
  return imgs2

def structure_tensor(imgs):
  imgs2=numpy.asarray([]).reshape(-1,1,28,28)
  for idx,img in enumerate(imgs):
    st=skimage.feature.structure_tensor(img.reshape(28,28));
    tmp=numpy.asarray([]).reshape(-1,28,28)
    # tmp=numpy.insert(tmp,0,st[0],0)
    # tmp=numpy.insert(tmp,0,st[1],0)
    # tmp=numpy.insert(tmp,0,st[2],0)
    tmp=numpy.insert(tmp,tmp.shape[0],img.reshape(28,28),0)
    imgs2=numpy.insert(imgs2,imgs2.shape[0],tmp,0)


  # print("====start========")
  # print(imgs2[0])
  # print("============")
  imgs2=numpy.swapaxes(imgs2,1,2)
  imgs2=numpy.swapaxes(imgs2,2,3)
  # print(imgs2[0])
  # print("====end========")
  imgs2=imgs2.reshape(imgs2.shape[0],28*28*1)
  return imgs2


  
  if __name__=='__main__':
  	pass