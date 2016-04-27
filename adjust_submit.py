import argparse
import sys
import csv




def argmax(array):
  v=max(array)
  for idx,value in enumerate(array):
    if value==v:return idx
  return -1


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-s','--submit_file',required=True)
  args = parser.parse_args()





  with open(args.submit_file,'rb') as f:
    csvReader= csv.reader(f,delimiter=',',quotechar='|')
    for row in csvReader:
     	sha1=row[0]
      	predict=[ float(i) for i in row[1:] ]
      	p_argmax=argmax(predict)
      	predict=[ 0.001 for i in predict ]
      	predict[p_argmax]=0.991
      	print(','.join([sha1]+[str(i) for i in predict]))