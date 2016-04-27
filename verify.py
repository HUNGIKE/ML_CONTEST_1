import argparse
import sys
import csv
import math
import numpy

def std(array):
	return numpy.std(array);

def argmax(array):
  v=max(array)
  for idx,value in enumerate(array):
    if value==v:return idx
  return -1

def getSubmitList(file_path):
    sList=[];
    with open(file_path,'rb') as f:
      csvReader= csv.reader(f,delimiter=',',quotechar='|')
      for row in csvReader:
        sList.append(row)
    return sList;

def getAnswerMap(file_path):
    ans=dict()
    with open(file_path,'rb') as f:
		csvReader= csv.reader(f,delimiter=',',quotechar='|')
		for row in csvReader:
			hash=row[0]
			label=int(row[1])
			ans[hash]=label
    return ans


def main():
  logloss=0.0;

  parser = argparse.ArgumentParser()
  parser.add_argument('-s','--submit_file',required=True)
  parser.add_argument('-a','--answer_file',required=True)
  args = parser.parse_args()

  ans=getAnswerMap(args.answer_file)
  submit=getSubmitList(args.submit_file)

  for item in submit:
    sha1=item[0]
    submit_ans=argmax(item[1:])
    true_ans=ans[sha1]

   
	    #print(sha1+',anser: '+str(true_ans)+',your_submit: '+str(submit_ans))
	   
    print(str(true_ans==submit_ans)+','+sha1+','+str(true_ans)+','+str(submit_ans)+','+str(item[1:])+','+str( std([float(i) for i in   item[1:]  ]) )  )

    p=float(item[ans[sha1]+1]);
    if p==0.0:p=0.0000000000000001
    logloss+=math.log( p )

  logloss/=len(submit);
  print('logloss='+str(logloss))





if __name__ == "__main__":
  sys.exit(main())
