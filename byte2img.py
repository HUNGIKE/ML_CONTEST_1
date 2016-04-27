#
#  Byte 2 image by Riz
#  
#  last modified : 2016.04.12
#
import os,sys,argparse,numpy as np
import matplotlib.pyplot as plt
import time
#
def resave_from_bytefile(inpath,outpath,filename):
  try:
    _bfile = np.fromfile(os.path.join(inpath,filename),dtype='ubyte').reshape((28,28))
  except:
    return (-1,filename,'error on numpy load file.')
  try:
    plt.imsave(os.path.join(outpath,filename),arr=_bfile,cmap='Greys_r')
  except:
    return (-1,filename,'error on save to path.')
  return (0,filename,'ok')
#
def check_path(args):
  if not os.path.exists(args.input_path):
    sys.stderr.write('[!] Input path does not exists.\n')
    exit(1)
  #
  if not os.path.exists(args.output_path):
    sys.stderr.write('[!] Output path does not exists.\n')
    exit(1)
#
def main():
  parser = argparse.ArgumentParser(description='Convert byte array to image for solution contest 2016.\n')
  parser.add_argument('-i','--input_path',required=True)
  parser.add_argument('-o','--output_path',required=True)
  args = parser.parse_args()
  #
  check_path(args)
  #
  _succeed, _failed = 0,0
  start_time = time.time()
  for i,x in enumerate(os.listdir(args.input_path)):
    _state = resave_from_bytefile(args.input_path,args.output_path,x)
    if _state[0] == -1:
      sys.stderr.write('[!] "%s" parse failed, reason = %s\n'%(x,_state[2]))
      sys.stderr.flush()
      _failed+=1
    else:
      _succeed+=1
    if i%100==0:
      sys.stdout.write('[+] parsed : %d\n'%i)
      sys.stdout.flush()
  end_time = time.time()-start_time
  sys.stdout.write('[*] done.\n')
  sys.stdout.write('   - Parsed : %d\n'%_succeed)
  sys.stdout.write('   - Parse failed : %d\n'%_failed)
  sys.stdout.write('   - Time used : %d seconds\n'%end_time)
  #op_state = [resave_from_bytefile(args.input_path,args.output_path,x) for x in os.listdir(args.input_path)]
#
if __name__ == "__main__":
  sys.exit(main())
