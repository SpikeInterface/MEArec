#!/usr/bin/env python3

import h5py
import numpy as np
import sys
import yaml
import json
import neo

def spiketrains_to_hdf5(spiketrains_folder,output_fname):
  F=h5py.File(output_fname,'w')

  with open(spiketrains_folder+'/info.yaml', 'r') as f:
      info = yaml.load(f)

  F.create_dataset('info',data=json.dumps(info))

  spiketrains=np.load(spiketrains_folder+'/gtst.npy')
  F.attrs['num_spiketrains']=len(spiketrains)
  for ii in range(len(spiketrains)):
      st=spiketrains[ii]
      F.create_dataset('spiketrains/{}/times'.format(ii),data=st.times.rescale('s').magnitude)
      F.create_dataset('spiketrains/{}/t_stop'.format(ii),data=st.t_stop.rescale('s').magnitude)
      annotations={}
      for key in st.annotations:
        if (type(st.annotations[key])==str) or (type(st.annotations[key])==int) or (type(st.annotations[key])==float):
          annotations[key]=st.annotations[key]
      F.create_dataset('spiketrains/{}/annotations'.format(ii),data=json.dumps(annotations))
  F.close()

def print_usage():
  print('Usage: spiketrains_to_hdf5.py [spiketrains_folder] [output_file.h5]')

if __name__ == "__main__":
  if len(sys.argv)<3:
    print_usage()
    sys.exit()
  arg1=sys.argv[1]
  arg2=sys.argv[2]
  spiketrains_to_hdf5(arg1,arg2)