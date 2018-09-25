#!/usr/bin/env python3

import h5py
import numpy as np
import sys
import yaml
import json
import os
import neo
import quantities

def hdf5_to_spiketrains(input_file,output_folder):
  if os.path.exists(output_folder):
    raise Exception('Output folder already exists: '+output_folder)

  os.mkdir(output_folder)

  with h5py.File(input_file,'r') as F:
    info=json.loads(str(F['info'][()]))
    with open(output_folder+'/info.yaml','w') as f:
      yaml.dump(info,f,default_flow_style=False)

    spiketrains=[]
    for ii in range(F.attrs['num_spiketrains']):
      times=np.array(F.get('spiketrains/{}/times'.format(ii)))
      t_stop=np.array(F.get('spiketrains/{}/t_stop'.format(ii)))
      spiketrains.append(neo.core.SpikeTrain(times,t_stop=t_stop,units=quantities.s))
    np.save(output_folder+'/gtst.npy',spiketrains)

def print_usage():
  print('Usage: hdf5_to_spiketrains.py [input_file.h5] [output_folder] ')

if __name__ == "__main__":
  if len(sys.argv)<3:
    print_usage()
    sys.exit()
  arg1=sys.argv[1]
  arg2=sys.argv[2]
  hdf5_to_spiketrains(arg1,arg2)
