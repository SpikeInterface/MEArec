#!/usr/bin/env python3

import h5py
import numpy as np
import sys
import yaml
import json

def recording_to_hdf5(recording_folder,output_fname):
  F=h5py.File(output_fname,'w')

  with open(recording_folder+'/info.yaml', 'r') as f:
      info = yaml.load(f)
      
  for key in info['General']:
      F.attrs[key]=info['General'][key] # this includes spiketrain_folder, fs, template_folder, duration, n_neurons, seed, electrode_name

  F.create_dataset('info',data=json.dumps(info))

  peaks=np.load(recording_folder+'/peaks.npy')
  F.create_dataset('peaks',data=peaks)
  positions=np.load(recording_folder+'/positions.npy')
  F.create_dataset('positions',data=positions)
  recordings=np.load(recording_folder+'/recordings.npy')
  F.create_dataset('recordings',data=recordings)
  sources=np.load(recording_folder+'/sources.npy')
  F.create_dataset('sources',data=sources)
  spiketrains=np.load(recording_folder+'/spiketrains.npy')
  for ii in range(len(spiketrains)):
      st=spiketrains[ii]
      F.create_dataset('spiketrains/{}/times'.format(ii),data=st.times)
  templates=np.load(recording_folder+'/templates.npy')
  F.create_dataset('templates',data=templates)
  templates=np.load(recording_folder+'/times.npy')
  F.create_dataset('times',data=templates)
  F.close()

def print_usage():
  print('Usage: recording_to_hdf5.py [recording_folder] [output_file.h5]')

if __name__ == "__main__":
  if len(sys.argv)<3:
    print_usage()
    sys.exit()
  arg1=sys.argv[1]
  arg2=sys.argv[2]
  recording_to_hdf5(arg1,arg2)