#!/usr/bin/env python3

import h5py
import numpy as np
import sys
import yaml
import json

def templates_to_hdf5(templates_folder,output_fname):
  F=h5py.File(output_fname,'w')

  with open(templates_folder+'/info.yaml', 'r') as f:
      info = yaml.load(f)
      
  for key in info['General']:
      F.attrs[key]=info['General'][key] # this includes spiketrain_folder, fs, template_folder, duration, n_neurons, seed, electrode_name

  F.create_dataset('info',data=json.dumps(info))

  celltypes=np.load(templates_folder+'/celltypes.npy')
  F.create_dataset('celltypes',data=celltypes)
  locations=np.load(templates_folder+'/locations.npy')
  F.create_dataset('locations',data=locations)
  rotations=np.load(templates_folder+'/rotations.npy')
  F.create_dataset('rotations',data=rotations)
  templates=np.load(templates_folder+'/templates.npy')
  F.create_dataset('templates',data=templates)

  F.close()

def print_usage():
  print('Usage: templates_to_hdf5.py [templates_folder] [output_file.h5]')

if __name__ == "__main__":
  if len(sys.argv)<3:
    print_usage()
    sys.exit()
  arg1=sys.argv[1]
  arg2=sys.argv[2]
  templates_to_hdf5(arg1,arg2)