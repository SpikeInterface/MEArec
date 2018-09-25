#!/usr/bin/env python3

import h5py
import numpy as np
import sys
import yaml
import json
import os

def hdf5_to_templates(input_file,output_folder):
  if os.path.exists(output_folder):
    raise Exception('Output folder already exists: '+output_folder)

  os.mkdir(output_folder)

  with h5py.File(input_file,'r') as F:
    info=json.loads(str(F['info'][()]))
    with open(output_folder+'/info.yaml','w') as f:
      yaml.dump(info,f,default_flow_style=False)

    celltypes=np.array(F.get('celltypes'))
    np.save(output_folder+'/celltypes.npy',celltypes)
    celltypes=np.array(F.get('locations'))
    np.save(output_folder+'/locations.npy',celltypes)
    celltypes=np.array(F.get('rotations'))
    np.save(output_folder+'/rotations.npy',celltypes)
    celltypes=np.array(F.get('templates'))
    np.save(output_folder+'/templates.npy',celltypes)

def print_usage():
  print('Usage: hdf5_to_templates.py [input_file.h5] [output_folder] ')

if __name__ == "__main__":
  if len(sys.argv)<3:
    print_usage()
    sys.exit()
  arg1=sys.argv[1]
  arg2=sys.argv[2]
  hdf5_to_templates(arg1,arg2)
