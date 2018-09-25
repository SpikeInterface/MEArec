#!/usr/bin/env python3

import h5py
import numpy as np
import sys
import yaml
import json
import os
import neo
import quantities

def hdf5_to_recording(input_file,output_folder):
  if os.path.exists(output_folder):
    raise Exception('Output folder already exists: '+output_folder)

  os.mkdir(output_folder)


  with h5py.File(input_file,'r') as F:
    info=json.loads(str(F['info'][()]))
    with open(output_folder+'/info.yaml','w') as f:
      yaml.dump(info,f,default_flow_style=False)

    peaks=np.array(F.get('peaks'))
    np.save(output_folder+'/peaks.npy',peaks)
    positions=np.array(F.get('positions'))
    np.save(output_folder+'/positions.npy',positions)
    recordings=np.array(F.get('recordings'))
    np.save(output_folder+'/recordings.npy',recordings)
    sources=np.array(F.get('sources'))
    np.save(output_folder+'/sources.npy',sources)
    templates=np.array(F.get('templates'))
    np.save(output_folder+'/templates.npy',templates)
    times=np.array(F.get('times'))
    np.save(output_folder+'/times.npy',times)
    spiketrains=[]
    for ii in range(F.attrs['n_neurons']):
      times=np.array(F.get('spiketrains/{}/times'.format(ii)))
      t_stop=np.array(F.get('spiketrains/{}/t_stop'.format(ii)))
      annotations_str=str(F.get('spiketrains/{}/annotations'.format(ii))[()])
      annotations=json.loads(annotations_str)
      st=neo.core.SpikeTrain(
        times,
        t_stop=t_stop,
        units=quantities.s
      )
      st.annotations=annotations
      spiketrains.append(st)
    np.save(output_folder+'/spiketrains.npy',spiketrains)

def print_usage():
  print('Usage: hdf5_to_recording.py [input_file.h5] [output_folder] ')

if __name__ == "__main__":
  if len(sys.argv)<3:
    print_usage()
    sys.exit()
  arg1=sys.argv[1]
  arg2=sys.argv[2]
  hdf5_to_recording(arg1,arg2)
