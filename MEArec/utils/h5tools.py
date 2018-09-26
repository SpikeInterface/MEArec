#!/usr/bin/env python3
import h5py
import numpy as np
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
      annotations_str=str(F.get('spiketrains/{}/annotations'.format(ii))[()])
      annotations=json.loads(annotations_str)
      st=neo.core.SpikeTrain(
        times,
        t_stop=t_stop,
        units=quantities.s
      )
      st.annotations=annotations
      spiketrains.append(st)
    np.save(output_folder+'/gtst.npy',spiketrains)

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


def recording_to_hdf5(recording_folder, output_fname):
    F = h5py.File(output_fname, 'w')

    with open(recording_folder + '/info.yaml', 'r') as f:
        info = yaml.load(f)

    for key in info['General']:
        F.attrs[key] = info['General'][
            key]  # this includes spiketrain_folder, fs, template_folder, duration, n_neurons, seed, electrode_name

    F.create_dataset('info', data=json.dumps(info))

    peaks = np.load(recording_folder + '/peaks.npy')
    F.create_dataset('peaks', data=peaks)
    positions = np.load(recording_folder + '/positions.npy')
    F.create_dataset('positions', data=positions)
    recordings = np.load(recording_folder + '/recordings.npy')
    F.create_dataset('recordings', data=recordings)
    sources = np.load(recording_folder + '/sources.npy')
    F.create_dataset('sources', data=sources)
    spiketrains = np.load(recording_folder + '/spiketrains.npy')
    for ii in range(len(spiketrains)):
        st = spiketrains[ii]
        F.create_dataset('spiketrains/{}/times'.format(ii), data=st.times.rescale('s').magnitude)
        F.create_dataset('spiketrains/{}/t_stop'.format(ii), data=st.t_stop)
        annotations_str = json.dumps(st.annotations)
        F.create_dataset('spiketrains/{}/annotations'.format(ii), data=annotations_str)
    templates = np.load(recording_folder + '/templates.npy')
    F.create_dataset('templates', data=templates)
    templates = np.load(recording_folder + '/times.npy')
    F.create_dataset('times', data=templates)
    F.close()

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


def templates_to_hdf5(templates_folder, output_fname):
    F = h5py.File(output_fname, 'w')

    with open(templates_folder + '/info.yaml', 'r') as f:
        info = yaml.load(f)

    for key in info['General']:
        F.attrs[key] = info['General'][
            key]  # this includes spiketrain_folder, fs, template_folder, duration, n_neurons, seed, electrode_name

    F.create_dataset('info', data=json.dumps(info))

    celltypes = np.load(templates_folder + '/celltypes.npy')
    F.create_dataset('celltypes', data=celltypes)
    locations = np.load(templates_folder + '/locations.npy')
    F.create_dataset('locations', data=locations)
    rotations = np.load(templates_folder + '/rotations.npy')
    F.create_dataset('rotations', data=rotations)
    templates = np.load(templates_folder + '/templates.npy')
    F.create_dataset('templates', data=templates)

    F.close()
