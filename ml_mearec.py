#!/usr/bin/env python3

import sys

from mlprocessors.registry import registry, register_processor

registry.namespace = "mearec"

from mlprocessors.core import Input, Output, Processor, IntegerParameter, FloatParameter, StringParameter, IntegerListParameter, BoolParameter

import os
import h5py
import numpy as np
from gen_spiketrains import SpikeTrainGenerator
from gen_recordings import RecordingGenerator
import json
from mountainlab_pytools import mlproc as mlp

@register_processor(registry)
class gen_spiketrains(Processor):
    """
        Generate spiketrains
    """
    VERSION='0.1.2'

    spiketrains_out = Output('MEArec spiketrains .npy output file -- neo format')

    duration = FloatParameter('Duration of spike trains (s)',optional=True,default=10)
    n_exc = IntegerParameter('Number of excitatory cells',optional=True,default=15)
    n_inh = IntegerParameter('Number of inhibitory cells',optional=True,default=5)
    f_exc = FloatParameter('Mean firing rate of excitatory cells (Hz)',optional=True,default=5)
    f_inh = FloatParameter('Mean firing rate of inhibitory cells (Hz)',optional=True,default=15)
    min_rate = FloatParameter('Minimum firing rate for all cells (Hz)',optional=True,default=0.5)
    st_exc = FloatParameter('Firing rate standard deviation of excitatory cells (Hz)',optional=True,default=1)
    st_inh = FloatParameter('Firing rate standard deviation of inhibitory cells (Hz)',optional=True,default=3)
    process = StringParameter('poisson or gamma',optional=True,default='poisson')
    t_start = FloatParameter('Starting time (s)',optional=True,default=0)
    ref_per = FloatParameter('Refractory period to remove spike violation (ms)',optional=True,default=2)

    def run(self):
        tmpdir=os.environ.get('ML_PROCESSOR_TEMPDIR')
        if not tmpdir:
            raise Exception('Environment variable not set: ML_PROCESSOR_TEMPDIR')
    
        params_dict=dict(
            duration=self.duration,
            n_exc=self.n_exc,
            n_inh=self.n_inh,
            f_exc=self.f_exc,
            f_inh=self.f_inh,
            st_exc=self.st_exc,
            st_inh=self.st_inh,
            process=self.process,
            t_start=self.t_start,
            ref_per=self.ref_per
        )

        spgen = SpikeTrainGenerator(params_dict)
        spgen.generate_spikes()
        spiketrains = spgen.all_spiketrains
        n_neurons = len(spiketrains)
        np.save(self.spiketrains_out,spiketrains)

        return True

@register_processor(registry)
class gen_recording(Processor):
    """
        Generate a MEArec recording
    """
    VERSION='0.1.3'

    templates = Input('MEArec templates .hdf5 file - generated using utils/templates_to_hdf5.py - if omitted, will download default',optional=True)
    spiketrains = Input('MEArec spiketrains .npy file')
    recording_out = Output('MEArec recording .hdf5 file')

    # recording generation parameters
    min_dist = FloatParameter('minimum distance between neurons',optional=True,default=25)
    min_amp = FloatParameter('minimum spike amplitude in uV',optional=True,default=50)
    noise_level = FloatParameter('noise standard deviation in uV',optional=True,default=10)
    filter = BoolParameter('if True it filters the recordings',optional=True,default=True)
    cutoff = IntegerListParameter('filter cutoff frequencies in Hz',optional=True,default=[300,6000])
    overlap_threshold = FloatParameter('threshold to consider two templates spatially overlapping (e.g 0.6 -> 60 percent of template B on largest electrode of template A)',optional=True,default=0.8)
    n_jitters = IntegerParameter('number of temporal jittered copies for each eap',optional=True,default=10)
    upsample = IntegerParameter('upsampling factor to extract jittered copies',optional=True,default=8)
    pad_len = IntegerListParameter('padding of templates in ms',optional=True,default=[3,3])
    modulation = StringParameter('# type of spike modulation [none (no modulation) | template (each spike instance is modulated with the same value on each electrode) | electrode (each electrode is modulated separately)]',optional=True,default='electrode')
    mrand = FloatParameter('mean of gaussian modulation (should be 1)',optional=True,default=1)
    sdrand = FloatParameter('standard deviation of gaussian modulation',optional=True,default=0.05)
    chunk_duration = FloatParameter('chunk duration in s for chunk processing (if 0 the entire recordings are generated at once)',optional=True,default=0)
    overlap = BoolParameter('if True it annotates overlapping spikes',optional=True,default=False)

    def run(self):
        tmpdir=os.environ.get('ML_PROCESSOR_TEMPDIR')
        if not tmpdir:
            raise Exception('Environment variable not set: ML_PROCESSOR_TEMPDIR')


        if not self.templates:
            print('Downloading templates (if needed)...')
            default_templates_url='kbucket://b5ecdf1474c5/MEArec/templates/templates_30_Neuronexus-32.h5'
            templates_path=mlp.realizeFile(default_templates_url)
            print('Done downloading templates (if needed)')
        else:
            templates_path=self.templates
        print('Using templates file: '+templates_path)

        params_dict=dict(
            min_dist=self.min_dist,
            min_amp=self.min_amp,
            noise_level=self.noise_level,
            noise_mode='uncorrelated',
            modulation=self.modulation,
            chunk_duration=self.chunk_duration,
            filter=self.filter,
            seed=np.random.randint(1, 10000),
            cutoff=self.cutoff,
            overlap_threshold=self.overlap_threshold,
            n_jitters=self.n_jitters,
            upsample=self.upsample,
            pad_len=self.pad_len,
            mrand=self.mrand,
            sdrand=self.sdrand,
            fs=None,
            depth_lim=None
        )
        params_dict['excitatory']=['STPC', 'TTPC1', 'TTPC2', 'UTPC']
        params_dict['inhibitory']=['BP', 'BTC', 'ChC', 'DBC', 'LBC', 'MC', 'NBC', 'NGC', 'SBC']
        
        templates_data={}
        with h5py.File(templates_path,'r') as F:
            templates_data['info']=json.loads(str(F['info'][()]))
            templates_data['celltypes']=np.array(F.get('celltypes'))
            templates_data['locations']=np.array(F.get('locations'))
            templates_data['rotations']=np.array(F.get('rotations'))
            templates_data['templates']=np.array(F.get('templates'))

        spiketrains_data={}
        spiketrains_data['spiketrains']=np.load(self.spiketrains)

        recgen = RecordingGenerator(templates_data, spiketrains_data, params_dict, self.overlap)
        info = recgen.info

        F=h5py.File(self.recording_out,'w')
        for key in info['General']:
          F.attrs[key]=info['General'][key] # this includes spiketrain_folder, fs, template_folder, duration, n_neurons, seed, electrode_name
        F.create_dataset('info',data=json.dumps(info))
        F.create_dataset('peaks',data=recgen.peaks)
        F.create_dataset('positions',data=recgen.positions)
        F.create_dataset('recordings',data=recgen.recordings)
        F.create_dataset('sources',data=recgen.sources)
        for ii in range(len(recgen.spiketrains)):
          st=recgen.spiketrains[ii]
          F.create_dataset('spiketrains/{}/times'.format(ii),data=st.times.rescale('s').magnitude)
          F.create_dataset('spiketrains/{}/t_stop'.format(ii),data=st.t_stop)
          annotations={}
          for key in st.annotations:
            if (type(st.annotations[key])==str) or (type(st.annotations[key])==int) or (type(st.annotations[key])==float):
              annotations[key]=st.annotations[key]
          F.create_dataset('spiketrains/{}/annotations'.format(ii),data=json.dumps(annotations))
        F.create_dataset('templates',data=recgen.templates)
        F.create_dataset('times',data=recgen.times)
        F.close()

        return True

if __name__ == "__main__":
    registry.process(sys.argv)
