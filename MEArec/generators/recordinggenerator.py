import numpy as np
import time
from copy import copy, deepcopy
from MEArec.tools import (select_templates, find_overlapping_templates, get_binary_cat,
                          resample_templates, jitter_templates, pad_templates, get_templates_features,
                          resample_spiketrains, compute_modulation, annotate_overlapping_spikes, extract_wf)

from .recgensteps import (chunk_convolution, chunk_uncorrelated_noise,
                          chunk_distance_correlated_noise, chunk_apply_filter)

import random
import string
import MEAutility as mu
import yaml
import os
from pprint import pprint
import h5py
import quantities as pq
from distutils.version import StrictVersion
import tempfile
from pathlib import Path

from joblib import Parallel, delayed

from MEArec.generators import SpikeTrainGenerator

if StrictVersion(yaml.__version__) >= StrictVersion('5.0.0'):
    use_loader = True
else:
    use_loader = False


class RecordingGenerator:
    """
    Class for generation of recordings called by the gen_recordings function.
    The list of parameters is in default_params/recordings_params.yaml.

    Parameters
    ----------
    spgen : SpikeTrainGenerator
        SpikeTrainGenerator object containing spike trains
    tempgen : TemplateGenerator
        TemplateGenerator object containing templates
    params : dict
        Dictionary with parameters to simulate recordings. Default values can be retrieved with
        mr.get_default_recording_params()
    rec_dict :  dict
        Dictionary to instantiate RecordingGenerator with existing data. It contains the following fields:
          - recordings : float (n_electrodes, n_samples)
          - spiketrains : list of neo.SpikeTrains (n_spiketrains)
          - templates : float (n_spiketrains, 3)
          - template_locations : float (n_spiketrains, 3)
          - template_rotations : float (n_spiketrains, 3)
          - template_celltypes : str (n_spiketrains)
          - channel_positions : float (n_electrodes, 3)
          - timestamps : float (n_samples)
          - voltage_peaks : float (n_spiketrains, n_electrodes)
          - spike_traces : float (n_spiketrains, n_samples)
    info :  dict
        Info dictionary to instantiate RecordingGenerator with existing data. Same fields as 'params'
    verbose : bool or int
        When verbose is 0 or False: no verbode
        When verbose is 1 or True: main verbose but no verbose each chunk
        When verbose is 2 : full verbose even each chunk
        
    """

    def __init__(self, spgen=None, tempgen=None, params=None, rec_dict=None, info=None, verbose=True):
        self._verbose = verbose

        if rec_dict is not None and info is not None:
            if 'recordings' in rec_dict.keys():
                self.recordings = rec_dict['recordings']
            else:
                self.recordings = np.array([])
            if 'spiketrains' in rec_dict.keys():
                self.spiketrains = deepcopy(rec_dict['spiketrains'])
            else:
                self.spiketrains = np.array([])
            if 'templates' in rec_dict.keys():
                self.templates = rec_dict['templates']
            else:
                self.templates = np.array([])
            if 'template_locations' in rec_dict.keys():
                self.template_locations = rec_dict['template_locations']
            else:
                self.template_locations = np.array([])
            if 'template_rotations' in rec_dict.keys():
                self.template_rotations = rec_dict['template_rotations']
            else:
                self.template_rotations = np.array([])
            if 'template_celltypes' in rec_dict.keys():
                self.template_celltypes = rec_dict['template_celltypes']
            else:
                self.template_celltypes = np.array([])
            if 'channel_positions' in rec_dict.keys():
                self.channel_positions = rec_dict['channel_positions']
            else:
                self.channel_positions = np.array([])
            if 'timestamps' in rec_dict.keys():
                self.timestamps = rec_dict['timestamps']
            else:
                self.timestamps = np.array([])
            if 'voltage_peaks' in rec_dict.keys():
                self.voltage_peaks = rec_dict['voltage_peaks']
            else:
                self.voltage_peaks = np.array([])
            if 'spike_traces' in rec_dict.keys():
                self.spike_traces = rec_dict['spike_traces']
            else:
                self.spike_traces = np.array([])
            self.info = deepcopy(info)
            self.params = deepcopy(info)
            if len(self.spiketrains) > 0:
                self.spgen = SpikeTrainGenerator(spiketrains=self.spiketrains, params=self.info['spiketrains'])
            self.tempgen = None
            if isinstance(self.recordings, h5py.Dataset):
                self.tmp_mode = 'h5'
            elif isinstance(self.recordings, np.memmap):
                self.tmp_mode = 'memmap'
            else:
                self.tmp_mode = None

        else:
            if spgen is None or tempgen is None:
                raise AttributeError("Specify SpikeTrainGenerator and TemplateGenerator objects!")
            if params is None:
                if self._verbose:
                    print("Using default parameters")
                params = {'spiketrains': {},
                          'celltypes': {},
                          'templates': {},
                          'recordings': {}}
            self.params = deepcopy(params)
            self.spgen = spgen
            self.tempgen = tempgen
            self.tmp_mode = None

        self.overlapping = []
        # temp file that should remove on delete
        self._to_remove_on_delete = []

    def __del__(self):
        self.recordings = None
        self.spike_traces = None

        for fname in self._to_remove_on_delete:
            if self._verbose:
                try:
                    os.remove(fname)
                    print('Deleted', fname)
                except:
                    print('Impossible to delete temp file:', fname)

    def generate_recordings(self, tmp_mode=None, tmp_folder=None, verbose=None, n_jobs=0):
        """
        Generates recordings
        Parameters
        ----------
        tmp_mode : None, 'h5' 'memmap'
            Use temporary file h5 memmap or None
            None is no temporary file and then use memory.
        tmp_folder: str or Path
            In case of tmp files, you can specify the folder.
            If None, then it is automatic using tempfile.mkdtemp()
        n_jobs: int if >1 then use joblib to execute chunk in parralel else in loop

        """
        self.tmp_mode = tmp_mode
        self.tmp_folder = tmp_folder
        self.n_jobs = n_jobs

        if tmp_mode is not None:
            tmp_prefix = ''.join([random.choice(string.ascii_letters) for i in range(5)]) + '_'

        if self.tmp_mode is not None:
            if self.tmp_folder is None:
                self.tmp_folder = Path(tempfile.mkdtemp())
            else:
                self.tmp_folder = Path(self.tmp_folder)

        if verbose is not None and isinstance(verbose, bool) or isinstance(verbose, int):
            self._verbose = verbose

        params = deepcopy(self.params)
        temp_params = self.params['templates']
        rec_params = self.params['recordings']
        st_params = self.params['spiketrains']
        if 'cell_types' in self.params.keys():
            celltype_params = self.params['cell_types']
        else:
            celltype_params = {}

        if 'rates' in st_params.keys():
            assert st_params['types'] is not None, "If 'rates' are provided as spiketrains parameters, " \
                                                   "corresponding 'types' ('E'-'I') must be provided"
            n_exc = st_params['types'].count('E')
            n_inh = st_params['types'].count('I')

        tempgen = self.tempgen
        spgen = self.spgen
        if tempgen is not None:
            eaps = tempgen.templates
            locs = tempgen.locations
            rots = tempgen.rotations
            celltypes = tempgen.celltypes
            temp_info = tempgen.info
            cut_outs = temp_info['params']['cut_out']
        else:
            temp_info = None
            cut_outs = self.params['templates']['cut_out']

        spiketrains = spgen.spiketrains

        n_neurons = len(spiketrains)

        if len(spiketrains) > 0:
            duration = spiketrains[0].t_stop - spiketrains[0].t_start
            only_noise = False
        else:
            if self._verbose:
                print('No spike trains provided: only simulating noise')
            only_noise = True
            duration = st_params['duration'] * pq.s

        if 'fs' not in rec_params.keys() and temp_info is not None:
            # when computed from templates fs is in kHz
            params['recordings']['fs'] = 1. / temp_info['params']['dt']
            fs = (params['recordings']['fs'] * pq.kHz).rescale('Hz')
            spike_fs = fs
        elif params['recordings']['fs'] is None and temp_info is not None:
            params['recordings']['fs'] = 1. / temp_info['params']['dt']
            fs = (params['recordings']['fs'] * pq.kHz).rescale('Hz')
            spike_fs = fs
        else:
            # In the rec_params fs is in Hz
            fs = params['recordings']['fs'] * pq.Hz
            if temp_info is not None:
                spike_fs = (1. / temp_info['params']['dt'] * pq.kHz).rescale('Hz')
            else:
                spike_fs = fs

        if 'dtype' not in rec_params.keys():
            params['recordings']['dtype'] = 'float32'
        elif rec_params['dtype'] is None:
            params['recordings']['dtype'] = 'float32'
        else:
            params['recordings']['dtype'] = rec_params['dtype']
        dtype = params['recordings']['dtype']

        if 'noise_mode' not in rec_params.keys():
            params['recordings']['noise_mode'] = 'uncorrelated'
        noise_mode = params['recordings']['noise_mode']

        if 'noise_color' not in rec_params.keys():
            params['recordings']['noise_color'] = False
        noise_color = params['recordings']['noise_color']

        if 'sync_rate' not in rec_params.keys():
            params['recordings']['sync_rate'] = None
        sync_rate = params['recordings']['sync_rate']

        if 'sync_jitt' not in rec_params.keys():
            params['recordings']['sync_jitt'] = 1
        sync_jitt = params['recordings']['sync_jitt'] * pq.ms

        if noise_mode == 'distance-correlated':
            if 'noise_half_distance' not in rec_params.keys():
                params['recordings']['noise_half_distance'] = 30
            half_dist = params['recordings']['noise_half_distance']

        if noise_mode == 'far-neurons':
            if 'far_neurons_n' not in rec_params.keys():
                params['recordings']['far_neurons_n'] = 300
            far_neurons_n = params['recordings']['far_neurons_n']
            if 'far_neurons_max_amp' not in rec_params.keys():
                params['recordings']['far_neurons_max_amp'] = 20
            far_neurons_max_amp = params['recordings']['far_neurons_max_amp']
            if 'far_neurons_noise_floor' not in rec_params.keys():
                params['recordings']['far_neurons_noise_floor'] = 0.5
            far_neurons_noise_floor = params['recordings']['far_neurons_noise_floor']
            if 'far_neurons_exc_inh_ratio' not in rec_params.keys():
                params['recordings']['far_neurons_exc_inh_ratio'] = 0.8
            far_neurons_exc_inh_ratio = params['recordings']['far_neurons_exc_inh_ratio']

        if noise_color:
            if 'color_peak' not in rec_params.keys():
                params['recordings']['color_peak'] = 500
            color_peak = params['recordings']['color_peak']
            if 'color_q' not in rec_params.keys():
                params['recordings']['color_q'] = 1
            color_q = params['recordings']['color_q']
            if 'color_noise_floor' not in rec_params.keys():
                params['recordings']['color_noise_floor'] = 1
            color_noise_floor = params['recordings']['color_noise_floor']
        else:
            color_peak, color_q, color_noise_floor = None, None, None

        if 'noise_level' not in rec_params.keys():
            params['recordings']['noise_level'] = 10
        noise_level = params['recordings']['noise_level']
        if self._verbose:
            print('Noise Level ', noise_level)

        if 'filter' not in rec_params.keys():
            params['recordings']['filter'] = True
        filter = params['recordings']['filter']

        if 'filter_cutoff' not in rec_params.keys():
            params['recordings']['filter_cutoff'] = [300., 6000.]
        cutoff = params['recordings']['filter_cutoff'] * pq.Hz

        if 'filter_order' not in rec_params.keys():
            params['recordings']['filter_order'] = 3
        order = params['recordings']['filter_order']

        if 'modulation' not in rec_params.keys():
            params['recordings']['modulation'] = 'electrode'
        elif params['recordings']['modulation'] not in ['none', 'electrode', 'template']:
            raise Exception("'modulation' can be 'none', 'template', or 'electrode'")
        modulation = params['recordings']['modulation']

        if 'bursting' not in rec_params.keys():
            params['recordings']['bursting'] = False
        bursting = params['recordings']['bursting']

        if 'shape_mod' not in rec_params.keys():
            params['recordings']['shape_mod'] = False
        shape_mod = params['recordings']['shape_mod']

        if bursting:
            if 'exp_decay' not in rec_params.keys():
                params['recordings']['exp_decay'] = 0.2
            exp_decay = params['recordings']['exp_decay']
            if 'n_burst_spikes' not in rec_params.keys():
                params['recordings']['n_burst_spikes'] = 10
            n_burst_spikes = params['recordings']['n_burst_spikes']
            if 'max_burst_duration' not in rec_params.keys():
                params['recordings']['max_burst_duration'] = 100
            max_burst_duration = 100 * pq.ms

            if rec_params['n_bursting'] is None:
                n_bursting = n_neurons
            else:
                n_bursting = rec_params['n_bursting']

            if shape_mod:
                if 'shape_stretch' not in rec_params.keys():
                    params['recordings']['shape_stretch'] = 30
                shape_stretch = params['recordings']['shape_stretch']
                if self._verbose:
                    print('Bursting with modulation sigmoid: ', shape_stretch)
            else:
                shape_stretch = None
        else:
            exp_decay = None
            n_burst_spikes = None
            max_burst_duration = None
            shape_stretch = None
            n_bursting = None

        chunk_noise_duration = params['recordings'].get('chunk_noise_duration', 0) * pq.s
        if chunk_noise_duration == 0 * pq.s:
            chunk_noise_duration = duration

        chunk_filter_duration = params['recordings'].get('chunk_filter_duration', 0) * pq.s
        if chunk_filter_duration == 0 * pq.s:
            chunk_filter_duration = duration

        chunk_conv_duration = params['recordings'].get('chunk_conv_duration', 0) * pq.s
        if chunk_noise_duration == 0 * pq.s:
            chunk_filter_duration = duration

        if 'mrand' not in rec_params.keys():
            params['recordings']['mrand'] = 1
        mrand = params['recordings']['mrand']

        if 'sdrand' not in rec_params.keys():
            params['recordings']['sdrand'] = 0.05
        sdrand = params['recordings']['sdrand']

        if 'overlap' not in rec_params.keys():
            params['recordings']['overlap'] = False
        overlap = params['recordings']['overlap']

        if 'extract_waveforms' not in rec_params.keys():
            params['recordings']['extract_waveforms'] = False
        extract_waveforms = params['recordings']['extract_waveforms']

        if 'seed' not in rec_params.keys():
            params['recordings']['seed'] = np.random.randint(1, 1000)
        elif params['recordings']['seed'] is None:
            params['recordings']['seed'] = np.random.randint(1, 1000)
        noise_seed = params['recordings']['seed']

        if 'xlim' not in temp_params.keys():
            params['templates']['xlim'] = None
        x_lim = params['templates']['xlim']

        if 'ylim' not in temp_params.keys():
            params['templates']['ylim'] = None
        y_lim = params['templates']['ylim']

        if 'zlim' not in temp_params.keys():
            params['templates']['zlim'] = None
        z_lim = params['templates']['zlim']

        if 'n_overlap_pairs' not in temp_params.keys():
            params['templates']['n_overlap_pairs'] = None
        n_overlap_pairs = params['templates']['n_overlap_pairs']

        if 'min_amp' not in temp_params.keys():
            params['templates']['min_amp'] = 50
        min_amp = params['templates']['min_amp']

        if 'max_amp' not in temp_params.keys():
            params['templates']['max_amp'] = np.inf
        max_amp = params['templates']['max_amp']

        if 'min_dist' not in temp_params.keys():
            params['templates']['min_dist'] = 25
        min_dist = params['templates']['min_dist']

        if 'overlap_threshold' not in temp_params.keys():
            params['templates']['overlap_threshold'] = 0.8
        overlap_threshold = params['templates']['overlap_threshold']

        if 'pad_len' not in temp_params.keys():
            params['templates']['pad_len'] = [3., 3.]
        pad_len = params['templates']['pad_len']

        if 'n_jitters' not in temp_params.keys():
            params['templates']['n_jitters'] = 10
        n_jitters = params['templates']['n_jitters']

        if 'upsample' not in temp_params.keys():
            params['templates']['upsample'] = 8
        upsample = params['templates']['upsample']

        if 'seed' not in temp_params.keys():
            params['templates']['seed'] = np.random.randint(1, 1000)
        elif params['templates']['seed'] is None:
            params['templates']['seed'] = np.random.randint(1, 1000)
        temp_seed = params['templates']['seed']

        if 'drifting' not in rec_params.keys():
            params['recordings']['drifting'] = False
        drifting = params['recordings']['drifting']

        if drifting:
            if temp_info is not None:
                assert temp_info['params']['drifting'], "For generating drifting recordings, templates must be drifting"
            else:
                if params['n_jitters'] == 1:
                    assert len(self.templates.shape) == 4
                else:
                    assert len(self.templates.shape) == 5
            preferred_dir = np.array(rec_params['preferred_dir'])
            preferred_dir = preferred_dir / np.linalg.norm(preferred_dir)
            angle_tol = rec_params['angle_tol']
            drift_velocity = rec_params['slow_drift_velocity']
            fast_drift_period = rec_params['fast_drift_period'] * pq.s
            fast_drift_max_jump = rec_params['fast_drift_max_jump']
            fast_drift_min_jump = rec_params['fast_drift_min_jump']
            drift_mode = rec_params['drift_mode']
            t_start_drift = rec_params['t_start_drift'] * pq.s
            if rec_params['n_drifting'] is None:
                n_drifting = n_neurons
            else:
                n_drifting = rec_params['n_drifting']
        else:
            # if drifting templates, but not recordings, consider initial template
            if temp_info is not None:
                if temp_info['params']['drifting']:
                    eaps = eaps[:, 0]
                    locs = locs[:, 0]
            elif len(self.templates.shape) == 5:
                self.templates = self.templates[:, 0]
                self.template_locs = self.template_locs[:, 0]
            preferred_dir = None
            angle_tol = None
            drift_velocity = None
            fast_drift_period = None
            fast_drift_max_jump = None
            fast_drift_min_jump = None
            t_start_drift = None
            n_drifting = None
            drift_mode = None

        # load MEA info
        if temp_info is not None:
            mea = mu.return_mea(info=temp_info['electrodes'])
            params['electrodes'] = temp_info['electrodes']
        else:
            mea = mu.return_mea(info=self.params['electrodes'])
            params['electrodes'] = self.params['electrodes']
        mea_pos = mea.positions
        n_elec = mea_pos.shape[0]
        n_samples = int(duration.rescale('s').magnitude * fs.rescale('Hz').magnitude)

        params['recordings'].update({'duration': float(duration.magnitude),
                                     'fs': float(fs.rescale('Hz').magnitude),
                                     'n_neurons': n_neurons})
        params['templates'].update({'cut_out': cut_outs})

        # create buffer h5/memmap/memmory
        if self.tmp_mode == 'h5':
            tmp_path = self.tmp_folder / (tmp_prefix + "mearec_tmp_file.h5")
            assert not os.path.exists(tmp_path), 'temporay file already exists'
            tmp_file = h5py.File(tmp_path, mode='w')
            recordings = tmp_file.create_dataset("recordings", (n_elec, n_samples), dtype=dtype)
            if not only_noise:
                spike_traces = tmp_file.create_dataset("spike_traces", (n_neurons, n_samples), dtype=dtype)

            self._to_remove_on_delete.append(tmp_path)

        elif self.tmp_mode == 'memmap':
            tmp_path_0 = self.tmp_folder / (tmp_prefix + "mearec_tmp_file_recordings.raw")
            recordings = np.memmap(tmp_path_0, shape=(n_samples, n_elec), dtype=dtype, mode='w+')
            recordings[:] = 0
            recordings = recordings.transpose()
            tmp_path_1 = self.tmp_folder / (tmp_prefix + "mearec_tmp_file_spike_traces.raw")
            if not only_noise:
                spike_traces = np.memmap(tmp_path_1, shape=(n_samples, n_neurons), dtype=dtype, mode='w+')
                spike_traces[:] = 0
                spike_traces = spike_traces.transpose()

            self._to_remove_on_delete.extend([tmp_path_0, tmp_path_1])

        else:
            recordings = np.zeros((n_elec, n_samples), dtype=dtype)
            spike_traces = np.zeros((n_neurons, n_samples), dtype=dtype)

        timestamps = np.arange(recordings.shape[1]) / fs

        #######################
        # Step 1: convolution #
        #######################
        if only_noise:
            spiketrains = np.array([])
            voltage_peaks = np.array([])
            spike_traces = np.array([])
            templates = np.array([])
            template_locs = np.array([])
            template_rots = np.array([])
            template_celltypes = np.array([])
            overlapping = np.array([])
        else:
            if tempgen is not None:
                if celltype_params is not None:
                    if 'excitatory' in celltype_params.keys() and 'inhibitory' in celltype_params.keys():
                        exc_categories = celltype_params['excitatory']
                        inh_categories = celltype_params['inhibitory']
                        bin_cat = get_binary_cat(celltypes, exc_categories, inh_categories)
                    else:
                        bin_cat = np.array(['U'] * len(celltypes))
                else:
                    bin_cat = np.array(['U'] * len(celltypes))

                if 'type' in spiketrains[0].annotations.keys():
                    n_exc = [st.annotations['type'] for st in spiketrains].count('E')
                    n_inh = n_neurons - n_exc
                if self._verbose:
                    print('Templates selection seed: ', temp_seed)
                np.random.seed(temp_seed)

                if drifting:
                    drift_directions = np.array([(p[-1] - p[0]) / np.linalg.norm(p[-1] - p[0]) for p in locs])
                    drift_velocity_ums = drift_velocity / 60.
                    velocity_vector = drift_velocity_ums * preferred_dir
                    if self._verbose:
                        print('Drift mode: ', drift_mode)
                        if 'slow' in drift_mode:
                            print('Slow drift velocity', drift_velocity, 'um/min')
                        if 'fast' in drift_mode:
                            print('Fast drift period', fast_drift_period)
                            print('Fast drift max jump',
                                  fast_drift_max_jump)  # 'Fast drift min jump', fast_drift_min_jump)
                    n_elec = eaps.shape[2]
                else:
                    drift_directions = None
                    preferred_dir = None
                    velocity_vector = None
                    n_elec = eaps.shape[1]

                if self._verbose:
                    print('Selecting cells')
                idxs_cells, selected_cat = select_templates(locs, eaps, bin_cat, n_exc, n_inh, x_lim=x_lim, y_lim=y_lim,
                                                            z_lim=z_lim, min_amp=min_amp, max_amp=max_amp,
                                                            min_dist=min_dist, drifting=drifting,
                                                            drift_dir=drift_directions,
                                                            preferred_dir=preferred_dir, angle_tol=angle_tol,
                                                            n_overlap_pairs=n_overlap_pairs,
                                                            overlap_threshold=overlap_threshold,
                                                            verbose=self._verbose)

                idxs_cells = sorted(idxs_cells)  # [np.argsort(selected_cat)]
                template_celltypes = celltypes[idxs_cells]
                template_locs = locs[idxs_cells]
                template_rots = rots[idxs_cells]
                templates_bin = bin_cat[idxs_cells]
                templates = eaps[idxs_cells]

                # find overlapping templates
                overlapping = find_overlapping_templates(templates, thresh=overlap_threshold)

                # peak images
                voltage_peaks = []
                for tem in templates:
                    dt = 1. / fs.magnitude
                    if not drifting:
                        feat = get_templates_features(tem, ['neg'], dt=dt)
                    else:
                        feat = get_templates_features(tem[0], ['neg'], dt=dt)
                    voltage_peaks.append(-np.squeeze(feat['neg']))
                voltage_peaks = np.array(voltage_peaks)

                # pad templates
                pad_samples = [int((pp * fs.rescale('kHz')).magnitude) for pp in pad_len]
                if self._verbose:
                    print('Padding template edges')
                t_pad = time.time()
                templates_pad = pad_templates(templates, pad_samples, drifting, self._verbose,
                                              parallel=True)

                if self._verbose:
                    print('Elapsed pad time:', time.time() - t_pad)

                # resample spikes
                t_rs = time.time()
                up = fs
                down = spike_fs
                spike_duration_pad = templates_pad.shape[-1]
                if up != down:
                    n_resample = int(spike_duration_pad * (up / down))
                    templates_rs = resample_templates(templates_pad, n_resample, up, down,
                                                      drifting, self._verbose)
                    if self._verbose:
                        print('Elapsed resample time:', time.time() - t_rs)
                else:
                    templates_rs = templates_pad

                if self._verbose:
                    print('Creating time jittering')
                jitter = 1. / fs
                t_j = time.time()
                templates = jitter_templates(templates_rs, upsample, fs, n_jitters, jitter,
                                             drifting, self._verbose, parallel=True)
                if self._verbose:
                    print('Elapsed jitter time:', time.time() - t_j)

                # find cut out samples for convolution after padding and resampling
                pre_peak_fraction = (pad_len[0] + cut_outs[0]) / (np.sum(pad_len) + np.sum(cut_outs))
                samples_pre_peak = int(pre_peak_fraction * templates.shape[-1])
                samples_post_peak = templates.shape[-1] - samples_pre_peak
                cut_outs_samples = [samples_pre_peak, samples_post_peak]

                # delete temporary preprocessed templates
                del templates_rs, templates_pad
            else:
                templates = self.templates
                pre_peak_fraction = (pad_len[0] + cut_outs[0]) / (np.sum(pad_len) + np.sum(cut_outs))
                samples_pre_peak = int(pre_peak_fraction * templates.shape[-1])
                samples_post_peak = templates.shape[-1] - samples_pre_peak
                cut_outs_samples = [samples_pre_peak, samples_post_peak]
                template_locs = self.template_locations
                template_rots = self.template_rotations
                template_celltypes = self.template_celltypes
                if celltype_params is not None:
                    if 'excitatory' in celltype_params.keys() and 'inhibitory' in celltype_params.keys():
                        exc_categories = celltype_params['excitatory']
                        inh_categories = celltype_params['inhibitory']
                        templates_bin = get_binary_cat(template_celltypes, exc_categories, inh_categories)
                    else:
                        templates_bin = np.array(['U'] * len(celltypes))
                else:
                    templates_bin = np.array(['U'] * len(celltypes))
                voltage_peaks = self.voltage_peaks
                overlapping = np.array([])
                if not drifting:
                    velocity_vector = None
                else:
                    drift_velocity_ums = drift_velocity / 60.
                    velocity_vector = drift_velocity_ums * preferred_dir
                    if self._verbose:
                        print('Drift velocity vector: ', velocity_vector)

            if sync_rate is not None:
                if self._verbose:
                    print('Modifying synchrony of spatially overlapping spikes')
                if self._verbose:
                    print('Overlapping templates: ', overlapping)
                for over in overlapping:
                    if self._verbose:
                        print('Overlapping pair: ', over)
                    spgen.add_synchrony(over, rate=sync_rate, verbose=self._verbose, time_jitt=sync_jitt)
                    # annotate new firing rates
                    fr1 = len(spgen.spiketrains[over[0]].times) / spgen.spiketrains[over[0]].t_stop
                    fr2 = len(spgen.spiketrains[over[1]].times) / spgen.spiketrains[over[1]].t_stop
                    spgen.spiketrains[over[0]].annotate(fr=fr1)
                    spgen.spiketrains[over[1]].annotate(fr=fr2)
            self.overlapping = overlapping

            # find SNR and annotate
            if self._verbose:
                print('Computing spike train SNR')
            for t_i, temp in enumerate(templates):
                min_peak = np.min(temp)
                snr = np.abs(min_peak / float(noise_level))
                spiketrains[t_i].annotate(snr=snr)

            if self._verbose:
                print('Adding spiketrain annotations')
            for i, st in enumerate(spiketrains):
                st.annotate(bintype=templates_bin[i], mtype=template_celltypes[i], soma_position=template_locs[i])

            if overlap:
                annotate_overlapping_spikes(spiketrains, overlapping_pairs=overlapping, verbose=self._verbose)

            amp_mod = []
            cons_spikes = []

            # modulated convolution
            if drifting:
                drifting_units = np.random.permutation(n_neurons)[:n_drifting]
            else:
                drifting_units = []
            if bursting:
                bursting_units = np.random.permutation(n_neurons)[:n_bursting]
            else:
                bursting_units = []

            if modulation == 'template':
                if self._verbose:
                    print('Template modulation')
                for i_s, st in enumerate(spiketrains):
                    if bursting and i_s in bursting_units:
                        if self._verbose:
                            print('Bursting unit: ', i_s)
                        amp, cons = compute_modulation(st, sdrand=sdrand,
                                                       n_spikes=n_burst_spikes, exp=exp_decay,
                                                       max_burst_duration=max_burst_duration)
                        amp_mod.append(amp)
                        cons_spikes.append(cons)
                        st.annotate(bursting=True)
                    else:
                        amp, cons = compute_modulation(st, mrand=mrand, sdrand=sdrand,
                                                       n_spikes=0)
                        amp_mod.append(amp)
                        cons_spikes.append(cons)
            elif modulation == 'electrode':
                if self._verbose:
                    print('Electrode modulaton')
                for i_s, st in enumerate(spiketrains):
                    if bursting and i_s in bursting_units:
                        if self._verbose:
                            print('Bursting unit: ', i_s)
                        amp, cons = compute_modulation(st, n_el=n_elec, mrand=mrand, sdrand=sdrand,
                                                       n_spikes=n_burst_spikes, exp=exp_decay,
                                                       max_burst_duration=max_burst_duration)
                        amp_mod.append(amp)
                        cons_spikes.append(cons)
                        st.annotate(bursting=True)
                    else:
                        amp, cons = compute_modulation(st, n_el=n_elec, mrand=mrand, sdrand=sdrand,
                                                       n_spikes=0)
                        amp_mod.append(amp)
                        cons_spikes.append(cons)

            spike_matrix = resample_spiketrains(spiketrains, fs=fs)

            # divide in chunks
            chunk_indexes = make_chunk_indexes(duration, chunk_conv_duration, fs)

            verbose = self._verbose >= 2
            # call the loop on chunks
            args = (spike_matrix, modulation, drifting, drift_mode, drifting_units, templates,
                    cut_outs_samples, template_locs, velocity_vector, fast_drift_period,
                    fast_drift_min_jump, fast_drift_max_jump, t_start_drift, fs, verbose,
                    amp_mod, bursting_units, shape_mod, shape_stretch,
                    True, voltage_peaks, dtype,)
            assignment_dict = {
                'recordings': recordings,
                'spike_traces': spike_traces}
            output_list = run_several_chunks(chunk_convolution, chunk_indexes, fs, timestamps, args,
                                             self.n_jobs, self.tmp_mode, self.tmp_folder, assignment_dict)

            # if drift then propagate annoations to spikestrains
            for st in np.arange(n_neurons):
                if drifting and st in drifting_units:
                    spiketrains[st].annotate(drifting=True)
                    spiketrains[st].annotate(template_idxs=output_list[-1]['template_idxs'][st])

        #################
        # Step 2: noise #
        #################
        if self._verbose:
            print('Adding noise')
            print('Noise seed: ', noise_seed)

        np.random.seed(noise_seed)

        # divide in chunks
        chunk_indexes = make_chunk_indexes(duration, chunk_noise_duration, fs)

        if noise_level == 0:
            if self._verbose:
                print('Noise level is set to 0')
        else:
            if self.tmp_mode == 'h5':
                tmp_path_noise = self.tmp_folder / (tmp_prefix + "mearec_tmp_noise_file.h5")
                tmp_noise_rec = h5py.File(tmp_path_noise, mode='w')
                additive_noise = tmp_noise_rec.create_dataset("recordings", (n_elec, n_samples), dtype=dtype)
                self._to_remove_on_delete.append(tmp_path_noise)
            elif self.tmp_mode == 'memmap':
                tmp_path_noise = self.tmp_folder / (tmp_prefix + "mearec_tmp_noise_file.raw")
                additive_noise = np.memmap(tmp_path_noise, shape=(n_samples, n_elec), dtype=dtype, mode='w+')
                additive_noise = additive_noise.transpose()
                self._to_remove_on_delete.append(tmp_path_noise)
            else:
                additive_noise = np.zeros((n_elec, n_samples), dtype=dtype)

            if noise_mode == 'uncorrelated':
                func = chunk_uncorrelated_noise
                num_chan = recordings.shape[0]
                args = (num_chan, noise_level, noise_color, color_peak, color_q, color_noise_floor,
                        fs.rescale('Hz').magnitude, dtype)
                assignment_dict = {'recordings': additive_noise}

                run_several_chunks(func, chunk_indexes, fs, timestamps, args,
                                   self.n_jobs, self.tmp_mode, self.tmp_folder, assignment_dict)

            elif noise_mode == 'distance-correlated':
                cov_dist = np.zeros((n_elec, n_elec))
                for i, el in enumerate(mea.positions):
                    for j, p in enumerate(mea.positions):
                        if i != j:
                            cov_dist[i, j] = (0.5 * half_dist) / np.linalg.norm(el - p)
                        else:
                            cov_dist[i, j] = 1

                func = chunk_distance_correlated_noise
                args = (noise_level, cov_dist, n_elec, noise_color, color_peak, color_q, color_noise_floor,
                        fs.rescale('Hz').magnitude, dtype)
                assignment_dict = {'recordings': additive_noise}

                run_several_chunks(func, chunk_indexes, fs, timestamps, args,
                                   self.n_jobs, self.tmp_mode, self.tmp_folder, assignment_dict)

            elif noise_mode == 'far-neurons':
                idxs_cells, selected_cat = select_templates(locs, eaps, bin_cat=None, n_exc=far_neurons_n, n_inh=0,
                                                            x_lim=x_lim, y_lim=y_lim, z_lim=z_lim, min_amp=0,
                                                            max_amp=far_neurons_max_amp, min_dist=1,
                                                            verbose=False)
                idxs_cells = sorted(idxs_cells)
                templates_noise = eaps[idxs_cells]
                template_noise_locs = locs[idxs_cells]
                if drifting:
                    templates_noise = templates_noise[:, 0]

                # pad spikes
                pad_samples = [int((pp * fs.rescale('kHz')).magnitude) for pp in pad_len]
                if self._verbose:
                    print('Padding noisy template edges')
                t_pad = time.time()
                templates_noise_pad = pad_templates(templates_noise, pad_samples, drifting, self._verbose,
                                                    parallel=True)
                if self._verbose:
                    print('Elapsed pad time:', time.time() - t_pad)

                # resample spikes
                t_rs = time.time()
                up = fs
                down = spike_fs
                spike_duration_pad = templates_noise_pad.shape[-1]
                if up != down:
                    n_resample = int(spike_duration_pad * (up / down))
                    templates_noise = resample_templates(templates_noise_pad, n_resample, up, down,
                                                         drifting, self._verbose)
                    if self._verbose:
                        print('Elapsed resample time:', time.time() - t_rs)
                else:
                    templates_noise = templates_noise_pad

                # find cut out samples for convolution after padding and resampling
                pre_peak_fraction = (pad_len[0] + cut_outs[0]) / (np.sum(pad_len) + np.sum(cut_outs))
                samples_pre_peak = int(pre_peak_fraction * templates.shape[-1])
                samples_post_peak = templates_noise.shape[-1] - samples_pre_peak
                cut_outs_samples = [samples_pre_peak, samples_post_peak]

                del templates_noise_pad

                # create noisy spiketrains
                if self._verbose:
                    print('Generating noisy spike trains')
                noisy_spiketrains_params = params['spiketrains']
                noisy_spiketrains_params['n_exc'] = int(far_neurons_n * far_neurons_exc_inh_ratio)
                noisy_spiketrains_params['n_inh'] = far_neurons_n - noisy_spiketrains_params['n_exc']
                noisy_spiketrains_params['seed'] = noise_seed
                spgen_noise = SpikeTrainGenerator(params=noisy_spiketrains_params)
                spgen_noise.generate_spikes()
                spiketrains_noise = spgen_noise.spiketrains

                spike_matrix_noise = resample_spiketrains(spiketrains_noise, fs=fs)
                if self._verbose:
                    print('Convolving noisy spike trains')
                templates_noise = templates_noise.reshape((templates_noise.shape[0], 1, templates_noise.shape[1],
                                                           templates_noise.shape[2]))

                chunk_indexes = make_chunk_indexes(duration, chunk_conv_duration, fs)

                # call the loop on chunks
                verbose = self._verbose >= 2
                args = (spike_matrix_noise, 'none', False, None, None, templates_noise,
                        cut_outs_samples, template_noise_locs, None, None, None, None, None, None,
                        verbose, None, None, False, None, False, None, dtype,)
                assignment_dict = {'recordings': additive_noise}
                run_several_chunks(chunk_convolution, chunk_indexes, fs, timestamps, args,
                                   self.n_jobs, self.tmp_mode, self.tmp_folder, assignment_dict)

                # removing mean
                for i, m in enumerate(np.mean(additive_noise, axis=1)):
                    if self.tmp_mode == 'h5':
                        additive_noise[i, ...] -= m
                    elif self.tmp_mode == 'memmap':
                        additive_noise[i, :] -= m
                    else:
                        additive_noise[i, :] -= m

                # adding noise floor
                for i, s in enumerate(np.std(additive_noise, axis=1)):
                    if self.tmp_mode == 'h5':
                        additive_noise[i, ...] += far_neurons_noise_floor * s * \
                                                  np.random.randn(additive_noise.shape[1])
                    elif self.tmp_mode == 'memmap':
                        additive_noise[i, :] += far_neurons_noise_floor * s * \
                                                np.random.randn(additive_noise.shape[1])
                    else:
                        additive_noise[i, :] += far_neurons_noise_floor * s * \
                                                np.random.randn(additive_noise.shape[1])

                # scaling noise
                noise_scale = noise_level / np.std(additive_noise, axis=1)
                if self._verbose:
                    print('Scaling to reach desired level')

                for i, n in enumerate(noise_scale):
                    if self.tmp_mode == 'h5':
                        additive_noise[i, ...] *= n
                    elif self.tmp_mode == 'memmap':
                        additive_noise[i, :] *= n
                    else:
                        additive_noise[i, :] *= n

            # Add it to recordings
            if self.tmp_mode == 'h5':
                recordings[...] += additive_noise
            elif self.tmp_mode == 'memmap':
                recordings += additive_noise
            else:
                recordings += additive_noise

        ##################
        # Step 3: filter #
        ##################
        if filter:
            if self._verbose:
                print('Filtering')
                if cutoff.size == 1:
                    print('High-pass cutoff', cutoff)
                elif cutoff.size == 2:
                    print('Band-pass cutoff', cutoff)

            chunk_indexes = make_chunk_indexes(duration, chunk_filter_duration, fs)

            # call the loop on chunks
            args = (recordings, cutoff, order, fs, dtype,)
            assignment_dict = {
                'filtered_chunk': recordings,
            }
            # Done in loop (as before) : this cannot be done in parralel because of bug transpose in joblib!!!!!!!!!!!!!
            output_list = run_several_chunks(chunk_apply_filter, chunk_indexes, fs, timestamps, args,
                                             # ~ self.n_jobs, self.tmp_mode, self.tmp_folder, assignment_dict)
                                             1, None, None, assignment_dict)

        #############################
        # Step 4: extract waveforms #
        #############################
        if not only_noise:
            if extract_waveforms:
                if self._verbose:
                    print('Extracting spike waveforms')
                extract_wf(spiketrains, recordings, fs=fs, timestamps=timestamps)

        params['templates']['overlapping'] = np.array(overlapping)
        self.recordings = recordings
        self.timestamps = timestamps
        self.channel_positions = mea_pos
        self.templates = np.squeeze(templates)
        self.template_locations = template_locs
        self.template_rotations = template_rots
        self.template_celltypes = template_celltypes
        self.spiketrains = spiketrains
        self.voltage_peaks = voltage_peaks
        self.spike_traces = spike_traces
        self.info = params

    def annotate_overlapping_spikes(self, parallel=True):
        """
        Annnotate spike trains with overlapping information.

        parallel : bool
            If True, spike trains are annotated in parallel
        """
        if self.info['templates']['overlapping'] is None or len(self.info['templates']['overlapping']) == 0:
            if self._verbose:
                print('Finding overlapping spikes')
            if len(self.templates.shape) == 3:
                templates = self.templates
            elif len(self.templates.shape) == 4:
                # drifting + no jitt or no drifting + jitt
                templates = self.templates[:, 0]
            elif len(self.templates.shape) == 5:
                # drifting + jitt
                templates = self.templates[:, 0, 0]
            self.overlapping = find_overlapping_templates(templates,
                                                          thresh=self.info['templates']['overlap_threshold'])
            print('Overlapping templates: ', self.overlapping)
            self.info['templates']['overlapping'] = self.overlapping
        annotate_overlapping_spikes(self.spiketrains, overlapping_pairs=self.overlapping, parallel=parallel)

    def extract_waveforms(self, cut_out=[0.5, 2]):
        """
        Extract waveforms from spike trains and recordings.

        Parameters
        ----------
        cut_out : float or list
            Ms before and after peak to cut out. If float the cut is symmetric.
        """
        fs = self.info['recordings']['fs'] * pq.Hz
        extract_wf(self.spiketrains, self.recordings, fs=fs, cut_out=cut_out)

    def extract_templates(self, cut_out=[0.5, 2], recompute=False):
        """
        Extract templates from spike trains.

        Parameters
        ----------
        cut_out : float or list
            Ms before and after peak to cut out. If float the cut is symmetric.
        recompute :  bool
            If True, templates are recomputed from extracted waveforms
        """
        fs = self.info['recordings']['fs'] * pq.Hz

        if not len(self.spiketrains) == 0:
            if self.spiketrains[0].waveforms is None:
                extract_wf(self.spiketrains, self.recordings, fs=fs, cut_out=cut_out)

        if self.tempgen is None or len(self.templates) == 0 or not recompute:
            wfs = [st.waveforms for st in self.spiketrains]
            templates = np.array([np.mean(wf, axis=0) for wf in wfs])
            if np.array(cut_out).size == 1:
                cut_out = [cut_out, cut_out]
            self.info['templates']['cut_out'] = cut_out
            self.info['templates']['pad_len'] = [0, 0]
            self.templates = templates[:, np.newaxis]
        else:
            raise Exception("templates are already computed. Use the 'recompute' argument to compute them from "
                            "extracted waveforms")


def make_chunk_indexes(total_duration, chunk_duration, fs):
    """
    Construct chunks list.
    Return a list of (start, stop) indexes.
    """
    fs_float = fs.rescale('Hz').magnitude
    chunk_size = int(chunk_duration.rescale('s').magnitude * fs_float)
    total_length = int(total_duration.rescale('s').magnitude * fs_float)

    if chunk_size == 0:
        chunk_indexes = [(0, total_length), ]
    else:
        n = int(np.floor(total_length / chunk_size))
        chunk_indexes = [(i * chunk_size, (i + 1) * chunk_size) for i in range(n)]
        if (total_length % chunk_size) > 0:
            chunk_indexes.append((n * chunk_size, total_length))

    return chunk_indexes


def run_several_chunks(func, chunk_indexes, fs, timestamps, args, n_jobs, tmp_mode, tmp_folder, assignment_dict):
    """
    Run a function on a list of chunks.
    
    this can be done in loop if n_jobs=1 (or 0)
    or in paralell if n_jobs>1
    
    The function can return
    
    
    """

    parallel_job = not n_jobs in (0, 1)
    # create task list
    arg_tasks = []
    karg_tasks = []
    for ch, (i_start, i_stop) in enumerate(chunk_indexes):

        chunk_start = (i_start / fs).rescale('s')

        arg_task = (ch, i_start, i_stop, chunk_start,) + args
        arg_tasks.append(arg_task)

        karg_task = dict(assignment_dict=assignment_dict, tmp_mode=tmp_mode, parallel_job=parallel_job)
        if tmp_mode is None:
            pass
        elif tmp_mode == 'h5':
            tmp_file = str(tmp_folder / ('tmp_chunk_' + str(ch) + '.h5'))
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
            karg_task['tmp_file'] = tmp_file
            karg_task['assignment_dict'] = {k: None for k in assignment_dict}  #  trick to not transmit the file
        elif tmp_mode == 'memmap':
            pass
        karg_tasks.append(karg_task)

    # run chunks
    if n_jobs in (0, 1):
        # simple loop
        output_list = []
        for ch, (i_start, i_stop) in enumerate(chunk_indexes):
            out = func(*arg_tasks[ch], **karg_tasks[ch])
            output_list.append(out)

            if tmp_mode is None:
                for key, full_arr in assignment_dict.items():
                    out_chunk = out[key]
                    full_arr[:, i_start:i_stop] += out_chunk
            elif tmp_mode == 'h5':
                tmp_file = karg_tasks[ch]['tmp_file']
                with h5py.File(tmp_file, 'r') as f:
                    for key, full_arr in assignment_dict.items():
                        out_chunk = f[key]
                        full_arr[:, i_start:i_stop] = out_chunk
                os.remove(tmp_file)
            elif tmp_mode == 'memmap':
                pass
                # Nothing to do here because done inside the func with FuncThenAddChunk

    else:
        # parallel
        output_list = Parallel(n_jobs=n_jobs)(
            delayed(func)(*arg_task, **karg_task) for arg_task, karg_task in zip(arg_tasks, karg_tasks))

        if tmp_mode == 'h5':
            for ch, arg_task in enumerate(arg_tasks):
                tmp_file = karg_tasks[ch]['tmp_file']
                with h5py.File(tmp_file, 'r') as f:
                    for key, full_arr in assignment_dict.items():
                        out_chunk = f[key]
                        full_arr[..., i_start:i_stop] += out_chunk
                os.remove(tmp_file)
        elif tmp_mode == 'memmap':
            pass
            # Nothing to do here because done inside the func
        else:
            # This case is very unefficient because it double the memory usage!!!!!!!
            for ch, (i_start, i_stop) in enumerate(chunk_indexes):
                for key, full_arr in assignment_dict.items():
                    full_arr[:, i_start:i_stop] += output_list[ch][key]

    return output_list
