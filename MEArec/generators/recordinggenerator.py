import numpy as np
import time
from copy import deepcopy
from MEArec.tools import (select_templates, find_overlapping_templates, get_binary_cat,
                          resample_templates, jitter_templates, pad_templates, get_templates_features,
                          compute_modulation, annotate_overlapping_spikes, extract_wf)

from .recgensteps import (chunk_convolution, chunk_uncorrelated_noise,
                          chunk_distance_correlated_noise, chunk_apply_filter)

import random
import string
import MEAutility as mu
import yaml
import os
import shutil
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
    """

    def __init__(self, spgen=None, tempgen=None, params=None, rec_dict=None, info=None):
        self._verbose = False
        self._verbose_1 = False
        self._verbose_2 = False

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
                if 'spiketrains' in self.info:
                    self.spgen = SpikeTrainGenerator(spiketrains=self.spiketrains, params=self.info['spiketrains'])
                else:
                    self.spgen = SpikeTrainGenerator(spiketrains=self.spiketrains, params={'custom': True})
            self.tempgen = None
            if isinstance(self.recordings, np.memmap):
                self.tmp_mode = 'memmap'
            else:
                self.tmp_mode = None

        else:
            if spgen is None or tempgen is None:
                raise AttributeError("Specify SpikeTrainGenerator and TemplateGenerator objects!")
            if params is None:
                params = {'spiketrains': {},
                          'celltypes': {},
                          'templates': {},
                          'recordings': {},
                          'seeds': {}}
            self.params = deepcopy(params)
            self.spgen = spgen
            self.tempgen = tempgen
            self.tmp_mode = None

        self.overlapping = []
        # temp file that should remove on delete
        self._to_remove_on_delete = []
        self.tmp_folder = None
        self._is_tmp_folder_local = False

    def __del__(self):
        self.recordings = None
        self.spike_traces = None

        if not self._is_tmp_folder_local:
            if self.tmp_folder is not None:
                try:
                    shutil.rmtree(self.tmp_folder)
                    if self._verbose >= 1:
                        print('Deleted', self.tmp_folder)
                except Exception as e:
                    if self._verbose >= 1:
                        print('Impossible to delete temp file:', self.tmp_folder, 'Error', e)
        else:
            for fname in self._to_remove_on_delete:
                try:
                    os.remove(fname)
                    if self._verbose >= 1:
                        print('Deleted', fname)
                except Exception as e:
                    if self._verbose >= 1:
                        print('Impossible to delete temp file:', fname, 'Error', e)

    def generate_recordings(self, tmp_mode=None, tmp_folder=None, verbose=None, n_jobs=0):
        """
        Generates recordings

        Parameters
        ----------
        tmp_mode : None, 'memmap'
            Use temporary file h5 memmap or None
            None is no temporary file and then use memory.
        tmp_folder: str or Path
            In case of tmp files, you can specify the folder.
            If None, then it is automatic using tempfile.mkdtemp()
        verbose: bool or int
            Determines the level of verbose. If 1 or True, low-level, if 2 high level, if False, not verbose
        n_jobs: int if >1 then use joblib to execute chunk in parallel else in loop
        """

        self.tmp_mode = tmp_mode
        self.tmp_folder = tmp_folder
        self.n_jobs = n_jobs

        if tmp_mode is not None:
            tmp_prefix = ''.join([random.choice(string.ascii_letters) for i in range(5)]) + '_'
        if self.tmp_mode is not None:
            if self.tmp_folder is None:
                self.tmp_folder = Path(tempfile.mkdtemp())
                self._is_tmp_folder_local = False
            else:
                self.tmp_folder = Path(self.tmp_folder)
                self._is_tmp_folder_local = True
        else:
            self._is_tmp_folder_local = False

        self._verbose = verbose
        if self._verbose is not None and isinstance(self._verbose, bool) or isinstance(self._verbose, int):
            verbose_1 = self._verbose >= 1
            verbose_2 = self._verbose >= 2
        elif isinstance(verbose, bool):
            if self._verbose:
                verbose_1 = True
                verbose_2 = False
            else:
                verbose_1 = False
                verbose_2 = False
        else:  # None
            verbose_1 = False
            verbose_2 = False
        self._verbose_1 = verbose_1
        self._verbose_2 = verbose_2

        params = deepcopy(self.params)
        temp_params = self.params['templates']
        rec_params = self.params['recordings']
        st_params = self.params['spiketrains']
        seeds = self.params['seeds']

        if 'cell_types' in self.params.keys():
            celltype_params = self.params['cell_types']
        else:
            celltype_params = {}

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
            if verbose_1:
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

        if verbose_1:
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
            max_burst_duration = params['recordings']['max_burst_duration'] * pq.ms

            if rec_params['n_bursting'] is None:
                n_bursting = n_neurons
            else:
                n_bursting = rec_params['n_bursting']

            if shape_mod:
                if 'shape_stretch' not in rec_params.keys():
                    params['recordings']['shape_stretch'] = 30
                shape_stretch = params['recordings']['shape_stretch']
                if verbose_1:
                    print('Bursting with modulation sigmoid: ', shape_stretch)
            else:
                shape_stretch = None
        else:
            exp_decay = None
            n_burst_spikes = None
            max_burst_duration = None
            shape_stretch = None
            n_bursting = None

        chunk_duration = params['recordings'].get('chunk_duration', 0) * pq.s
        if chunk_duration == 0 * pq.s:
            chunk_duration = duration

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

        if 'drifting' not in rec_params.keys():
            params['recordings']['drifting'] = False
        drifting = params['recordings']['drifting']

        # set seeds
        if 'templates' not in seeds.keys():
            temp_seed = np.random.randint(1, 1000)
        elif seeds['templates'] is None:
            temp_seed = np.random.randint(1, 1000)
        else:
            temp_seed = seeds['templates']

        if 'convolution' not in seeds.keys():
            conv_seed = np.random.randint(1, 1000)
        elif seeds['convolution'] is None:
            conv_seed = np.random.randint(1, 1000)
        else:
            conv_seed = seeds['convolution']

        if 'noise' not in seeds.keys():
            noise_seed = np.random.randint(1, 1000)
        elif seeds['noise'] is None:
            noise_seed = np.random.randint(1, 1000)
        else:
            noise_seed = seeds['noise']

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
            if 'fast' in drift_mode:
                if chunk_duration > 0 and chunk_duration != duration:
                    print('Disabling chunking for fast drifts')
                    chunk_duration = duration
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
        if self.tmp_mode == 'memmap':
            tmp_path_0 = self.tmp_folder / (tmp_prefix + "mearec_tmp_file_recordings.raw")
            recordings = np.memmap(tmp_path_0, shape=(n_samples, n_elec), dtype=dtype, mode='w+')
            recordings[:] = 0
            # recordings = recordings.transpose()
            tmp_path_1 = self.tmp_folder / (tmp_prefix + "mearec_tmp_file_spike_traces.raw")
            if not only_noise:
                spike_traces = np.memmap(tmp_path_1, shape=(n_samples, n_neurons), dtype=dtype, mode='w+')
                spike_traces[:] = 0
                # spike_traces = spike_traces.transpose()
            # file names for templates
            tmp_templates_pad = self.tmp_folder / (tmp_prefix + "templates_pad.raw")
            tmp_templates_rs = self.tmp_folder / (tmp_prefix + "templates_resample.raw")
            tmp_templates_jit = self.tmp_folder / (tmp_prefix + "templates_jitter.raw")
            self._to_remove_on_delete.extend([tmp_path_0, tmp_path_1,
                                              tmp_templates_pad, tmp_templates_rs, tmp_templates_jit])
        else:
            recordings = np.zeros((n_samples, n_elec), dtype=dtype)
            spike_traces = np.zeros((n_samples, n_neurons), dtype=dtype)
            tmp_templates_pad = None
            tmp_templates_rs = None
            tmp_templates_jit = None

        timestamps = np.arange(recordings.shape[0]) / fs

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

                if 'cell_type' in spiketrains[0].annotations.keys():
                    n_exc = [st.annotations['cell_type'] for st in spiketrains].count('E')
                    n_inh = n_neurons - n_exc
                    st_types = np.array([st.annotations['cell_type'] for st in spiketrains])
                elif 'rates' in st_params.keys():
                    assert st_params['types'] is not None, "If 'rates' are provided as spiketrains parameters, " \
                                                           "corresponding 'types' ('E'-'I') must be provided"
                    n_exc = st_params['types'].count('E')
                    n_inh = st_params['types'].count('I')
                    st_types = np.array(st_params['types'])
                else:
                    if self._verbose:
                        print('Setting random number of excitatory and inhibitory neurons as cell_type info is missing')
                    n_exc = np.random.randint(n_neurons)
                    n_inh = n_neurons - n_exc
                    st_types = np.array(['E'] * n_exc + ['I'] * n_inh)

                e_idx = np.where(st_types == 'E')
                i_idx = np.where(st_types == 'I')
                if len(e_idx) > 0 and len(i_idx) > 0:
                    if not np.all([[e < i for e in e_idx[0]] for i in i_idx[0]]):
                        if verbose_1:
                            print('Re-arranging spike trains: Excitatory first, Inhibitory last')
                        order = np.argsort(st_types)
                        new_spiketrains = []
                        for idx in order:
                            new_spiketrains.append(spiketrains[idx])
                        spgen.spiketrains = new_spiketrains
                        spiketrains = new_spiketrains

                if verbose_1:
                    print('Templates selection seed: ', temp_seed)
                np.random.seed(temp_seed)

                if drifting:
                    drift_directions = np.array([(p[-1] - p[0]) / np.linalg.norm(p[-1] - p[0]) for p in locs])
                    drift_velocity_ums = drift_velocity / 60.
                    velocity_vector = drift_velocity_ums * preferred_dir
                    if verbose_1:
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

                if n_neurons > 100 or drifting:
                    parallel_templates = True
                else:
                    parallel_templates = False

                if verbose_1:
                    print('Selecting cells')
                idxs_cells, selected_cat = select_templates(locs, eaps, bin_cat, n_exc, n_inh, x_lim=x_lim, y_lim=y_lim,
                                                            z_lim=z_lim, min_amp=min_amp, max_amp=max_amp,
                                                            min_dist=min_dist, drifting=drifting,
                                                            drift_dir=drift_directions,
                                                            preferred_dir=preferred_dir, angle_tol=angle_tol,
                                                            n_overlap_pairs=n_overlap_pairs,
                                                            overlap_threshold=overlap_threshold,
                                                            verbose=verbose_2)

                if not np.any('U' in  selected_cat):
                    assert selected_cat.count('E') == n_exc and selected_cat.count('I') == n_inh
                    # Reorder templates according to E-I types
                    reordered_idx_cells = np.array(idxs_cells)[np.argsort(selected_cat)]
                else:
                    reordered_idx_cells = idxs_cells

                template_celltypes = celltypes[reordered_idx_cells]
                template_locs = np.array(locs)[reordered_idx_cells]
                template_rots = np.array(rots)[reordered_idx_cells]
                template_bin = np.array(bin_cat)[reordered_idx_cells]
                templates = np.array(eaps)[reordered_idx_cells]

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
                if verbose_1:
                    print('Padding template edges')
                t_pad = time.time()
                templates_pad = pad_templates(templates, pad_samples, drifting, dtype, verbose_2,
                                              tmp_file=tmp_templates_pad, parallel=parallel_templates)

                if verbose_1:
                    print('Elapsed pad time:', time.time() - t_pad)

                # resample spikes
                t_rs = time.time()
                up = fs
                down = spike_fs
                spike_duration_pad = templates_pad.shape[-1]
                if up != down:
                    n_resample = int(spike_duration_pad * (up / down))
                    templates_rs = resample_templates(templates_pad, n_resample, up, down, drifting, dtype,
                                                      verbose_2, tmp_file=tmp_templates_rs,
                                                      parallel=parallel_templates)
                    if verbose_1:
                        print('Elapsed resample time:', time.time() - t_rs)
                else:
                    templates_rs = templates_pad

                if verbose_1:
                    print('Creating time jittering')
                jitter = 1. / fs
                t_j = time.time()
                templates = jitter_templates(templates_rs, upsample, fs, n_jitters, jitter, drifting, dtype,
                                             verbose_2, tmp_file=tmp_templates_jit, parallel=parallel_templates)
                if verbose_1:
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
                        template_bin = get_binary_cat(template_celltypes, exc_categories, inh_categories)
                    else:
                        template_bin = np.array(['U'] * len(celltypes))
                else:
                    template_bin = np.array(['U'] * len(celltypes))
                voltage_peaks = self.voltage_peaks
                overlapping = np.array([])
                if not drifting:
                    velocity_vector = None
                else:
                    drift_velocity_ums = drift_velocity / 60.
                    velocity_vector = drift_velocity_ums * preferred_dir
                    if verbose_1:
                        print('Drift velocity vector: ', velocity_vector)

            if sync_rate is not None:
                if verbose_1:
                    print('Modifying synchrony of spatially overlapping spikes')
                if verbose_1:
                    print('Overlapping templates: ', overlapping)
                for over in overlapping:
                    if verbose_1:
                        print('Overlapping pair: ', over)
                    spgen.add_synchrony(over, rate=sync_rate, verbose=verbose_2, time_jitt=sync_jitt)
                    # annotate new firing rates
                    fr1 = len(spgen.spiketrains[over[0]].times) / spgen.spiketrains[over[0]].t_stop
                    fr2 = len(spgen.spiketrains[over[1]].times) / spgen.spiketrains[over[1]].t_stop
                    spgen.spiketrains[over[0]].annotate(fr=fr1)
                    spgen.spiketrains[over[1]].annotate(fr=fr2)
            self.overlapping = overlapping

            # find SNR and annotate
            if verbose_1:
                print('Computing spike train SNR')
            for t_i, temp in enumerate(templates):
                min_peak = np.min(temp)
                snr = np.abs(min_peak / float(noise_level))
                spiketrains[t_i].annotate(snr=snr)

            if verbose_1:
                print('Adding spiketrain annotations')
            for i, st in enumerate(spiketrains):
                st.annotate(bintype=template_bin[i], mtype=template_celltypes[i], soma_position=template_locs[i])

            if overlap:
                annotate_overlapping_spikes(spiketrains, overlapping_pairs=overlapping, verbose=verbose_2)

            if verbose_1:
                print('Convolution seed: ', conv_seed)
            np.random.seed(conv_seed)

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
                if verbose_1:
                    print('Template modulation')
                for i_s, st in enumerate(spiketrains):
                    if bursting and i_s in bursting_units:
                        if verbose_1:
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
                if verbose_1:
                    print('Electrode modulaton')
                for i_s, st in enumerate(spiketrains):
                    if bursting and i_s in bursting_units:
                        if verbose_1:
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

            spike_idxs = []
            for st in spiketrains:
                spike_idxs.append((st.times * fs).magnitude.astype('int'))

            # divide in chunks
            chunk_indexes = make_chunk_indexes(duration, chunk_duration, fs)
            seed_list_conv = [np.random.randint(1000) for i in np.arange(len(chunk_indexes))]

            pad_samples_conv = templates.shape[-1]
            # call the loop on chunks
            args = (spike_idxs, pad_samples_conv, modulation, drifting, drift_mode, drifting_units, templates,
                    cut_outs_samples, template_locs, velocity_vector, fast_drift_period,
                    fast_drift_min_jump, fast_drift_max_jump, t_start_drift, fs, verbose_2,
                    amp_mod, bursting_units, shape_mod, shape_stretch,
                    True, voltage_peaks, dtype, seed_list_conv,)
            assignment_dict = {
                'recordings': recordings,
                'spike_traces': spike_traces}
            output_list = run_several_chunks(chunk_convolution, chunk_indexes, fs, timestamps, args,
                                             self.n_jobs, self.tmp_mode, self.tmp_folder, assignment_dict)

            # if drift then propagate annoations to spikestrains
            for st in np.arange(n_neurons):
                if drifting and st in drifting_units:
                    spiketrains[st].annotate(drifting=True)
                    template_idxs = np.array([], dtype='int')
                    for out in output_list:
                        template_idxs = np.concatenate((template_idxs, out['template_idxs'][st]))
                    assert len(template_idxs) == len(spiketrains[st])
                    spiketrains[st].annotate(template_idxs=template_idxs)

        #################
        # Step 2: noise #
        #################
        if verbose_1:
            print('Adding noise')
            print('Noise seed: ', noise_seed)

        np.random.seed(noise_seed)

        if noise_level == 0:
            if verbose_1:
                print('Noise level is set to 0')
        else:
            # divide in chunks
            chunk_indexes = make_chunk_indexes(duration, chunk_duration, fs)
            seed_list_noise = [np.random.randint(1000) for i in np.arange(len(chunk_indexes))]

            if self.tmp_mode == 'memmap':
                tmp_path_noise = self.tmp_folder / (tmp_prefix + "mearec_tmp_noise_file.raw")
                additive_noise = np.memmap(tmp_path_noise, shape=(n_samples, n_elec), dtype=dtype, mode='w+')
                # additive_noise = additive_noise.transpose()
                self._to_remove_on_delete.append(tmp_path_noise)
            else:
                additive_noise = np.zeros((n_samples, n_elec), dtype=dtype)

            if noise_mode == 'uncorrelated':
                func = chunk_uncorrelated_noise
                args = (n_elec, noise_level, noise_color, color_peak, color_q, color_noise_floor,
                        fs.rescale('Hz').magnitude, dtype, seed_list_noise,)
                assignment_dict = {'additive_noise': additive_noise}

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
                        fs.rescale('Hz').magnitude, dtype, seed_list_noise,)
                assignment_dict = {'additive_noise': additive_noise}

                run_several_chunks(func, chunk_indexes, fs, timestamps, args,
                                   self.n_jobs, self.tmp_mode, self.tmp_folder, assignment_dict)

            elif noise_mode == 'far-neurons':
                if self.tmp_mode == 'memmap':
                    # file names for templates
                    tmp_templates_noise_pad = self.tmp_folder / (tmp_prefix + "templates_noise_pad.raw")
                    tmp_templates_noise_rs = self.tmp_folder / (tmp_prefix + "templates_noise_resample.raw")
                    self._to_remove_on_delete.extend([tmp_templates_noise_pad, tmp_templates_noise_rs])
                else:
                    tmp_templates_noise_pad = None
                    tmp_templates_noise_rs = None
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
                if verbose_1:
                    print('Padding noisy template edges')
                t_pad = time.time()
                templates_noise_pad = pad_templates(templates_noise, pad_samples, drifting, dtype, verbose_2,
                                                    tmp_file=tmp_templates_noise_pad, parallel=True)
                if verbose_1:
                    print('Elapsed pad time:', time.time() - t_pad)

                # resample templates
                t_rs = time.time()
                up = fs
                down = spike_fs
                spike_duration_pad = templates_noise_pad.shape[-1]
                if up != down:
                    n_resample = int(spike_duration_pad * (up / down))
                    templates_noise = resample_templates(templates_noise_pad, n_resample, up, down,
                                                         drifting, dtype, verbose_2,
                                                         tmp_file=tmp_templates_noise_rs)
                    if verbose_1:
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
                if verbose_1:
                    print('Generating noisy spike trains')
                noisy_spiketrains_params = params['spiketrains']
                noisy_spiketrains_params['n_exc'] = int(far_neurons_n * far_neurons_exc_inh_ratio)
                noisy_spiketrains_params['n_inh'] = far_neurons_n - noisy_spiketrains_params['n_exc']
                noisy_spiketrains_params['seed'] = noise_seed
                spgen_noise = SpikeTrainGenerator(params=noisy_spiketrains_params)
                spgen_noise.generate_spikes()
                spiketrains_noise = spgen_noise.spiketrains

                spike_idxs_noise = []
                for st in spiketrains_noise:
                    spike_idxs_noise.append((st.times * fs).magnitude.astype('int'))

                if verbose_1:
                    print('Convolving noisy spike trains')
                templates_noise = templates_noise.reshape((templates_noise.shape[0], 1, templates_noise.shape[1],
                                                           templates_noise.shape[2]))

                # call the loop on chunks
                args = (spike_idxs_noise, 0, 'none', False, None, None, templates_noise,
                        cut_outs_samples, template_noise_locs, None, None, None, None, None, None,
                        verbose_2, None, None, False, None, False, None, dtype, seed_list_noise,)
                assignment_dict = {'recordings': additive_noise}
                run_several_chunks(chunk_convolution, chunk_indexes, fs, timestamps, args,
                                   self.n_jobs, self.tmp_mode, self.tmp_folder, assignment_dict)

                # removing mean
                for i, m in enumerate(np.mean(additive_noise, axis=0)):
                    additive_noise[:, i] -= m
                # adding noise floor
                for i, s in enumerate(np.std(additive_noise, axis=0)):
                    additive_noise[:, i] += far_neurons_noise_floor * s * \
                                                np.random.randn(additive_noise.shape[0])
                # scaling noise
                noise_scale = noise_level / np.std(additive_noise, axis=0)
                if verbose_1:
                    print('Scaling to reach desired level')
                for i, n in enumerate(noise_scale):
                    additive_noise[:, i] *= n

            # Add it to recordings
            recordings += additive_noise

        ##################
        # Step 3: filter #
        ##################
        if filter:
            if verbose_1:
                print('Filtering')
                if cutoff.size == 1:
                    print('High-pass cutoff', cutoff)
                elif cutoff.size == 2:
                    print('Band-pass cutoff', cutoff)

            chunk_indexes = make_chunk_indexes(duration, chunk_duration, fs)

            # compute pad samples as 3 times the low-cutoff period
            if cutoff.size == 1:
                pad_samples_filt = 3 * int((1. / cutoff * fs).magnitude)
            elif cutoff.size == 2:
                pad_samples_filt = 3 * int((1. / cutoff[0] * fs).magnitude)

            # call the loop on chunks
            args = (recordings, pad_samples_filt, cutoff, order, fs, dtype,)
            assignment_dict = {
                'filtered_chunk': recordings,
            }
            # Done in loop (as before) : this cannot be done in parralel because of bug transpose in joblib!!!!!!!!!!!!!
            run_several_chunks(chunk_apply_filter, chunk_indexes, fs, timestamps, args,
                               self.n_jobs, self.tmp_mode, self.tmp_folder, assignment_dict)

        # assign class variables
        params['templates']['overlapping'] = np.array(overlapping)
        self.recordings = recordings.transpose()
        self.timestamps = timestamps
        self.channel_positions = mea_pos
        self.templates = np.squeeze(templates)
        self.template_locations = template_locs
        self.template_rotations = template_rots
        self.template_celltypes = template_celltypes
        self.spiketrains = spiketrains
        self.voltage_peaks = voltage_peaks
        self.spike_traces = spike_traces.transpose()
        self.info = params

        #############################
        # Step 4: extract waveforms #
        #############################
        if not only_noise:
            if extract_waveforms:
                if verbose_1:
                    print('Extracting spike waveforms')
                self.extract_waveforms()

    def annotate_overlapping_spikes(self, parallel=True):
        """
        Annnotate spike trains with overlapping information.

        parallel : bool
            If True, spike trains are annotated in parallel
        """
        if self.info['templates']['overlapping'] is None or len(self.info['templates']['overlapping']) == 0:
            if self._verbose_1:
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

    # create task list
    arg_tasks = []
    karg_tasks = []
    for ch, (i_start, i_stop) in enumerate(chunk_indexes):

        chunk_start = (i_start / fs).rescale('s')

        arg_task = (ch, i_start, i_stop, chunk_start,) + args
        arg_tasks.append(arg_task)

        karg_task = dict(assignment_dict=assignment_dict, tmp_mode=tmp_mode)
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
                    full_arr[i_start:i_stop] += out_chunk
            elif tmp_mode == 'memmap':
                pass
                # Nothing to do here because done inside the func with FuncThenAddChunk

    else:
        # parallel
        output_list = Parallel(n_jobs=n_jobs)(
            delayed(func)(*arg_task, **karg_task) for arg_task, karg_task in zip(arg_tasks, karg_tasks))

        if tmp_mode == 'memmap':
            pass
            # Nothing to do here because done inside the func
        else:
            # This case is very unefficient because it double the memory usage!!!!!!!
            for ch, (i_start, i_stop) in enumerate(chunk_indexes):
                for key, full_arr in assignment_dict.items():
                    full_arr[i_start:i_stop] += output_list[ch][key]

    return output_list
