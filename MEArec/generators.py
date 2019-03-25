from __future__ import print_function, division

import numpy as np
import neo
import elephant.spike_train_generation as stg
import elephant.conversion as conv
import elephant.statistics as stat
import scipy.signal as ss
import time
from copy import copy, deepcopy
from MEArec.tools import *
import MEAutility as mu
import threading
import shutil
import yaml
from pprint import pprint
import quantities as pq
from quantities import Quantity
from distutils.version import StrictVersion

if StrictVersion(yaml.__version__) >= StrictVersion('5.0.0'):
    use_loader = True
else:
    use_loader = False


class simulationThread(threading.Thread):
    def __init__(self, threadID, name, simulate_script, numb, tot, cell_model,
                 model_folder, intraonly, params, verbose):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.sim_script = simulate_script
        self.numb = numb
        self.tot = tot
        self.cell_model = cell_model
        self.model_folder = model_folder
        self.intra = intraonly
        self.params = params
        self.verbose = verbose

    def run(self):
        print("Starting " + self.name)
        print('\n\n', self.cell_model, self.numb + 1, '/', self.tot, '\n\n')
        os.system('python %s %s %s %s %s' \
                  % (self.sim_script, join(self.model_folder, self.cell_model), self.intra, self.params, self.verbose))
        print("Exiting " + self.name)


class TemplateGenerator:
    '''
    Class for generation of templates called by the gen_templates function.
    The list of parameters is in default_params/templates_params.yaml.
    '''
    def __init__(self, cell_models_folder=None, templates_folder=None, temp_dict=None, info=None,
                 params=None, intraonly=False, parallel=True, delete_tmp=True, verbose=False):
        self.verbose = verbose
        if temp_dict is not None and info is not None:
            if 'templates' in temp_dict.keys():
                self.templates = temp_dict['templates']
            if 'locations' in temp_dict.keys():
                self.locations = temp_dict['locations']
            if 'rotations' in temp_dict.keys():
                self.rotations = temp_dict['rotations']
            if 'celltypes' in temp_dict.keys():
                self.celltypes = temp_dict['celltypes']
            self.info = info
        else:
            if cell_models_folder is None:
                raise AttributeError("Specify cell folder!")
            if params is None:
                if self.verbose:
                    print("Using default parameters")
                self.params = {}
            else:
                self.params = deepcopy(params)
            self.cell_model_folder = cell_models_folder
            self.templates_folder = templates_folder
            self.simulation_params = {'intraonly': intraonly, 'parallel': parallel, 'delete_tmp': delete_tmp}

    def generate_templates(self):
        '''
        Generate templates.
        '''
        cell_models_folder = self.cell_model_folder
        templates_folder = self.templates_folder
        intraonly = self.simulation_params['intraonly']
        parallel = self.simulation_params['parallel']
        delete_tmp = self.simulation_params['delete_tmp']

        if os.path.isdir(cell_models_folder):
            cell_models = [f for f in os.listdir(join(cell_models_folder)) if 'mods' not in f]
            if len(cell_models) == 0:
                raise AttributeError(cell_models_folder, ' contains no cell models!')
        else:
            raise NotADirectoryError('Cell models folder: does not exist!')

        this_dir, this_filename = os.path.split(__file__)
        simulate_script = join(this_dir, 'simulate_cells.py')
        self.params['cell_models_folder'] = cell_models_folder
        self.params['templates_folder'] = templates_folder

        # Compile NEURON models (nrnivmodl)
        if not os.path.isdir(join(cell_models_folder, 'mods')):
            if self.verbose:
                print('Compiling NEURON models')
            os.system('python %s compile %s' % (simulate_script, cell_models_folder))

        if 'sim_time' not in self.params.keys():
            self.params['sim_time'] = 1
        if 'target_spikes' not in self.params.keys():
            self.params['target_spikes'] = [3, 50]
        if 'cut_out' not in self.params.keys():
            self.params['cut_out'] = [2, 5]
        if 'dt' not in self.params.keys():
            self.params['dt'] = 2 ** -5
        if 'delay' not in self.params.keys():
            self.params['delay'] = 10
        if 'weights' not in self.params.keys():
            self.params['weights'] = [0.25, 1.75]

        if 'rot' not in self.params.keys():
            self.params['rot'] = 'physrot'
        if 'probe' not in self.params.keys():
            available_mea = mu.return_mea_list()
            probe = available_mea[np.random.randint(len(available_mea))]
            if self.verbose:
                print("Probe randomly set to: %s" % probe)
            self.params['probe'] = probe
        if 'ncontacts' not in self.params.keys():
            self.params['ncontacts'] = 1
        if 'overhang' not in self.params.keys():
            self.params['overhang'] = 1
        if 'xlim' not in self.params.keys():
            self.params['xlim'] = [10, 80]
        if 'ylim' not in self.params.keys():
            self.params['ylim'] = None
        if 'zlim' not in self.params.keys():
            self.params['zlim'] = None
        if 'offset' not in self.params.keys():
            self.params['offset'] = 0
        if 'det_thresh' not in self.params.keys():
            self.params['det_thresh'] = 30
        if 'n' not in self.params.keys():
            self.params['n'] = 50
        if 'seed' not in self.params.keys():
            self.params['seed'] = np.random.randint(1, 10000)
        elif self.params['seed'] is None:
            self.params['seed'] = np.random.randint(1, 10000)
        if templates_folder is None:
            self.params['templates_folder'] = os.getcwd()
            templates_folder = self.params['templates_folder']
        else:
            self.params['templates_folder'] = templates_folder
        if 'drifting' not in self.params.keys():
            self.params['drifting'] = False

        rot = self.params['rot']
        n = self.params['n']
        probe = self.params['probe']

        tmp_params_path = 'tmp_params_path'
        with open(tmp_params_path, 'w') as f:
            yaml.dump(self.params, f)

        # Simulate neurons and EAP for different cell models sparately
        if parallel:
            start_time = time.time()
            if self.verbose:
                print('Parallel')
            tot = len(cell_models)
            threads = []
            for numb, cell_model in enumerate(cell_models):
                threads.append(simulationThread(numb, "Thread-" + str(numb), simulate_script,
                                                numb, tot, cell_model, cell_models_folder, intraonly,
                                                tmp_params_path, self.verbose))
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            print('\n\n\nSimulation time: ', time.time() - start_time, '\n\n\n')
        else:
            start_time = time.time()
            for numb, cell_model in enumerate(cell_models):
                if self.verbose:
                    print('\n\n', cell_model, numb + 1, '/', len(cell_models), '\n\n')
                os.system('python %s %s %s %s %s' \
                          % (simulate_script, join(cell_models_folder, cell_model), intraonly, tmp_params_path,
                             self.verbose))
            print('\n\n\nSimulation time: ', time.time() - start_time, '\n\n\n')

        tmp_folder = join(templates_folder, rot, 'tmp_%d_%s' % (n, probe))
        templates, locations, rotations, celltypes = load_tmp_eap(tmp_folder)
        if delete_tmp:
            shutil.rmtree(tmp_folder)
            os.remove(tmp_params_path)

        self.info = {}

        self.templates = templates
        self.locations = locations
        self.rotations = rotations
        self.celltypes = celltypes
        self.info['params'] = self.params
        self.info['electrodes'] = mu.return_mea_info(probe)


class SpikeTrainGenerator:
    '''
    Class for generation of spike trains called by the gen_recordings function.
    The list of parameters is in default_params/recordings_params.yaml (spiketrains field).
    '''
    def __init__(self, params=None, spiketrains=None, verbose=False):
        self.verbose = verbose
        if params is None:
            if self.verbose:
                print("Using default parameters")
            self.params = {}
        if spiketrains is None:
            self.params = deepcopy(params)
            if self.verbose:
                print('Spiketrains seed: ', self.params['seed'])
            np.random.seed(self.params['seed'])

            if 't_start' not in self.params.keys():
                params['t_start'] = 0
            self.params['t_start'] = params['t_start'] * pq.s
            if 'duration' not in self.params.keys():
                params['duration'] = 10
            self.params['t_stop'] = self.params['t_start'] + params['duration'] * pq.s
            if 'min_rate' not in self.params.keys():
                params['min_rate'] = 0.1
            self.params['min_rate'] = params['min_rate'] * pq.Hz
            if 'ref_per' not in self.params.keys():
                params['ref_per'] = 2
            self.params['ref_per'] = params['ref_per'] * pq.ms
            if 'process' not in self.params.keys():
                params['process'] = 'poisson'
            self.params['process'] = params['process']
            if 'gamma_shape' not in self.params.keys() and params['process'] == 'gamma':
                params['gamma_shape'] = 2
                self.params['gamma_shape'] = params['gamma_shape']

            if 'rates' in self.params.keys():  # all firing rates are provided
                self.params['rates'] = self.params['rates'] * pq.Hz
                self.n_neurons = len(self.params['rates'])
            else:
                rates = []
                types = []
                if 'f_exc' not in self.params.keys():
                    params['f_exc'] = 5
                self.params['f_exc'] = params['f_exc'] * pq.Hz
                if 'f_inh' not in self.params.keys():
                    params['f_inh'] = 15
                self.params['f_inh'] = params['f_inh'] * pq.Hz
                if 'st_exc' not in self.params.keys():
                    params['st_exc'] = 1
                self.params['st_exc'] = params['st_exc'] * pq.Hz
                if 'st_inh' not in self.params.keys():
                    params['st_inh'] = 3
                self.params['st_inh'] = params['st_inh'] * pq.Hz
                if 'n_exc' not in self.params.keys():
                    params['n_exc'] = 15
                self.params['n_exc'] = params['n_exc']
                if 'n_inh' not in self.params.keys():
                    params['n_inh'] = 5
                self.params['n_inh'] = params['n_inh']

                for exc in np.arange(self.params['n_exc']):
                    rate = self.params['st_exc'] * np.random.randn() + self.params['f_exc']
                    if rate < self.params['min_rate']:
                        rate = self.params['min_rate']
                    rates.append(rate)
                    types.append('e')
                for inh in np.arange(self.params['n_inh']):
                    rate = self.params['st_inh'] * np.random.randn() + self.params['f_inh']
                    if rate < self.params['min_rate']:
                        rate = self.params['min_rate']
                    rates.append(rate)
                    types.append('i')
                self.params['rates'] = rates
                self.params['types'] = types
                self.n_neurons = len(self.params['rates'])

            self.changing = False
            self.intermittent = False

            self.info = params
            self.spiketrains = False
        else:
            self.all_spiketrains = spiketrains
            self.spiketrains = True

    def set_spiketrain(self, idx, spiketrain):
        '''
        Sets spike train idx to new spiketrain.

        Parameters
        ----------
        idx : int
            Index of spike train to set
        spiketrain : neo.SpikeTrain
            New spike train

        '''
        self.all_spiketrains[idx] = spiketrain

    def generate_spikes(self):
        '''
        Generate spike trains based on default_params of the SpikeTrainGenerator class.
        self.all_spiketrains contains the newly generated spike trains
        '''

        if not self.spiketrains:
            self.all_spiketrains = []
            idx = 0
            for n in np.arange(self.n_neurons):
                if not self.changing and not self.intermittent:
                    rate = self.params['rates'][n]
                    if self.params['process'] == 'poisson':
                        st = stg.homogeneous_poisson_process(rate,
                                                             self.params['t_start'], self.params['t_stop'])
                    elif self.params['process'] == 'gamma':
                        st = stg.homogeneous_gamma_process(self.params['gamma_shape'], rate,
                                                           self.params['t_start'], self.params['t_stop'])
                else:
                    raise NotImplementedError('Changing and intermittent spiketrains are not impleented yet')
                self.all_spiketrains.append(st)
                self.all_spiketrains[-1].annotate(fr=rate)
                if 'n_exc' in self.params.keys() and 'n_inh' in self.params.keys():
                    if idx < self.params['n_exc']:
                        self.all_spiketrains[-1].annotate(type='E')
                    else:
                        self.all_spiketrains[-1].annotate(type='I')
                idx += 1

            # check consistency and remove spikes below refractory period
            for idx, st in enumerate(self.all_spiketrains):
                isi = stat.isi(st)
                idx_remove = np.where(isi < self.params['ref_per'])[0]
                spikes_to_remove = len(idx_remove)
                unit = st.times.units

                while spikes_to_remove > 0:
                    new_times = np.delete(st.times, idx_remove[0]) * unit
                    st = neo.SpikeTrain(new_times, t_start=self.params['t_start'], t_stop=self.params['t_stop'])
                    isi = stat.isi(st)
                    idx_remove = np.where(isi < self.params['ref_per'])[0]
                    spikes_to_remove = len(idx_remove)

                st.annotations = self.all_spiketrains[idx].annotations
                self.set_spiketrain(idx, st)
        else:
            print("SpikeTrainGenerator initialized with existing spike trains!")

    def add_synchrony(self, idxs, rate=0.05, time_jitt=1*pq.ms, verbose=False):
        '''
        Adds synchronous spikes between pairs of spike trains at a certain rate.

        Parameters
        ----------
        idxs : list or array
            Spike train indexes to add synchrony to
        rate : float
            Rate of added synchrony spike to spike train idxs[1] for each spike of idxs[0]
        time_jitt : quantity
            Maximum time jittering between added spikes
        verbose : bool
            If True output is verbose

        Returns
        -------
        sync_rate : float
            New synchrony rate
        fr1 : quantity
            Firing rate spike train 1
        fr2 : quantity
            Firing rate spike train 2

        '''
        idx1 = idxs[0]
        idx2 = idxs[1]
        st1 = self.all_spiketrains[idx1]
        st2 = self.all_spiketrains[idx2]
        times1 = st1.times
        times2 = st2.times
        t_start = st2.t_start
        t_stop = st2.t_stop
        unit = times2.units
        added_spikes_t1 = 0
        added_spikes_t2 = 0

        all_times_shuffle = np.concatenate((times1, times2))
        all_times_shuffle = all_times_shuffle[np.random.permutation(len(all_times_shuffle))] * unit

        sync_rate = compute_sync_rate(st1, st2, time_jitt)
        for t in all_times_shuffle:
            if sync_rate <= rate:
                # check time difference
                if t in times1:
                    t_diff = np.abs(t.rescale(pq.ms).magnitude - times2.rescale(pq.ms).magnitude)
                    if np.all(t_diff > self.params['ref_per']):
                        t1_jitt = time_jitt.rescale(unit).magnitude * np.random.rand(1) + t.rescale(unit).magnitude - \
                                  (time_jitt.rescale(unit) / 2).magnitude
                        times2 = np.sort(np.concatenate((np.array(times2), np.array(t1_jitt))))
                        times2 = times2 * unit
                        st2 = neo.SpikeTrain(times2, t_start=t_start, t_stop=t_stop)
                        added_spikes_t1 += 1
                elif t in times2:
                    t_diff = np.abs(t.rescale(pq.ms).magnitude - times1.rescale(pq.ms).magnitude)
                    if np.all(t_diff > self.params['ref_per']):
                        t2_jitt = time_jitt.rescale(unit).magnitude * np.random.rand(1) + t.rescale(unit).magnitude - \
                                  (time_jitt.rescale(unit) / 2).magnitude
                        times1 = np.sort(np.concatenate((np.array(times1), np.array(t2_jitt))))
                        times1 = times1 * unit
                        st1 = neo.SpikeTrain(times1, t_start=t_start, t_stop=t_stop)
                        added_spikes_t2 += 1
                sync_rate = compute_sync_rate(st1, st2, time_jitt)
            else:
                break
        st1.annotations = self.all_spiketrains[idx1].annotations
        st2.annotations = self.all_spiketrains[idx2].annotations
        self.set_spiketrain(idx1, st1)
        self.set_spiketrain(idx2, st2)
        if verbose:
            print("Added", added_spikes_t1, "spikes to spike train", idxs[0],
                  "and", added_spikes_t2, "spikes to spike train", idxs[1])

        fr1 = len(st1.times) / st1.t_stop
        fr2 = len(st2.times) / st2.t_stop

        return sync_rate, fr1, fr2


class RecordingGenerator:
    '''
    Class for generation of recordings called by the gen_recordings function.
    The list of parameters is in default_params/recordings_params.yaml.
    '''
    def __init__(self, spgen=None, tempgen=None, params=None, rec_dict=None, info=None, verbose=True):
        self.verbose = verbose
        if rec_dict is not None and info is not None:
            if 'recordings' in rec_dict.keys():
                self.recordings = rec_dict['recordings']
            else:
                self.recordings = np.array([])
            if 'spiketrains' in rec_dict.keys():
                self.spiketrains = rec_dict['spiketrains']
            else:
                self.spiketrains = np.array([])
            if 'templates' in rec_dict.keys():
                self.templates = rec_dict['templates']
            else:
                self.templates = np.array([])
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
            self.info = info
        else:
            if spgen is None or tempgen is None:
                raise AttributeError("Specify SpikeGenerator and TemplateGenerator objects!")
            if params is None:
                if self.verbose:
                    print("Using default parameters")
                params = {'spiketrains': {},
                          'celltypes': {},
                          'templates': {},
                          'recordings': {}}
            self.params = deepcopy(params)
            self.spgen = spgen
            self.tempgen = tempgen

    def generate_recordings(self):
        '''
        Generate recordings.
        '''
        params = deepcopy(self.params)
        temp_params = self.params['templates']
        rec_params = self.params['recordings']
        st_params = self.params['spiketrains']
        celltype_params = self.params['cell_types']

        if 'rates' in st_params.keys():
            assert st_params['types'] is not None, "If 'rates' are provided as spiketrains parameters, " \
                                                   "corresponding 'types' ('E'-'I') must be provided"
            n_exc = st_params['types'].count('E')
            n_inh = st_params['types'].count('I')

        tempgen = self.tempgen
        spgen = self.spgen
        eaps = tempgen.templates
        locs = tempgen.locations
        rots = tempgen.rotations
        celltypes = tempgen.celltypes
        temp_info = tempgen.info

        spiketrains = spgen.all_spiketrains

        n_neurons = len(spiketrains)
        cut_outs = temp_info['params']['cut_out']
        if len(spiketrains) > 0:
            duration = spiketrains[0].t_stop - spiketrains[0].t_start
            only_noise = False
        else:
            if self.verbose:
                print('No spike trains provided: only simulating noise')
            only_noise = True
            duration = st_params['duration'] * pq.s

        if 'fs' not in rec_params.keys():
            params['recordings']['fs'] = 1. / temp_info['params']['dt']
            fs = params['recordings']['fs'] * pq.kHz
        elif params['recordings']['fs'] is None:
            params['recordings']['fs'] = 1. / temp_info['params']['dt']
            fs = params['recordings']['fs'] * pq.kHz
        else:
            fs = params['recordings']['fs'] * pq.kHz

        if 'noise_mode' not in rec_params.keys():
            params['recordings']['noise_mode'] = 'uncorrelated'
        noise_mode = params['recordings']['noise_mode']

        if 'noise_color' not in rec_params.keys():
            params['recordings']['noise_color'] = False
        noise_color = params['recordings']['noise_color']

        if 'sync_rate' not in rec_params.keys():
            params['recordings']['sync_rate'] = 0
        sync_rate = params['recordings']['sync_rate']

        if 'sync_jitt' not in rec_params.keys():
            params['recordings']['sync_jitt'] = 1
        sync_jitt = params['recordings']['sync_jitt'] * pq.ms

        if noise_mode == 'distance-correlated':
            if 'half_distance' not in rec_params.keys():
                params['recordings']['half_dist'] = 30
                half_dist = 30

        if noise_color is True:
            if 'color_peak' not in rec_params.keys():
                params['recordings']['color_peak'] = 500
            color_peak = params['recordings']['color_peak']
            if 'color_q' not in rec_params.keys():
                params['recordings']['color_q'] = 1
            color_q = params['recordings']['color_q']
            if 'random_noise_floor' not in rec_params.keys():
                params['recordings']['random_noise_floor'] = 0.5
            random_noise_floor = params['recordings']['random_noise_floor']

        if 'noise_level' not in rec_params.keys():
            params['recordings']['noise_level'] = 10
        noise_level = params['recordings']['noise_level']
        if self.verbose:
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
        modulation = params['recordings']['modulation']

        if 'shape_mod' not in rec_params.keys():
            params['recordings']['shape_mod'] = False
        shape_mod = params['recordings']['shape_mod']

        if shape_mod:
            if 'bursting_fc' not in rec_params.keys():
                params['recordings']['bursting_fc'] = [500., 12000.]
            bursting_fc = params['recordings']['bursting_fc']
            if self.verbose:
                print('Bursting with modulation frequencies: ', bursting_fc, ' Hz')
        else:
            bursting_fc = None

        if 'isi' in modulation:
            if 'exp_decay' not in rec_params.keys():
                params['recordings']['exp_decay'] = 0.2
            exp_decay = params['recordings']['exp_decay']
            if 'n_burst_spikes' not in rec_params.keys():
                params['recordings']['n_burst_spikes'] = 10
            n_burst_spikes = params['recordings']['n_burst_spikes']
            if 'max_burst_duration' not in rec_params.keys():
                params['recordings']['max_burst_duration'] = 100
            max_burst_duration = 100 * pq.ms

        if 'chunk_noise_duration' not in rec_params.keys():
            params['recordings']['chunk_noise_duration'] = 0
        chunk_noise_duration = params['recordings']['chunk_noise_duration'] * pq.s

        if 'chunk_filter_duration' not in rec_params.keys():
            params['recordings']['chunk_filter_duration'] = 0
        chunk_filter_duration = params['recordings']['chunk_filter_duration'] * pq.s

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

        if 'drifting' not in rec_params.keys():
            params['recordings']['drifting'] = False
        drifting = params['recordings']['drifting']

        if drifting:
            assert temp_info['params']['drifting']
            preferred_dir = np.array(rec_params['preferred_dir'])
            preferred_dir = preferred_dir / np.linalg.norm(preferred_dir)
            angle_tol = rec_params['angle_tol']
            drift_velocity = rec_params['drift_velocity']
            t_start_drift = rec_params['t_start_drift'] * pq.s
            if rec_params['n_drifting'] is None:
                n_drifting = n_neurons
            else:
                n_drifting = rec_params['n_drifting']
        else:
            # if drifting templates, but not recordings, consider initial template
            if temp_info['params']['drifting']:
                eaps = eaps[:, 0]
            preferred_dir = None
            angle_tol = None
            drift_velocity = None
            t_start_drift = None
            n_drifting = None

        if 'xlim' not in temp_params.keys():
            params['templates']['xlim'] = None
        x_lim = params['templates']['xlim']

        if 'ylim' not in temp_params.keys():
            params['templates']['ylim'] = None
        y_lim = params['templates']['ylim']

        if 'zlim' not in temp_params.keys():
            params['templates']['zlim'] = None
        z_lim = params['templates']['zlim']

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

        if celltype_params is not None:
            if 'excitatory' in celltype_params.keys() and 'inhibitory' in celltype_params.keys():
                exc_categories = celltype_params['excitatory']
                inh_categories = celltype_params['inhibitory']
                bin_cat = get_binary_cat(celltypes, exc_categories, inh_categories)
            else:
                bin_cat = np.array(['U'] * len(celltypes))
        else:
            bin_cat = np.array(['U'] * len(celltypes))

        # load MEA info
        electrode_name = temp_info['electrodes']['electrode_name']
        elinfo = mu.return_mea_info(electrode_name)
        offset = temp_info['params']['offset']
        elinfo['offset'] = offset
        mea = mu.return_mea(info=elinfo)
        mea_pos = mea.positions
        n_elec = mea_pos.shape[0]
        n_samples = int(duration.rescale('s').magnitude * fs.rescale('Hz').magnitude)

        params['recordings'].update({'duration': float(duration.magnitude),
                                     'fs': float(fs.rescale('Hz').magnitude),
                                     'n_neurons': n_neurons})
        params['electrodes'] = temp_info['electrodes']

        spike_duration = np.sum(temp_info['params']['cut_out']) * pq.ms
        spike_fs = 1. / temp_info['params']['dt'] * pq.kHz

        if self.verbose:
            print('Selecting cells')

        recordings = np.zeros((n_elec, n_samples))
        timestamps = np.arange(recordings.shape[1]) / fs

        if not only_noise:
            if 'type' in spiketrains[0].annotations.keys():
                n_exc = [st.annotations['type'] for st in spiketrains].count('E')
                n_inh = n_neurons - n_exc
            if self.verbose:
                print('Templates selection seed: ', temp_seed)
            np.random.seed(temp_seed)

            if drifting:
                drift_directions = np.array([(p[-1] - p[0]) / np.linalg.norm(p[-1] - p[0]) for p in locs])
                drift_velocity_ums = drift_velocity / 60.
                velocity_vector = drift_velocity_ums * preferred_dir
                if self.verbose:
                    print('Drift velocity vector: ', velocity_vector)
                n_elec = eaps.shape[2]
            else:
                drift_directions = None
                preferred_dir = None
                velocity_vector = None
                n_elec = eaps.shape[1]

            idxs_cells, selected_cat = select_templates(locs, eaps, bin_cat, n_exc, n_inh, x_lim=x_lim, y_lim=y_lim,
                                                        z_lim=z_lim, min_amp=min_amp, max_amp=max_amp,
                                                        min_dist=min_dist, drifting=drifting,
                                                        drift_dir=drift_directions,
                                                        preferred_dir=preferred_dir, angle_tol=angle_tol,
                                                        verbose=self.verbose)

            idxs_cells = np.array(idxs_cells)[np.argsort(selected_cat)]
            template_celltypes = celltypes[idxs_cells]
            template_locs = locs[idxs_cells]
            templates_bin = bin_cat[idxs_cells]
            templates = eaps[idxs_cells]

            # peak images
            voltage_peaks = []
            for tem in templates:
                dt = 1. / fs.magnitude
                if not drifting:
                    feat = get_templates_features(tem, ['na'], dt=dt)
                else:
                    feat = get_templates_features(tem[0], ['na'], dt=dt)
                voltage_peaks.append(-np.squeeze(feat['na']))
            voltage_peaks = np.array(voltage_peaks)

            # resample spikes
            up = fs
            down = spike_fs
            sampling_ratio = float(up / down)
            pad_samples = [int((pp * fs).magnitude) for pp in pad_len]
            n_resample = int((fs * spike_duration).magnitude)
            if not drifting:
                if templates.shape[2] != n_resample:
                    templates_pol = np.zeros((templates.shape[0], templates.shape[1], n_resample))
                    if self.verbose:
                        print('Resampling spikes')
                    for t, tem in enumerate(templates):
                        tem_pad = np.pad(tem, [(0, 0), pad_samples], 'edge')
                        tem_poly = ss.resample_poly(tem_pad, up, down, axis=1)
                        templates_pol[t, :] = tem_poly[:,
                                              int(sampling_ratio * pad_samples[0]):int(sampling_ratio * pad_samples[0])
                                                                                   + n_resample]
                else:
                    templates_pol = templates
            else:
                if templates.shape[3] != n_resample:
                    templates_pol = np.zeros((templates.shape[0], templates.shape[1], templates.shape[2], n_resample))
                    if self.verbose:
                        print('Resampling spikes')
                    for t, tem in enumerate(templates):
                        for ts, tem_single in enumerate(tem):
                            tem_pad = np.pad(tem_single, [(0, 0), pad_samples], 'edge')
                            tem_poly = ss.resample_poly(tem_pad, up, down, axis=1)
                            templates_pol[t, ts, :] = tem_poly[:,int(sampling_ratio *
                                                                     pad_samples[0]):int(sampling_ratio *
                                                                                         pad_samples[0]) + n_resample]
                else:
                    templates_pol = templates

            templates_pad = []
            if self.verbose:
                print('Padding template edges')
            for t, tem in enumerate(templates_pol):
                if not drifting:
                    tem = cubic_padding(tem, pad_len, fs)
                    templates_pad.append(tem)
                else:
                    if self.verbose:
                        print('Padding edges: neuron ', t + 1, ' of ', len(templates_pol))
                    templates_pad_p = []
                    for tem_p in tem:
                        tem_p = cubic_padding(tem_p, pad_len, fs)
                        templates_pad_p.append(tem_p)
                    templates_pad.append(templates_pad_p)
            templates_pad = np.array(templates_pad)

            if self.verbose:
                print('Creating time jittering')
            jitter = 1. / fs
            templates_jitter = []
            if not drifting:
                for temp in templates_pad:
                    temp_up = ss.resample_poly(temp, upsample, 1, axis=1)
                    nsamples_up = temp_up.shape[1]
                    temp_jitt = []
                    for n in np.arange(n_jitters):
                        # align waveform
                        shift = int((jitter * (np.random.random() - 0.5) * upsample * fs).magnitude)
                        if shift > 0:
                            t_jitt = np.pad(temp_up, [(0, 0), (np.abs(shift), 0)], 'constant')[:, :nsamples_up]
                        elif shift < 0:
                            t_jitt = np.pad(temp_up, [(0, 0), (0, np.abs(shift))], 'constant')[:, -nsamples_up:]
                        else:
                            t_jitt = temp_up
                        temp_down = ss.decimate(t_jitt, upsample, axis=1)
                        temp_jitt.append(temp_down)
                    templates_jitter.append(temp_jitt)
            else:
                for t, temp in enumerate(templates_pad):
                    if self.verbose:
                        print('Jittering: neuron ', t + 1, ' of ', len(templates_pol))
                    templates_jitter_p = []
                    for tem_p in temp:
                        temp_up = ss.resample_poly(tem_p, upsample, 1, axis=1)
                        nsamples_up = temp_up.shape[1]
                        temp_jitt = []
                        for n in np.arange(n_jitters):
                            # align waveform
                            shift = int((jitter * np.random.randn() * upsample * fs).magnitude)
                            if shift > 0:
                                t_jitt = np.pad(temp_up, [(0, 0), (np.abs(shift), 0)], 'constant')[:, :nsamples_up]
                            elif shift < 0:
                                t_jitt = np.pad(temp_up, [(0, 0), (0, np.abs(shift))], 'constant')[:, -nsamples_up:]
                            else:
                                t_jitt = temp_up
                            temp_down = ss.decimate(t_jitt, upsample, axis=1)
                            temp_jitt.append(temp_down)

                        templates_jitter_p.append(temp_jitt)
                    templates_jitter.append(templates_jitter_p)
            templates = np.array(templates_jitter)

            cut_outs_samples = np.array(cut_outs * fs.rescale('kHz').magnitude, dtype=int) + pad_samples

            del templates_pol, templates_pad, templates_jitter
            if drifting:
                del templates_jitter_p, templates_pad_p

            overlapping_computed = False
            if sync_rate != 0:
                if self.verbose:
                    print('Adding synchrony on overlapping spikes')
                overlapping = find_overlapping_templates(templates, thresh=overlap_threshold)
                overlapping_computed = True
                if self.verbose:
                    print('Overlapping templates: ', overlapping)
                for over in overlapping:
                    if self.verbose:
                        print('Overlapping pair: ', over)
                    spgen.add_synchrony(over, rate=sync_rate, verbose=self.verbose, time_jitt=sync_jitt)
                    # annotate new firing rates
                    fr1 = len(spgen.all_spiketrains[over[0]].times) / spgen.all_spiketrains[over[0]].t_stop
                    fr2 = len(spgen.all_spiketrains[over[1]].times) / spgen.all_spiketrains[over[1]].t_stop
                    spgen.all_spiketrains[over[0]].annotate(fr=fr1)
                    spgen.all_spiketrains[over[1]].annotate(fr=fr2)
            else:
                overlapping = []
            self.overlapping = overlapping

            # find SNR and annotate
            if self.verbose:
                print('Computing spike train SNR')
            for t_i, temp in enumerate(templates):
                min_peak = np.min(temp)
                snr = np.abs(min_peak / float(noise_level))
                spiketrains[t_i].annotate(snr=snr)

            if self.verbose:
                print('Adding spiketrain annotations')
            for i, st in enumerate(spiketrains):
                st.annotate(bintype=templates_bin[i], mtype=template_celltypes[i], soma_position=template_locs[i])
            if overlap:
                if not overlapping_computed:
                    if self.verbose:
                        print('Finding overlapping spikes')
                    overlapping = find_overlapping_templates(templates, thresh=overlap_threshold)
                annotate_overlapping_spikes(spiketrains, overlapping_pairs=overlapping, verbose=True)

            amp_mod = []
            cons_spikes = []

            if modulation == 'template':
                if self.verbose:
                    print('Template modulation')
                for st in spiketrains:
                    amp, cons = compute_modulation(st, mrand=mrand, sdrand=sdrand,
                                                   n_spikes=0)
                    amp_mod.append(amp)
                    cons_spikes.append(cons)
            elif modulation == 'electrode':
                if self.verbose:
                    print('Electrode modulaton')
                for st in spiketrains:
                    amp, cons = compute_modulation(st, n_el=n_elec, mrand=mrand, sdrand=sdrand,
                                                   n_spikes=0)
                    amp_mod.append(amp)
                    cons_spikes.append(cons)
            elif modulation == 'template-isi':
                if self.verbose:
                    print('Template-ISI modulation')
                for st in spiketrains:
                    amp, cons = compute_modulation(st, sdrand=sdrand,
                                                   n_spikes=n_burst_spikes, exp=exp_decay,
                                                   max_burst_duration=max_burst_duration)
                    amp_mod.append(amp)
                    cons_spikes.append(cons)
            elif modulation == 'electrode-isi':
                if self.verbose:
                    print('Electrode-ISI modulation')
                for st in spiketrains:
                    amp, cons = compute_modulation(st, n_el=n_elec, mrand=mrand, sdrand=sdrand,
                                                   n_spikes=n_burst_spikes, exp=exp_decay,
                                                   max_burst_duration=max_burst_duration)
                    amp_mod.append(amp)
                    cons_spikes.append(cons)

            spike_matrix = resample_spiketrains(spiketrains, fs=fs)
            n_samples = spike_matrix.shape[1]

            final_loc = []
            final_idxs = []

            # # modulated convolution
            spike_traces = []
            if drifting:
                drifting_units = np.random.permutation(n_neurons)[:n_drifting]
            else:
                drifting_units = []

            for st, spike_bin in enumerate(spike_matrix):
                if modulation == 'none':
                    # reset random seed to keep sampling of jitter spike same
                    seed = np.random.randint(10000)
                    np.random.seed(seed)

                    if drifting and st in drifting_units:
                        rec, fin_pos, final_idx, mix = convolve_drifting_templates_spiketrains(st, spike_bin,
                                                                                               templates[st],
                                                                                               cut_out=
                                                                                               cut_outs_samples,
                                                                                               fs=fs,
                                                                                               loc=
                                                                                               template_locs[st],
                                                                                               v_drift=
                                                                                               velocity_vector,
                                                                                               t_start_drift=
                                                                                               t_start_drift,
                                                                                               verbose=self.verbose)
                        recordings += rec
                        final_loc.append(fin_pos)
                        final_idxs.append(final_idx)
                        np.random.seed(seed)
                        spike_traces.append(convolve_single_template(st, spike_bin,
                                                                     templates[st, 0, :,
                                                                     np.argmax(voltage_peaks[st])],
                                                                     cut_out=cut_outs_samples))
                        spiketrains[st].annotate(drifting=True)
                        spiketrains[st].annotate(initial_soma_position=template_locs[st, 0])
                        spiketrains[st].annotate(final_soma_position=fin_pos)
                        spiketrains[st].annotate(final_idx=final_idx)
                    else:
                        if drifting:
                            template = templates[st, 0]
                        else:
                            template = templates[st]
                        recordings += convolve_templates_spiketrains(st, spike_bin, template,
                                                                     cut_out=cut_outs_samples, verbose=self.verbose)
                        np.random.seed(seed)
                        spike_traces.append(convolve_single_template(st, spike_bin,
                                                                     template[:, np.argmax(voltage_peaks[st])],
                                                                     cut_out=cut_outs_samples))
                        if drifting:
                            final_loc.append(template_locs[st, 0])
                            final_idxs.append(0)
                elif 'electrode' in modulation:
                    seed = np.random.randint(10000)
                    np.random.seed(seed)

                    if drifting and st in drifting_units:
                        rec, fin_pos, final_idx, mix = convolve_drifting_templates_spiketrains(st, spike_bin,
                                                                                               templates[st],
                                                                                               cut_out=
                                                                                               cut_outs_samples,
                                                                                               modulation=True,
                                                                                               mod_array=
                                                                                               amp_mod[st],
                                                                                               fs=fs,
                                                                                               loc=
                                                                                               template_locs[st],
                                                                                               v_drift=
                                                                                               velocity_vector,
                                                                                               t_start_drift=
                                                                                               t_start_drift,
                                                                                               bursting=shape_mod,
                                                                                               fc=bursting_fc,
                                                                                               verbose=self.verbose)
                        recordings += rec
                        final_loc.append(fin_pos)
                        final_idxs.append(final_idx)
                        np.random.seed(seed)
                        spike_traces.append(convolve_single_template(st, spike_bin,
                                                                     templates[st, 0, :,
                                                                     np.argmax(voltage_peaks[st])],
                                                                     cut_out=cut_outs_samples,
                                                                     modulation=True,
                                                                     mod_array=amp_mod[st][:,
                                                                               np.argmax(voltage_peaks[st])]))
                        spiketrains[st].annotate(drifting=True)
                        spiketrains[st].annotate(initial_soma_position=template_locs[st, 0])
                        spiketrains[st].annotate(final_soma_position=fin_pos)
                        spiketrains[st].annotate(final_idx=final_idx)
                    else:
                        if drifting:
                            template = templates[st, 0]
                        else:
                            template = templates[st]
                        recordings += convolve_templates_spiketrains(st, spike_bin, template,
                                                                     cut_out=cut_outs_samples,
                                                                     modulation=True,
                                                                     mod_array=amp_mod[st],
                                                                     bursting=shape_mod,
                                                                     fc=bursting_fc,
                                                                     fs=fs, verbose=self.verbose)
                        np.random.seed(seed)
                        spike_traces.append(convolve_single_template(st, spike_bin,
                                                                     template[:, np.argmax(voltage_peaks[st])],
                                                                     cut_out=cut_outs_samples,
                                                                     modulation=True,
                                                                     mod_array=amp_mod[st][:,
                                                                               np.argmax(voltage_peaks[st])],
                                                                     bursting=shape_mod,
                                                                     fc=bursting_fc,
                                                                     fs=fs))
                        if drifting:
                            final_loc.append(template_locs[st, 0])
                            final_idxs.append(0)
                elif 'template' in modulation:
                    seed = np.random.randint(10000)
                    np.random.seed(seed)
                    if drifting and st in drifting_units:
                        rec, fin_pos, final_idx, mix = convolve_drifting_templates_spiketrains(st, spike_bin,
                                                                                               templates[st],
                                                                                               cut_out=
                                                                                               cut_outs_samples,
                                                                                               modulation=True,
                                                                                               mod_array=
                                                                                               amp_mod[st],
                                                                                               fs=fs,
                                                                                               loc=
                                                                                               template_locs[st],
                                                                                               v_drift=
                                                                                               velocity_vector,
                                                                                               t_start_drift=
                                                                                               t_start_drift,
                                                                                               bursting=shape_mod,
                                                                                               fc=bursting_fc,
                                                                                               verbose=self.verbose)
                        recordings += rec
                        final_idxs.append(final_idx)
                        final_loc.append(fin_pos)
                        np.random.seed(seed)
                        spike_traces.append(convolve_single_template(st, spike_bin,
                                                                     templates[st, 0, :,
                                                                     np.argmax(voltage_peaks[st])],
                                                                     cut_out=cut_outs_samples,
                                                                     modulation=True,
                                                                     mod_array=amp_mod[st]))
                        spiketrains[st].annotate(drifting=True)
                        spiketrains[st].annotate(initial_soma_position=template_locs[st, 0])
                        spiketrains[st].annotate(final_soma_position=fin_pos)
                        spiketrains[st].annotate(final_idx=final_idx)
                    else:
                        if drifting:
                            template = templates[st, 0]
                        else:
                            template = templates[st]
                        recordings += convolve_templates_spiketrains(st, spike_bin, template,
                                                                     cut_out=cut_outs_samples,
                                                                     modulation=True,
                                                                     mod_array=amp_mod[st],
                                                                     bursting=shape_mod,
                                                                     fc=bursting_fc,
                                                                     fs=fs, verbose=self.verbose)
                        np.random.seed(seed)
                        spike_traces.append(convolve_single_template(st, spike_bin,
                                                                     template[:, np.argmax(voltage_peaks[st])],
                                                                     cut_out=cut_outs_samples,
                                                                     modulation=True,
                                                                     mod_array=amp_mod[st],
                                                                     bursting=shape_mod,
                                                                     fc=bursting_fc,
                                                                     fs=fs))
                        if drifting:
                            final_loc.append(template_locs[st, 0])
                            final_idxs.append(0)
            spike_traces = np.array(spike_traces)

            if drifting:
                templates_drift = np.zeros((templates.shape[0], final_idx + 1,
                                            templates.shape[2], templates.shape[3],
                                            templates.shape[4]))
                for i, st in enumerate(spiketrains):
                    if i in drifting_units:
                        templates_drift[i] = templates[i, :(final_idx + 1)]
                    else:
                        templates_drift[i] = np.array([templates[i, 0]] * (final_idx + 1))
                templates = templates_drift
        else:
            spiketrains = np.array([])
            voltage_peaks = np.array([])
            spike_traces = np.array([])
            templates = np.array([])
            overlapping = np.array([])

        if self.verbose:
            print('Adding noise')
        # divide in chunks
        chunks_noise = []
        if duration > chunk_noise_duration and chunk_noise_duration != 0:
            start = 0 * pq.s
            finished = False
            while not finished:
                chunks_noise.append([start, start + chunk_noise_duration])
                start = start + chunk_noise_duration
                if start >= duration:
                    finished = True

        if self.verbose:
            print('Noise seed: ', noise_seed)
        if noise_color:
            if self.verbose:
                print('Coloring noise with peak: ', color_peak, ' quality factor: ', color_q,
                      ' and random noise level: ', random_noise_floor)
        np.random.seed(noise_seed)
        if noise_level > 0:
            if noise_mode == 'uncorrelated':
                if len(chunks_noise) > 0:
                    for ch, chunk in enumerate(chunks_noise):
                        if self.verbose:
                            print('Generating noise in: ', chunk[0], chunk[1], ' chunk')
                        idxs = np.where((timestamps >= chunk[0]) & (timestamps < chunk[1]))[0]
                        additive_noise = noise_level * np.random.randn(recordings.shape[0],
                                                                       len(idxs))
                        if noise_color:
                            # iir peak filter
                            b_iir, a_iir = ss.iirpeak(color_peak, Q=color_q, fs=fs.rescale('Hz').magnitude)
                            additive_noise = ss.filtfilt(b_iir, a_iir, additive_noise, axis=1, padlen=1000)
                            additive_noise = additive_noise + random_noise_floor * np.std(additive_noise) * \
                                             np.random.randn(additive_noise.shape[0], additive_noise.shape[1])
                            additive_noise = additive_noise * (noise_level / np.std(additive_noise))
                        recordings[:, idxs] += additive_noise
                else:
                    additive_noise = noise_level * np.random.randn(recordings.shape[0],
                                                                   recordings.shape[1])
                    if noise_color:
                        # iir peak filter
                        b_iir, a_iir = ss.iirpeak(color_peak, Q=color_q, fs=fs.rescale('Hz').magnitude)
                        additive_noise = ss.filtfilt(b_iir, a_iir, additive_noise, axis=1)
                        additive_noise = additive_noise + random_noise_floor * np.std(additive_noise) \
                                         * np.random.randn(additive_noise.shape[0], additive_noise.shape[1])
                        additive_noise = additive_noise * (noise_level / np.std(additive_noise))

                    recordings += additive_noise
            elif noise_mode == 'distance-correlated':
                cov_dist = np.zeros((n_elec, n_elec))
                for i, el in enumerate(mea.positions):
                    for j, p in enumerate(mea.positions):
                        if i != j:
                            cov_dist[i, j] = (0.5 * half_dist) / np.linalg.norm(el - p)
                        else:
                            cov_dist[i, j] = 1

                if len(chunks_noise) > 0:
                    for ch, chunk in enumerate(chunks_noise):
                        if self.verbose:
                            print('Generating noise in: ', chunk[0], chunk[1], ' chunk')
                        idxs = np.where((timestamps >= chunk[0]) & (timestamps < chunk[1]))[0]
                        additive_noise = noise_level * np.random.multivariate_normal(np.zeros(n_elec), cov_dist,
                                                                                     size=(len(idxs))).T
                        if noise_color:
                            # iir peak filter
                            b_iir, a_iir = ss.iirpeak(color_peak, Q=color_q, fs=fs.rescale('Hz').magnitude)
                            additive_noise = ss.filtfilt(b_iir, a_iir, additive_noise, axis=1)
                            additive_noise = additive_noise + random_noise_floor * np.std(additive_noise) \
                                             * np.random.randn(additive_noise.shape[0], additive_noise.shape[1])
                            additive_noise = additive_noise * (noise_level / np.std(additive_noise))
                        recordings[:, idxs] += additive_noise
                else:
                    additive_noise = noise_level * np.random.multivariate_normal(np.zeros(n_elec), cov_dist,
                                                                                 size=recordings.shape[1]).T
                    if noise_color:
                        # iir peak filter
                        b_iir, a_iir = ss.iirpeak(color_peak, Q=color_q, fs=fs.rescale('Hz').magnitude)
                        additive_noise = ss.filtfilt(b_iir, a_iir, additive_noise, axis=1)
                        additive_noise = additive_noise + random_noise_floor * np.std(additive_noise) \
                                         * np.random.randn(additive_noise.shape[0], additive_noise.shape[1])
                        additive_noise = additive_noise * (noise_level / np.std(additive_noise))

                    recordings += additive_noise
        else:
            if self.verbose:
                print('Noise level is set to 0')

        if filter:
            if self.verbose:
                print('Filtering')
                if cutoff.size == 1:
                    print('High-pass cutoff', cutoff)
                elif cutoff.size == 2:
                    print('Band-pass cutoff', cutoff)
            chunks_filter = []
            if duration > chunk_filter_duration and chunk_filter_duration != 0:
                start = 0 * pq.s
                finished = False
                while not finished:
                    chunks_filter.append([start, start + chunk_filter_duration])
                    start = start + chunk_filter_duration
                    if start >= duration:
                        finished = True
            if len(chunks_filter) > 0:
                for ch, chunk in enumerate(chunks_filter):
                    if self.verbose:
                        print('Filtering in: ', chunk[0], chunk[1], ' chunk')
                    idxs = np.where((timestamps >= chunk[0]) & (timestamps < chunk[1]))[0]
                    if cutoff.size == 1:
                        recordings[:, idxs] = filter_analog_signals(recordings[:, idxs], freq=cutoff, fs=fs,
                                                                    filter_type='highpass', order=order)
                    elif cutoff.size == 2:
                        if fs / 2. < cutoff[1]:
                            recordings[:, idxs] = filter_analog_signals(recordings[:, idxs], freq=cutoff[0], fs=fs,
                                                                        filter_type='highpass', order=order)
                        else:
                            recordings[:, idxs] = filter_analog_signals(recordings[:, idxs], freq=cutoff, fs=fs)
            else:
                if cutoff.size == 1:
                    recordings = filter_analog_signals(recordings, freq=cutoff, fs=fs, filter_type='highpass',
                                                       order=order)
                elif cutoff.size == 2:
                    if fs / 2. < cutoff[1]:
                        recordings = filter_analog_signals(recordings, freq=cutoff[0], fs=fs,
                                                           filter_type='highpass', order=order)
                    else:
                        recordings = filter_analog_signals(recordings, freq=cutoff, fs=fs, order=order)

        if not only_noise:
            if extract_waveforms:
                if self.verbose:
                    print('Extracting spike waveforms')
                extract_wf(spiketrains, recordings, fs=fs, timestamps=timestamps)

        params['templates']['overlapping'] = np.array(overlapping)
        self.recordings = recordings
        self.timestamps = timestamps
        self.channel_positions = mea_pos
        self.templates = np.squeeze(templates)
        self.spiketrains = spiketrains
        self.voltage_peaks = voltage_peaks
        self.spike_traces = spike_traces
        self.info = params

    def annotate_overlapping_spike(self, parallel=True):
        '''
        Annnotate spike trains with overlapping information.
        '''
        annotate_overlapping_spikes(self.spiketrains, overlapping_pairs=self.overlapping, parallel=parallel)


def gen_recordings(params=None, templates=None, tempgen=None, verbose=True):
    '''
    Generates recordings.

    Parameters
    ----------
    templates : str
        Path to generated templates
    params : dict OR str
        Dictionary containing recording parameters OR path to yaml file containing parameters
    tempgen : TemplateGenerator
        Template generator object
    verbose : bool
        If True output is verbose

    Returns
    -------
    RecordingGenerator
        Generated recording generator object

    '''
    t_start = time.time()
    if isinstance(params, str):
        if os.path.isfile(params) and (params.endswith('yaml') or params.endswith('yml')):
            with open(params, 'r') as pf:
                if use_loader:
                    params_dict = yaml.load(pf, Loader=yaml.FullLoader)
                else:
                    params_dict = yaml.load(pf)
    elif isinstance(params, dict):
        params_dict = params
    else:
        params_dict = {}

    if 'spiketrains' not in params_dict:
        params_dict['spiketrains'] = {}
    if 'templates' not in params_dict:
        params_dict['templates'] = {}
    if 'recordings' not in params_dict:
        params_dict['recordings'] = {}
    if 'cell_types' not in params_dict:
        params_dict['cell_types'] = {}

    if tempgen is None and templates is None:
        raise AttributeError("Provide either 'templates' or 'tempgen' TemplateGenerator object")

    if tempgen is None:
        if os.path.isdir(templates):
            tempgen = load_templates(templates, verbose=False)
        elif templates.endswith('h5') or templates.endswith('hdf5'):
            tempgen = load_templates(templates, verbose=False)
        else:
            raise AttributeError("'templates' is not a folder or an hdf5 file")

    if 'seed' in params_dict['spiketrains']:
        if params_dict['spiketrains']['seed'] is None:
            params_dict['spiketrains']['seed'] = np.random.randint(1, 10000)
    else:
        params_dict['spiketrains'].update({'seed': np.random.randint(1, 10000)})

    if 'seed' in params_dict['templates']:
        if params_dict['templates']['seed'] is None:
            params_dict['templates']['seed'] = np.random.randint(1, 10000)
    else:
        params_dict['templates'].update({'seed': np.random.randint(1, 10000)})

    if 'seed' in params_dict['recordings']:
        if params_dict['recordings']['seed'] is None:
            params_dict['recordings']['seed'] = np.random.randint(1, 10000)
    else:
        params_dict['recordings'].update({'recordings': np.random.randint(1, 10000)})

    # Generate spike trains
    spgen = SpikeTrainGenerator(params_dict['spiketrains'], verbose=verbose)
    spgen.generate_spikes()
    spiketrains = spgen.all_spiketrains

    params_dict['spiketrains'] = spgen.info
    # Generate recordings
    recgen = RecordingGenerator(spgen, tempgen, params_dict, verbose=verbose)
    recgen.generate_recordings()

    print('Elapsed time: ', time.time() - t_start)

    return recgen


def gen_templates(cell_models_folder, params=None, templates_folder=None,
                  intraonly=False, parallel=True, delete_tmp=True, verbose=True):
    '''

    Parameters
    ----------
    tmp_params_path : str
        path to tmp params.yaml (same as templates_params with extra 'simulate_script', 'templates_folder' fields)
    intraonly : bool
        if True only intracellular simulation is run
    parallel : bool
        if True multi-threading is used
    delete_tmp :
        if True the temporary files are deleted
    verbose : bool
        If True output is verbose

    Returns
    -------
    TemplateGenerator
        Generated template generator object

    '''
    if isinstance(params, str):
        if os.path.isfile(params) and (params.endswith('yaml') or params.endswith('yml')):
            with open(params, 'r') as pf:
                if use_loader:
                    params_dict = yaml.load(pf, Loader=yaml.FullLoader)
                else:
                    params_dict = yaml.load(pf)
    elif isinstance(params, dict):
        params_dict = params
    else:
        params_dict = None

    if templates_folder is not None:
        if not os.path.isdir(templates_folder):
            os.makedirs(templates_folder)

    tempgen = TemplateGenerator(cell_models_folder=cell_models_folder,
                                params=params_dict,
                                templates_folder=templates_folder,
                                intraonly=intraonly,
                                parallel=parallel,
                                delete_tmp=delete_tmp,
                                verbose=verbose)
    tempgen.generate_templates()

    return tempgen
