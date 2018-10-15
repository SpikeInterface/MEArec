from __future__ import print_function, division

import neo
import elephant.spike_train_generation as stg
import elephant.conversion as conv
import elephant.statistics as stat
import matplotlib.pylab as plt
import scipy.signal as ss
import time
import multiprocessing
from copy import copy
from MEArec.tools import *
import threading
import shutil
from pprint import pprint

from .simulate_cells import run_cell_model, calc_extracellular

class simulationThread(threading.Thread):
    def __init__(self, threadID, name, simulate_script, numb, tot, cell_model, model_folder, intraonly, params):
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
    def run(self):
        print ("Starting " + self.name)
        print('\n\n', self.cell_model, self.numb + 1, '/', self.tot, '\n\n')
        os.system('python %s %s %s %s' \
                % (self.sim_script, join(self.model_folder, self.cell_model), self.intra, self.params))
        print ("Exiting " + self.name)


class TemplateGenerator:
    def __init__(self, *, cell_models_folder=None, templates_folder=None, temp_dict=None, info=None,
                 params=None, intraonly=False, parallel=True, delete_tmp=True):
        if temp_dict is not None and info is not None:
            self.templates = temp_dict['templates']
            self.locations = temp_dict['locations']
            self.rotations = temp_dict['rotations']
            self.celltypes = temp_dict['celltypes']
            self.info = info
        else:
            if cell_models_folder is None:
                raise AttributeError("Specify cell folder!")
            if params is None:
                print("Using default parameters")
                params = {}

            if os.path.isdir(cell_models_folder):
                cell_models = [f for f in os.listdir(join(cell_models_folder)) if 'mods' not in f]
                if len(cell_models) == 0:
                    raise AttributeError(cell_models_folder, ' contains no cell models!')
            else:
                raise NotADirectoryError('Cell models folder: does not exist!')

            this_dir, this_filename = os.path.split(__file__)
            simulate_script = join(this_dir, 'simulate_cells.py')
            params['cell_models_folder'] = cell_models_folder

            # Compile NEURON models (nrnivmodl)
            if not os.path.isdir(join(cell_models_folder, 'mods')):
                print('Compiling NEURON models')
                os.system('python %s compile %s' % (simulate_script, cell_models_folder))

            if 'sim_time' not in params.keys():
                params['sim_time'] = 1
            if 'target_spikes' not in params.keys():
                params['target_spikes'] = [3, 50]
            if 'cut_out' not in params.keys():
                params['cut_out'] = [2, 5]
            if 'dt' not in params.keys():
                params['dt'] = 2 ** -5
            if 'delay' not in params.keys():
                params['delay'] = 10
            if 'weights' not in params.keys():
                params['weights'] = [0.25, 1.75]

            if 'rot' not in params.keys():
                params['rot'] = 'physrot'
            if 'probe' not in params.keys():
                available_mea = MEA.return_mea_list()
                probe = available_mea[np.random.randint(len(available_mea))]
                print("Probe randomly set to: %s" % probe)
                params['probe'] = probe
            if 'ncontacts' not in params.keys():
                params['ncontacts'] = 1
            if 'overhang' not in params.keys():
                params['overhang'] = 1
            if 'xlim' not in params.keys():
                params['xlim'] = [10, 80]
            if 'ylim' not in params.keys():
                params['ylim'] = None
            if 'zlim' not in params.keys():
                params['zlim'] = None
            if 'offset' not in params.keys():
                params['offset'] = 0
            if 'det_thresh' not in params.keys():
                params['det_thresh'] = 30
            if 'n' not in params.keys():
                params['n'] = 50
            if 'seed' not in params.keys():
                params['seed'] = np.random.randint(1, 10000)
            elif params['seed'] is None:
                params['seed'] = np.random.randint(1, 10000)
            if templates_folder is None:
                params['templates_folder'] = os.getcwd()
                templates_folder = params['templates_folder']
            else:
                params['templates_folder'] = templates_folder

            rot = params['rot']
            n = params['n']
            probe = params['probe']
            tot = len(cell_models)

            tmp_params_path = 'tmp_params_path'
            with open(tmp_params_path, 'w') as f:
                yaml.dump(params, f)

            # Simulate neurons and EAP for different cell models sparately
            if parallel:
                start_time = time.time()
                print('Parallel')
                tot = len(cell_models)
                threads = []
                for numb, cell_model in enumerate(cell_models):
                    threads.append(simulationThread(numb, "Thread-" + str(numb), simulate_script,
                                                    numb, tot, cell_model, cell_models_folder, intraonly, tmp_params_path))
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                print('\n\n\nSimulation time: ', time.time() - start_time, '\n\n\n')
            else:
                start_time = time.time()
                for numb, cell_model in enumerate(cell_models):
                    print('\n\n', cell_model, numb + 1, '/', len(cell_models), '\n\n')
                    os.system('python %s %s %s %s' \
                              % (simulate_script, join(cell_models_folder, cell_model), intraonly, tmp_params_path))
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
            self.info['params'] = params
            self.info['electrodes'] = MEA.return_mea_info(probe)


class SpikeTrainGenerator:
    def __init__(self, params=None):
        '''
        Spike Train Generator: class to create poisson or gamma spike trains

        Parameters
        ----------
        n_exc: number of excitatory cells
        n_inh: number of inhibitory cells
        f_exc: mean firing rate of excitatory cells
        f_inh: mean firing rate of inhibitory cells
        st_exc: firing rate standard deviation of excitatory cells
        st_inh: firing rate standard deviation of inhibitory cells
        process: 'poisson' - 'gamma'
        gamma_shape: shape param for gamma distribution
        t_start: starting time (s)
        t_stop: stopping time (s)
        ref_period: refractory period to remove spike violation
        n_add: number of units to add at t_add time
        t_add: time to add units
        n_remove: number of units to remove at t_remove time
        t_remove: time to remove units
        '''
        if params is None:
            print("Using default parameters")
            params = {}
        self.params = copy(params)

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

            for exc in range(self.params['n_exc']):
                rate = self.params['st_exc'] * np.random.randn() + self.params['f_exc']
                if rate < self.params['min_rate']:
                    rate = self.params['min_rate']
                rates.append(rate)
                types.append('e')
            for inh in range(self.params['n_inh']):
                rate = self.params['st_inh'] * np.random.randn() + self.params['f_inh']
                if rate < self.params['min_rate']:
                    rate = self.params['min_rate']
                rates.append(rate)
                types.append('i')
            self.params['rates'] = rates
            self.params['types'] = types
            self.n_neurons = len(self.params['rates'])

        np.random.seed(self.params['seed'])

        self.changing = False
        self.intermittent = False

        self.info = params

        #TODO add spike synchrony and overlapping synchrony
        #self.sync_rate = float(sync)
        #
        # self.save = save
        #
        # #print( 'Modulation: ', self.modulation
        #

        # self.changing = False
        # self.n_add = n_add
        # self.n_remove = n_remove
        # self.t_add = int(t_add) * pq.s
        # self.t_remove = int(t_remove) * pq.s
        #
        # self.intermittent = False
        # self.n_int = n_int
        # self.t_int = t_int
        # self.t_burst = t_burst
        # self.t_int_sd = t_int
        # self.t_burst_sd = t_burst
        # self.f_int = f_int
        #
        # self.idx = 0
        #
        #
        # if n_add != 0:
        #     if t_add == 0:
        #         raise Exception('Provide time to add units')
        #     else:
        #         self.changing = True
        # if n_remove != 0:
        #     if t_remove == 0:
        #         raise Exception('Provide time to remove units')
        #     else:
        #         self.changing = True
        # if n_int != 0:
        #     self.intermittent = True
        #
        #
        # if self.changing:
        #     n_tot = n_exc + n_inh
        #     perm_idxs = np.random.permutation(np.arange(n_tot))
        #     self.idxs_add = perm_idxs[:self.n_add]
        #     self.idxs_remove = perm_idxs[-self.n_remove:]
        # else:
        #     self.idxs_add = []
        #     self.idxs_remove = []
        #
        # if self.intermittent:
        #     n_tot = n_exc + n_inh
        #     perm_idxs = np.random.permutation(np.arange(n_tot))
        #     self.idxs_int = perm_idxs[:self.n_int]
        # else:
        #     self.idxs_int = []

    def set_spiketrain(self, idx, spiketrain):
        '''
        Sets spike train idx to new spiketrain
        Parameters
        ----------
        idx: index of spike train to set
        spiketrain: new spike train

        Returns
        -------

        '''
        self.all_spiketrains[idx] = spiketrain

    def generate_spikes(self):
        '''
        Generate spike trains based on default_params of the SpikeTrainGenerator class.
        self.all_spiketrains contains the newly generated spike trains

        Returns
        -------

        '''

        self.all_spiketrains = []
        idx = 0
        for n in range(self.n_neurons):
            if not self.changing and not self.intermittent:
                rate = self.params['rates'][n]
                if self.params['process'] == 'poisson':
                    st = stg.homogeneous_poisson_process(rate,
                                                         self.params['t_start'], self.params['t_stop'])
                elif self.params['process'] == 'gamma':
                    st = stg.homogeneous_gamma_process(self.params['gamma_shape'], rate, self.params['rates'][n],
                                                       self.params['t_start'], self.params['t_stop'])
            else:
                raise NotImplementedError('Changing and intermittent spiketrains are not impleented yet')
            self.all_spiketrains.append(st)
            self.all_spiketrains[-1].annotate(freq=rate)
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

    def raster_plots(self, marker='|', markersize=5, mew=2):
        '''
        Plots raster plots of spike trains

        Parameters
        ----------
        marker: marker type (def='|')
        markersize: marker size (def=5)
        mew: marker edge width (def=2)

        Returns
        -------
        ax: matplotlib axes

        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, spiketrain in enumerate(self.all_spiketrains):
            t = spiketrain.rescale(pq.s)
            if i < self.params['n_exc']:
                ax.plot(t, i * np.ones_like(t), color='b', marker=marker, ls='', markersize=markersize, mew=mew)
            else:
                ax.plot(t, i * np.ones_like(t), color='r', marker=marker, ls='', markersize=markersize, mew=mew)
        ax.axis('tight')
        ax.set_xlim([self.params['t_start'].rescale(pq.s), self.params['t_stop'].rescale(pq.s)])
        ax.set_xlabel('Time (ms)', fontsize=16)
        ax.set_ylabel('Spike Train Index', fontsize=16)
        plt.gca().tick_params(axis='both', which='major', labelsize=14)

        return ax

    def resample_spiketrains(self, fs=None, T=None):
        '''
        Resamples spike trains. Provide either fs or T parameters
        Parameters
        ----------
        fs: new sampling frequency (quantity)
        T: new period (quantity)

        Returns
        -------
        matrix with resampled binned spike trains

        '''
        resampled_mat = []
        if not fs and not T:
            print('Provide either sampling frequency fs or time period T')
        elif fs:
            if not isinstance(fs, Quantity):
                raise ValueError("fs must be of type pq.Quantity")
            binsize = 1. / fs
            binsize.rescale('ms')
            resampled_mat = []
            for sts in self.all_spiketrains:
                spikes = conv.BinnedSpikeTrain(sts, binsize=binsize).to_array()
                resampled_mat.append(np.squeeze(spikes))
        elif T:
            binsize = T
            if not isinstance(T, Quantity):
                raise ValueError("T must be of type pq.Quantity")
            binsize.rescale('ms')
            resampled_mat = []
            for sts in self.all_spiketrains:
                spikes = conv.BinnedSpikeTrain(sts, binsize=binsize).to_array()
                resampled_mat.append(np.squeeze(spikes))
        return np.array(resampled_mat)

    def add_synchrony(self, idxs, rate=0.05):
        '''
        Adds synchronous spikes between pairs of spike trains at a certain rate
        Parameters
        ----------
        idxs: list or array with the 2 indices
        rate: probability of adding a synchronous spike to spike train idxs[1] for each spike of idxs[0]

        Returns
        -------

        '''
        idx1 = idxs[0]
        idx2 = idxs[1]
        st1 = self.all_spiketrains[idx1]
        st2 = self.all_spiketrains[idx2]
        times2 = st2.times
        t_start = st2.t_start
        t_stop = st2.t_stop
        unit = times2.units
        for t1 in st1:
            rand = np.random.rand()
            if rand <= rate:
                # check time difference
                t_diff = np.abs(t1.rescale(pq.ms).magnitude - times2.rescale(pq.ms).magnitude)
                if np.all(t_diff > self.params['ref_period']):
                    times2 = np.sort(np.concatenate((np.array(times2), np.array([t1]))))
                    times2 = times2 * unit
                    st2 = neo.SpikeTrain(times2, t_start=t_start, t_stop=t_stop)
                    self.set_spiketrain(idx2, st2)

    def bursting_st(self, freq=None, min_burst=3, max_burst=10):
        pass

class RecordingGenerator:
    def __init__(self, spgen=None, tempgen=None, params=None, rec_dict=None, info=None):
        '''

        Parameters
        ----------
        spgen
        tempgen
        params
        rec_dict
        info
        '''
        if rec_dict is not None and info is not None:
            self.recordings = rec_dict['recordings']
            self.spiketrains = rec_dict['spiketrains']
            self.templates = rec_dict['templates']
            self.positions = rec_dict['positions']
            self.times = rec_dict['times']
            self.peaks = rec_dict['peaks']
            self.sources = rec_dict['sources']
            self.info = info
        else:
            if spgen is None or tempgen is None:
                raise AttributeError("Specify SpikeGenerator and TemplateGenerator objects!")
            if params is None:
                print("Using default parameters")
                params = {'spiketrains': {},
                          'celltypes': {},
                          'templates': {},
                          'recordings': {}}
            self.params = copy(params)

            temp_params = self.params['templates']
            rec_params = self.params['recordings']
            st_params = self.params['spiketrains']
            celltype_params = self.params['cell_types']

            eaps = tempgen.templates
            locs = tempgen.locations
            rots = tempgen.rotations
            celltypes = tempgen.celltypes
            temp_info = tempgen.info

            spiketrains = spgen.all_spiketrains

            n_neurons = len(spiketrains)
            cut_outs = temp_info['params']['cut_out']
            duration = spiketrains[0].t_stop - spiketrains[0].t_start

            if 'fs' not in rec_params.keys():
                params['recordings']['fs'] = 1. / temp_info['params']['dt']
            elif params['recordings']['fs'] is None:
                params['recordings']['fs'] = 1. / temp_info['params']['dt']
            fs = params['recordings']['fs'] * pq.kHz

            if 'noise_level' not in rec_params.keys():
                params['recordings']['noise_level'] = 15
            noise_level =  params['recordings']['noise_level']

            if 'noise_mode' not in rec_params.keys():
                params['recordings']['noise_mode'] = 'uncorrelated'
            noise_mode = params['recordings']['noise_mode']

            if 'filter' not in rec_params.keys():
                params['recordings']['filter'] = True
            filter = params['recordings']['noise_mode']

            if 'cutoff' not in rec_params.keys():
                params['recordings']['cutoff'] = [300., 6000.]
            cutoff = params['recordings']['cutoff'] * pq.Hz

            if 'modulation' not in rec_params.keys():
                params['recordings']['modulation'] = 'electrode'
            modulation = params['recordings']['modulation']

            if 'chunk_duration' not in rec_params.keys():
                params['recordings']['chunk_duration'] = 0
            chunk_duration = params['recordings']['chunk_duration'] * pq.s

            if 'mrand' not in rec_params.keys():
                params['recordings']['mrand'] = 1
            mrand = params['recordings']['mrand']

            if 'sdrand' not in rec_params.keys():
                params['recordings']['sdrand'] = 0.05
            sdrand = params['recordings']['sdrand']

            if 'overlap' not in rec_params.keys():
                params['recordings']['overlap'] = False
            overlap = params['recordings']['overlap']

            if 'seed' not in rec_params.keys():
                params['recordings']['seed'] = np.random.randint(1,1000)
            elif params['recordings']['seed'] is None:
                params['recordings']['seed'] = np.random.randint(1,1000)
            noise_seed = params['recordings']['seed']

            if 'x_lim' not in temp_params.keys():
                params['templates']['x_lim'] = None
            x_lim =  params['templates']['x_lim']

            if 'y_lim' not in temp_params.keys():
                params['templates']['y_lim'] = None
            y_lim = params['templates']['y_lim']

            if 'z_lim' not in temp_params.keys():
                params['templates']['z_lim'] = None
            z_lim = params['templates']['z_lim']

            if 'min_amp' not in temp_params.keys():
                params['templates']['min_amp'] = 50
            min_amp = params['templates']['min_amp']

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
                params['templates']['seed'] = np.random.randint(1,1000)
            elif params['templates']['seed'] is None:
                params['templates']['seed'] = np.random.randint(1,1000)
            temp_seed = params['templates']['seed']

            parallel=False

            if 'excitatory' in celltype_params.keys() and 'inhibitory' in celltype_params.keys():
                exc_categories = celltype_params['excitatory']
                inh_categories = celltype_params['inhibitory']
                bin_cat = get_binary_cat(celltypes, exc_categories, inh_categories)
            else:
                bin_cat = np.array(['U'] * len(celltypes))

            # load MEA info
            electrode_name = temp_info['electrodes']['electrode_name']
            elinfo = MEA.return_mea_info(electrode_name)
            offset = temp_info['params']['offset']
            elinfo['offset'] = offset
            mea = MEA.return_mea(info=elinfo)
            mea_pos = mea.positions

            params['recordings'].update({'electrode_name': electrode_name,
                                         'duration': float(duration.magnitude),
                                         'fs': float(fs.rescale('Hz').magnitude),
                                         'n_neurons': n_neurons})


            n_elec = eaps.shape[1]
            # this is fixed from recordings
            spike_duration = np.sum(temp_info['params']['cut_out'])*pq.ms
            spike_fs = 1./temp_info['params']['dt']*pq.kHz

            print('Selecting cells')
            if 'type' in spiketrains[0].annotations.keys():
                n_exc = [st.annotations['type'] for st in spiketrains].count('E')
                n_inh = n_neurons - n_exc

            np.random.seed(temp_seed)
            idxs_cells = select_templates(locs, eaps, bin_cat, n_exc, n_inh, x_lim=x_lim, y_lim=y_lim, z_lim=z_lim,
                                          min_amp=min_amp, min_dist=min_dist, verbose=False)
            template_celltypes = celltypes[idxs_cells]
            template_locs = locs[idxs_cells]
            templates_bin = bin_cat[idxs_cells]
            templates = eaps[idxs_cells]

            # peak images
            peak = []
            for tem in templates:
                dt = 2 ** -5
                feat = get_EAP_features(tem, ['Na'], dt=dt)
                peak.append(-np.squeeze(feat['na']))
            peak = np.array(peak)

            up = fs
            down = spike_fs
            sampling_ratio = float(up/down)
            # resample spikes
            resample=False
            pad_samples = [int((pp*fs).magnitude) for pp in pad_len]
            n_resample = int((fs * spike_duration).magnitude)
            if templates.shape[2] != n_resample:
                templates_pol = np.zeros((templates.shape[0], templates.shape[1], n_resample))
                print('Resampling spikes')
                for t, tem in enumerate(templates):
                    tem_pad = np.pad(tem, [(0,0), pad_samples], 'edge')
                    tem_poly = ss.resample_poly(tem_pad, up, down, axis=1)
                    templates_pol[t, :] = tem_poly[:, int(sampling_ratio*pad_samples[0]):int(sampling_ratio*pad_samples[0])
                                                                                         +n_resample]
                resample=True
            else:
                templates_pol = templates

            templates_pad = []
            templates_spl = []
            print('Padding template edges')
            for t, tem in enumerate(templates_pol):
                tem, _ = cubic_padding(tem, pad_len, fs)
                templates_pad.append(tem)

            print('Creating time jittering')
            jitter = 1. / fs
            templates_jitter = []
            for temp in templates_pad:
                temp_up = ss.resample_poly(temp, upsample, 1., axis=1)
                nsamples_up = temp_up.shape[1]
                temp_jitt = []
                for n in range(n_jitters):
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
            templates = np.array(templates_jitter)

            cut_outs_samples = np.array(cut_outs * fs.rescale('kHz').magnitude, dtype=int) + pad_samples

        #
            # if self.sync_rate != 0:
            #     #print( 'Adding synchrony on overlapping spikes'
            #     self.overlapping = find_overlapping_spikes(self.templates, thresh=self.overlap_threshold)
            #
            #     for over in self.overlapping:
            #         self.spgen.add_synchrony(over, rate=self.sync_rate)
            # else:
            #     self.overlapping = []

            # find SNR and annotate
            print('Computing spike train SNR')
            for t_i, temp in enumerate(templates):
                min_peak = np.min(temp)
                snr = np.abs(min_peak/float(noise_level))
                spiketrains[t_i].annotate(snr=snr)
                # print( min_peak, snr)
        #
        #
        #     if self.plot_figures and self.sync_rate != 0:
        #         ax = self.spgen.raster_plots()
        #         ax.set_title('After synchrony')
        #
            print('Adding spiketrain annotations')
            for i, st in enumerate(spiketrains):
                st.annotate(bintype=templates_bin[i], mtype=template_celltypes[i], loc=template_locs[i])
            if overlap:
                print('Finding temporally overlapping spikes')
                overlapping = find_overlapping_templates(templates, thresh=overlap_threshold)
                annotate_overlapping_spikes(spiketrains, overlapping_pairs=overlapping, verbose=True)


            amp_mod = []
            cons_spikes = []
            exp = 0.3
            n_isi = 5
            mem_isi = 10*pq.ms
            if modulation == 'isi':
                print('ISI modulation')
                for st in spiketrains:
                    amp, cons = ISI_amplitude_modulation(st, mrand=mrand, sdrand=sdrand,
                                                         n_spikes=n_isi, exp=exp, mem_ISI=mem_isi)
                    amp_mod.append(amp)
                    cons_spikes.append(cons)
            elif modulation == 'template':
                print('Template modulation')
                for st in spiketrains:
                    amp, cons = ISI_amplitude_modulation(st, mrand=mrand, sdrand=sdrand,
                                                         n_spikes=0, exp=exp, mem_ISI=mem_isi)
                    amp_mod.append(amp)
                    cons_spikes.append(cons)
            elif modulation == 'electrode':
                print('Electrode modulaton')
                for st in spiketrains:
                    amp, cons = ISI_amplitude_modulation(st, n_el=n_elec, mrand=mrand, sdrand=sdrand,
                                                         n_spikes=0, exp=exp, mem_ISI=mem_isi)
                    amp_mod.append(amp)
                    cons_spikes.append(cons)

            spike_matrix = resample_spiketrains(spiketrains, fs=fs)
            n_samples = spike_matrix.shape[1]

            #print( 'Generating clean recordings'
            recordings = np.zeros((n_elec, n_samples))
            times = np.arange(recordings.shape[1]) / fs


            # modulated convolution
            pool = multiprocessing.Pool(n_neurons)
            t_start = time.time()
            gt_spikes = []

            # divide in chunks
            chunks = []
            if duration > chunk_duration and chunk_duration != 0:
                start = 0*pq.s
                finished = False
                while not finished:
                    chunks.append([start, start+chunk_duration])
                    start=start+chunk_duration
                    if start >= duration:
                        finished = True
                print('Chunks: ', chunks)


            if len(chunks) > 0:
                recording_chunks = []
                for ch, chunk in enumerate(chunks):
                    #print( 'Generating chunk ', ch+1, ' of ', len(chunks)
                    idxs = np.where((times>=chunk[0]) & (times<chunk[1]))[0]
                    spike_matrix_chunk = spike_matrix[:, idxs]
                    rec_chunk=np.zeros((n_elec, len(idxs)))
                    amp_chunk = []
                    for i, st in enumerate(spiketrains):
                        idxs = np.where((st >= chunk[0]) & (st < chunk[1]))[0]
                        if modulation != 'none':
                            amp_chunk.append(amp_mod[i][idxs])

                    if not parallel:
                        for st, spike_bin in enumerate(spike_matrix_chunk):
                            #print( 'Convolving with spike ', st, ' out of ', spike_matrix_chunk.shape[0]
                            if modulation == 'none':
                                rec_chunk += convolve_templates_spiketrains(st, spike_bin, templates[st],
                                                                            cut_out=cut_outs_samples)
                            else:
                                rec_chunk += convolve_templates_spiketrains(st, spike_bin, templates[st],
                                                                            cut_out=cut_outs_samples,
                                                                                       modulation=True,
                                                                                       amp_mod=amp_chunk[st])
                    else:
                        if modulation == 'none':
                            results = [pool.apply_async(convolve_templates_spiketrains, (st, spike_bin, templates[st],))
                                       for st, spike_bin in enumerate(spike_matrix_chunk)]
                        else:
                            results = [pool.apply_async(convolve_templates_spiketrains,
                                                        (st, spike_bin, templates[st], True, amp))
                                       for st, (spike_bin, amp) in enumerate(zip(spike_matrix_chunk, amp_chunk))]
                        for r in results:
                            rec_chunk += r.get()

                    recording_chunks.append(rec_chunk)
                recordings = np.hstack(recording_chunks)
            else:
                for st, spike_bin in enumerate(spike_matrix):
                    print('Convolving with spike ', st, ' out of ', spike_matrix.shape[0])
                    if modulation == 'none':
                        # reset random seed to keep sampling of jitter spike same
                        seed = np.random.randint(10000)
                        np.random.seed(seed)

                        recordings += convolve_templates_spiketrains(st, spike_bin, templates[st],
                                                                     cut_out=cut_outs_samples)
                        np.random.seed(seed)
                        gt_spikes.append(convolve_single_template(st, spike_bin,
                                                                  templates[st, :, np.argmax(peak[st])],
                                                                  cut_out=cut_outs_samples))
                    elif modulation == 'electrode':
                        seed = np.random.randint(10000)
                        np.random.seed(seed)
                        recordings += convolve_templates_spiketrains(st, spike_bin, templates[st], cut_out=cut_outs_samples,
                                                                          modulation=True,
                                                                          amp_mod=amp_mod[st])
                        np.random.seed(seed)
                        gt_spikes.append(convolve_single_template(st, spike_bin,
                                                                  templates[st, :, np.argmax(peak[st])],
                                                                  cut_out=cut_outs_samples,
                                                                  modulation=True,
                                                                  amp_mod=amp_mod[st][:,
                                                                          np.argmax(peak[st])]))
                    elif modulation == 'template' or modulation == 'all':
                        seed = np.random.randint(10000)
                        np.random.seed(seed)
                        recordings += convolve_templates_spiketrains(st, spike_bin, templates[st], cut_out=cut_outs_samples,
                                                                          modulation=True,
                                                                          amp_mod=amp_mod[st])
                        np.random.seed(seed)
                        gt_spikes.append(convolve_single_template(st, spike_bin,
                                                                  templates[st, :, np.argmax(peak[st])],
                                                                  cut_out=cut_outs_samples,
                                                                  modulation=True,
                                                                  amp_mod=amp_mod[st]))

            pool.close()
            gt_spikes = np.array(gt_spikes)

            print('Elapsed time ', time.time() - t_start)
            clean_recordings = copy(recordings)

            print('Adding noise')
            np.random.seed(noise_seed)
            if noise_level > 0:
                if noise_mode == 'uncorrelated':
                    additive_noise = noise_level * np.random.randn(recordings.shape[0],
                                                                             recordings.shape[1])
                    recordings += additive_noise
                elif noise_mode == 'correlated-dist':
                    # TODO divide in chunks
                    cov_dist = np.zeros((n_elec, n_elec))
                    for i, el in enumerate(mea_pos):
                        for j, p in enumerate(mea_pos):
                            if i != j:
                                cov_dist[i, j] = (0.5*np.min(mea_pitch))/np.linalg.norm(el - p)
                            else:
                                cov_dist[i, j] = 1

                    additive_noise = np.random.multivariate_normal(np.zeros(n_elec), cov_dist,
                                                                   size=(recordings.shape[0], recordings.shape[1]))
                    recordings += additive_noise
                elif noise_level == 'experimental':
                    pass
                    #print( 'experimental noise model'
            else:
                print('Noise level is set to 0')

            if filter:
                print('Filtering signals')
                if fs/2. < cutoff[1]:
                    recordings = filter_analog_signals(recordings, freq=cutoff[0], fs=fs, filter_type='highpass')
                else:
                    recordings = filter_analog_signals(recordings, freq=cutoff, fs=fs)

            print('Extracting spike waveforms')
            extract_wf(spiketrains, recordings, times, fs)

            self.recordings = recordings
            self.times = times
            self.positions = mea_pos
            self.templates = templates
            self.spiketrains = spiketrains
            self.peaks = peak
            self.sources = gt_spikes
            self.info = params

def gen_recordings(params=None, templates=None, tempgen=None):
    '''

    Parameters
    ----------
    templates: str
        Path to generated templates
    params: dict OR str
        Dictionary containing recording parameters OR path to yaml file containing parameters

    Returns
    -------
    recgen: RecordingGenerator

    '''
    if isinstance(params, str):
        if os.path.isfile(params) and (params.endswith('yaml') or params.endswith('yml')):
            with open(params, 'r') as pf:
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
    spgen = SpikeTrainGenerator(params_dict['spiketrains'])
    spgen.generate_spikes()
    spiketrains = spgen.all_spiketrains

    params_dict['spiketrains'] = spgen.info
    # Generate recordings
    recgen = RecordingGenerator(spgen, tempgen, params_dict)

    return recgen


def gen_templates(cell_models_folder, params=None, templates_folder=None,
                  intraonly=False, parallel=True, delete_tmp=True):
    '''

    Parameters
    ----------
    tmp_params_path: str
        path to tmp params.yaml (same as templates_params with extra 'simulate_script', 'templates_folder' fields)
    intraonly: bool
        if True only intracellular simulation is run
    parallel:
        if True multi-threading is used
    delete_tmp:
        if True the temporary files are deleted

    Returns
    -------
    templates: np.array
        Array containing num_eaps x num_channels templates
    locations: np.array
        Array containing num_eaps x 3 (x,y,z) soma positions
    rotations: np.array
        Array containing num_eaps x 3 (x_rot, y_rot. z_rot) rotations
    celltypes: np.array
        Array containing num_eaps cell-types

    '''
    if isinstance(params, str):
        if os.path.isfile(params) and (params.endswith('yaml') or params.endswith('yml')):
            with open(params, 'r') as pf:
                params_dict = yaml.load(pf)
    elif isinstance(params, dict):
        params_dict = params
    else:
        params_dict=None

    if templates_folder is not None:
        if not os.path.isdir(templates_folder):
            os.makedirs(templates_folder)

    tempgen = TemplateGenerator(cell_models_folder=cell_models_folder,
                                params=params_dict,
                                templates_folder=templates_folder,
                                intraonly=intraonly,
                                parallel=parallel,
                                delete_tmp=delete_tmp)

    return tempgen