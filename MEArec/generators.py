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
from .tools import *
import threading
import shutil

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


class SpikeTrainGenerator:
    def __init__(self, params):
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

        self.params = copy(params)

        # Check quantities
        if 't_start' not in self.params.keys():
            self.params['t_start'] = 0 * pq.s
        else:
            self.params['t_start'] = self.params['t_start'] * pq.s
        if 'duration' in self.params.keys():
            self.params['t_stop'] = self.params['t_start'] + self.params['duration'] * pq.s
        if 'min_rate' not in self.params.keys():
            self.params['min_rate'] = 0.1 * pq.Hz
        else:
            self.params['min_rate'] = self.params['min_rate'] * pq.Hz
        if 'ref_per' not in self.params.keys():
            self.params['ref_per'] = 2 * pq.ms
        else:
            self.params['ref_per'] = self.params['ref_per'] * pq.ms
        if 'rates' in self.params.keys():  # all firing rates are provided
            self.params['rates'] = self.params['rates'] * pq.Hz
            self.n_neurons = len(self.params['rates'])
        else:
            rates = []
            types = []
            if 'f_exc' not in self.params.keys():
                self.params['f_exc'] = 5 * pq.Hz
            else:
                self.params['f_exc'] = self.params['f_exc'] * pq.Hz
            if 'f_inh' not in self.params.keys():
                self.params['f_inh'] = 15 * pq.Hz
            else:
                self.params['f_inh'] = self.params['f_inh'] * pq.Hz
            if 'st_exc' not in self.params.keys():
                self.params['st_exc'] = 1 * pq.Hz
            else:
                self.params['st_exc'] = self.params['st_exc'] * pq.Hz
            if 'st_inh' not in self.params.keys():
                self.params['st_inh'] = 3 * pq.Hz
            else:
                self.params['st_inh'] = self.params['st_inh'] * pq.Hz
            if 'n_exc' not in self.params.keys():
                self.params['n_exc'] = 15
            if 'n_inh' not in self.params.keys():
                self.params['n_inh'] = 5

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
                    st = stg.homogeneous_gamma_process(rate, self.params['rates'][n],
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
    def __init__(self, spiketrains, params):
        '''

        Parameters
        ----------
        template_folder
        spiketrain_folder
        params
        '''
        temp_params = params['templates']
        rec_params = params['recordings']
        st_params = params['spiketrains']
        celltype_params = params['cell_types']

        templates_folder = temp_params['templates_folder']
        eaps, locs, rots, celltypes, temp_info = load_templates(templates_folder)
        n_neurons = len(spiketrains)
        cut_outs = temp_info['params']['cut_out']

        if rec_params['fs'] is None:
            fs = 1. / temp_info['params']['dt'] * pq.kHz
        else:
            fs = rec_params['fs'] * pq.kHz
        noise_level = rec_params['noise_level']
        noise_mode = rec_params['noise_mode']
        duration = spiketrains[0].t_stop - spiketrains[0].t_start
        filter = rec_params['filter']
        cutoff = rec_params['cutoff'] * pq.Hz
        modulation = rec_params['modulation']
        chunk_duration = rec_params['chunk_duration']*pq.s
        mrand = rec_params['mrand']
        sdrand = rec_params['sdrand']
        overlap = rec_params['overlap']
        if rec_params['seed'] is not None:
            noise_seed = rec_params['seed']
        else:
            rec_params['seed'] = np.random.randint(1,1000)
            noise_seed = rec_params['seed']

        x_lim = temp_params['x_lim']
        y_lim = temp_params['y_lim']
        z_lim = temp_params['z_lim']
        min_amp = temp_params['min_amp']
        min_dist = temp_params['min_dist']
        overlap_threshold = temp_params['overlap_threshold']
        pad_len = temp_params['pad_len'] * pq.ms
        n_jitters = temp_params['n_jitters']
        upsample = temp_params['upsample']

        if temp_params['seed'] is not None:
            noise_seed = temp_params['seed']
        else:
            temp_params['seed'] = np.random.randint(1,1000)
            noise_seed = temp_params['seed']

        #TODO add spike synchrony and overlapping synchony
        #self.sync_rate = float(sync)
        #
        # self.save = save
        #
        # #print( 'Modulation: ', self.modulation
        #
        parallel=False
        #
        # all_categories = ['BP', 'BTC', 'ChC', 'DBC', 'LBC', 'MC', 'NBC',
        #                   'NGC', 'SBC', 'STPC', 'TTPC1', 'TTPC2', 'UTPC']

        exc_categories = celltype_params['excitatory']
        inh_categories = celltype_params['inhibitory']
        bin_cat = get_binary_cat(celltypes, exc_categories, inh_categories)

        # print(exc_categories, inh_categories, bin_cat)

        # load MEA info
        electrode_name = temp_info['electrodes']['electrode_name']
        elinfo = MEA.return_mea_info(electrode_name)
        offset = temp_info['params']['offset']
        elinfo['offset'] = offset
        mea = MEA.return_mea(info=elinfo)
        mea_pos = mea.positions

        n_elec = eaps.shape[1]
        # this is fixed from recordings
        spike_duration = np.sum(temp_info['params']['cut_out'])*pq.ms
        spike_fs = 1./temp_info['params']['dt']*pq.kHz

        print('Selecting cells')
        if 'type' in spiketrains[0].annotations.keys():
            n_exc = [st.annotations['type'] for st in spiketrains].count('E')
            n_inh = n_neurons - n_exc

        # TODO add y_lim and z_lim
        idxs_cells = select_templates(locs, eaps, bin_cat, n_exc, n_inh, x_lim=x_lim, min_amp=min_amp,
                                      min_dist=min_dist, verbose=True)
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
            print('electrode')
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

        params['recordings'].update({'electrode_name': electrode_name,
                                     'duration': float(duration.magnitude),
                                     'fs': float(fs.rescale('Hz').magnitude),
                                     'n_neurons': n_neurons})

        self.info = params


def gen_recordings(templates_folder, params):
    '''

    Parameters
    ----------
    templates_folder: str
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

    if 'recording_folder' in params_dict['recordings'].keys():
        recordings_folder = params_dict['recordings']['recordings_folder']
    else:
        print("'recording_folder' not specified: using cwd")
        recordings_folder = os.getcwd()

    if os.path.isdir(templates_folder):
        params_dict['templates'].update({'templates_folder': templates_folder})
    else:
        raise AttributeError("'templates_folder' is not a folder")

    if params_dict['spiketrains']['seed'] is None:
        params_dict['spiketrains']['seed'] = np.random.randint(1, 10000)

    if params_dict['templates']['seed'] is None:
        params_dict['templates']['seed'] = np.random.randint(1, 10000)

    if params_dict['recordings']['seed'] is None:
        params_dict['recordings']['seed'] = np.random.randint(1, 10000)

    # Generate spike trains
    spgen = SpikeTrainGenerator(params_dict['spiketrains'])
    spgen.generate_spikes()
    spiketrains = spgen.all_spiketrains

    # Generate recordings
    recgen = RecordingGenerator(spiketrains, params_dict)

    return recgen


def gen_templates(tmp_params_path, intraonly, parallel, delete_tmp=True):
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
    with open(tmp_params_path, 'r') as f:
        params = yaml.load(f)

    cell_models = params['cell_models']
    simulate_script = params['simulate_script']
    cell_models_folder = params['cell_models_folder']
    templates_folder = params['templates_folder']
    rot = params['rot']
    n = params['n']
    probe = params['probe']
    tot = len(cell_models)

    import pprint

    pprint.pprint(params)

    # Simulate neurons and EAP for different cell models sparately
    if parallel:
        start_time = time.time()
        print('Parallel')
        tot = len(cell_models)
        threads = []
        for numb, cell_model in enumerate(cell_models):
            threads.append(simulationThread(numb, "Thread-"+str(numb), simulate_script,
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
            os.system('python %s %s %s %s'\
                      % (simulate_script, join(cell_models_folder, cell_model), intraonly, tmp_params_path))
        print('\n\n\nSimulation time: ', time.time() - start_time, '\n\n\n')

    if os.path.isfile(tmp_params_path):
        os.remove(tmp_params_path)

    tmp_folder = join(templates_folder, rot, 'tmp_%d_%s' % (n, probe))
    templates, locations, rotations, celltypes = load_tmp_eap(tmp_folder)
    if delete_tmp:
        shutil.rmtree(tmp_folder)

    return templates, locations, rotations, celltypes