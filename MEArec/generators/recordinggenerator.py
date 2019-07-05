import numpy as np
import scipy.signal as ss
import time
from copy import copy, deepcopy
from MEArec.tools import *
import MEAutility as mu
import yaml
import os
from pprint import pprint
import quantities as pq
from distutils.version import StrictVersion
import tempfile
from MEArec.generators import SpikeTrainGenerator

if StrictVersion(yaml.__version__) >= StrictVersion('5.0.0'):
    use_loader = True
else:
    use_loader = False


class RecordingGenerator:
    """
    Class for generation of recordings called by the gen_recordings function.
    The list of parameters is in default_params/recordings_params.yaml.
    """

    def __init__(self, spgen=None, tempgen=None, params=None, rec_dict=None, info=None, tmp_h5=True, verbose=True):
        self._verbose = verbose
        self._tmp_h5 = tmp_h5
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
            self.info = info
            self.params = deepcopy(info)
            if len(self.spiketrains) > 0:
                self.spgen = SpikeTrainGenerator(spiketrains=self.spiketrains, params=self.info['spiketrains'])
            self.tempgen = None
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

    def generate_recordings(self):
        """
        Generate recordings.
        """
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
                if 'bursting_sigmoid' not in rec_params.keys():
                    params['recordings']['bursting_sigmoid'] = 30
                bursting_sigmoid = params['recordings']['bursting_sigmoid']
                if self._verbose:
                    print('Bursting with modulation sigmoid: ', bursting_sigmoid)
            else:
                bursting_sigmoid = None
        else:
            exp_decay = None
            n_burst_spikes = None
            max_burst_duration = None
            bursting_sigmoid = None
            n_bursting = None

        if 'chunk_noise_duration' not in rec_params.keys():
            params['recordings']['chunk_noise_duration'] = 0
        chunk_noise_duration = params['recordings']['chunk_noise_duration'] * pq.s

        if 'chunk_filter_duration' not in rec_params.keys():
            params['recordings']['chunk_filter_duration'] = 0
        chunk_filter_duration = params['recordings']['chunk_filter_duration'] * pq.s

        if 'chunk_conv_duration' not in rec_params.keys():
            params['recordings']['chunk_conv_duration'] = 0
        chunk_conv_duration = params['recordings']['chunk_conv_duration'] * pq.s

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
            drift_velocity = rec_params['drift_velocity']
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
            t_start_drift = None
            n_drifting = None

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

        if not only_noise:
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
                        print('Drift velocity vector: ', velocity_vector)
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

                idxs_cells = np.array(idxs_cells)[np.argsort(selected_cat)]
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
                        feat = get_templates_features(tem, ['na'], dt=dt)
                    else:
                        feat = get_templates_features(tem[0], ['na'], dt=dt)
                    voltage_peaks.append(-np.squeeze(feat['na']))
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
                voltage_peaks = self.voltage_peaks
                overlapping = np.array([])
                if not drifting:
                    velocity_vector = None
                else:
                    drift_directions = np.array([(p[-1] - p[0]) / np.linalg.norm(p[-1] - p[0]) for p in template_locs])
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
            if tempgen is not None:
                for i, st in enumerate(spiketrains):
                    st.annotate(bintype=templates_bin[i], mtype=template_celltypes[i], soma_position=template_locs[i])

            if overlap:
                annotate_overlapping_spikes(spiketrains, overlapping_pairs=overlapping, verbose=True)

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
            chunks_rec = []
            if duration > chunk_conv_duration and chunk_conv_duration != 0:
                if self._verbose:
                    print('Splitting in chunks of', chunk_conv_duration, 's')
                start = 0 * pq.s
                finished = False
                while not finished:
                    chunks_rec.append([start, start + chunk_conv_duration])
                    start = start + chunk_conv_duration
                    if start >= duration:
                        finished = True

            if self._tmp_h5:
                temp_dir = Path(tempfile.mkdtemp())
                tmp_rec_path = temp_dir / "mearec_tmp_file.h5"
                tmp_rec = h5py.File(tmp_rec_path)
                recordings = tmp_rec.create_dataset("recordings", (n_elec, n_samples), dtype=dtype)
                spike_traces = tmp_rec.create_dataset("spike_traces", (n_neurons, n_samples), dtype=dtype)
                # timestamps = tmp_rec.create_dataset("timestamps", data=np.arange(recordings.shape[1]) / fs)
            else:
                tmp_rec = None
                tmp_rec_path = None
                temp_dir = None
                recordings = np.zeros((n_elec, n_samples))
                spike_traces = np.zeros((n_neurons, n_samples))
            timestamps = np.arange(recordings.shape[1]) / fs
            self._tmp_rec = tmp_rec
            self._tmp_rec_path = tmp_rec_path
            self._temp_dir = temp_dir

            if len(chunks_rec) > 0:
                import multiprocessing
                threads = []
                manager = multiprocessing.Manager()
                output_dict = manager.dict()
                tempfiles = dict()
                for ch, chunk in enumerate(chunks_rec):
                    if self._verbose:
                        print('Convolving in: ', chunk[0], chunk[1], ' chunk')
                    idxs = np.where((timestamps >= chunk[0]) & (timestamps < chunk[1]))[0]
                    if self._tmp_h5:
                        tempfiles[ch] = self._temp_dir / ('rec_' + str(ch))
                    else:
                        tempfiles[ch] = None
                    p = multiprocessing.Process(target=chunk_convolution, args=(ch, idxs,
                                                                                output_dict, spike_matrix,
                                                                                modulation, drifting,
                                                                                drifting_units, templates,
                                                                                cut_outs_samples,
                                                                                template_locs, velocity_vector,
                                                                                t_start_drift, fs, self._verbose,
                                                                                amp_mod, bursting_units, shape_mod,
                                                                                bursting_sigmoid, chunk[0], True,
                                                                                voltage_peaks, tempfiles[ch],))
                    p.start()
                    threads.append(p)
                for p in threads:
                    p.join()
                # retrieve annotated spiketrains
                for ch, chunk in enumerate(chunks_rec):
                    if self._verbose:
                        print('Extracting data from chunk', ch + 1, 'out of', len(chunks_rec))
                    if not self._tmp_h5:
                        rec = output_dict[ch]['rec']
                        spike_trace = output_dict[ch]['spike_traces']
                        idxs = output_dict[ch]['idxs']
                        recordings[:, idxs] += rec
                        spike_traces[:, idxs] = spike_trace
                    else:
                        idxs = output_dict[ch]['idxs']
                        tmp_ch_file = h5py.File(tempfiles[ch], 'r')
                        recordings[:, idxs] = tmp_ch_file['recordings']
                        spike_traces[:, idxs] = tmp_ch_file['spike_traces']
                        os.remove(tempfiles[ch])
                    if ch == len(chunks_rec) - 1 and drifting:
                        for st in np.arange(n_neurons):
                            final_locs = output_dict[ch]['final_locs']
                            final_idxs = output_dict[ch]['final_idxs']
                            if st in drifting_units:
                                spiketrains[st].annotate(drifting=True)
                                spiketrains[st].annotate(initial_soma_position=template_locs[st, 0])
                                spiketrains[st].annotate(final_soma_position=final_locs[st])
                                spiketrains[st].annotate(final_idx=final_idxs[st])
            else:
                # convolve in single chunk
                output_dict = dict()
                idxs = np.arange(spike_matrix.shape[1])
                ch = 0
                # reorder this
                chunk_convolution(ch=ch, idxs=idxs, output_dict=output_dict, spike_matrix=spike_matrix,
                                  modulation=modulation, drifting=drifting, drifting_units=drifting_units,
                                  templates=templates, cut_outs_samples=cut_outs_samples, template_locs=template_locs,
                                  velocity_vector=velocity_vector, t_start_drift=t_start_drift, fs=fs, amp_mod=amp_mod,
                                  bursting_units=bursting_units, shape_mod=shape_mod, bursting_sigmoid=bursting_sigmoid,
                                  chunk_start=0 * pq.s, extract_spike_traces=True, voltage_peaks=voltage_peaks,
                                  tmp_mearec_file=tmp_rec, verbose=self._verbose)
                if not self._tmp_h5:
                    recordings = output_dict[ch]['rec']
                    spike_traces = output_dict[ch]['spike_traces']
                for st in np.arange(n_neurons):
                    if drifting and st in drifting_units:
                        final_locs = output_dict[ch]['final_locs']
                        final_idxs = output_dict[ch]['final_idxs']
                        spiketrains[st].annotate(drifting=True)
                        spiketrains[st].annotate(initial_soma_position=template_locs[st, 0])
                        spiketrains[st].annotate(final_soma_position=final_locs[st])
                        spiketrains[st].annotate(final_idx=final_idxs[st])
            if drifting:
                templates_drift = np.zeros((templates.shape[0], np.max(final_idxs) + 1,
                                            templates.shape[2], templates.shape[3],
                                            templates.shape[4]))
                for i, st in enumerate(spiketrains):
                    if i in drifting_units:
                        if final_idxs[i] == np.max(final_idxs):
                            templates_drift[i] = templates[i, :(final_idxs[i] + 1)]
                        else:
                            templates_drift[i] = np.vstack((templates[i, :(final_idxs[i] + 1)],
                                                            np.array([templates[i, final_idxs[i]]]
                                                                     * (np.max(final_idxs) - final_idxs[i]))))
                    else:
                        templates_drift[i] = np.array([templates[i, 0]] * (np.max(final_idxs) + 1))
                templates = templates_drift
        else:
            recordings = np.zeros((n_elec, n_samples))
            timestamps = np.arange(recordings.shape[1]) / fs
            spiketrains = np.array([])
            voltage_peaks = np.array([])
            spike_traces = np.array([])
            templates = np.array([])
            template_locs = np.array([])
            template_rots = np.array([])
            template_celltypes = np.array([])
            overlapping = np.array([])
            tmp_rec = None
            self._tmp_rec = None

        if self._verbose:
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

        if self._verbose:
            print('Noise seed: ', noise_seed)
        np.random.seed(noise_seed)
        if noise_level > 0:
            if noise_mode == 'uncorrelated':
                if len(chunks_noise) > 0:
                    for ch, chunk in enumerate(chunks_noise):
                        if self._verbose:
                            print('Generating noise in: ', chunk[0], chunk[1], ' chunk')
                        idxs = np.where((timestamps >= chunk[0]) & (timestamps < chunk[1]))[0]
                        additive_noise = noise_level * np.random.randn(recordings.shape[0],
                                                                       len(idxs)).astype(dtype)
                        if noise_color:
                            if self._verbose:
                                print('Coloring noise with peak: ', color_peak, ' quality factor: ', color_q,
                                      ' and random noise level: ', color_noise_floor)
                            # iir peak filter
                            b_iir, a_iir = ss.iirpeak(color_peak, Q=color_q, fs=fs.rescale('Hz').magnitude)
                            additive_noise = ss.filtfilt(b_iir, a_iir, additive_noise, axis=1, padlen=1000)
                            additive_noise = additive_noise + color_noise_floor * np.std(additive_noise) * \
                                             np.random.randn(additive_noise.shape[0], additive_noise.shape[1])
                            additive_noise = additive_noise * (noise_level / np.std(additive_noise))
                        if tmp_rec is not None:
                            recordings[...][:, idxs] += additive_noise
                        else:
                            recordings[:, idxs] += additive_noise
                else:
                    additive_noise = noise_level * np.random.randn(recordings.shape[0],
                                                                   recordings.shape[1]).astype(dtype)
                    if noise_color:
                        if self._verbose:
                            print('Coloring noise with peak: ', color_peak, ' quality factor: ', color_q,
                                  ' and random noise level: ', color_noise_floor)
                        # iir peak filter
                        b_iir, a_iir = ss.iirpeak(color_peak, Q=color_q, fs=fs.rescale('Hz').magnitude)
                        additive_noise = ss.filtfilt(b_iir, a_iir, additive_noise, axis=1)
                        additive_noise = additive_noise + color_noise_floor * np.std(additive_noise) \
                                         * np.random.randn(additive_noise.shape[0], additive_noise.shape[1])
                        additive_noise = additive_noise * (noise_level / np.std(additive_noise))
                    if tmp_rec is not None:
                        recordings[...] += additive_noise
                    else:
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
                        if self._verbose:
                            print('Generating noise in: ', chunk[0], chunk[1], ' chunk')
                        idxs = np.where((timestamps >= chunk[0]) & (timestamps < chunk[1]))[0]
                        additive_noise = noise_level * np.random.multivariate_normal(np.zeros(n_elec), cov_dist,
                                                                                     size=(len(idxs))).astype(dtype).T
                        if noise_color:
                            if self._verbose:
                                print('Coloring noise with peak: ', color_peak, ' quality factor: ', color_q,
                                      ' and random noise level: ', color_noise_floor)
                            # iir peak filter
                            b_iir, a_iir = ss.iirpeak(color_peak, Q=color_q, fs=fs.rescale('Hz').magnitude)
                            additive_noise = ss.filtfilt(b_iir, a_iir, additive_noise, axis=1)
                            additive_noise = additive_noise + color_noise_floor * np.std(additive_noise) * \
                                             np.random.multivariate_normal(np.zeros(n_elec), cov_dist,
                                                                           size=(len(idxs))).T
                        additive_noise = additive_noise * (noise_level / np.std(additive_noise))
                        if tmp_rec is not None:
                            recordings[...][:, idxs] += additive_noise
                        else:
                            recordings[:, idxs] += additive_noise
                else:
                    additive_noise = noise_level * np.random.multivariate_normal(np.zeros(n_elec), cov_dist,
                                                                                 size=recordings.shape[1]). \
                        astype(dtype).T
                    if noise_color:
                        if self._verbose:
                            print('Coloring noise with peak: ', color_peak, ' quality factor: ', color_q,
                                  ' and random noise level: ', color_noise_floor)
                        # iir peak filter
                        b_iir, a_iir = ss.iirpeak(color_peak, Q=color_q, fs=fs.rescale('Hz').magnitude)
                        additive_noise = ss.filtfilt(b_iir, a_iir, additive_noise, axis=1)
                        additive_noise = additive_noise + color_noise_floor * np.std(additive_noise) * \
                                         np.random.multivariate_normal(np.zeros(n_elec), cov_dist,
                                                                       size=recordings.shape[1]).T
                    additive_noise = additive_noise * (noise_level / np.std(additive_noise))

                    if tmp_rec is not None:
                        recordings[...] = recordings + additive_noise
                    else:
                        recordings += additive_noise

            elif noise_mode == 'far-neurons':
                idxs_cells, selected_cat = select_templates(locs, eaps, bin_cat=None, n_exc=far_neurons_n, n_inh=0,
                                                            x_lim=x_lim, y_lim=y_lim, z_lim=z_lim, min_amp=0,
                                                            max_amp=far_neurons_max_amp, min_dist=1,
                                                            verbose=False)
                templates_noise = eaps[np.array(idxs_cells)]
                template_noise_locs = locs[np.array(idxs_cells)]
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

                chunks_rec = []
                if duration > chunk_conv_duration and chunk_conv_duration != 0:
                    if self._verbose:
                        print('Splitting in chunks of', chunk_conv_duration, 's')
                    start = 0 * pq.s
                    finished = False
                    while not finished:
                        chunks_rec.append([start, start + chunk_conv_duration])
                        start = start + chunk_conv_duration
                        if start >= duration:
                            finished = True

                additive_noise = np.zeros((n_elec, n_samples))
                # for st, spike_bin in enumerate(spike_matrix):
                if len(chunks_rec) > 0:
                    import multiprocessing
                    threads = []
                    manager = multiprocessing.Manager()
                    output_dict = manager.dict()
                    for ch, chunk in enumerate(chunks_rec):
                        if self._verbose:
                            print('Convolving in: ', chunk[0], chunk[1], ' chunk')
                        idxs = np.where((timestamps >= chunk[0]) & (timestamps < chunk[1]))[0]
                        p = multiprocessing.Process(target=chunk_convolution, args=(ch, idxs,
                                                                                    output_dict, spike_matrix_noise,
                                                                                    'none', False,
                                                                                    None, templates_noise,
                                                                                    cut_outs_samples,
                                                                                    template_noise_locs, None,
                                                                                    None, None, self._verbose,
                                                                                    None, None, False,
                                                                                    None, chunk[0], False,
                                                                                    voltage_peaks))
                        p.start()
                        threads.append(p)
                    for p in threads:
                        p.join()
                    # retrieve annotated spiketrains
                    if not self._tmp_h5:
                        for ch, chunk in enumerate(chunks_rec):
                            rec = output_dict[ch]['rec']
                            idxs = output_dict[ch]['idxs']
                            additive_noise[:, idxs] += rec
                else:
                    # convolve in single chunk
                    output_dict = dict()
                    idxs = np.arange(spike_matrix_noise.shape[1])
                    ch = 0
                    # reorder this
                    chunk_convolution(ch, idxs, output_dict, spike_matrix_noise, 'none', False, None,
                                      templates_noise, cut_outs_samples, template_noise_locs, None, None, None,
                                      self._verbose, None, None, False, None, 0 * pq.s, False, voltage_peaks)
                    additive_noise = output_dict[ch]['rec']

                # remove mean
                for i, m in enumerate(np.mean(additive_noise, axis=1)):
                    additive_noise[i] -= m

                # adding noise floor
                additive_noise += far_neurons_noise_floor * np.std(additive_noise) * \
                                  np.random.randn(additive_noise.shape[0], additive_noise.shape[1])

                noise_scale = noise_level / np.std(additive_noise)
                if self._verbose:
                    print('Scaling to reach desired level by: ', noise_scale)
                additive_noise *= noise_scale
                if tmp_rec is not None:
                    recordings[...] = recordings + additive_noise
                else:
                    recordings += additive_noise
        else:
            if self._verbose:
                print('Noise level is set to 0')

        if filter:
            if self._verbose:
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
                    if self._verbose:
                        print('Filtering in: ', chunk[0], chunk[1], ' chunk')
                    idxs = np.where((timestamps >= chunk[0]) & (timestamps < chunk[1]))[0]
                    if not self._tmp_h5:
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
                            recordings[...][:, idxs] = filter_analog_signals(recordings[:, idxs], freq=cutoff, fs=fs,
                                                                             filter_type='highpass', order=order)
                        elif cutoff.size == 2:
                            if fs / 2. < cutoff[1]:
                                recordings[...][:, idxs] = filter_analog_signals(recordings[:, idxs], freq=cutoff[0],
                                                                                 fs=fs, filter_type='highpass',
                                                                                 order=order)
                            else:
                                recordings[...][:, idxs] = filter_analog_signals(recordings[:, idxs], freq=cutoff,
                                                                                 fs=fs)
            else:
                if not self._tmp_h5:
                    if cutoff.size == 1:
                        recordings = filter_analog_signals(recordings, freq=cutoff, fs=fs, filter_type='highpass',
                                                           order=order)
                    elif cutoff.size == 2:
                        if fs / 2. < cutoff[1]:
                            recordings = filter_analog_signals(recordings, freq=cutoff[0], fs=fs,
                                                               filter_type='highpass', order=order)
                        else:
                            recordings = filter_analog_signals(recordings, freq=cutoff, fs=fs, order=order)
                else:
                    if cutoff.size == 1:
                        recordings[...] = filter_analog_signals(recordings, freq=cutoff, fs=fs, filter_type='highpass',
                                                                order=order)
                    elif cutoff.size == 2:
                        if fs / 2. < cutoff[1]:
                            recordings[...] = filter_analog_signals(recordings, freq=cutoff[0], fs=fs,
                                                                    filter_type='highpass', order=order)
                        else:
                            recordings[...] = filter_analog_signals(recordings, freq=cutoff, fs=fs, order=order)

        if not only_noise:
            if extract_waveforms:
                if self._verbose:
                    print('Extracting spike waveforms')
                extract_wf(spiketrains, recordings, fs=fs, timestamps=timestamps)

        # TODO save to tmp_h5 all
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
        Extract waveforms from spike trains.

        Parameters
        ----------
        cut_out : float or list
            Ms before and after peak to cut out. If float the cut is symmetric.
        """
        fs = self.info['recordings']['fs'] * pq.Hz
        extract_wf(self.spiketrains, self.recordings, fs=fs, cut_out=cut_out)