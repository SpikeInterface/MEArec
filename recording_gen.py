from __future__ import print_function, division

'''
Generate recordings from biophysically detailed simulations of EAPs
'''
import numpy as np
import os, sys
from os.path import join
import matplotlib.pyplot as plt
import elephant
import scipy.signal as ss
import scipy.stats as stats
import quantities as pq
import yaml
import time
import multiprocessing
from copy import copy

import MEAutility as MEA
import click
from tools import *

root_folder = os.getcwd()

#TODO add bursting events

class RecordingGenerator:
    def __init__(self, template_folder, spiketrain_folder, params):
        # save=False, spike_folder=None, fs=None, noise_mode=None, n_neurons=None, p_exc=None,
        #          bound_x=[], min_amp=None, noise_level=None, duration=None, f_exc=None, f_inh=None,
        #          filter=True, over=None, sync=None, modulation='none', min_dist=None, plot_figures=True,
        #          seed=2904):
        '''

        Parameters
        ----------
        save: save flag (True-False)
        spike_folder: folder containing spikes or CNN model with validation_data folder
        fs: sampling frequency (if None taken from spikes)
        noise_mode: noise generation (uncorrelated: independent gaussian noise - correlated-dist: noise correlated with distance)
        n_neurons: number of cells
        p_exc: percent of excitatory cells
        bound_x: boundaries for x direction in um (e.g. [10 60])
        min_amp: minimum amplitude of templates in uV
        noise_level: rms noise level
        duration: duration in s
        f_exc: average frequency of excitatory cells
        f_inh: average frequency of inhibtory cells
        filter: filter or not (True-False)
        over: threshold to consider 2 templates dpatially overlapping (e.g. 0.6)
        sync: rate of added synchrony on overlapping spikes
        modulation: modulation type (none - noise-all (electrodes modulated separately) - noise (templates modulated separately))
        min_dist: minimum distance between cells in um
        plot_figures: plot figures or not
        seed: random seed to select cells
        '''

        eaps, locs, rots, celltypes, temp_info = load_templates(template_folder)
        spiketrains, spike_info = load_spiketrains(spiketrain_folder)
        n_neurons = len(spiketrains)

        seed = params['seed']
        np.random.seed(seed)
        chunk_duration = params['chunk_duration']*pq.s

        fs = params['fs'] * pq.kHz
        depth_lim = params['depth_lim']
        min_amp = params['min_amp']
        min_dist = params['min_dist']
        noise_level = params['noise_level']
        noise_mode = params['noise_mode']
        duration = spiketrains[0].t_stop - spiketrains[0].t_start
        filter = params['filter']
        cutoff = params['cutoff'] * pq.Hz
        overlap_threshold = params['overlap_threshold']
        modulation = params['modulation']

        #TODO add spike synchrony and overlapping synchony
        #self.sync_rate = float(sync)
        #
        # self.save = save
        #
        # #print( 'Modulation: ', self.modulation
        #
        # parallel=False
        #
        # all_categories = ['BP', 'BTC', 'ChC', 'DBC', 'LBC', 'MC', 'NBC',
        #                   'NGC', 'SBC', 'STPC', 'TTPC1', 'TTPC2', 'UTPC']

        exc_categories = params['excitatory']
        inh_categories = params['inhibitory']
        bin_cat = get_binary_cat(celltypes, exc_categories, inh_categories)

        # print(exc_categories, inh_categories, bin_cat)

        # load MEA info
        electrode_name = temp_info['Electrodes']['electrode_name']
        elinfo = MEA.return_mea_info(electrode_name)
        x_plane = temp_info['Params']['xplane']
        mea_pos, mea_dim, mea_pitch = MEA.return_mea(electrode_name)

        n_elec = eaps.shape[1]
        # this is fixed from recordings
        spike_duration = np.sum(temp_info['Params']['cut_out'])*pq.ms
        spike_fs = 1./temp_info['General']['dt']*pq.kHz

        print('Selecting cells')
        if 'type' in spiketrains[0].annotations.keys():
            n_exc = [st.annotations['type'] for st in spiketrains].count('exc')
            n_inh = n_neurons - n_exc
            # print(n_exc, n_inh)
        #print( n_exc, ' excitatory and ', n_inh, ' inhibitory'

        idxs_cells = select_cells(locs, eaps, bin_cat, n_exc, n_inh, bound_x=depth_lim, min_amp=min_amp,
                                  min_dist=min_dist)
        template_celltypes = celltypes[idxs_cells]
        template_locs = locs[idxs_cells]
        templates_bin = bin_cat[idxs_cells]
        templates = eaps[idxs_cells]

        # print(templates.shape, templates_bin, template_celltypes, template_locs)

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
        pad_len = params['pad_len'] * pq.ms
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
        n_jitters = params['n_jitters']
        upsample = params['upsample']
        jitter = 1. / fs
        templates_jitter = []
        for temp in templates_pad:
            temp_up = ss.resample_poly(temp, upsample, 1., axis=1)
            nsamples_up = temp_up.shape[1]
            temp_jitt = []
            for n in range(n_jitters):
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
            templates_jitter.append(temp_jitt)
        templates = np.array(templates_jitter)

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
        print('Finding temporally overlapping spikes')
        overlapping = find_overlapping_spikes(templates, thresh=overlap_threshold)
        # print(overlapping)
        # annotate_overlapping(spiketrains, overlapping_pairs=overlapping, verbose=True)


        amp_mod = []
        cons_spikes = []
        mrand = params['mrand']
        sdrand = params['sdrand']
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
                            rec_chunk += convolve_templates_spiketrains(st, spike_bin, templates[st])
                        else:
                            rec_chunk += convolve_templates_spiketrains(st, spike_bin, templates[st],
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

                    recordings += convolve_templates_spiketrains(st, spike_bin, templates[st])
                    np.random.seed(seed)
                    gt_spikes.append(convolve_single_template(st, spike_bin,
                                                              templates[st, :, np.argmax(peak[st])]))
                elif modulation == 'electrode':
                    seed = np.random.randint(10000)
                    np.random.seed(seed)
                    recordings += convolve_templates_spiketrains(st, spike_bin, templates[st],
                                                                      modulation=True,
                                                                      amp_mod=amp_mod[st])
                    np.random.seed(seed)
                    gt_spikes.append(convolve_single_template(st, spike_bin,
                                                              templates[st, :, np.argmax(peak[st])],
                                                              modulation=True,
                                                              amp_mod=amp_mod[st][:,
                                                                      np.argmax(peak[st])]))
                elif modulation == 'template' or modulation == 'all':
                    seed = np.random.randint(10000)
                    np.random.seed(seed)
                    recordings += convolve_templates_spiketrains(st, spike_bin, templates[st],
                                                                      modulation=True,
                                                                      amp_mod=amp_mod[st])
                    np.random.seed(seed)
                    gt_spikes.append(convolve_single_template(st, spike_bin,
                                                              templates[st, :, np.argmax(peak[st])],
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

        ''' save meta data in old fashioned format'''
        self.recordings = recordings
        self.templates = templates
        self.spiketrains = spiketrains
        self.peaks = peak
        self.sources = gt_spikes

        general = {'spiketrain_folder': str(spiketrain_folder), 'template_folder': str(template_folder),
                   'n_neurons': n_neurons, 'electrode_name': str(electrode_name),'fs': float(fs.magnitude),
                   'duration': float(duration.magnitude), 'seed': seed}

        templates = {'pad_len': [float(pl.magnitude) for pl in pad_len], 'depth_lim': depth_lim,
                     'min_amp': min_amp, 'min_dist': min_dist}


        # synchrony = {'overlap_threshold': self.overlap_threshold,
        #              'overlap_pairs_str': str([list(ov) for ov in self.overlapping]),
        #              'overlap_pairs': self.overlapping,
        #              'sync_rate': self.sync_rate}

        modulation = {'modulation': modulation,'mrand': mrand, 'sdrand': sdrand}

        noise = {'noise_mode': noise_mode, 'noise_level': noise_level}

        if filter:
            filter = {'filter': filter, 'cutoff': [float(co.magnitude) for co in cutoff]}
        else:
            filter = {'filter': filter}

        # create dictionary for yaml file
        info = {'General': general, 'Templates': templates, 'Modulation': modulation,
                'Filter': filter, 'Noise': noise}

        self.info = info

        # self.rec_path = join(rec_dir, self.rec_name)
        # os.makedirs(self.rec_path)
        # # Save meta_info yaml
        #
        # print('Saved ', self.rec_path)

@click.command()
@click.option('--templates', '-t', default=None,
              help='eap templates path')
@click.option('--spiketrains', '-st', default=None,
              help='spike trains path')
@click.option('--params', '-prm', default=None,
              help='path to params.yaml (otherwise default params are used and some of the parameters can be overwritten with the following options)')
@click.option('--default', is_flag=True,
              help='shows default values for simulation')
@click.option('--fname', '-fn', default=None,
              help='recording filename')
@click.option('--fs', default=None, type=float,
              help='sampling frequency in kHz (default from templates sampling frequency)')
@click.option('--min-dist', '-md', default=None, type=int,
              help='minumum distance between neuron in um (default=25)')
@click.option('--noise-lev', '-nl', default=None, type=int,
              help='noise level in uV (default=10)')
@click.option('--modulation', '-m', default=None, type=click.Choice(['none', 'template', 'electrode']),
              help='modulation type')
@click.option('--chunk', '-ch', default=None, type=float,
              help='chunk duration in s for chunk processing (default 0)')
@click.option('--seed', '-s', default=None, type=int,
              help='random seed (default randint(1,1000))')
@click.option('--no-filt', is_flag=True,
              help='if True no filter is applied')
def run(params, **kwargs):
    """Generates recordings from TEMPLATES and SPIKETRAINS"""
    # Retrieve params file
    if params is None:
        with open(join('params/recording_params.yaml'), 'r') as pf:
            params_dict = yaml.load(pf)
    else:
        with open(params, 'r') as pf:
            params_dict = yaml.load(pf)

    if kwargs['default'] is True:
        print(params_dict)
        return

    recording_folder = params_dict['recording_folder']

    if kwargs['templates'] is None or kwargs['spiketrains'] is None:
        print('Provide eap templates and spiketrains paths')
        return
    else:
        template_folder = kwargs['templates']
        spiketrain_folder = kwargs['spiketrains']
        templates, locs, rots, celltypes, temp_info = load_templates(template_folder)
        spiketrains, spike_info = load_spiketrains(spiketrain_folder)
        print('Number of templates: ', len(templates))
        print('Number of spike trains: ', len(spiketrains))

    if kwargs['min_dist'] is not None:
        params_dict['min_dist'] = kwargs['min_dist']
    if kwargs['noise_lev'] is not None:
        params_dict['noise_lev'] = kwargs['noise_lev']
    if kwargs['modulation'] is not None:
        params_dict['modulation'] = kwargs['modulation']
    if kwargs['chunk'] is not None:
        params_dict['chunk_duration'] = kwargs['chunk']
    if kwargs['no_filt'] is True:
        params_dict['filter'] = False
    if kwargs['fs'] is not None:
        params_dict['fs'] = kwargs['fs']
    else:
        params_dict['fs'] = 1. / temp_info['General']['dt']
    if kwargs['seed'] is not None:
        params_dict['seed'] = kwargs['seed']
    else:
        params_dict['seed'] = np.random.randint(1, 10000)

    recgen = RecordingGenerator(template_folder, spiketrain_folder, params_dict)
    info = recgen.info

    n_neurons = info['General']['n_neurons']
    electrode_name = info['General']['electrode_name']
    duration = info['General']['duration'] # remove s
    noise_level = info['Noise']['noise_level']

    if kwargs['fname'] is None:
        fname = 'recordings_%dcells_%s_%s_%.1fuV_%s' % (n_neurons, electrode_name, duration,
                                                       noise_level, time.strftime("%d-%m-%Y:%H:%M"))
    else:
        fname = kwargs['fname']

    rec_path = join(recording_folder, fname)
    if not os.path.isdir(rec_path):
        os.makedirs(rec_path)

    np.save(join(rec_path, 'recordings'), recgen.recordings)
    np.save(join(rec_path, 'spiketrains'), recgen.spiketrains)
    np.save(join(rec_path, 'peak'), recgen.peaks)
    np.save(join(rec_path, 'sources'), recgen.sources)
    np.save(join(rec_path, 'templates'), recgen.templates)

    with open(join(rec_path, 'info.yaml'), 'w') as f:
        yaml.dump(info, f, default_flow_style=False)

if __name__ == '__main__':
    run()
