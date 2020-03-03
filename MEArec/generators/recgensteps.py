"""
This module group several sub function for recording generator.

All this function work on chunk of signals.
They can be call in loop mode or with joblib.

Important:

When tmp_mode=='memmap' : theses functions must assign and add directly the buffer.
When tmp_mode is Noe : theses functions return the buffer and the assignament is done externally.



"""
import h5py
import numpy as np
import scipy.signal

from MEArec.tools import (filter_analog_signals, convolve_templates_spiketrains,
                          convolve_single_template, convolve_drifting_templates_spiketrains)


class FuncThenAddChunk:
    """
    Helper for functions that do chunk to assign one or several chunks at the 
    good place.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kargs):
        return_dict = self.func(*args)

        ch, i_start, i_stop, = args[:3]

        assignment_dict = kargs['assignment_dict']
        tmp_mode = kargs['tmp_mode']

        if tmp_mode is None:
            pass
        elif tmp_mode == 'memmap':
            for key, full_arr in assignment_dict.items():
                out_chunk = return_dict.pop(key)
                full_arr[i_start:i_stop] += out_chunk

        return return_dict


def chunk_convolution_(ch, i_start, i_stop, chunk_start,
                       st_idxs, pad_samples, modulation, drifting, drift_mode, drifting_units, templates,
                       cut_outs_samples, template_locs, velocity_vector, fast_drift_period, fast_drift_min_jump,
                       fast_drift_max_jump, t_start_drift, fs, verbose, amp_mod, bursting_units, shape_mod,
                       shape_stretch, extract_spike_traces, voltage_peaks, dtype, seed_list):
    """
    Perform full convolution for all spike trains by chunk.

    Parameters
    ----------
    ch: int
        Chunk id
    i_start: int
        first index of chunk
    i_stop: 
        last index of chunk (exclude)
    chunk_start: quantity
        Start time for current chunk
    st_idxs: list
        list of spike indexes for each spike train
    pad_samples: int
        Number of samples for padding
    modulation: str
        Modulation type
    drifting: bool
        If True drifting is performed
    drift_mode :  str
        Drift mode ['slow' | 'fast' | 'slow+fast']
    drifting_units: list
        List of drifting units (if None all units are drifted)
    templates: np.array
        Templates
    cut_outs_samples: list
        List with number of samples to cut out before and after spike peaks
    template_locs: np.array
        For drifting, array with drifting locations
    velocity_vector: np.array
        For drifting, drifring direction
    fast_drift_period : Quantity
        Periods between fast drifts
    fast_drift_min_jump : float
        Min 'jump' in um for fast drifts
    fast_drift_max_jump : float
        Max 'jump' in um for fast drifts
    t_start_drift: quantity
        For drifting, start drift time
    fs: quantity
        Sampling frequency
    verbose: bool
        If True output is verbose
    amp_mod: np.array
        Array with modulation values
    bursting_units : list
        List of bursting units
    shape_mod: bool
        If True waveforms are modulated in shape
    shape_stretch: float
        Low and high frequency for bursting
    extract_spike_traces: bool
        If True (default), spike traces are extracted
    voltage_peaks: np.array
        Array containing the voltage values at the peak
    
    """
    if verbose:
        print('Start convolutions for chunk', ch)
    # set seed
    np.random.seed(seed_list[ch])

    # checking boundaries not needed here because the recordings are created on the fly
    i_start_pad = i_start - pad_samples
    i_stop_pad = i_stop + pad_samples
    n_samples = i_stop_pad - i_start_pad

    template_idxs = []
    if extract_spike_traces:
        spike_traces = np.zeros((n_samples, len(st_idxs)), dtype=dtype)
    if len(templates.shape) == 4:
        n_elec = templates.shape[2]
    elif len(templates.shape) == 5:
        n_elec = templates.shape[3]
    else:
        raise AttributeError("Wrong 'templates' shape!")

    recordings = np.zeros((n_samples, n_elec), dtype=dtype)

    for st, st_idx in enumerate(st_idxs):
        if extract_spike_traces:
            max_electrode = np.argmax(voltage_peaks[st])

        seed = np.random.randint(10000)
        np.random.seed(seed)

        if modulation in ['electrode', 'template']:
            mod_bool = True
            if bursting_units is not None:
                if st in bursting_units and shape_mod:
                    unit_burst = True
                else:
                    unit_burst = False
            else:
                unit_burst = False
            mod_array = amp_mod[st]
        else:  # modulation 'none'
            mod_bool = False
            mod_array = None
            unit_burst = False

        spike_idx_in_chunk_pad = np.where((st_idx > i_start_pad) & (st_idx <= i_stop_pad))
        spike_idx_in_chunk = np.where((st_idx > i_start) & (st_idx <= i_stop))

        if len(spike_idx_in_chunk_pad) > 0:
            st_idx_in_chunk_pad = st_idx[spike_idx_in_chunk_pad[0]] - i_start_pad

            if drifting and st in drifting_units:
                recordings, template_idx = convolve_drifting_templates_spiketrains(st, st_idx_in_chunk_pad,
                                                                                   templates[st],
                                                                                   n_samples=n_samples,
                                                                                   cut_out=cut_outs_samples,
                                                                                   modulation=mod_bool,
                                                                                   mod_array=mod_array,
                                                                                   fs=fs,
                                                                                   loc=template_locs[st],
                                                                                   drift_mode=drift_mode,
                                                                                   slow_drift_velocity=velocity_vector,
                                                                                   fast_drift_period=fast_drift_period,
                                                                                   fast_drift_min_jump=fast_drift_min_jump,
                                                                                   fast_drift_max_jump=fast_drift_max_jump,
                                                                                   t_start_drift=t_start_drift,
                                                                                   chunk_start=chunk_start,
                                                                                   bursting=unit_burst,
                                                                                   shape_stretch=shape_stretch,
                                                                                   verbose=verbose,
                                                                                   recordings=recordings)
                np.random.seed(seed)
                if extract_spike_traces:
                    spike_traces[:, st] = convolve_single_template(st, st_idx_in_chunk_pad,
                                                                   templates[st, 0, :, max_electrode],
                                                                   n_samples=n_samples,
                                                                   cut_out=cut_outs_samples,
                                                                   modulation=mod_bool,
                                                                   mod_array=mod_array,
                                                                   bursting=unit_burst,
                                                                   shape_stretch=shape_stretch)

                # only keep template idxs inside the chunk
                if len(spike_idx_in_chunk) > 0:
                    if len(spike_idx_in_chunk_pad[0]) != len(spike_idx_in_chunk[0]):
                        common_idxs = [i for i, idx in enumerate(spike_idx_in_chunk_pad[0])
                                       if idx in spike_idx_in_chunk[0]]
                        template_idx = template_idx[common_idxs]
                        assert len(template_idx) == len(spike_idx_in_chunk[0])
                else:
                    template_idx = np.array([])
            else:
                if drifting:
                    template = templates[st, 0]
                else:
                    template = templates[st]
                recordings = convolve_templates_spiketrains(st, st_idx_in_chunk_pad, template,
                                                            n_samples=n_samples,
                                                            cut_out=cut_outs_samples,
                                                            modulation=mod_bool,
                                                            mod_array=mod_array,
                                                            bursting=unit_burst,
                                                            shape_stretch=shape_stretch,
                                                            verbose=verbose,
                                                            recordings=recordings)
                np.random.seed(seed)
                if extract_spike_traces:
                    spike_traces[:, st] = convolve_single_template(st, st_idx_in_chunk_pad,
                                                                   template[:, max_electrode],
                                                                   n_samples=n_samples,
                                                                   cut_out=cut_outs_samples,
                                                                   modulation=mod_bool,
                                                                   mod_array=mod_array,
                                                                   bursting=unit_burst,
                                                                   shape_stretch=shape_stretch)
                template_idx = np.array([])
        else:
            if verbose:
                print('No spikes found in chunk', ch, 'for spike train', st)
            template_idx = np.array([])
        template_idxs.append(template_idx)

    if verbose:
        print('Done all convolutions for chunk', ch)

    return_dict = dict()
    if pad_samples > 0:
        return_dict['recordings'] = recordings[pad_samples:-pad_samples]
    else:
        return_dict['recordings'] = recordings
    if extract_spike_traces:
        return_dict['spike_traces'] = spike_traces[pad_samples:-pad_samples]
    return_dict['template_idxs'] = template_idxs

    return return_dict


chunk_convolution = FuncThenAddChunk(chunk_convolution_)


def chunk_uncorrelated_noise_(ch, i_start, i_stop, chunk_start,
                              num_chan, noise_level, noise_color, color_peak, color_q, color_noise_floor, fs, dtype,
                              seed_list):
    np.random.seed(seed_list[ch])
    length = i_stop - i_start
    additive_noise = noise_level * np.random.randn(length, num_chan).astype(dtype)

    if noise_color:
        # iir peak filter
        b_iir, a_iir = scipy.signal.iirpeak(color_peak, Q=color_q, fs=fs)
        additive_noise = scipy.signal.filtfilt(b_iir, a_iir, additive_noise, axis=0, padlen=1000)
        additive_noise = additive_noise.astype(dtype)
        additive_noise += color_noise_floor * np.std(additive_noise) * \
                          np.random.randn(additive_noise.shape[0], additive_noise.shape[1])
        additive_noise = additive_noise * (noise_level / np.std(additive_noise))

    return_dict = {}
    return_dict['additive_noise'] = additive_noise

    return return_dict


chunk_uncorrelated_noise = FuncThenAddChunk(chunk_uncorrelated_noise_)


def chunk_distance_correlated_noise_(ch, i_start, i_stop, chunk_start,
                                     noise_level, cov_dist, n_elec, noise_color, color_peak, color_q, color_noise_floor,
                                     fs, dtype, seed_list):
    np.random.seed(seed_list[ch])
    length = i_stop - i_start

    additive_noise = noise_level * np.random.multivariate_normal(np.zeros(n_elec), cov_dist,
                                                                 size=length).astype(dtype)
    if noise_color:
        # iir peak filter
        b_iir, a_iir = scipy.signal.iirpeak(color_peak, Q=color_q, fs=fs)
        additive_noise = scipy.signal.filtfilt(b_iir, a_iir, additive_noise, axis=0, padlen=1000)
        additive_noise = additive_noise + color_noise_floor * np.std(additive_noise) * \
                         np.random.multivariate_normal(np.zeros(n_elec), cov_dist,
                                                       size=length)
    additive_noise = additive_noise * (noise_level / np.std(additive_noise))

    return_dict = {}
    return_dict['additive_noise'] = additive_noise

    return return_dict


chunk_distance_correlated_noise = FuncThenAddChunk(chunk_distance_correlated_noise_)


def chunk_apply_filter_(ch, i_start, i_stop, chunk_start,
                        recordings, pad_samples, cutoff, order, fs, dtype):
    n_samples = recordings.shape[0]

    # compute padding idxs
    if pad_samples > 0:
        if i_start > pad_samples:
            i_start_pad = i_start - pad_samples
            pad_start_samples = pad_samples
        else:
            i_start_pad = 0
            pad_start_samples = i_start
        if n_samples - i_stop > pad_samples:
            i_stop_pad = i_stop + pad_samples
            pad_stop_samples = pad_samples
        else:
            i_stop_pad = n_samples
            pad_stop_samples = n_samples - i_stop
    else:
        i_start_pad = i_start
        i_stop_pad = i_stop

    if cutoff.size == 1:
        filtered_chunk = filter_analog_signals(recordings[i_start_pad:i_stop_pad], freq=cutoff, fs=fs,
                                               filter_type='highpass', order=order)
    elif cutoff.size == 2:
        if fs / 2. < cutoff[1]:
            filtered_chunk = filter_analog_signals(recordings[i_start_pad:i_stop_pad], freq=cutoff[0], fs=fs,
                                                   filter_type='highpass', order=order)
        else:
            filtered_chunk = filter_analog_signals(recordings[i_start_pad:i_stop_pad], freq=cutoff, fs=fs)

    filtered_chunk = filtered_chunk.astype(dtype)
    if pad_samples > 0:
        filtered_chunk = filtered_chunk[pad_start_samples:filtered_chunk.shape[0] - pad_stop_samples]

    return_dict = {}
    return_dict['filtered_chunk'] = filtered_chunk

    return return_dict


chunk_apply_filter = FuncThenAddChunk(chunk_apply_filter_)
