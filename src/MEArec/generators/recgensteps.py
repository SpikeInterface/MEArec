"""
This module group several sub function for recording generator.

All this function work on chunk of signals.
They can be call in loop mode or with joblib.

Important:

When tmp_mode=='memmap' : these functions must assign and add directly the buffer.
When tmp_mode is None : these functions return the buffer and the assignament is done externally.



"""
import h5py
import numpy as np
import scipy.signal

from MEArec.tools import (compute_drift_idxs_from_drift_list,
                          convolve_single_template,
                          convolve_templates_spiketrains,
                          filter_analog_signals)


class FuncThenAddChunk:
    """
    Helper function to compute and add to an existing array by chunk.

    E.g., used for convolution, additive noise
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kargs):
        return_dict = self.func(*args)

        (
            ch,
            i_start,
            i_stop,
        ) = args[:3]

        assignment_dict = kargs["assignment_dict"]
        tmp_mode = kargs["tmp_mode"]

        if tmp_mode is None:
            pass
        elif tmp_mode == "memmap":
            for key, full_arr in assignment_dict.items():
                out_chunk = return_dict.pop(key)
                full_arr[i_start:i_stop] += out_chunk.astype(full_arr.dtype)

        return return_dict


class FuncThenReplaceChunk:
    """
    Helper function to compute by and replace an existing array by chunk.

    E.g., used for convolution, additive noise
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kargs):
        return_dict = self.func(*args)

        (
            ch,
            i_start,
            i_stop,
        ) = args[:3]

        assignment_dict = kargs["assignment_dict"]
        tmp_mode = kargs["tmp_mode"]

        if tmp_mode is None:
            pass
        elif tmp_mode == "memmap":
            for key, full_arr in assignment_dict.items():
                out_chunk = return_dict.pop(key)
                full_arr[i_start:i_stop] = out_chunk.astype(full_arr.dtype)

        return return_dict


def chunk_convolution_(
    ch,
    i_start,
    i_stop,
    fs,
    lsb,
    st_idxs,
    pad_samples,
    modulation,
    drifting,
    drifting_units,
    templates,
    cut_outs_samples,
    drift_list,
    verbose,
    amp_mod,
    bursting_units,
    shape_mod,
    shape_stretch,
    extract_spike_traces,
    voltage_peaks,
    dtype,
    seed_list,
):
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
    fs: float
        Sampling frequency in Hz
    lsb: int or None
        If given, the signal is quantized given the lsb value
    st_idxs: list
        list of spike indexes for each spike train
    pad_samples: int
        Number of samples for padding
    modulation: str
        Modulation type
    drifting: bool
        If True drifting is performed
    drifting_units: list
        List of drifting units (if None all units are drifted)
    templates: np.array
        Templates
    cut_outs_samples: list
        List with number of samples to cut out before and after spike peaks
    drift_list: list of dict
        List of drift dictionaries
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
        print("Start convolutions for chunk", ch)
    # set seed
    np.random.seed(seed_list[ch])
    traces_dtype = np.float32

    # checking boundaries not needed here because the recordings are created on the fly
    i_start_pad = i_start - pad_samples
    i_stop_pad = i_stop + pad_samples
    n_samples = i_stop_pad - i_start_pad

    template_idxs = []
    if extract_spike_traces:
        spike_traces = np.zeros((n_samples, len(st_idxs)), dtype=traces_dtype)
    if len(templates.shape) == 4:
        n_elec = templates.shape[2]
    elif len(templates.shape) == 5:
        n_elec = templates.shape[3]
        drift_steps = templates.shape[1]
        default_drift_ind = drift_steps // 2
    else:
        raise AttributeError("Wrong 'templates' shape!")

    recordings = np.zeros((n_samples, n_elec), dtype=traces_dtype)

    for st, st_idx in enumerate(st_idxs):
        if extract_spike_traces:
            max_electrode = np.argmax(voltage_peaks[st])

        seed = np.random.randint(10000)
        np.random.seed(seed)

        if modulation in ["electrode", "template"]:
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

        (spike_in_chunk,) = np.nonzero((st_idx >= i_start_pad) & (st_idx < i_stop_pad))

        if len(spike_in_chunk) > 0:
            st_idx_in_chunk_pad = st_idx[spike_in_chunk] - i_start_pad

            if drifting:
                drift_idxs_in_chunk = np.zeros(len(st_idx_in_chunk_pad), dtype="uint16")

                spike_times_in_chunk = st_idx[spike_in_chunk]
                drift_idxs_in_chunk = compute_drift_idxs_from_drift_list(st, spike_times_in_chunk, drift_list, fs)

                # 4d
                template = templates[st]
                if extract_spike_traces:
                    central_template = templates[st, default_drift_ind, :, max_electrode, :]
            else:
                drift_idxs_in_chunk = None
                # drift_fs = None
                if templates.ndim == 4:
                    drift_vector = None
                    # 3d no drift
                    template = templates[st]
                    if extract_spike_traces:
                        central_template = templates[st, :, max_electrode, :]
                elif templates.ndim == 5:
                    drift_vector = None
                    # 3d no drift
                    template = templates[st, default_drift_ind]
                    if extract_spike_traces:
                        central_template = templates[st, default_drift_ind, :, max_electrode, :]
                else:
                    raise Exception(f"templates.shape no 4 or 5 {templates.shape}")

            recordings = convolve_templates_spiketrains(
                st,
                st_idx_in_chunk_pad,
                template,
                n_samples=n_samples,
                cut_out=cut_outs_samples,
                modulation=mod_bool,
                mod_array=mod_array,
                bursting=unit_burst,
                shape_stretch=shape_stretch,
                verbose=verbose,
                recordings=recordings,
                drift_idxs=drift_idxs_in_chunk,
            )
            np.random.seed(seed)
            if extract_spike_traces:
                spike_traces[:, st] = convolve_single_template(
                    st,
                    st_idx_in_chunk_pad,
                    central_template,
                    n_samples=n_samples,
                    cut_out=cut_outs_samples,
                    modulation=mod_bool,
                    mod_array=mod_array,
                    bursting=unit_burst,
                    shape_stretch=shape_stretch,
                )
        else:
            if verbose:
                print("No spikes found in chunk", ch, "for spike train", st)

    if verbose:
        print("Done all convolutions for chunk", ch)

    return_dict = dict()
    if pad_samples > 0:
        recordings_ret = recordings[pad_samples:-pad_samples]
    else:
        recordings_ret = recordings

    if extract_spike_traces:
        return_dict["spike_traces"] = spike_traces[pad_samples:-pad_samples]

    if lsb is not None:
        recordings_ret = np.floor_divide(recordings_ret, lsb) * lsb

    return_dict["recordings"] = recordings_ret.astype(dtype)
    return_dict["template_idxs"] = template_idxs

    return return_dict


chunk_convolution = FuncThenAddChunk(chunk_convolution_)


def chunk_uncorrelated_noise_(
    ch,
    i_start,
    i_stop,
    fs,
    lsb,
    num_chan,
    noise_level,
    noise_color,
    color_peak,
    color_q,
    color_noise_floor,
    dtype,
    seed_list,
):
    np.random.seed(seed_list[ch])
    length = i_stop - i_start
    additive_noise = noise_level * np.random.randn(length, num_chan)

    if noise_color:
        # iir peak filter
        b_iir, a_iir = scipy.signal.iirpeak(color_peak, Q=color_q, fs=float(fs))
        additive_noise = scipy.signal.filtfilt(b_iir, a_iir, additive_noise, axis=0, padlen=1000)
        additive_noise += (
            color_noise_floor
            * np.std(additive_noise)
            * np.random.randn(additive_noise.shape[0], additive_noise.shape[1])
        )
        additive_noise = additive_noise * (noise_level / np.std(additive_noise))

    return_dict = {}
    if lsb is not None:
        additive_noise = np.floor_divide(additive_noise, lsb) * lsb
    return_dict["additive_noise"] = additive_noise.astype(dtype)

    return return_dict


chunk_uncorrelated_noise = FuncThenAddChunk(chunk_uncorrelated_noise_)


def chunk_distance_correlated_noise_(
    ch,
    i_start,
    i_stop,
    fs,
    lsb,
    noise_level,
    cov_dist,
    n_elec,
    noise_color,
    color_peak,
    color_q,
    color_noise_floor,
    dtype,
    seed_list,
):
    np.random.seed(seed_list[ch])
    length = i_stop - i_start

    additive_noise = noise_level * np.random.multivariate_normal(np.zeros(n_elec), cov_dist, size=length)
    if noise_color:
        # iir peak filter
        b_iir, a_iir = scipy.signal.iirpeak(color_peak, Q=color_q, fs=float(fs))
        additive_noise = scipy.signal.filtfilt(b_iir, a_iir, additive_noise, axis=0, padlen=1000)
        additive_noise = additive_noise + color_noise_floor * np.std(additive_noise) * np.random.multivariate_normal(
            np.zeros(n_elec), cov_dist, size=length
        )
    additive_noise = additive_noise * (noise_level / np.std(additive_noise))

    return_dict = {}
    if lsb is not None:
        additive_noise = np.floor_divide(additive_noise, lsb) * lsb
    return_dict["additive_noise"] = additive_noise.astype(dtype)

    return return_dict


chunk_distance_correlated_noise = FuncThenAddChunk(chunk_distance_correlated_noise_)


def chunk_apply_filter_(ch, i_start, i_stop, fs, lsb, recordings, pad_samples, cutoff, order, mode, dtype):
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
        filtered_chunk = filter_analog_signals(
            recordings[i_start_pad:i_stop_pad], freq=cutoff, fs=fs, filter_type="highpass", mode=mode, order=order
        )
    elif cutoff.size == 2:
        if fs / 2.0 < cutoff[1]:
            filtered_chunk = filter_analog_signals(
                recordings[i_start_pad:i_stop_pad],
                freq=cutoff[0],
                fs=fs,
                filter_type="highpass",
                mode=mode,
                order=order,
            )
        else:
            filtered_chunk = filter_analog_signals(
                recordings[i_start_pad:i_stop_pad], freq=cutoff, fs=fs, filter_type="bandpass", mode=mode, order=order
            )

    filtered_chunk = filtered_chunk
    if pad_samples > 0:
        filtered_chunk = filtered_chunk[pad_start_samples : filtered_chunk.shape[0] - pad_stop_samples]

    return_dict = {}
    if lsb is not None:
        filtered_chunk = np.floor_divide(filtered_chunk, lsb) * lsb
    return_dict["filtered_chunk"] = filtered_chunk.astype(dtype)

    return return_dict


chunk_apply_filter = FuncThenReplaceChunk(chunk_apply_filter_)
