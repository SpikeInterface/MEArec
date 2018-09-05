# from __future__ import print_function
# Helper functions

import numpy as np
import quantities as pq
from quantities import Quantity
import yaml
import elephant
# import scipy.signal as ss
# from scipy.optimize import curve_fit
import os
from os.path import join
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
from neuroplot import *

def load(filename):
    '''Generic loading of cPickled objects from file'''
    import pickle
    
    filen = open(filename,'rb')
    obj = pickle.load(filen)
    filen.close()
    return obj


def load_EAP_data(templates_folder, celltypes=None, samples_per_cat=None):
    """ Loading extracellular action potential data from files

    Parameters:
    -----------
    templates_folder : str
        Path to folder containing the data.
    samples_per_cat : int (optional, default None)
        Number of samples per category to be loaded. If ``None``,
        all available samples are loaded

    Returns:
    --------
    spikes_list : array_like
        Loaded EAPs
    loc_list : array_like
        Positions of the neurons evoking the loaded EAPs.
    rot_list : array_like
        Rotations (angles) of the neurons evoking the loaded EAPs.
    celltype_list : array_like (dtype=str)
        Categories of the neurons evoking the loaded EAPs.
    loaded_categories : list
        List of loaded categories.
    """
    print("Loading spike data ...")
    spikelist = [f for f in os.listdir(templates_folder) if f.startswith('eap')]
    loclist = [f for f in os.listdir(templates_folder) if f.startswith('pos')]
    rotlist = [f for f in os.listdir(templates_folder) if f.startswith('rot')]
    infolist = [f for f in os.listdir(templates_folder) if f.startswith('info')]

    spikes_list = []
    loc_list = []
    rot_list = []
    cat_list = []

    spikelist = sorted(spikelist)
    loclist = sorted(loclist)
    rotlist = sorted(rotlist)
    infolist = sorted(infolist)

    loaded_categories = set()
    ignored_categories = set()

    for idx, f in enumerate(spikelist):
        celltype = f.split('-')[1][:-4]

        print('loading cell type: ', f)
        if celltypes is not None:
            if celltype in celltypes:
                spikes = np.load(join(templates_folder, f))
                locs = np.load(join(templates_folder, loclist[idx]))
                rots = np.load(join(templates_folder, rotlist[idx]))

                if samples_per_cat is None or samples_per_cat > len(spikes):
                    samples_to_read = len(spikes)
                else:
                    samples_to_read = samples_per_cat

                spikes_list.extend(spikes[:samples_to_read])
                rot_list.extend(rots[:samples_to_read])
                loc_list.extend(locs[:samples_to_read])
                cat_list.extend([celltype] * samples_to_read)
                loaded_categories.add(celltype)
            else:
                ignored_categories.add(celltype)
        else:
            spikes = np.load(join(templates_folder, f))
            locs = np.load(join(templates_folder, loclist[idx]))
            rots = np.load(join(templates_folder, rotlist[idx]))

            if samples_per_cat is None or samples_per_cat > len(spikes):
                samples_to_read = len(spikes)
            else:
                samples_to_read = samples_per_cat

            spikes_list.extend(spikes[:samples_to_read])
            rot_list.extend(rots[:samples_to_read])
            loc_list.extend(locs[:samples_to_read])
            cat_list.extend([celltype] * samples_to_read)
            loaded_categories.add(celltype)

    # load info
    with open(join(templates_folder, infolist[0]), 'r') as fl:
        info = yaml.load(fl)
        info['General'].pop('cell name', None)

    print("Done loading spike data ...")
    print("Loaded categories", loaded_categories)
    print("Ignored categories", ignored_categories)
    return np.array(spikes_list), np.array(loc_list), np.array(rot_list), np.array(cat_list), loaded_categories, info


def apply_pca(X, n_comp):
    from sklearn.decomposition import PCA

    # whiten data
    pca = PCA(n_components=n_comp)
    data = pca.fit_transform(np.transpose(X))

    return np.transpose(data), pca.components_

def whiten_data(X, n_comp=None):
    '''

    Parameters
    ----------
    X: nfeatures x nsa
    n_comp: number of components

    Returns
    -------

    '''
    # whiten data
    if n_comp==None:
        n_comp = np.min(X.shape)

    n_feat, n_samples = X.shape

    pca = PCA(n_components=n_comp, whiten=True)
    data = pca.fit_transform(X.T)
    eigvecs = pca.components_
    eigvals = pca.explained_variance_
    sphere = np.matmul(np.diag(1. / np.sqrt(eigvals)), eigvecs)

    return np.transpose(data), eigvecs, eigvals, sphere

# def whiten_data(X, n_comp=None):
#     '''
#
#     Parameters
#     ----------
#     X: nsa x nfeatures
#     n_comp: number of components
#
#     Returns
#     -------
#
#     '''
#     # whiten data
#     if n_comp==None:
#         n_comp = np.min(X.shape)
#     pca = PCA(n_components=n_comp, whiten=True)
#     data = pca.fit_transform(X.T)
#
#     return np.transpose(data), pca.components_, pca.explained_variance_ratio_


############ RECSTIMSIM ######################


def load_validation_data(validation_folder,load_mcat=False):
    print "Loading validation spike data ..."

    spikes = np.load(join(validation_folder, 'val_spikes.npy'))  # [:spikes_per_cell, :, :]
    feat = np.load(join(validation_folder, 'val_feat.npy'))  # [:spikes_per_cell, :, :]
    locs = np.load(join(validation_folder, 'val_loc.npy'))  # [:spikes_per_cell, :]
    rots = np.load(join(validation_folder, 'val_rot.npy'))  # [:spikes_per_cell, :]
    cats = np.load(join(validation_folder, 'val_cat.npy'))
    if load_mcat:
        mcats = np.load(join(validation_folder, 'val_mcat.npy'))
        print "Done loading spike data ..."
        return np.array(spikes), np.array(feat), np.array(locs), np.array(rots), np.array(cats),np.array(mcats)
    else:
        print "Done loading spike data ..."
        return np.array(spikes), np.array(feat), np.array(locs), np.array(rots), np.array(cats)

def get_binary_cat(categories, excit, inhib):
    binary_cat = []
    for i, cat in enumerate(categories):
        if cat in excit:
            binary_cat.append('EXCIT')
        elif cat in inhib:
            binary_cat.append('INHIB')

    return np.array(binary_cat, dtype=str)


def get_EAP_features(EAP, feat_list, dt=None, EAP_times=None, threshold_detect=5., normalize=False):
    ''' extract features specified in feat_list from EAP
    '''
    reference_mode = 't0'
    if EAP_times is not None and dt is not None:
        test_dt = (EAP_times[-1] - EAP_times[0]) / (len(EAP_times) - 1)
        if dt != test_dt:
            raise ValueError('EAP_times and dt do not match.')
    elif EAP_times is not None:
        dt = (EAP_times[-1] - EAP_times[0]) / (len(EAP_times) - 1)
    elif dt is not None:
        EAP_times = np.arange(EAP.shape[-1]) * dt
    else:
        raise NotImplementedError('Please, specify either dt or EAP_times.')

    if len(EAP.shape) == 1:
        EAP = np.reshape(EAP, [1, 1, -1])
    elif len(EAP.shape) == 2:
        EAP = np.reshape(EAP, [1, EAP.shape[0], EAP.shape[1]])
    if len(EAP.shape) != 3:
        raise ValueError('Cannot handle EAPs with shape', EAP.shape)

    if normalize:
        signs = np.sign(np.min(EAP.reshape([EAP.shape[0], -1]), axis=1))
        norm = np.abs(np.min(EAP.reshape([EAP.shape[0], -1]), axis=1))
        EAP = np.array([EAP[i] / n if signs[i] > 0 else EAP[i] / n - 2. for i, n in enumerate(norm)])

    features = {}

    amps = np.zeros((EAP.shape[0], EAP.shape[1]))
    na_peak = np.zeros((EAP.shape[0], EAP.shape[1]))
    rep_peak = np.zeros((EAP.shape[0], EAP.shape[1]))
    if 'W' in feat_list:
        features['widths'] = np.zeros((EAP.shape[0], EAP.shape[1]))
    if 'F' in feat_list:
        features['fwhm'] = np.zeros((EAP.shape[0], EAP.shape[1]))
    if 'R' in feat_list:
        features['ratio'] = np.zeros((EAP.shape[0], EAP.shape[1]))
    if 'S' in feat_list:
        features['speed'] = np.zeros((EAP.shape[0], EAP.shape[1]))
    if 'Aids' in feat_list:
        features['Aids'] = np.zeros((EAP.shape[0], EAP.shape[1], 2), dtype=int)
    if 'Fids' in feat_list:
        features['Fids'] = []
    if 'FV' in feat_list:
        features['fwhm_V'] = np.zeros((EAP.shape[0], EAP.shape[1]))
    if 'Na' in feat_list:
        features['na'] = np.zeros((EAP.shape[0], EAP.shape[1]))
    if 'Rep' in feat_list:
        features['rep'] = np.zeros((EAP.shape[0], EAP.shape[1]))

    for i in range(EAP.shape[0]):
        # For AMP feature
        min_idx = np.array([np.unravel_index(EAP[i, e].argmin(), EAP[i, e].shape)[0] for e in
                            range(EAP.shape[1])])
        max_idx = np.array([np.unravel_index(EAP[i, e, min_idx[e]:].argmax(),
                                             EAP[i, e, min_idx[e]:].shape)[0]
                            + min_idx[e] for e in range(EAP.shape[1])])
        # for na and rep
        min_elid, min_idx_na = np.unravel_index(EAP[i].argmin(), EAP[i].shape)
        max_idx_rep = EAP[i, min_elid, min_idx_na:].argmax() + min_idx_na
        na_peak[i, :] = EAP[i, :, min_idx_na]
        rep_peak[i, :] = EAP[i, :, max_idx_rep]

        if 'Aids' in feat_list:
            features['Aids'][i] = np.vstack((min_idx, max_idx)).T

        amps[i, :] = np.array([EAP[i, e, max_idx[e]] - EAP[i, e, min_idx[e]] for e in range(EAP.shape[1])])
        # If below 'detectable threshold, set amp and width to 0
        if normalize:
            too_low = np.where(amps[i, :] < threshold_detect / norm[i])
        else:
            too_low = np.where(amps[i, :] < threshold_detect)
        amps[i, too_low] = 0

        if 'R' in feat_list:
            min_id_ratio = np.array([np.unravel_index(EAP[i, e, min_idx_na:].argmin(),
                                                      EAP[i, e, min_idx_na:].shape)[0]
                                     + min_idx_na for e in range(EAP.shape[1])])
            max_id_ratio = np.array([np.unravel_index(EAP[i, e, min_idx_na:].argmax(),
                                                      EAP[i, e, min_idx_na:].shape)[0]
                                     + min_idx_na for e in range(EAP.shape[1])])
            features['ratio'][i, :] = np.array([np.abs(EAP[i, e, max_id_ratio[e]]) /
                                                np.abs(EAP[i, e, min_id_ratio[e]])
                                                for e in range(EAP.shape[1])])
            # If below 'detectable threshold, set amp and width to 0
            too_low = np.where(amps[i, :] < threshold_detect)
            features['ratio'][i, too_low] = 1
        if 'S' in feat_list:
            features['speed'][i, :] = np.array((min_idx - min_idx_na) * dt)
            features['speed'][i, too_low] = min_idx_na * dt

        if 'W' in feat_list:
            features['widths'][i, :] = np.abs(EAP_times[max_idx] - EAP_times[min_idx])
            features['widths'][i, too_low] = EAP.shape[2] * dt  # EAP_times[-1]-EAP_times[0]

        if 'F' in feat_list:
            min_peak = np.min(EAP[i], axis=1)
            if reference_mode == 't0':
                # reference voltage is zeroth voltage entry
                fwhm_ref = np.array([EAP[i, e, 0] for e in range(EAP.shape[1])])
            elif reference_mode == 'maxd2EAP':
                # reference voltage is taken at peak onset
                # peak onset is defined as id of maximum 2nd derivative of EAP
                peak_onset = np.array([np.argmax(savitzky_golay(EAP[i, e], 5, 2, deriv=2)[:min_idx[e]])
                                       for e in range(EAP.shape[1])])
                fwhm_ref = np.array([EAP[i, e, peak_onset[e]] for e in range(EAP.shape[1])])
            else:
                raise NotImplementedError('Reference mode ' + reference_mode + ' for FWHM calculation not implemented.')
            fwhm_V = (fwhm_ref + min_peak) / 2.
            id_trough = [np.where(EAP[i, e] < fwhm_V[e])[0] for e in range(EAP.shape[1])]
            if 'Fids' in feat_list:
                features['Fids'].append(id_trough)
            if 'FV' in feat_list:
                features['fwhm_V'][i, :] = fwhm_V

            # linear interpolation due to little number of data points during peak

            # features['fwhm'][i,:] = np.array([(len(t)+1)*dt+(fwhm_V[e]-EAP[i,e,t[0]-1])/(EAP[i,e,t[0]]-EAP[i,e,t[0]-1])*dt -(fwhm_V[e]-EAP[i,e,t[-1]])/(EAP[i,e,min(t[-1]+1,EAP.shape[2]-1)]-EAP[i,e,t[-1]])*dt if len(t)>0 else np.infty for e,t in enumerate(id_trough)])

            # no linear interpolation
            features['fwhm'][i, :] = [(id_trough[e][-1] - id_trough[e][0]) * dt if len(id_trough[e]) > 1 \
                                          else EAP.shape[2] * dt for e in range(EAP.shape[1])]
            features['fwhm'][i, too_low] = EAP.shape[2] * dt  # EAP_times[-1]-EAP_times[0]

    if 'A' in feat_list:
        features.update({'amps': amps})
    if 'Na' in feat_list:
        features.update({'na': na_peak})
    if 'Rep' in feat_list:
        features.update({'rep': rep_peak})

    return features



############ SPYICA ######################3

def filter_analog_signals(anas, freq, fs, filter_type='bandpass', order=3, copy_signal=False):
    """Filters analog signals with zero-phase Butterworth filter.
    The function raises an Exception if the required filter is not stable.

    Parameters
    ----------
    anas : np.array
           2d array of analog signals
    freq : list or float
           cutoff frequency-ies in Hz
    fs : float
         sampling frequency
    filter_type : string
                  'lowpass', 'highpass', 'bandpass', 'bandstop'
    order : int
            filter order

    Returns
    -------
    anas_filt : filtered signals
    """
    from scipy.signal import butter, filtfilt
    fn = fs / 2.
    fn = fn.rescale(pq.Hz)
    freq = freq.rescale(pq.Hz)
    band = freq / fn

    b, a = butter(order, band, btype=filter_type)

    if np.all(np.abs(np.roots(a)) < 1) and np.all(np.abs(np.roots(a)) < 1):
        print 'Filtering signals with ', filter_type, ' filter at ', freq, '...'
        if len(anas.shape) == 2:
            anas_filt = filtfilt(b, a, anas, axis=1)
        elif len(anas.shape) == 1:
            anas_filt = filtfilt(b, a, anas)
        return anas_filt
    else:
        raise ValueError('Filter is not stable')


def select_cells(loc, spikes, bin_cat, n_exc, n_inh, min_dist=25, bound_x=[], min_amp=None, drift=False,
                 drift_dir_ang=[], preferred_dir=None, ang_tol=30, verbose=False):
    pos_sel = []
    idxs_sel = []
    exc_idxs = np.where(bin_cat == 'EXCIT')[0]
    inh_idxs = np.where(bin_cat == 'INHIB')[0]

    if not min_amp:
        min_amp = 0

    if drift:
        if len(drift_dir_ang) == 0 or preferred_dir == None:
            raise Exception('For drift selection provide drifting angles and preferred drift direction')

    for (idxs, num) in zip([exc_idxs, inh_idxs], [n_exc, n_inh]):
        n_sel = 0
        iter = 0
        while n_sel < num:
            # randomly draw a cell
            id_cell = idxs[np.random.permutation(len(idxs))[0]]
            dist = np.array([np.linalg.norm(loc[id_cell] - p) for p in pos_sel])

            iter += 1

            if np.any(dist < min_dist):
                if verbose:
                    print 'distance violation', dist, iter
                pass
            else:
                amp = np.max(np.abs(spikes[id_cell]))
                if not drift:
                    if len(bound_x) == 0:
                        if amp > min_amp:
                            # save cell
                            pos_sel.append(loc[id_cell])
                            idxs_sel.append(id_cell)
                            n_sel += 1
                        else:
                            if verbose:
                                print 'amp violation', amp, iter
                    else:
                        if loc[id_cell][0] > bound_x[0] and loc[id_cell][0] < bound_x[1] and amp > min_amp:
                            # save cell
                            pos_sel.append(loc[id_cell])
                            idxs_sel.append(id_cell)
                            n_sel += 1
                        else:
                            if verbose:
                                print 'boundary violation', loc[id_cell], iter
                else:
                    # drift
                    if len(bound_x) == 0:
                        if amp > min_amp:
                            # save cell
                            if np.abs(drift_dir_ang[id_cell] - preferred_dir) < ang_tol:
                                pos_sel.append(loc[id_cell])
                                idxs_sel.append(id_cell)
                                n_sel += 1
                            else:
                                if verbose:
                                    print 'drift violation', loc[id_cell], iter
                        else:
                            if verbose:
                                print 'amp violation', amp, iter
                    else:
                        if loc[id_cell][0] > bound_x[0] and loc[id_cell][0] < bound_x[1] and amp > min_amp:
                            # save cell
                            if np.abs(drift_dir_ang[id_cell] - preferred_dir) < ang_tol:
                                pos_sel.append(loc[id_cell])
                                idxs_sel.append(id_cell)
                                n_sel += 1
                            else:
                                if verbose:
                                    print 'drift violation', loc[id_cell], iter
                        else:
                            if verbose:
                                print 'boundary violation', loc[id_cell], iter
    return idxs_sel


def find_overlapping_spikes(spikes, thresh=0.7, parallel=True):
    overlapping_pairs = []

    for i in range(spikes.shape[0] - 1):
        if parallel:
            import multiprocessing
            nprocesses = len(spikes)
            # t_start = time.time()
            pool = multiprocessing.Pool(nprocesses)
            results = [pool.apply_async(overlapping(i, spikes,))
                       for i, sp_times in enumerate(sst)]
            overlapping_pairs = []
            for result in results:
                overlapping_pairs.extend(result.get())
        else:
            temp_1 = spikes[i]
            max_ptp = (np.array([np.ptp(t) for t in temp_1]).max())
            max_ptp_idx = (np.array([np.ptp(t) for t in temp_1]).argmax())

            for j in range(i + 1, spikes.shape[0]):
                temp_2 = spikes[j]
                ptp_on_max = np.ptp(temp_2[max_ptp_idx])

                max_ptp_2 = (np.array([np.ptp(t) for t in temp_2]).max())

                max_peak = np.max([ptp_on_max, max_ptp])
                min_peak = np.min([ptp_on_max, max_ptp])

                if min_peak > thresh * max_peak and ptp_on_max > thresh * max_ptp_2:
                    overlapping_pairs.append([i, j])  # , max_ptp_idx, max_ptp, ptp_on_max

    return np.array(overlapping_pairs)


def overlapping(i, spikes):
    overlapping_pairs = []
    temp_1 = spikes[i]
    max_ptp = (np.array([np.ptp(t) for t in temp_1]).max())
    max_ptp_idx = (np.array([np.ptp(t) for t in temp_1]).argmax())

    for j in range(i + 1, spikes.shape[0]):
        temp_2 = spikes[j]
        ptp_on_max = np.ptp(temp_2[max_ptp_idx])

        max_ptp_2 = (np.array([np.ptp(t) for t in temp_2]).max())

        max_peak = np.max([ptp_on_max, max_ptp])
        min_peak = np.min([ptp_on_max, max_ptp])

        if min_peak > thresh * max_peak and ptp_on_max > thresh * max_ptp_2:
            overlapping_pairs.append([i, j])  # , max_ptp_idx, max_ptp, ptp_on_max

    return overlapping_pairs


def cubic_padding(spike, pad_len, fs, percent_mean=0.2):
    '''
    Cubic spline padding on left and right side to 0

    Parameters
    ----------
    spike
    pad_len
    fs

    Returns
    -------
    padded_template

    '''
    import scipy.interpolate as interp
    n_pre = int(pad_len[0] * fs)
    n_post = int(pad_len[1] * fs)

    padded_template = np.zeros((spike.shape[0], int(n_pre) + spike.shape[1] + n_post))
    splines = np.zeros((spike.shape[0], int(n_pre) + spike.shape[1] + n_post))

    for i, sp in enumerate(spike):
        # Remove inital offset
        padded_sp = np.zeros(n_pre + len(sp) + n_post)
        padded_t = np.arange(len(padded_sp))
        initial_offset = np.mean(sp[0])
        sp -= initial_offset

        x_pre = float(n_pre)
        x_pre_pad = np.arange(n_pre)
        x_post = float(n_post)
        x_post_pad = np.arange(n_post)[::-1]

        off_pre = sp[0]
        off_post = sp[-1]
        m_pre = sp[0] / x_pre
        m_post = sp[-1] / x_post

        padded_sp[:n_pre] = m_pre * x_pre_pad
        padded_sp[n_pre:-n_post] = sp
        padded_sp[-n_post:] = m_post * x_post_pad

        f = interp.interp1d(padded_t, padded_sp, kind='cubic')
        splines[i] = f(np.arange(len(padded_sp)))

        padded_template[i, :n_pre] = f(x_pre_pad)
        padded_template[i, n_pre:-n_post] = sp
        padded_template[i, -n_post:] = f(np.arange(n_pre + len(sp), n_pre + len(sp) + n_post))

    return padded_template, splines



def detect_and_align(sources, fs, recordings, t_start=None, t_stop=None, n_std=5, ref_period=2*pq.ms, upsample=8):
    '''

    Parameters
    ----------
    sources
    fs
    recordings
    t_start
    t_stop
    n_std
    ref_period
    upsample

    Returns
    -------

    '''
    import scipy.signal as ss
    import quantities as pq
    import neo

    idx_spikes = []
    idx_sources = []
    spike_trains = []
    times = (np.arange(sources.shape[1]) / fs).rescale('ms')
    unit = times[0].rescale('ms').units

    for s_idx, s in enumerate(sources):
        thresh = -n_std * np.median(np.abs(s) / 0.6745)
        # print s_idx, thresh
        idx_spike = np.where(s < thresh)[0]
        idx_spikes.append(idx_spike)

        n_pad = int(2 * pq.ms * fs.rescale('kHz'))
        sp_times = []
        sp_wf = []
        sp_rec_wf = []
        sp_amp = []
        first_spike = True

        for t in range(len(idx_spike) - 1):
            idx = idx_spike[t]
            # find single waveforms crossing thresholds
            if idx_spike[t + 1] - idx > 1 or t == len(idx_spike) - 2:  # single spike
                if idx - n_pad > 0 and idx + n_pad < len(s):
                    spike = s[idx - n_pad:idx + n_pad]
                    t_spike = times[idx - n_pad:idx + n_pad]
                    spike_rec = recordings[:, idx - n_pad:idx + n_pad]
                elif idx - n_pad < 0:
                    spike = s[:idx + n_pad]
                    spike = np.pad(spike, (np.abs(idx - n_pad), 0), 'constant')
                    t_spike = times[:idx + n_pad]
                    t_spike = np.pad(t_spike, (np.abs(idx - n_pad), 0), 'constant') * unit
                    spike_rec = recordings[:, :idx + n_pad]
                    spike_rec = np.pad(spike_rec, ((0, 0), (np.abs(idx - n_pad), 0)), 'constant')
                elif idx + n_pad > len(s):
                    spike = s[idx - n_pad:]
                    spike = np.pad(spike, (0, idx + n_pad - len(s)), 'constant')
                    t_spike = times[idx - n_pad:]
                    t_spike = np.pad(t_spike, (0, idx + n_pad - len(s)), 'constant') * unit
                    spike_rec = recordings[:, idx - n_pad:]
                    spike_rec = np.pad(spike_rec, ((0, 0), (0, idx + n_pad - len(s))), 'constant')

                if first_spike:
                    nsamples = len(spike)
                    nsamples_up = nsamples*upsample
                    first_spike = False

                # upsample and find minimum
                if upsample > 1:
                    spike_up = ss.resample_poly(spike, upsample, 1)
                    # times_up = ss.resample_poly(t_spike, upsample, 1)*unit
                    t_spike_up = np.linspace(t_spike[0].magnitude, t_spike[-1].magnitude, num=len(spike_up)) * unit
                else:
                    spike_up = spike
                    t_spike_up = t_spike

                min_idx_up = np.argmin(spike_up)
                min_amp_up = np.min(spike_up)
                min_time_up = t_spike_up[min_idx_up]

                min_idx = np.argmin(spike)
                min_amp = np.min(spike)
                min_time = t_spike[min_idx]

                # align waveform
                shift = nsamples_up//2 - min_idx_up
                if shift > 0:
                    spike_up = np.pad(spike_up, (np.abs(shift), 0), 'constant')[:nsamples_up]
                elif shift < 0:
                    spike_up = np.pad(spike_up, (0, np.abs(shift)), 'constant')[-nsamples_up:]

                if len(sp_times) != 0:
                    if min_time_up - sp_times[-1] > ref_period:
                        sp_wf.append(spike_up)
                        sp_rec_wf.append(spike_rec)
                        sp_amp.append(min_amp_up)
                        sp_times.append(min_time_up)
                else:
                    sp_wf.append(spike_up)
                    sp_rec_wf.append(spike_rec)
                    sp_amp.append(min_amp_up)
                    sp_times.append(min_time_up)

        if t_start and t_stop:
            for i, sp in enumerate(sp_times):
                if sp.magnitude * unit < t_start:
                    sp_times[i] = t_start.rescale('ms')
                if  sp.magnitude * unit > t_stop:
                    sp_times[i] = t_stop.rescale('ms')
        elif t_stop:
            for i, sp in enumerate(sp_times):
                if sp > t_stop:
                    sp_times[i] = t_stop.rescale('ms')
        else:
            t_start = 0 * pq.s
            t_stop = sp_times[-1]

        spiketrain = neo.SpikeTrain([sp.magnitude for sp in sp_times] * unit, t_start=0 * pq.s, t_stop=t_stop,
                                    waveforms=np.array(sp_rec_wf))

        spiketrain.annotate(ica_amp=np.array(sp_amp))
        spiketrain.annotate(ica_wf=np.array(sp_wf))
        spike_trains.append(spiketrain)
        idx_sources.append(s_idx)

    return spike_trains


def extract_wf(sst, recordings, times, fs, upsample=8, ica=False, sources=[]):
    '''

    Parameters
    ----------
    sst
    sources
    recordings
    times
    fs
    upsample

    Returns
    -------

    '''
    import scipy.signal as ss
    import quantities as pq

    n_pad = int(2 * pq.ms * fs.rescale('kHz'))
    unit = times[0].rescale('ms').units

    nChs, nPts = recordings.shape

    if ica:
        if len(sources) == 0:
            raise Exception('Provide IC sources for IC waveforms')
        for (st, s) in zip(sst, sources):
            sp_wf = []
            sp_rec_wf = []
            sp_amp = []
            first_spike = True

            for t in st:
                idx = np.where(times>t)[0][0]
                # find single waveforms crossing thresholds
                if idx - n_pad > 0 and idx + n_pad < nPts:
                    spike = s[idx - n_pad:idx + n_pad]
                    t_spike = times[idx - n_pad:idx + n_pad]
                    spike_rec = recordings[:, idx - n_pad:idx + n_pad]
                elif idx - n_pad < 0:
                    spike = s[:idx + n_pad]
                    spike = np.pad(spike, (np.abs(idx - n_pad), 0), 'constant')
                    t_spike = times[:idx + n_pad]
                    t_spike = np.pad(t_spike, (np.abs(idx - n_pad), 0), 'constant') * unit
                    spike_rec = recordings[:, :idx + n_pad]
                    spike_rec = np.pad(spike_rec, ((0, 0), (np.abs(idx - n_pad), 0)), 'constant')
                elif idx + n_pad > nPts:
                    spike = s[idx - n_pad:]
                    spike = np.pad(spike, (0, idx + n_pad - nPts), 'constant')
                    t_spike = times[idx - n_pad:]
                    t_spike = np.pad(t_spike, (0, idx + n_pad - nPts), 'constant') * unit
                    spike_rec = recordings[:, idx - n_pad:]
                    spike_rec = np.pad(spike_rec, ((0, 0), (0, idx + n_pad - nPts)), 'constant')

                if first_spike:
                    nsamples = len(spike)
                    nsamples_up = nsamples*upsample
                    first_spike = False

                min_ic_amp = np.min(spike)
                sp_wf.append(spike)
                sp_rec_wf.append(spike_rec)
                sp_amp.append(min_ic_amp)

            st.waveforms = np.array(sp_rec_wf)
            st.annotate(ica_amp=np.array(sp_amp))
            st.annotate(ica_wf=np.array(sp_wf))
    else:
        for st in sst:
            sp_rec_wf = []
            sp_amp = []
            first_spike = True

            for t in st:
                idx = np.where(times > t)[0][0]
                # find single waveforms crossing thresholds
                if idx - n_pad > 0 and idx + n_pad < nPts:
                    t_spike = times[idx - n_pad:idx + n_pad]
                    spike_rec = recordings[:, idx - n_pad:idx + n_pad]
                elif idx - n_pad < 0:
                    t_spike = times[:idx + n_pad]
                    t_spike = np.pad(t_spike, (np.abs(idx - n_pad), 0), 'constant') * unit
                    spike_rec = recordings[:, :idx + n_pad]
                    spike_rec = np.pad(spike_rec, ((0, 0), (np.abs(idx - n_pad), 0)), 'constant')
                elif idx + n_pad > nPts:
                    t_spike = times[idx - n_pad:]
                    t_spike = np.pad(t_spike, (0, idx + n_pad - nPts), 'constant') * unit
                    spike_rec = recordings[:, idx - n_pad:]
                    spike_rec = np.pad(spike_rec, ((0, 0), (0, idx + n_pad - nPts)), 'constant')
                if first_spike:
                    nsamples = len(spike_rec)
                    nsamples_up = nsamples * upsample
                    first_spike = False

                min_amp = np.min(spike_rec)
                sp_rec_wf.append(spike_rec)
                sp_amp.append(min_amp)

            st.waveforms = np.array(sp_rec_wf)




def reject_duplicate_spiketrains(sst, percent_threshold=0.5, min_spikes=3, sources=None, parallel=False,
                                 nprocesses=None):
    '''

    Parameters
    ----------
    sst
    percent_threshold
    min_spikes

    Returns
    -------

    '''
    import neo
    import multiprocessing
    import time

    if nprocesses is None:
        nprocesses = len(sst)

    spike_trains = []
    idx_sources = []
    duplicates = []

    if parallel:
        # t_start = time.time()
        pool = multiprocessing.Pool(nprocesses)
        results = [pool.apply_async(find_duplicates, (i, sp_times, sst,))
                   for i, sp_times in enumerate(sst)]
        duplicates = []
        for result in results:
            duplicates.extend(result.get())

        # print 'Parallel: ', time.time() - t_start
    else:
        # t_start = time.time()
        for i, sp_times in enumerate(sst):
            # check if overlapping with another source
            t_jitt = 1 * pq.ms
            counts = []
            for j, sp in enumerate(sst):
                count = 0
                if i != j:
                    for t_i in sp_times:
                        id_over = np.where((sp > t_i - t_jitt) & (sp < t_i + t_jitt))[0]
                        if len(id_over) != 0:
                            count += 1
                    if count >= percent_threshold * len(sp_times):
                        if [i, j] not in duplicates and [j, i] not in duplicates:
                            print 'Found duplicate spike trains: ', i, j, count
                            duplicates.append([i, j])
                    counts.append(count)
        # print 'Sequential: ', time.time() - t_start

    duplicates = np.array(duplicates)
    discard = []
    if len(duplicates) > 0:
        for i, sp_times in enumerate(sst):
            if i not in duplicates:
                # rej ect spiketrains with less than 3 spikes...
                if len(sp_times) >= min_spikes:
                    spike_trains.append(sp_times)
                    idx_sources.append(i)
            else:
                # Keep spike train with largest number of spikes among duplicates
                idxs = np.argwhere(duplicates==i)
                max_len = []
                c_max = 0
                st_idx = []
                for idx in idxs:
                    row = idx[0]
                    st_idx.append(duplicates[row][0])
                    st_idx.append(duplicates[row][1])
                    c_ = np.max([len(sst[duplicates[row][0]]), len(sst[duplicates[row][1]])])
                    i_max = np.argmax([len(sst[duplicates[row][0]]), len(sst[duplicates[row][1]])])
                    if len(max_len) == 0 or c_ > c_max:
                        max_len = [row, i_max]
                        c_max = c_
                index = duplicates[max_len[0], max_len[1]]
                if index not in idx_sources and index not in discard:
                    spike_trains.append(sst[index])
                    idx_sources.append(index)
                    [discard.append(d) for d in st_idx if d != index and d not in discard]


    else:
        spike_trains = sst
        idx_sources = range(len(sst))

    return spike_trains, idx_sources, duplicates


def find_duplicates(i, sp_times, sst, percent_threshold=0.5, t_jitt=1*pq.ms):
    counts = []
    duplicates = []
    for j, sp in enumerate(sst):
        count = 0
        if i != j:
            for t_i in sp_times:
                id_over = np.where((sp > t_i - t_jitt) & (sp < t_i + t_jitt))[0]
                if len(id_over) != 0:
                    count += 1
            if count >= percent_threshold * len(sp_times):
                if [i, j] not in duplicates and [j, i] not in duplicates:
                    print 'Found duplicate spike trains: ', i, j, count
                    duplicates.append([i, j])
            counts.append(count)

    return duplicates


def integrate_sources(sources):
    '''

    Parameters
    ----------
    sources

    Returns
    -------

    '''
    integ_source = np.zeros_like(sources)

    for s, sor in enumerate(sources):
        partial_sum = 0
        for t, s_t in enumerate(sor):
            partial_sum += s_t
            integ_source[s, t] = partial_sum

    return integ_source


def clean_sources(sources, kurt_thresh=0.7, skew_thresh=0.5, remove_correlated=True):
    '''

    Parameters
    ----------
    s
    corr_thresh
    skew_thresh

    Returns
    -------

    '''
    import scipy.stats as stat
    import scipy.signal as ss

    sk = stat.skew(sources, axis=1)
    ku = stat.kurtosis(sources, axis=1)

    high_sk = np.where(np.abs(sk) >= skew_thresh)[0]
    low_sk = np.where(np.abs(sk) < skew_thresh)[0]
    high_ku = np.where(ku >= kurt_thresh)[0]
    low_ku = np.where(ku < kurt_thresh)[0]

    idxs = np.unique(np.concatenate((high_sk, high_ku)))

    # sources_sp = sources[high_sk]
    # sources_disc = sources[low_sk]
    # # compute correlation matrix
    # corr = np.zeros((sources_sp.shape[0], sources_sp.shape[0]))
    # mi = np.zeros((sources_sp.shape[0], sources_sp.shape[0]))
    # max_lag = np.zeros((sources_sp.shape[0], sources_sp.shape[0]))
    # for i in range(sources_sp.shape[0]):
    #     s_i = sources_sp[i]
    #     for j in range(i + 1, sources_sp.shape[0]):
    #         s_j = sources_sp[j]
    #         cmat = crosscorrelation(s_i, s_j, maxlag=50)
    #         # cmat = ss.correlate(s_i, s_j)
    #         corr[i, j] = np.max(np.abs(cmat))
    #         max_lag[i, j] = np.argmax(np.abs(cmat))
    #         mi[i, j] = calc_MI(s_i, s_j, bins=100)

    # sources_keep = sources[idxs]
    # corr_idx = np.argwhere(corr > corr_thresh)
    # sk_keep = stat.skew(sources_keep, axis=1)

    # # remove smaller skewnesses
    # remove_ic = []
    # for idxs in corr_idx:
    #     sk_pair = sk_keep[idxs]
    #     remove_ic.append(idxs[np.argmin(np.abs(sk_pair))])
    # remove_ic = np.array(remove_ic)
    #
    # if len(remove_ic) != 0 and remove_correlated:
    #     mask = np.array([True] * len(sources_keep))
    #     mask[remove_ic] = False
    #
    #     spike_sources = sources_keep[mask]
    #     source_idx = high_sk[mask]
    # else:
    # source_idx = high_sk

    spike_sources = sources[idxs]
    sk_sp = stat.skew(spike_sources, axis=1)

    # invert sources with positive skewness
    spike_sources[sk_sp > 0] = -spike_sources[sk_sp > 0]

    return spike_sources, idxs #, corr_idx, corr, mi



def crosscorrelation(x, y, maxlag):
    """
    Cross correlation with a maximum number of lags.

    `x` and `y` must be one-dimensional numpy arrays with the same length.

    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]

    The return vaue has length 2*maxlag + 1.
    """
    from numpy.lib.stride_tricks import as_strided

    py = np.pad(y.conj(), 2*maxlag, mode='constant')
    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                   strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')
    return T.dot(px)


def bin_spiketimes(spike_times, fs=None, T=None, t_stop=None):
    '''

    Parameters
    ----------
    spike_times
    fs
    T

    Returns
    -------

    '''
    import elephant.conversion as conv
    import neo
    resampled_mat = []
    binned_spikes = []
    spiketrains = []

    if isinstance(spike_times[0], neo.SpikeTrain):
        unit = spike_times[0].units
        spike_times = [st.times.magnitude for st in spike_times]*unit
    for st in spike_times:
        if t_stop:
            t_st = t_stop.rescale(pq.ms)
        else:
            t_st = st[-1].rescale(pq.ms)
        st_pq = [s.rescale(pq.ms).magnitude for s in st]*pq.ms
        spiketrains.append(neo.SpikeTrain(st_pq, t_st))
    if not fs and not T:
        print 'Provide either sampling frequency fs or time period T'
    elif fs:
        if not isinstance(fs, Quantity):
            raise ValueError("fs must be of type pq.Quantity")
        binsize = 1./fs
        binsize.rescale('ms')
        resampled_mat = []
        spikes = conv.BinnedSpikeTrain(spiketrains, binsize=binsize)
        for sts in spiketrains:
            spikes = conv.BinnedSpikeTrain(sts, binsize=binsize)
            resampled_mat.append(np.squeeze(spikes.to_array()))
            binned_spikes.append(spikes)
    elif T:
        binsize = T
        if not isinstance(T, Quantity):
            raise ValueError("T must be of type pq.Quantity")
        binsize.rescale('ms')
        resampled_mat = []
        for sts in spiketrains:
            spikes = conv.BinnedSpikeTrain(sts, binsize=binsize)
            resampled_mat.append(np.squeeze(spikes.to_array()))
            binned_spikes.append(spikes)


    return np.array(resampled_mat), binned_spikes


def ISI_amplitude_modulation(st, n_el=1, mrand=1, sdrand=0.05, n_spikes=1, exp=0.5, mem_ISI = 10*pq.ms):
    '''

    Parameters
    ----------
    st
    mrand
    sdrand
    n_spikes
    exp
    mem_ISI

    Returns
    -------

    '''

    import elephant.statistics as stat

    if n_el == 1:
        ISI = stat.isi(st).rescale('ms')
        # mem_ISI = 2*mean_ISI
        amp_mod = np.zeros(len(st))
        amp_mod[0] = sdrand*np.random.randn() + mrand
        cons = np.zeros(len(st))

        for i, isi in enumerate(ISI):
            if n_spikes == 1:
                if isi > mem_ISI:
                    amp_mod[i + 1] = sdrand * np.random.randn() + mrand
                else:
                    amp_mod[i + 1] = isi.magnitude ** exp * (1. / mem_ISI.magnitude ** exp) + sdrand * np.random.randn()
            else:
                consecutive = 0
                bursting=True
                while consecutive < n_spikes and bursting:
                    if i-consecutive >= 0:
                        if ISI[i-consecutive] > mem_ISI:
                            bursting = False
                        else:
                            consecutive += 1
                    else:
                        bursting = False

                if consecutive == 0:
                    amp_mod[i + 1] = sdrand * np.random.randn() + mrand
                elif consecutive==1:
                    amp = (isi / float(consecutive)) ** exp * (1. / mem_ISI.magnitude ** exp)
                    # scale std by amp
                    amp_mod[i + 1] = amp + amp * sdrand * np.random.randn()
                else:
                    if i != len(ISI):
                        isi_mean = np.mean(ISI[i-consecutive+1:i+1])
                    else:
                        isi_mean = np.mean(ISI[i - consecutive + 1:])
                    amp = (isi_mean/float(consecutive)) ** exp * (1. / mem_ISI.magnitude ** exp)
                    # scale std by amp
                    amp_mod[i + 1] = amp + amp * sdrand * np.random.randn()

                cons[i+1] = consecutive
    else:
        if n_spikes == 0:
            amp_mod = []
            cons = []
            for i, sp in enumerate(st):
                amp_mod.append(sdrand * np.random.randn(n_el) + mrand)

    return np.array(amp_mod), cons

def cluster_spike_amplitudes(sst, metric='cal', min_sihlo=0.8, min_cal=100, max_clusters=4,
                             alg='kmeans', features='amp', ncomp=3, keep_all=False):
    '''

    Parameters
    ----------
    spike_amps
    sst
    metric
    min_sihlo
    min_cal
    max_clusters
    alg
    keep_all

    Returns
    -------

    '''
    from sklearn.metrics import silhouette_score, calinski_harabaz_score
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    import neo
    from copy import copy

    spike_wf = np.array([sp.annotations['ica_wf'] for sp in sst])
    spike_amps = [sp.annotations['ica_amp'] for sp in sst]
    nclusters = np.zeros(len(spike_amps))
    silhos = np.zeros(len(spike_amps))
    cal_hars = np.zeros(len(spike_amps))

    reduced_amps = []
    reduced_sst = []
    keep_id = []

    if features == 'amp':
        for i, amps in enumerate(spike_amps):
            silho = 0
            cal_har = 0
            keep_going = True

            if len(amps) > 2:
                for k in range(2, max_clusters):
                    if alg=='kmeans':
                        kmeans_new = KMeans(n_clusters=k, random_state=0)
                        kmeans_new.fit(amps.reshape(-1, 1))
                        labels_new = kmeans_new.predict(amps.reshape(-1, 1))
                    elif alg=='mog':
                        gmm_new = GaussianMixture(n_components=k, covariance_type='full')
                        gmm_new.fit(amps.reshape(-1, 1))
                        labels_new = gmm_new.predict(amps.reshape(-1, 1))

                    if len(np.unique(labels_new)) > 1:
                        silho_new = silhouette_score(amps.reshape(-1, 1), labels_new)
                        cal_har_new = calinski_harabaz_score(amps.reshape(-1, 1), labels_new)
                        if metric == 'silho':
                            if silho_new > silho:
                                silho = silho_new
                                nclusters[i] = k
                                if alg == 'kmeans':
                                    kmeans = kmeans_new
                                elif alg == 'mog':
                                    gmm = gmm_new
                                labels = labels_new
                            else:
                                keep_going=False
                        elif metric == 'cal':
                            if cal_har_new > cal_har:
                                cal_har = cal_har_new
                                nclusters[i] = k
                                if alg == 'kmeans':
                                    kmeans = kmeans_new
                                elif alg == 'mog':
                                    gmm = gmm_new
                                labels = labels_new
                            else:
                                keep_going=False
                    else:
                        keep_going=False
                        nclusters[i] = 1

                    if not keep_going:
                        break

                if nclusters[i] != 1:
                    if metric == 'silho':
                        if silho < min_sihlo:
                            nclusters[i] = 1
                            reduced_sst.append(sst[i])
                            reduced_amps.append(amps)
                            keep_id.append(range(len(sst[i])))
                        else:
                            if keep_all:
                                for clust in np.unique(labels):
                                    idxs = np.where(labels == clust)[0]
                                    reduced_sst.append(sst[i][idxs])
                                    reduced_amps.append(amps[idxs])
                                    keep_id.append(idxs)
                            else:
                                highest_clust = np.argmin(kmeans.cluster_centers_)
                                highest_idx = np.where(labels==highest_clust)[0]
                                reduced_sst.append(sst[i][highest_idx])
                                reduced_amps.append(amps[highest_idx])
                                keep_id.append(highest_idx)
                    elif metric == 'cal':
                        if cal_har < min_cal:
                            nclusters[i] = 1
                            sst[i].annotate(ica_source=i)
                            reduced_sst.append(sst[i])
                            reduced_amps.append(amps)
                            keep_id.append(range(len(sst[i])))
                        else:
                            if keep_all:
                                for clust in np.unique(labels):
                                    idxs = np.where(labels == clust)[0]
                                    red_spikes = sst[i][idxs]
                                    red_spikes.annotations = copy(sst[i].annotations)
                                    if 'ica_amp' in red_spikes.annotations:
                                        red_spikes.annotate(ica_amp=red_spikes.annotations['ica_amp'][idxs])
                                    if 'ica_wf' in red_spikes.annotations:
                                        red_spikes.annotate(ica_wf=red_spikes.annotations['ica_wf'][idxs])
                                    red_spikes.annotate(ica_source=i)
                                    reduced_sst.append(red_spikes)
                                    reduced_amps.append(amps[idxs])
                                    keep_id.append(idxs)
                            else:
                                if alg == 'kmeans':
                                    highest_clust = np.argmin(kmeans.cluster_centers_)
                                elif alg == 'mog':
                                    highest_clust = np.argmin(gmm.means_)
                                idxs = np.where(labels == highest_clust)[0]
                                red_spikes = sst[i][idxs]
                                red_spikes.annotations = copy(sst[i][idxs].annotations)
                                if 'ica_amp' in red_spikes.annotations:
                                    red_spikes.annotate(ica_amp=red_spikes.annotations['ica_amp'][idxs])
                                if 'ica_wf' in red_spikes.annotations:
                                    red_spikes.annotate(ica_wf=red_spikes.annotations['ica_wf'][idxs])
                                red_spikes.annotate(ica_source=i)
                                reduced_sst.append(red_spikes)
                                reduced_amps.append(amps[idxs])
                                keep_id.append(idxs)
                    silhos[i] = silho
                    cal_hars[i] = cal_har
                else:
                    red_spikes = copy(sst[i])
                    red_spikes.annotations = copy(sst[i].annotations)
                    red_spikes.annotate(ica_source=i)
                    reduced_sst.append(red_spikes)
                    reduced_amps.append(amps)
                    keep_id.append(range(len(sst[i])))
            else:
                red_spikes = copy(sst[i])
                red_spikes.annotations = copy(sst[i].annotations)
                red_spikes.annotate(ica_source=i)
                reduced_sst.append(red_spikes)
                reduced_amps.append(amps)
                keep_id.append(range(len(sst[i])))

    elif features == 'pca':
        for i, wf in enumerate(spike_wf):
            # apply pca on ica_wf
            wf_pca, comp = apply_pca(wf.T, n_comp=ncomp)
            wf_pca = wf_pca.T
            amps = spike_amps[i]

            silho = 0
            cal_har = 0
            keep_going = True

            if len(wf_pca) > 2:
                for k in range(2, max_clusters):
                    if alg == 'kmeans':
                        kmeans_new = KMeans(n_clusters=k, random_state=0)
                        kmeans_new.fit(wf_pca)
                        labels_new = kmeans_new.predict(wf_pca)
                    elif alg == 'mog':
                        gmm_new = GaussianMixture(n_components=k, covariance_type='full')
                        gmm_new.fit(wf_pca)
                        labels_new = gmm_new.predict(wf_pca)

                    if len(np.unique(labels_new)) > 1:
                        silho_new = silhouette_score(wf_pca, labels_new)
                        cal_har_new = calinski_harabaz_score(wf_pca, labels_new)
                        if metric == 'silho':
                            if silho_new > silho:
                                silho = silho_new
                                nclusters[i] = k
                                if alg == 'kmeans':
                                    kmeans = kmeans_new
                                elif alg == 'mog':
                                    gmm = gmm_new
                                labels = labels_new
                            else:
                                keep_going = False
                        elif metric == 'cal':
                            if cal_har_new > cal_har:
                                cal_har = cal_har_new
                                if metric == 'cal':
                                    nclusters[i] = k
                                    if alg == 'kmeans':
                                        kmeans = kmeans_new
                                    elif alg == 'mog':
                                        gmm = gmm_new
                                labels = labels_new
                            else:
                                keep_going = False
                    else:
                        keep_going = False
                        nclusters[i] = 1

                    if not keep_going:
                        break

                if nclusters[i] != 1:
                    if metric == 'silho':
                        if silho < min_sihlo:
                            nclusters[i] = 1
                            red_spikes = sst[i]
                            red_spikes.annotations = copy(sst[i].annotations)
                            red_spikes.annotate(ica_source=i)
                            reduced_sst.append(red_spikes)
                            reduced_amps.append(amps)
                            keep_id.append(range(len(sst[i])))
                        else:
                            if keep_all:
                                for clust in np.unique(labels):
                                    idxs = np.where(labels == clust)[0]
                                    red_spikes = sst[i][idxs]
                                    red_spikes.annotations = copy(sst[i].annotations)
                                    if 'ica_amp' in red_spikes.annotations:
                                        red_spikes.annotate(ica_amp=red_spikes.annotations['ica_amp'][idxs])
                                    if 'ica_wf' in red_spikes.annotations:
                                        red_spikes.annotate(ica_wf=red_spikes.annotations['ica_wf'][idxs])
                                    red_spikes.annotate(ica_source=i)
                                    reduced_amps.append(amps[idxs])
                                    keep_id.append(idxs)
                            else:
                                highest_clust = np.argmin(kmeans.cluster_centers_)
                                idxs = np.where(labels == highest_clust)[0]
                                red_spikes.annotations = copy(sst[i].annotations)
                                if 'ica_amp' in red_spikes.annotations:
                                    red_spikes.annotate(ica_amp=red_spikes.annotations['ica_amp'][idxs])
                                if 'ica_wf' in red_spikes.annotations:
                                    red_spikes.annotate(ica_wf=red_spikes.annotations['ica_wf'][idxs])
                                red_spikes.annotate(ica_source=i)
                                reduced_amps.append(amps[highest_idx])
                                keep_id.append(highest_idx)
                    elif metric == 'cal':
                        if cal_har < min_cal:
                            nclusters[i] = 1
                            red_spikes = copy(sst[i])
                            red_spikes.annotations = copy(sst[i].annotations)
                            red_spikes.annotate(ica_source=i)
                            reduced_sst.append(red_spikes)
                            reduced_amps.append(amps)
                            keep_id.append(range(len(sst[i])))
                        else:
                            if keep_all:
                                for clust in np.unique(labels):
                                    idxs = np.where(labels == clust)[0]
                                    red_spikes = sst[i][idxs]
                                    red_spikes.annotations = copy(sst[i].annotations)
                                    if 'ica_amp' in red_spikes.annotations:
                                        red_spikes.annotate(ica_amp=red_spikes.annotations['ica_amp'][idxs])
                                    if 'ica_wf' in red_spikes.annotations:
                                        red_spikes.annotate(ica_wf=red_spikes.annotations['ica_wf'][idxs])
                                    red_spikes.annotate(ica_source=i)
                                    reduced_sst.append(red_spikes)
                                    reduced_amps.append(amps[idxs])
                                    keep_id.append(idxs)
                            else:
                                # for PCA the sign might be inverted
                                if alg == 'kmeans':
                                    highest_clust = np.argmax(np.abs(kmeans.cluster_centers_))
                                elif alg == 'mog':
                                    highest_clust = np.argmax(np.abs(gmm.means_))
                                idxs = np.where(labels == highest_clust)[0]
                                red_spikes = sst[i][idxs]
                                red_spikes.annotations = copy(sst[i].annotations)
                                if 'ica_amp' in red_spikes.annotations:
                                    red_spikes.annotate(ica_amp=red_spikes.annotations['ica_amp'][idxs])
                                if 'ica_wf' in red_spikes.annotations:
                                    red_spikes.annotate(ica_wf=red_spikes.annotations['ica_wf'][idxs])
                                red_spikes.annotate(ica_source=i)
                                reduced_sst.append(red_spikes)
                                reduced_amps.append(amps[idxs])
                                keep_id.append(idxs)
                    silhos[i] = silho
                    cal_hars[i] = cal_har
                else:
                    red_spikes = copy(sst[i])
                    red_spikes.annotations = copy(sst[i].annotations)
                    red_spikes.annotate(ica_source=i)
                    reduced_sst.append(red_spikes)
                    reduced_amps.append(amps)
                    keep_id.append(range(len(sst[i])))
            else:
                red_spikes = copy(sst[i])
                red_spikes.annotations = copy(sst[i].annotations)
                red_spikes.annotate(ica_source=i)
                reduced_sst.append(red_spikes)
                reduced_amps.append(amps)
                keep_id.append(range(len(sst[i])))

    if metric == 'silho':
        score = silhos
    elif metric == 'cal':
        score = cal_hars

    return reduced_sst, reduced_amps, nclusters, keep_id, score

def cc_max_spiketrains(st, st_id, other_st, max_lag=20*pq.ms):
    '''

    Parameters
    ----------
    st
    st_id
    other_st

    Returns
    -------

    '''
    from elephant.spike_train_correlation import cch, corrcoef

    cc_vec = np.zeros(len(other_st))
    for p, p_st in enumerate(other_st):
        # cc, bin_ids = cch(st, p_st, kernel=np.hamming(100))
        cc, bin_ids = cch(st, p_st, kernel=np.hamming(10), window=[-max_lag, max_lag])
        # central_bin = len(cc) // 2
        # normalize by number of spikes
        # cc_vec[p] = np.max(cc[central_bin-10:central_bin+10]) #/ (len(st) + len(p_st))
        cc_vec[p] = np.max(cc) #/ (len(st) + len(p_st))

    return st_id, cc_vec

def evaluate_spiketrains(gtst, sst, t_jitt = 1*pq.ms, overlapping=False, parallel=True, nprocesses=None,
                         pairs=[], t_start=0*pq.s):
    '''

    Parameters
    ----------
    gtst
    sst
    t_jitt
    overlapping
    parallel
    nprocesses

    Returns
    -------

    '''
    import neo
    import multiprocessing
    from elephant.spike_train_correlation import cch, corrcoef
    from scipy.optimize import linear_sum_assignment
    import time

    if nprocesses is None:
        num_cores = len(gtst)
    else:
        num_cores = nprocesses

    t_stop = gtst[0].t_stop

    if t_start > 0:
        sst_clip = [st.time_slice(t_start=t_start, t_stop=st.t_stop) for st in sst]
        gtst_clip = [st.time_slice(t_start=t_start, t_stop=st.t_stop) for st in gtst]
    else:
        sst_clip = sst
        gtst_clip = gtst

    if len(pairs) == 0:
        print 'Computing correlations between spiketrains'

        or_mat, original_st = bin_spiketimes(gtst_clip, T=1*pq.ms, t_stop=t_stop)
        pr_mat, predicted_st = bin_spiketimes(sst_clip, T=1*pq.ms, t_stop=t_stop)
        cc_matr = np.zeros((or_mat.shape[0], pr_mat.shape[0]))

        if parallel:
            pool = multiprocessing.Pool(nprocesses)
            results = [pool.apply_async(cc_max_spiketrains, (st, st_id, predicted_st,))
                       for st_id, st in enumerate(original_st)]

            idxs = []
            cc_vecs = []
            for result in results:
                idxs.append(result.get()[0])
                cc_vecs.append(result.get()[1])

            for (id, cc_vec) in zip(idxs, cc_vecs):
                cc_matr[id] = [c / (len(gtst_clip[id]) + len(sst_clip[i])) for i, c in enumerate(cc_vec)]
            pool.close()
        else:
            max_lag = 20*pq.ms
            for o, o_st in enumerate(original_st):
                for p, p_st in enumerate(predicted_st):
                    # cc, bin_ids = cch(o_st, p_st, kernel=np.hamming(100))
                    cc, bin_ids = cch(o_st, p_st, kernel=np.hamming(10), window=[-max_lag, max_lag])
                    # normalize by number of spikes
                    cc_matr[o, p] = np.max(cc) / (len(gtst_clip[o]) + len(sst_clip[p])) # (abs(len(gtst[o]) - len(sst[p])) + 1)
        cc_matr /= np.max(cc_matr)

        print 'Pairing spike trains'
        t_hung_st = time.time()
        cc2 = cc_matr ** 2
        col_ind, row_ind = linear_sum_assignment(-cc2)
        put_pairs = -1 * np.ones((len(gtst), 2)).astype('int')

        for i in range(len(gtst)):
            # if i in row_ind:
            idx = np.where(i == col_ind)
            if len(idx[0]) != 0:
                if cc2[col_ind[idx], row_ind[idx]] > 0.1:
                    put_pairs[i] = [int(col_ind[idx]), int(row_ind[idx])]
        t_hung_end = time.time()

        # # shift best match by max lag
        # for gt_i, gt in enumerate(gtst_clip):
        #     pair = put_pairs[gt_i]
        #     if pair[0] != -1:
        #         o_st = original_st[gt_i]
        #         p_st = predicted_st[pair[1]]
        #
        #         cc, bin_ids = cch(o_st, p_st, kernel=np.hamming(50))
        #         central_bin = len(cc) // 2
        #         # normalize by number of spikes
        #         max_lag = np.argmax(cc[central_bin - 5:central_bin + 5])
        #         optimal_shift = (-5 + max_lag) * pq.ms
        #         sst_clip[pair[1]] -= optimal_shift
        #         idx_after = np.where(sst_clip[pair[1]] > sst_clip[pair[1]].t_stop)[0]
        #         idx_before = np.where(sst_clip[pair[1]] < sst_clip[pair[1]].t_start)[0]
        #         if len(idx_after) > 0:
        #             sst_clip[pair[1]][idx_after] = sst_clip[pair[1]].t_stop
        #         if len(idx_before) > 0:
        #             sst_clip[pair[1]][idx_before] = sst_clip[pair[1]].t_start
    else:
        put_pairs = pairs

    # print 'Standard: ', t_standard_end - t_standard_st
    # print 'Hungarian: ', t_hung_end - t_hung_st

    # raise Exception()

    [gt.annotate(paired=False) for gt in gtst_clip]
    [st.annotate(paired=False) for st in sst_clip]
    for pp in put_pairs:
        if pp[0] != -1:
            gtst_clip[pp[0]].annotate(paired=True)
        if pp[1] != -1:
            sst_clip[pp[1]].annotate(paired=True)


    # Evaluate
    for i, gt in enumerate(gtst_clip):
        lab_gt = np.array(['UNPAIRED'] * len(gt))
        gt.annotate(labels=lab_gt)
    for i, st in enumerate(sst_clip):
        lab_st = np.array(['UNPAIRED'] * len(st))
        st.annotate(labels=lab_st)

    print 'Finding TP'
    for gt_i, gt in enumerate(gtst_clip):
        if put_pairs[gt_i, 0] != -1:
            lab_gt = gt.annotations['labels']
            st_sel = sst_clip[put_pairs[gt_i, 1]]
            lab_st = sst_clip[put_pairs[gt_i, 1]].annotations['labels']
            # from gtst: TP, TPO, TPSO, FN, FNO, FNSO
            for sp_i, t_sp in enumerate(gt):
                id_sp = np.where((st_sel > t_sp - t_jitt) & (st_sel < t_sp + t_jitt))[0]
                if len(id_sp) == 1:
                    if 'overlap' in gt.annotations.keys():
                        if gt.annotations['overlap'][sp_i] == 'NO':
                            lab_gt[sp_i] = 'TP'
                            lab_st[id_sp] = 'TP'
                        elif gt.annotations['overlap'][sp_i] == 'O':
                            lab_gt[sp_i] = 'TPO'
                            lab_st[id_sp] = 'TPO'
                        elif gt.annotations['overlap'][sp_i] == 'SO':
                            lab_gt[sp_i] = 'TPSO'
                            lab_st[id_sp] = 'TPSO'
                    else:
                        lab_gt[sp_i] = 'TP'
                        lab_st[id_sp] = 'TP'
            sst_clip[put_pairs[gt_i, 1]].annotate(labels=lab_st)
        else:
            lab_gt = np.array(['FN'] * len(gt))
        gt.annotate(labels=lab_gt)


    # find CL-CLO-CLSO
    print 'Finding CL'
    for gt_i, gt in enumerate(gtst_clip):
        lab_gt = gt.annotations['labels']
        for l_gt, lab in enumerate(lab_gt):
            if lab == 'UNPAIRED':
                for st_i, st in enumerate(sst_clip):
                    if st.annotations['paired']:
                        t_up = gt[l_gt]
                        id_sp = np.where((st > t_up - t_jitt) & (st < t_up + t_jitt))[0]
                        lab_st = st.annotations['labels']
                        if len(id_sp) == 1 and lab_st[id_sp] == 'UNPAIRED':
                            if 'overlap' in gt.annotations.keys():
                                if gt.annotations['overlap'][l_gt] == 'NO':
                                    lab_gt[l_gt] = 'CL_' + str(gt_i) + '_' + str(st_i)
                                    if lab_st[id_sp] == 'UNPAIRED':
                                        lab_st[id_sp] = 'CL_NP'
                                elif gt.annotations['overlap'][l_gt] == 'O':
                                    lab_gt[l_gt] = 'CLO_' + str(gt_i) + '_' + str(st_i)
                                    if lab_st[id_sp] == 'UNPAIRED':
                                        lab_st[id_sp] = 'CLO_NP'
                                elif gt.annotations['overlap'][l_gt] == 'SO':
                                    lab_gt[l_gt] = 'CLSO_' + str(gt_i) + '_' + str(st_i)
                                    if lab_st[id_sp] == 'UNPAIRED':
                                        lab_st[id_sp] = 'CLSO_NP'
                            else:
                                lab_gt[l_gt] = 'CL_' + str(gt_i) + '_' + str(st_i)
                                # print 'here'
                                if lab_st[id_sp] == 'UNPAIRED':
                                    lab_st[id_sp] = 'CL_NP'
                        st.annotate(labels=lab_st)
        gt.annotate(labels=lab_gt)

    print 'Finding FP and FN'
    for gt_i, gt in enumerate(gtst_clip):
        lab_gt = gt.annotations['labels']
        for l_gt, lab in enumerate(lab_gt):
            if lab == 'UNPAIRED':
                if 'overlap' in gt.annotations.keys():
                    if gt.annotations['overlap'][l_gt] == 'NO':
                        lab_gt[l_gt] = 'FN'
                    elif gt.annotations['overlap'][l_gt] == 'O':
                        lab_gt[l_gt] = 'FNO'
                    elif gt.annotations['overlap'][l_gt] == 'SO':
                        lab_gt[l_gt] = 'FNSO'
                else:
                    lab_gt[l_gt] = 'FN'
        gt.annotate(labels=lab_gt)

    for st_i, st in enumerate(sst_clip):
        lab_st = st.annotations['labels']
        for st_i, lab in enumerate(lab_st):
            if lab == 'UNPAIRED':
                    lab_st[st_i] = 'FP'
        st.annotate(labels=lab_st)

    TOT_GT = sum([len(gt) for gt in gtst_clip])
    TOT_ST = sum([len(st) for st in sst_clip])
    total_spikes = TOT_GT + TOT_ST

    TP = sum([len(np.where('TP' == gt.annotations['labels'])[0]) for gt in gtst_clip])
    TPO = sum([len(np.where('TPO' == gt.annotations['labels'])[0]) for gt in gtst_clip])
    TPSO = sum([len(np.where('TPSO' == gt.annotations['labels'])[0]) for gt in gtst_clip])

    print 'TP :', TP, TPO, TPSO, TP+TPO+TPSO

    CL = sum([len([i for i, v in enumerate(gt.annotations['labels']) if 'CL' in v]) for gt in gtst_clip])
         # + sum([len(np.where('CL' == st.annotations['labels'])[0]) for st in sst])
    CLO = sum([len([i for i, v in enumerate(gt.annotations['labels']) if 'CLO' in v]) for gt in gtst_clip])
          # + sum([len(np.where('CLO' == st.annotations['labels'])[0]) for st in sst])
    CLSO = sum([len([i for i, v in enumerate(gt.annotations['labels']) if 'CLSO' in v]) for gt in gtst_clip])
           # + sum([len(np.where('CLSO' == st.annotations['labels'])[0]) for st in sst_clip])

    print 'CL :', CL, CLO, CLSO, CL+CLO+CLSO

    FN = sum([len(np.where('FN' == gt.annotations['labels'])[0]) for gt in gtst_clip])
    FNO = sum([len(np.where('FNO' == gt.annotations['labels'])[0]) for gt in gtst_clip])
    FNSO = sum([len(np.where('FNSO' == gt.annotations['labels'])[0]) for gt in gtst_clip])

    print 'FN :', FN, FNO, FNSO, FN+FNO+FNSO


    FP = sum([len(np.where('FP' == st.annotations['labels'])[0]) for st in sst_clip])

    print 'FP :', FP

    print 'TOTAL: ', TOT_GT, TOT_ST, TP+TPO+TPSO+CL+CLO+CLSO+FN+FNO+FNSO+FP

    counts = {'TP': TP, 'TPO': TPO, 'TPSO': TPSO,
              'CL': CL, 'CLO': CLO, 'CLSO': CLSO,
              'FN': FN, 'FNO': FNO, 'FNSO': FNSO,
              'FP': FP, 'TOT': total_spikes, 'TOT_GT': TOT_GT, 'TOT_ST': TOT_ST}


    return counts, put_pairs #, cc_matr

def confusion_matrix(gtst, sst, pairs, plot_fig=True, xlabel=None, ylabel=None):
    '''

    Parameters
    ----------
    gtst
    sst
    pairs 1D array with paired sst to gtst

    Returns
    -------

    '''
    conf_matrix = np.zeros((len(gtst)+1, len(sst)+1), dtype=int)
    idxs_pairs_clean = np.where(pairs != -1)
    idxs_pairs_dirty = np.where(pairs == -1)
    pairs_clean = pairs[idxs_pairs_clean]
    gtst_clean = np.array(gtst)[idxs_pairs_clean]
    gtst_extra = np.array(gtst)[idxs_pairs_dirty]

    gtst_idxs = np.append(idxs_pairs_clean, idxs_pairs_dirty)
    sst_idxs = pairs_clean
    sst_extra = []

    for gt_i, gt in enumerate(gtst_clean):
        if gt.annotations['paired']:
            tp = len(np.where('TP' == gt.annotations['labels'])[0])
            conf_matrix[gt_i, gt_i] =  int(tp)
            for st_i, st in enumerate(sst):
                cl_str = str(gt_i) + '_' + str(st_i)
                cl = len([i for i, v in enumerate(gt.annotations['labels']) if 'CL' in v and cl_str in v])
                if cl != 0:
                    st_p = np.where(st_i == pairs_clean)
                    conf_matrix[gt_i, st_p] = int(cl)
        fn = len(np.where('FN' == gt.annotations['labels'])[0])
        conf_matrix[gt_i, -1] = int(fn)
    for gt_i, gt in enumerate(gtst_extra):
        fn = len(np.where('FN' == gt.annotations['labels'])[0])
        conf_matrix[gt_i+len(gtst_clean), -1] = int(fn)
    for st_i, st in enumerate(sst):
        fp = len(np.where('FP' == st.annotations['labels'])[0])
        st_p = np.where(st_i == pairs_clean)[0]
        if len(st_p) != 0:
            conf_matrix[-1, st_p] = fp
        else:
            sst_extra.append(int(st_i))
            conf_matrix[-1, len(pairs_clean) + len(sst_extra) - 1] = fp

    if plot_fig:
        fig, ax = plt.subplots()
        # Using matshow here just because it sets the ticks up nicely. imshow is faster.
        ax.matshow(conf_matrix, cmap='Greens')

        for (i, j), z in np.ndenumerate(conf_matrix):
            if z != 0:
                if z > np.max(conf_matrix)/2.:
                    ax.text(j, i, '{:d}'.format(z), ha='center', va='center', color='white')
                else:
                    ax.text(j, i, '{:d}'.format(z), ha='center', va='center', color='black')
                    # ,   bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

        ax.axhline(int(len(gtst)-1)+0.5, color='black')
        ax.axvline(int(len(sst)-1)+0.5, color='black')

        # Major ticks
        ax.set_xticks(np.arange(0, len(sst) + 1))
        ax.set_yticks(np.arange(0, len(gtst) + 1))
        ax.xaxis.tick_bottom()
        # Labels for major ticks
        ax.set_xticklabels(np.append(np.append(sst_idxs, sst_extra).astype(int), 'FN'), fontsize=12)
        ax.set_yticklabels(np.append(gtst_idxs, 'FP'), fontsize=12)

        if xlabel==None:
            ax.set_xlabel('Sorted spike trains', fontsize=15)
        else:
            ax.set_xlabel(xlabel, fontsize=20)
        if ylabel==None:
            ax.set_ylabel('Ground truth spike trains', fontsize=15)
        else:
            ax.set_ylabel(ylabel, fontsize=20)

    return conf_matrix, ax



def matcorr(x, y, rmmean=False, weighting=None):
    from scipy.optimize import linear_sum_assignment

    m, n = x.shape
    p, q = y.shape
    m = np.min([m,p])

    if m != n or  p!=q:
        # print 'matcorr(): Matrices are not square: using max abs corr method (2).'
        method = 2

    if n != q:
      raise Exception('Rows in the two input matrices must be the same length.')

    if rmmean:
      x = x - np.mean(x, axis=1) # optionally remove means
      y = y - np.mean(y, axis=1)

    dx = np.sum(x**2, axis=1)
    dy = np.sum(y**2, axis=1)
    dx[np.where(dx==0)] = 1
    dy[np.where(dy==0)] = 1
    # raise Exception()
    corrs = np.matmul(x, y.T)/np.sqrt(dx[:, np.newaxis]*dy[np.newaxis, :])

    if weighting != None:
        if any(corrs.shape != weighting.shape):
            print 'matcorr(): weighting matrix size must match that of corrs'
        else:
            corrs = corrs * weighting

    cc = np.abs(corrs)

    # Performs Hungarian algorithm matching
    col_ind, row_ind = linear_sum_assignment(-cc.T)

    idx = np.argsort(-cc[row_ind, col_ind])
    corr = corrs[row_ind, col_ind][idx]
    indy = np.arange(m)[idx]
    indx = row_ind[idx]

    return corr, indx, indy, corrs


def evaluate_PI(ic_unmix, gt_mix):
    '''

    Parameters
    ----------
    gt_mix
    ic_mix

    Returns
    -------

    '''
    H = np.matmul(ic_unmix, gt_mix)
    C = H**2
    N = np.min([gt_mix.shape[0], ic_unmix.shape[0]])

    PI = (N - 0.5*(np.sum(np.max(C, axis=0)/np.sum(C, axis=0)) + np.sum(np.max(C, axis=1)/np.sum(C, axis=1))))/(N-1)

    return PI, C


def evaluate_sum_CC(ic_mix, gt_mix, ic_sources, gt_sources, n_sources): # ):
    '''

    Parameters
    ----------
    ic_unmix
    gt_mix
    ic_source
    gt_source

    Returns
    -------

    '''
    correlation, idx_truth, idx_, corr_m = matcorr(gt_mix, ic_mix)
    correlation, idx_truth, idx_, corr_s = matcorr(gt_sources, ic_sources)

    # corr_mix = np.corrcoef(ic_mix, gt_mix)
    # corr_sources = np.corrcoef(ic_sources, gt_sources)
    #
    # id_sources = ic_mix.shape[0]
    #
    # corr_cross_mix = corr_mix[id_sources:, :id_sources] ** 2
    # corr_cross_sources = corr_sources[id_sources:, :id_sources] ** 2
    corr_cross_mix = corr_m**2
    corr_cross_sources = corr_s**2

    mix_CC_mean_gt = np.trace(corr_cross_mix)/n_sources
    mix_CC_mean_id = np.trace(corr_cross_mix)/len(ic_sources)
    sources_CC_mean = np.trace(corr_cross_sources)/n_sources

    return mix_CC_mean_gt, mix_CC_mean_id, sources_CC_mean, corr_cross_mix, corr_cross_sources



def annotate_overlapping(gtst, t_jitt = 1*pq.ms, overlapping_pairs=None, verbose=False, parallel=True):
    '''

    Parameters
    ----------
    gtst
    t_jitt
    overlapping_pairs
    verbose

    Returns
    -------

    '''
    nprocesses = len(gtst)
    if parallel:
        import multiprocessing
        # t_start = time.time()
        pool = multiprocessing.Pool(nprocesses)
        results = [pool.apply_async(annotate(i, st_i, gtst, overlapping_pairs, t_jitt, ))
                   for i, st_i in enumerate(gtst)]
    else:
        # find overlapping spikes
        for i, st_i in enumerate(gtst):
            if verbose:
                print 'SPIKETRAIN ', i
            over = np.array(['NO'] * len(st_i))
            for i_sp, t_i in enumerate(st_i):
                for j, st_j in enumerate(gtst):
                    if i != j:
                        # find overlapping
                        id_over = np.where((st_j > t_i - t_jitt) & (st_j < t_i + t_jitt))[0]
                        if not np.any(overlapping_pairs):
                            if len(id_over) != 0:
                                over[i_sp] = 'O'
                                # if verbose:
                                #     print 'found overlap! spike 1: ', i, t_i, ' spike 2: ', j, st_j[id_over]
                        else:
                            pair = [i, j]
                            pair_i = [j, i]
                            if np.any([np.all(pair == p) for p in overlapping_pairs]) or \
                                    np.any([np.all(pair_i == p) for p in overlapping_pairs]):
                                if len(id_over) != 0:
                                    over[i_sp] = 'SO'
                                    # if verbose:
                                    #     print 'found spatial overlap! spike 1: ', i, t_i, ' spike 2: ', j, st_j[id_over]
                            else:
                                if len(id_over) != 0:
                                    over[i_sp] = 'O'
                                    # if verbose:
                                    #     print 'found overlap! spike 1: ', i, t_i, ' spike 2: ', j, st_j[id_over]
            st_i.annotate(overlap=over)


def annotate(i, st_i, gtst, overlapping_pairs, t_jitt):
    print 'SPIKETRAIN ', i
    over = np.array(['NO'] * len(st_i))
    for i_sp, t_i in enumerate(st_i):
        for j, st_j in enumerate(gtst):
            if i != j:
                # find overlapping
                id_over = np.where((st_j > t_i - t_jitt) & (st_j < t_i + t_jitt))[0]
                if not np.any(overlapping_pairs):
                    if len(id_over) != 0:
                        over[i_sp] = 'O'
                        # if verbose:
                        #     print 'found overlap! spike 1: ', i, t_i, ' spike 2: ', j, st_j[id_over]
                else:
                    pair = [i, j]
                    pair_i = [j, i]
                    if np.any([np.all(pair == p) for p in overlapping_pairs]) or \
                            np.any([np.all(pair_i == p) for p in overlapping_pairs]):
                        if len(id_over) != 0:
                            over[i_sp] = 'SO'
                            # if verbose:
                            #     print 'found spatial overlap! spike 1: ', i, t_i, ' spike 2: ', j, st_j[id_over]
                    else:
                        if len(id_over) != 0:
                            over[i_sp] = 'O'
                            # if verbose:
                            #     print 'found overlap! spike 1: ', i, t_i, ' spike 2: ', j, st_j[id_over]
    st_i.annotate(overlap=over)

def raster_plots(st, bintype=False, ax=None, overlap=False, labels=False, color_st=None, color=None, fs=10,
                 marker='|', mew=2, markersize=5):
    '''

    Parameters
    ----------
    st
    bintype
    ax

    Returns
    -------

    '''
    import matplotlib.pylab as plt
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    for i, spiketrain in enumerate(st):
        t = spiketrain.rescale(pq.s)
        if bintype:
            if spiketrain.annotations['bintype'] == 'EXCIT':
                ax.plot(t, i * np.ones_like(t), 'b', marker=marker, mew=mew, markersize=markersize, ls='')
            elif spiketrain.annotations['bintype'] == 'INHIB':
                ax.plot(t, i * np.ones_like(t), 'r', marker=marker, mew=mew, markersize=markersize, ls='')
        else:
            if not overlap and not labels:
                if np.any(color_st):
                    import seaborn as sns
                    colors = sns.color_palette("Paired", len(color_st))
                    if i in color_st:
                        idx = np.where(color_st==i)[0][0]
                        ax.plot(t, i * np.ones_like(t), marker=marker, mew=mew, color=colors[idx], markersize=5, ls='')
                    else:
                        ax.plot(t, i * np.ones_like(t), 'k', marker=marker, mew=mew, markersize=markersize, ls='')
                elif color is not None:
                    if isinstance(color, list) or isinstance(color, np.ndarray):
                        ax.plot(t, i * np.ones_like(t), color=color[i], marker=marker, mew=mew, markersize=markersize, ls='')
                    else:
                        ax.plot(t, i * np.ones_like(t), color=color, marker=marker, mew=mew, markersize=markersize, ls='')
                else:
                    ax.plot(t, i * np.ones_like(t), 'k', marker=marker, mew=mew, markersize=markersize, ls='')
            elif overlap:
                for j, t_sp in enumerate(spiketrain):
                    if spiketrain.annotations['overlap'][j] == 'SO':
                        ax.plot(t_sp, i, 'r', marker=marker, mew=mew, markersize=markersize, ls='')
                    elif spiketrain.annotations['overlap'][j] == 'O':
                        ax.plot(t_sp, i, 'g', marker=marker, mew=mew, markersize=markersize, ls='')
                    elif spiketrain.annotations['overlap'][j] == 'NO':
                        ax.plot(t_sp, i, 'k', marker=marker, mew=mew, markersize=markersize, ls='')
            elif labels:
                for j, t_sp in enumerate(spiketrain):
                    if 'TP' in spiketrain.annotations['labels'][j]:
                        ax.plot(t_sp, i, 'g', marker=marker, mew=mew, markersize=markersize, ls='')
                    elif 'CL' in spiketrain.annotations['labels'][j]:
                        ax.plot(t_sp, i, 'y', marker=marker, mew=mew, markersize=markersize, ls='')
                    elif 'FN' in spiketrain.annotations['labels'][j]:
                        ax.plot(t_sp, i, 'r', marker=marker, mew=mew, markersize=markersize, ls='')
                    elif 'FP' in spiketrain.annotations['labels'][j]:
                        ax.plot(t_sp, i, 'm', marker=marker, mew=mew, markersize=markersize, ls='')
                    else:
                        ax.plot(t_sp, i, 'k', marker=marker, mew=mew, markersize=markersize, ls='')

    ax.axis('tight')
    ax.set_xlim([st[0].t_start.rescale(pq.s), st[0].t_stop.rescale(pq.s)])
    ax.set_xlabel('Time (ms)', fontsize=fs)
    ax.set_ylabel('Spike Train Index', fontsize=fs)
    plt.gca().tick_params(axis='both', which='major')

    return ax

def threshold_spike_sorting(recordings, threshold):
    '''

    Parameters
    ----------
    recordings
    threshold

    Returns
    -------

    '''
    spikes = {}
    for i_rec, rec in enumerate(recordings):
        sp_times = []
        if isinstance(threshold, list):
            idx_spikes = np.where(rec < threshold[i_rec])
        else:
            idx_spikes = np.where(rec < threshold)
        if len(idx_spikes[0]) != 0:
            idx_spikes = idx_spikes[0]
            for t, idx in enumerate(idx_spikes):
                # find single waveforms crossing thresholds
                if t == 0:
                    sp_times.append(idx)
                elif idx - idx_spikes[t - 1] > 1:  # or t == len(idx_spike) - 2:  # single spike
                    sp_times.append(idx)

            spikes.update({i_rec: sp_times})

    return spikes


def SNR_evaluation(gtst, sst, sources):
    '''
    Evaluates the SNR increase from ICA processing by comparing original SNR (from ground truth spike trains) and
    SNR computed from the ICA sources.
    Parameters
    ----------
    gtst
    sst
    sources

    Returns
    -------

    '''
    pass



###### KLUSTA #########
def save_binary_format(filename, signal, spikesorter='klusta', dtype='float32'):
    """Saves analog signals into klusta (time x chan) or spyking
    circus (chan x time) binary format (.dat)

    Parameters
    ----------
    filename : string
               absolute path (_klusta.dat or _spycircus.dat are appended)
    signal : np.array
             2d array of analog signals
    spikesorter : string
                  'klusta' or 'spykingcircus'

    Returns
    -------
    """
    if spikesorter is 'klusta':
        if not filename.endswith('dat'):
            filename += '.dat'
        print 'saving ', filename
        with open(filename, 'wb') as f:
            np.transpose(np.array(signal, dtype=dtype)).tofile(f)
    elif spikesorter is 'spykingcircus':
        if not filename.endswith('dat'):
            filename += '.dat'
        print 'saving ', filename
        with open(filename, 'wb') as f:
            np.array(signal, dtype=dtype).tofile(f)
    elif spikesorter is 'yass':
        if not filename.endswith('dat'):
            filename += '.dat'
        print 'saving ', filename
        with open(filename, 'wb') as f:
            np.transpose(np.array(signal, dtype=dtype)).tofile(f)
    elif spikesorter == 'kilosort' or spikesorter == 'none':
        if not filename.endswith('dat'):
            filename += '.dat'
        print 'saving ', filename
        with open(filename, 'wb') as f:
            np.transpose(np.array(signal, dtype=dtype)).tofile(f)
    return filename


def create_klusta_prm(pathname, prb_path, nchan=32, fs=30000,
                      klusta_filter=True, filter_low=300, filter_high=6000):
    """Creates klusta .prm files, with spikesorting parameters

    Parameters
    ----------
    pathname : string
               absolute path (_klusta.dat or _spycircus.dat are appended)
    prbpath : np.array
              2d array of analog signals
    nchan : int
            number of channels
    fs: float
        sampling frequency
    klusta_filter : bool
        filter with klusta or not
    filter_low: float
                low cutoff frequency (if klusta_filter is True)
    filter_high : float
                  high cutoff frequency (if klusta_filter is True)
    Returns
    -------
    full_filename : absolute path of .prm file
    """
    assert pathname is not None
    abspath = os.path.abspath(pathname)
    assert prb_path is not None
    prb_path = os.path.abspath(prb_path)
    full_filename = abspath + '.prm'

    if isinstance(fs, Quantity):
        fs = fs.rescale('Hz').magnitude

    extract_s_before = int(5*1e-4*fs)
    extract_s_after = int(1*1e-3*fs)

    print full_filename
    print('Saving ', full_filename)
    with open(full_filename, 'w') as f:
        f.write('\n')
        f.write('experiment_name = ' + "r'" + str(abspath) + '_klusta' + "'" + '\n')
        f.write('prb_file = ' + "r'" + str(prb_path) + "'")
        f.write('\n')
        f.write('\n')
        f.write("traces = dict(\n\traw_data_files=[experiment_name + '.dat'],\n\tvoltage_gain=1.,"
                "\n\tsample_rate="+str(fs)+",\n\tn_channels="+str(nchan)+",\n\tdtype='float32',\n)")
        f.write('\n')
        f.write('\n')
        f.write("spikedetekt = dict(")
        if klusta_filter:
            f.write("\n\tfilter_low="+str(filter_low)+",\n\tfilter_high="+str(filter_high)+","
                    "\n\tfilter_butter_order=3,\n\tfilter_lfp_low=0,\n\tfilter_lfp_high=300,\n")
        f.write("\n\tchunk_size_seconds=1,\n\tchunk_overlap_seconds=.015,\n"
                "\n\tn_excerpts=50,\n\texcerpt_size_seconds=1,"
                "\n\tthreshold_strong_std_factor=4.5,\n\tthreshold_weak_std_factor=2,\n\tdetect_spikes='negative',"
                "\n\n\tconnected_component_join_size=1,\n"
                "\n\textract_s_before="+str(extract_s_before)+",\n\textract_s_after="+str(extract_s_after)+",\n"
                "\n\tn_features_per_channel=3,\n\tpca_n_waveforms_max=10000,\n)")
        f.write('\n')
        f.write('\n')
        f.write("klustakwik2 = dict(\n\tnum_starting_clusters=50,\n)")
                # "\n\tnum_cpus=4,)")
    return full_filename


def export_prb_file(n_elec, electrode_name, pathname,
                    pos=None, adj_dist=None, graph=True, geometry=True, separate_channels=False,
                    spikesorter='klusta', radius=100):
    '''

    Parameters
    ----------
    n_elec
    electrode_name
    pathname
    pos
    adj_dist
    graph
    geometry
    separate_channels
    spikesorter
    radius

    Returns
    -------

    '''

    assert pathname is not None
    abspath = os.path.abspath(pathname)
    full_filename = join(abspath, electrode_name + '.prb')

    # find adjacency graph
    if graph:
        if pos is not None and adj_dist is not None:
            adj_graph = []
            for el1, el_pos1 in enumerate(pos):
                for el2, el_pos2 in enumerate(pos):
                    if el1 != el2:
                        if np.linalg.norm(el_pos1 - el_pos2) < adj_dist:
                            adj_graph.append((el1, el2))

    print 'Saving ', full_filename
    with open(full_filename, 'w') as f:
        f.write('\n')
        if spikesorter=='spykingcircus':
            f.write('total_nb_channels = ' + str(n_elec) + '\n')
            f.write('radius = ' + str(radius) + '\n')
        f.write('channel_groups = {\n')
        if not separate_channels:
            f.write("    0: ")
            f.write("\n        {\n")
            f.write("           'channels': " + str(range(n_elec)) + ',\n')
            if graph:
                f.write("           'graph':  " + str(adj_graph) + ',\n')
            else:
                f.write("           'graph':  [],\n")
            if geometry:
                f.write("           'geometry':  {\n")
                for i, pos in enumerate(pos):
                    f.write('               ' + str(i) +': ' + str(tuple(pos[1:])) + ',\n')
                f.write('           }\n')
            f.write('       }\n}')
            f.write('\n')
        else:
            for elec in range(n_elec):
                f.write('    ' + str(elec) + ': ')
                f.write("\n        {\n")
                f.write("           'channels': [" + str(elec) + '],\n')
                f.write("           'graph':  [],\n")
                f.write('        },\n')
            f.write('}\n')

    return full_filename


def extract_adjacency(pos, adj_dist):
    '''

    Parameters
    ----------
    pos
    adj_dist

    Returns
    -------

    '''
    if pos is not None and adj_dist is not None:
        adj_graph = []
        for el1, el_pos1 in enumerate(pos):
            adjacent_electrodes = []
            for el2, el_pos2 in enumerate(pos):
                if el1 != el2:
                    if np.linalg.norm(el_pos1 - el_pos2) < adj_dist:
                        adjacent_electrodes.append(el2)
            adj_graph.append(adjacent_electrodes)
    return adj_graph


def calc_MI(x, y, bins):
    '''

    Parameters
    ----------
    x
    y
    bins

    Returns
    -------

    '''
    from sklearn.metrics import mutual_info_score

    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


##### Methods from spike_sorting.py ###

def compute_performance(counts):

    tp_rate = float(counts['TP']) / counts['TOT_GT'] * 100
    tpo_rate = float(counts['TPO']) / counts['TOT_GT'] * 100
    tpso_rate = float(counts['TPSO']) / counts['TOT_GT'] * 100
    tot_tp_rate = float(counts['TP'] + counts['TPO'] + counts['TPSO']) / counts['TOT_GT'] * 100

    cl_rate = float(counts['CL']) / counts['TOT_GT'] * 100
    clo_rate = float(counts['CLO']) / counts['TOT_GT'] * 100
    clso_rate = float(counts['CLSO']) / counts['TOT_GT'] * 100
    tot_cl_rate = float(counts['CL'] + counts['CLO'] + counts['CLSO']) / counts['TOT_GT'] * 100

    fn_rate = float(counts['FN']) / counts['TOT_GT'] * 100
    fno_rate = float(counts['FNO']) / counts['TOT_GT'] * 100
    fnso_rate = float(counts['FNSO']) / counts['TOT_GT'] * 100
    tot_fn_rate = float(counts['FN'] + counts['FNO'] + counts['FNSO']) / counts['TOT_GT'] * 100

    fp_gt = float(counts['FP']) / counts['TOT_GT'] * 100
    fp_st = float(counts['FP']) / counts['TOT_ST'] * 100

    accuracy = tot_tp_rate / (tot_tp_rate + tot_fn_rate + fp_gt) * 100
    sensitivity = tot_tp_rate / (tot_tp_rate + tot_fn_rate) * 100
    miss_rate = tot_fn_rate / (tot_tp_rate + tot_fn_rate) * 100
    precision = tot_tp_rate / (tot_tp_rate + fp_gt) * 100
    false_discovery_rate = fp_gt / (tot_tp_rate + fp_gt) * 100

    print 'PERFORMANCE: \n'
    print '\nTP: ', tp_rate, ' %'
    print 'TPO: ', tpo_rate, ' %'
    print 'TPSO: ', tpso_rate, ' %'
    print 'TOT TP: ', tot_tp_rate, ' %'

    print '\nCL: ', cl_rate, ' %'
    print 'CLO: ', clo_rate, ' %'
    print 'CLSO: ', clso_rate, ' %'
    print 'TOT CL: ', tot_cl_rate, ' %'

    print '\nFN: ', fn_rate, ' %'
    print 'FNO: ', fno_rate, ' %'
    print 'FNSO: ', fnso_rate, ' %'
    print 'TOT FN: ', tot_fn_rate, ' %'

    print '\nFP (%GT): ', fp_gt, ' %'
    print '\nFP (%ST): ', fp_st, ' %'

    print '\nACCURACY: ', accuracy, ' %'
    print 'SENSITIVITY: ', sensitivity, ' %'
    print 'MISS RATE: ', miss_rate, ' %'
    print 'PRECISION: ', precision, ' %'
    print 'FALSE DISCOVERY RATE: ', false_discovery_rate, ' %'

    performance = {'tot_tp': tot_tp_rate, 'tot_cl': tot_cl_rate, 'tot_fn': tot_fn_rate, 'tot_fp': fp_gt,
                   'accuracy': accuracy, 'sensitivity': sensitivity, 'precision': precision, 'miss_rate': miss_rate,
                   'false_disc_rate': false_discovery_rate}

    return performance


def unit_SNR(sst, sources, times):
    '''

    Parameters
    ----------
    sst
    sources

    Returns
    -------

    '''
    between_spiks = 5*pq.ms
    noise_source = []

    for (st, s) in zip(sst, sources):
        t_idx = []
        for (t_pre, t_post) in zip(st[:-1], st[1:]):
            t_idx = np.append(t_idx, np.where((times > t_pre + between_spiks) & (times < t_post - between_spiks)))
        noise_source.append(s[t_idx.astype(int)])
        sd = np.std(noise_source[-1])
        mean_ic_amp = np.abs(np.mean(st.annotations['ica_amp']))
        # print t_idx[0]/32000., mean_ic_amp, sd
        st.annotate(ica_snr=mean_ic_amp/sd)

    return np.array(noise_source)


def interpolate_template(temp, mea_pos, mea_dim, x_shift=0, y_shift=0):
    from scipy import interpolate
    x = mea_pos[:, 1]
    y = mea_pos[:, 2]

    func = []
    for t in range(temp.shape[1]):
        func.append(interpolate.interp2d(x, y, temp[:, t], kind='cubic'))

    x1 = x + x_shift
    y1 = y + y_shift
    t_shift = []
    for f in func:
        t_shift.append([f(x_, y_) for (x_, y_) in zip(x1, y1)])

    return np.squeeze(np.array(t_shift)).swapaxes(0,1)


def convolve_single_template(spike_id, spike_bin, template, modulation=False, amp_mod=None):
    if len(template.shape) == 2:
        njitt = template.shape[0]
        len_spike = template.shape[1]
    else:
        len_spike = template.shape[0]
    spike_pos = np.where(spike_bin == 1)[0]
    n_samples = len(spike_bin)
    gt_source = np.zeros(n_samples)

    if len(template.shape) == 2:
        rand_idx = np.random.randint(njitt)
        print 'rand_idx: ', rand_idx
        temp_jitt = template[rand_idx]
        if not modulation:
            for pos, spos in enumerate(spike_pos):
                if spos - len_spike // 2 >= 0 and spos - len_spike // 2 + len_spike <= n_samples:
                    gt_source[spos - len_spike // 2:spos - len_spike // 2 + len_spike] +=  temp_jitt
                elif spos - len_spike // 2 < 0:
                    diff = -(spos - len_spike // 2)
                    gt_source[:spos - len_spike // 2 + len_spike] += temp_jitt[diff:]
                else:
                    diff = n_samples - (spos - len_spike // 2)
                    gt_source[spos - len_spike // 2:] += temp_jitt[:diff]
        else:
            print 'Template-Electrode modulation'
            for pos, spos in enumerate(spike_pos):
                if spos - len_spike // 2 >= 0 and spos - len_spike // 2 + len_spike <= n_samples:
                    gt_source[spos - len_spike // 2:spos - len_spike // 2 + len_spike] += amp_mod[pos]*temp_jitt
                elif spos - len_spike // 2 < 0:
                    diff = -(spos - len_spike // 2)
                    gt_source[:spos - len_spike // 2 + len_spike] += amp_mod[pos]*temp_jitt[diff:]
                else:
                    diff = n_samples - (spos - len_spike // 2)
                    gt_source[spos - len_spike // 2:] += amp_mod[pos]*temp_jitt[:diff]
    else:
        print 'No modulation'
        for pos, spos in enumerate(spike_pos):
            if spos - len_spike // 2 >= 0 and spos - len_spike // 2 + len_spike <= n_samples:
                gt_source[spos - len_spike // 2:spos - len_spike // 2 + len_spike] += template
            elif spos - len_spike // 2 < 0:
                diff = -(spos - len_spike // 2)
                gt_source[:spos - len_spike // 2 + len_spike] += template[diff:]
            else:
                diff = n_samples - (spos - len_spike // 2)
                gt_source[spos - len_spike // 2:] += template[:diff]

    return gt_source


def convolve_templates_spiketrains(spike_id, spike_bin, template, modulation=False, amp_mod=None, recordings=[]):
    print 'START: convolution with spike ', spike_id
    if len(template.shape) == 3:
        njitt = template.shape[0]
        n_elec = template.shape[1]
        len_spike = template.shape[2]
    else:
        n_elec = template.shape[0]
        len_spike = template.shape[1]
    n_samples = len(spike_bin)
    if len(recordings) == 0:
        recordings = np.zeros((n_elec, n_samples))

    # recordings_test = np.zeros((n_elec, n_samples))
    if not modulation:
        spike_pos = np.where(spike_bin == 1)[0]
        amp_mod = np.ones_like(spike_pos)
        if len(template.shape) == 3:
            rand_idx = np.random.randint(njitt)
            print 'rand_idx: ', rand_idx
            temp_jitt = template[rand_idx]
            print 'No modulation'
            for pos, spos in enumerate(spike_pos):
                if spos - len_spike // 2 >= 0 and spos - len_spike // 2 + len_spike <= n_samples:
                    recordings[:, spos - len_spike // 2:spos - len_spike // 2 + len_spike] += amp_mod[pos] * temp_jitt
                elif spos - len_spike // 2 < 0:
                    diff = -(spos - len_spike // 2)
                    recordings[:, :spos - len_spike // 2 + len_spike] += amp_mod[pos] * temp_jitt[:, diff:]
                else:
                    diff = n_samples - (spos - len_spike // 2)
                    recordings[:, spos - len_spike // 2:] += amp_mod[pos] * temp_jitt[:, :diff]
        else:
            print 'No jitter'
            for pos, spos in enumerate(spike_pos):
                if spos - len_spike // 2 >= 0 and spos - len_spike // 2 + len_spike <= n_samples:
                    recordings[:, spos - len_spike // 2:spos - len_spike // 2 + len_spike] += amp_mod[
                                                                                                  pos] * template
                elif spos - len_spike // 2 < 0:
                    diff = -(spos - len_spike // 2)
                    recordings[:, :spos - len_spike // 2 + len_spike] += amp_mod[pos] * template[:, diff:]
                else:
                    diff = n_samples - (spos - len_spike // 2)
                    recordings[:, spos - len_spike // 2:] += amp_mod[pos] * template[:, :diff]
    else:
        assert amp_mod is not None
        spike_pos = np.where(spike_bin == 1)[0]
        if len(template.shape) == 3:
            rand_idx = np.random.randint(njitt)
            print 'rand_idx: ', rand_idx
            temp_jitt = template[rand_idx]
            if not isinstance(amp_mod[0], (list, tuple, np.ndarray)):
                print 'Template modulation'
                for pos, spos in enumerate(spike_pos):
                    if spos-len_spike//2 >= 0 and spos-len_spike//2+len_spike <= n_samples:
                        recordings[:, spos-len_spike//2:spos-len_spike//2+len_spike] +=  amp_mod[pos] * temp_jitt
                    elif spos-len_spike//2 < 0:
                        diff = -(spos-len_spike//2)
                        recordings[:, :spos - len_spike // 2 + len_spike] += amp_mod[pos] * temp_jitt[:, diff:]
                    else:
                        diff = n_samples-(spos - len_spike // 2)
                        recordings[:, spos - len_spike // 2:] += amp_mod[pos] * temp_jitt[:, :diff]
            else:
                print 'Electrode modulation'
                for pos, spos in enumerate(spike_pos):
                    if spos-len_spike//2 >= 0 and spos-len_spike//2+len_spike <= n_samples:
                        recordings[:, spos - len_spike // 2:spos - len_spike // 2 + len_spike] += \
                            [a * t for (a, t) in zip(amp_mod[pos], temp_jitt)]
                    elif spos-len_spike//2 < 0:
                        diff = -(spos-len_spike//2)
                        recordings[:, :spos - len_spike // 2 + len_spike] += \
                            [a * t for (a, t) in zip(amp_mod[pos], temp_jitt[:, diff:])]
                        # recordings[:, :spos - len_spike // 2 + len_spike] += amp_mod[pos] * template[:, diff:]
                    else:
                        diff = n_samples-(spos - len_spike // 2)
                        recordings[:, spos - len_spike // 2:] += \
                            [a * t for (a, t) in zip(amp_mod[pos], temp_jitt[:, :diff])]
        else:
            if not isinstance(amp_mod[0], (list, tuple, np.ndarray)):
                print 'Template modulation'
                for pos, spos in enumerate(spike_pos):
                    if spos - len_spike // 2 >= 0 and spos - len_spike // 2 + len_spike <= n_samples:
                        recordings[:, spos - len_spike // 2:spos - len_spike // 2 + len_spike] += amp_mod[
                                                                                                      pos] * template
                    elif spos - len_spike // 2 < 0:
                        diff = -(spos - len_spike // 2)
                        recordings[:, :spos - len_spike // 2 + len_spike] += amp_mod[pos] * template[:, diff:]
                    else:
                        diff = n_samples - (spos - len_spike // 2)
                        recordings[:, spos - len_spike // 2:] += amp_mod[pos] * template[:, :diff]

            else:
                print 'Electrode modulation'
                for pos, spos in enumerate(spike_pos):
                    if spos-len_spike//2 >= 0 and spos-len_spike//2+len_spike <= n_samples:
                        recordings[:, spos - len_spike // 2:spos - len_spike // 2 + len_spike] += \
                            [a * t for (a, t) in zip(amp_mod[pos], template)]
                    elif spos-len_spike//2 < 0:
                        diff = -(spos-len_spike//2)
                        recordings[:, : spos - len_spike // 2 + len_spike] += \
                            [a * t for (a, t) in zip(amp_mod[pos], template[:, diff:])]
                        # recordings[:, :spos - len_spike // 2 + len_spike] += amp_mod[pos] * template[:, diff:]
                    else:
                        diff = n_samples-(spos - len_spike // 2)
                        recordings[:, spos - len_spike // 2:] += \
                            [a * t for (a, t) in zip(amp_mod[pos], template[:, :diff])]
                        # recordings[:, spos - len_spike // 2:] += amp_mod[pos] * template[:, :diff]


    print 'DONE: convolution with spike ', spike_id

    return recordings


def convolve_drifting_templates_spiketrains(spike_id, spike_bin, template, fs, loc, v_drift, t_start_drift,
                                            modulation=False, amp_mod=None, recordings=[], n_step_sec=1):
    print 'START: convolution with spike ', spike_id
    if len(template.shape) == 4:
        njitt = template.shape[1]
        n_elec = template.shape[2]
        len_spike = template.shape[3]
    elif len(template.shape) == 3:
        n_elec = template.shape[1]
        len_spike = template.shape[2]
    n_samples = len(spike_bin)
    n_step_sec = 1
    dur = (n_samples / fs).rescale('s').magnitude
    t_steps = np.arange(0, dur, n_step_sec)
    n_step_sample = n_step_sec * int(fs.magnitude)
    dt = 2 ** -5

    mixing = np.zeros((int(n_samples/float(fs.rescale('Hz').magnitude)), n_elec))
    if len(recordings) == 0:
        recordings = np.zeros((n_elec, n_samples))

    # recordings_test = np.zeros((n_elec, n_samples))
    if not modulation:
        spike_pos = np.where(spike_bin == 1)[0]
        amp_mod = np.ones_like(spike_pos)
        if len(template.shape) == 4:
            rand_idx = np.random.randint(njitt)
            print 'rand_idx: ', rand_idx
            print 'No modulation'
            for pos, spos in enumerate(spike_pos):
                sp_time = spos / fs
                if sp_time < t_start_drift:
                    print sp_time, 'No drift', loc[0]
                    temp_idx = 0
                    temp_jitt = template[temp_idx, rand_idx]
                else:
                    # compute current position
                    new_pos = np.array(loc[0, 1:] + v_drift * (sp_time - t_start_drift).rescale('s').magnitude)
                    temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc[:, 1:]])
                    print sp_time, temp_idx, 'Drifting', new_pos, loc[temp_idx, 1:]
                    temp_jitt = template[temp_idx, rand_idx]

                if spos - len_spike // 2 >= 0 and spos - len_spike // 2 + len_spike <= n_samples:
                    recordings[:, spos - len_spike // 2:spos - len_spike // 2 + len_spike] += amp_mod[pos] * temp_jitt
                elif spos - len_spike // 2 < 0:
                    diff = -(spos - len_spike // 2)
                    recordings[:, :spos - len_spike // 2 + len_spike] += amp_mod[pos] * temp_jitt[:, diff:]
                else:
                    diff = n_samples - (spos - len_spike // 2)
                    recordings[:, spos - len_spike // 2:] += amp_mod[pos] * temp_jitt[:, :diff]

            for i, t in enumerate(t_steps):
                if t < t_start_drift:
                    temp_idx = 0
                    temp_jitt = template[temp_idx, rand_idx]
                else:
                    # compute current position
                    new_pos = np.array(loc[0, 1:] + v_drift * (t - t_start_drift.rescale('s').magnitude))
                    temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc[:, 1:]])
                    temp_jitt = template[temp_idx, rand_idx]

                feat = get_EAP_features(np.squeeze(temp_jitt), ['Na'], dt=dt)
                mixing[i] = -np.squeeze(feat['na'])
        else:
            print 'No jitter'
            for pos, spos in enumerate(spike_pos):
                sp_time = spos / fs
                if sp_time < t_start_drift:
                    temp_idx = 0
                    temp = template[temp_idx]
                else:
                    # compute current position
                    new_pos = np.array(loc[0, 1:] + v_drift * (sp_time - t_start_drift).rescale('s').magnitude)
                    temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc[:, 1:]])
                    temp = template[temp_idx]
                if spos - len_spike // 2 >= 0 and spos - len_spike // 2 + len_spike <= n_samples:
                    recordings[:, spos - len_spike // 2:spos - len_spike // 2 + len_spike] += amp_mod[pos] * temp
                elif spos - len_spike // 2 < 0:
                    diff = -(spos - len_spike // 2)
                    recordings[:, :spos - len_spike // 2 + len_spike] += amp_mod[pos] * temp[:, diff:]
                else:
                    diff = n_samples - (spos - len_spike // 2)
                    recordings[:, spos - len_spike // 2:] += amp_mod[pos] * temp[:, :diff]
            for i, t in enumerate(t_steps):
                if t < t_start_drift:
                    temp_idx = 0
                    temp_jitt = template[temp_idx]
                else:
                    # compute current position
                    new_pos = np.array(loc[0, 1:] + v_drift * (t - t_start_drift.rescale('s').magnitude))
                    temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc[:, 1:]])
                    temp_jitt = template[temp_idx]

                feat = get_EAP_features(np.squeeze(temp_jitt), ['Na'], dt=dt)
                mixing[i] = -np.squeeze(feat['na'])
    else:
        assert amp_mod is not None
        spike_pos = np.where(spike_bin == 1)[0]
        if len(template.shape) == 4:
            rand_idx = np.random.randint(njitt)
            print 'rand_idx: ', rand_idx
            if not isinstance(amp_mod[0], (list, tuple, np.ndarray)):
                print 'Template modulation'
                for pos, spos in enumerate(spike_pos):
                    sp_time = spos / fs
                    if sp_time < t_start_drift:
                        print sp_time, 'No drift', loc[0]
                        temp_idx = 0
                        temp_jitt = template[temp_idx, rand_idx]
                    else:
                        # compute current position
                        new_pos = np.array(loc[0, 1:] + v_drift * (sp_time - t_start_drift).rescale('s').magnitude)
                        temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc[:, 1:]])
                        temp_jitt = template[temp_idx, rand_idx]
                        print sp_time, temp_idx, 'Drifting', new_pos, loc[temp_idx, 1:]
                    if spos - len_spike // 2 >= 0 and spos - len_spike // 2 + len_spike <= n_samples:
                        recordings[:, spos - len_spike // 2:spos - len_spike // 2 + len_spike] += amp_mod[pos] \
                                                                                                  * temp_jitt
                    elif spos - len_spike // 2 < 0:
                        diff = -(spos - len_spike // 2)
                        recordings[:, :spos - len_spike // 2 + len_spike] += amp_mod[pos] * temp_jitt[:, diff:]
                    else:
                        diff = n_samples - (spos - len_spike // 2)
                        recordings[:, spos - len_spike // 2:] += amp_mod[pos] * temp_jitt[:, :diff]
            else:
                print 'Electrode modulation'
                for pos, spos in enumerate(spike_pos):
                    sp_time = spos / fs
                    if sp_time < t_start_drift:
                        temp_idx = 0
                        temp_jitt = template[temp_idx, rand_idx]
                        print sp_time, 'No drift', loc[0]
                        if spos - len_spike // 2 >= 0 and spos - len_spike // 2 + len_spike <= n_samples:
                            recordings[:, spos - len_spike // 2:spos - len_spike // 2 + len_spike] += \
                                [a * t for (a, t) in zip(amp_mod[pos], temp_jitt)]
                        elif spos - len_spike // 2 < 0:
                            diff = -(spos - len_spike // 2)
                            recordings[:, :spos - len_spike // 2 + len_spike] += \
                                [a * t for (a, t) in zip(amp_mod[pos], temp_jitt[:, diff:])]
                            # recordings[:, :spos - len_spike // 2 + len_spike] += amp_mod[pos] * template[:, diff:]
                        else:
                            diff = n_samples - (spos - len_spike // 2)
                            recordings[:, spos - len_spike // 2:] += \
                                [a * t for (a, t) in zip(amp_mod[pos], temp_jitt[:, :diff])]
                    else:
                        # compute current position
                        new_pos = np.array(loc[0, 1:] + v_drift * (sp_time - t_start_drift).rescale('s').magnitude)
                        temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc[:, 1:]])
                        new_temp_jitt = template[temp_idx, rand_idx]
                        print sp_time, temp_idx, 'Drifting', new_pos, loc[temp_idx, 1:]
                        if spos - len_spike // 2 >= 0 and spos - len_spike // 2 + len_spike <= n_samples:
                            recordings[:, spos - len_spike // 2:spos - len_spike // 2 + len_spike] += \
                                [a * t for (a, t) in zip(amp_mod[pos], new_temp_jitt)]
                        elif spos - len_spike // 2 < 0:
                            diff = -(spos - len_spike // 2)
                            recordings[:, :spos - len_spike // 2 + len_spike] += \
                                [a * t for (a, t) in zip(amp_mod[pos], new_temp_jitt[:, diff:])]
                            # recordings[:, :spos - len_spike // 2 + len_spike] += amp_mod[pos] * template[:, diff:]
                        else:
                            diff = n_samples - (spos - len_spike // 2)
                            recordings[:, spos - len_spike // 2:] += \
                                [a * t for (a, t) in zip(amp_mod[pos], new_temp_jitt[:, :diff])]
                for i, t in enumerate(t_steps):
                    if t < t_start_drift:
                        temp_idx = 0
                        temp_jitt = template[temp_idx, rand_idx]
                    else:
                        # compute current position
                        new_pos = np.array(loc[0, 1:] + v_drift * (t - t_start_drift.rescale('s').magnitude))
                        temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc[:, 1:]])
                        temp_jitt = template[temp_idx, rand_idx]

                    feat = get_EAP_features(np.squeeze(temp_jitt), ['Na'], dt=dt)
                    mixing[i] = -np.squeeze(feat['na'])
        else:
            print 'No jitter'
            if not isinstance(amp_mod[0], (list, tuple, np.ndarray)):
                print 'Template modulation'
                for pos, spos in enumerate(spike_pos):
                    sp_time = spos / fs
                    if sp_time < t_start_drift:
                        temp_idx = 0
                        temp = template[temp_idx]
                    else:
                        # compute current position
                        new_pos = np.array(pos[0, 1:] + v * (sp_time - t_start_drift))
                        temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc[:, 1:]])
                        temp = template[temp_idx]
                    if spos - len_spike // 2 >= 0 and spos - len_spike // 2 + len_spike <= n_samples:
                        recordings[:, spos - len_spike // 2:spos - len_spike // 2 + len_spike] += amp_mod[pos] * temp
                    elif spos - len_spike // 2 < 0:
                        diff = -(spos - len_spike // 2)
                        recordings[:, :spos - len_spike // 2 + len_spike] += amp_mod[pos] * temp[:, diff:]
                    else:
                        diff = n_samples - (spos - len_spike // 2)
                        recordings[:, spos - len_spike // 2:] += amp_mod[pos] * temp[:, :diff]

            else:
                print 'Electrode modulation'
                for pos, spos in enumerate(spike_pos):
                    sp_time = spos / fs
                    if sp_time < t_start_drift:
                        temp_idx = 0
                        temp = template[temp_idx]
                    else:
                        # compute current position
                        new_pos = np.array(pos[0, 1:] + v * (sp_time - t_start_drift))
                        temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc[:, 1:]])
                        temp = template[temp_idx]
                    if spos - len_spike // 2 >= 0 and spos - len_spike // 2 + len_spike <= n_samples:
                        recordings[:, spos - len_spike // 2:spos - len_spike // 2 + len_spike] += \
                            [a * t for (a, t) in zip(amp_mod[pos], temp)]
                    elif spos - len_spike // 2 < 0:
                        diff = -(spos - len_spike // 2)
                        recordings[:, :spos - len_spike // 2 + len_spike] += \
                            [a * t for (a, t) in zip(amp_mod[pos], temp[:, diff:])]
                        # recordings[:, :spos - len_spike // 2 + len_spike] += amp_mod[pos] * template[:, diff:]
                    else:
                        diff = n_samples - (spos - len_spike // 2)
                        recordings[:, spos - len_spike // 2:] += \
                            [a * t for (a, t) in zip(amp_mod[pos], temp[:, :diff])]

            for i, t in enumerate(t_steps):
                if t < t_start_drift:
                    temp_idx = 0
                    temp_jitt = template[temp_idx]
                else:
                    # compute current position
                    new_pos = np.array(loc[0, 1:] + v_drift * (t - t_start_drift.rescale('s').magnitude))
                    temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc[:, 1:]])
                    temp_jitt = template[temp_idx]

    final_pos = loc[temp_idx]
    print 'DONE: convolution with spike ', spike_id

    return recordings, final_pos, mixing




def find_consistent_sorces(source_idx, thresh=0.5):
    '''
    Returns sources that appear at least thresh % of the times
    Parameters
    ----------
    source_idx
    thresh

    Returns
    -------

    '''
    len_no_empty = len([s for s in source_idx if len(s)>0])

    s_dict = {}
    for s in source_idx:
        for id in s:
            if id not in s_dict.keys():
                s_dict.update({id: 1})
            else:
                s_dict[id] += 1

    consistent_sources = []
    for id in s_dict.keys():
        if s_dict[id] >= thresh*len_no_empty:
            consistent_sources.append(id)

    return np.sort(consistent_sources)


def plot_mixing(mixing, mea_dim, mea_name=None, gs=None, fig=None):
    '''

    Parameters
    ----------
    mixing
    mea_dim
    mea_pitch

    Returns
    -------

    '''
    from neuroplot import plot_weight

    n_sources = len(mixing)
    cols = int(np.ceil(np.sqrt(n_sources)))
    rows = int(np.ceil(n_sources / float(cols)))

    if fig is None and gs is None:
        fig = plt.figure()

    if gs is not None:
        from matplotlib import gridspec
        gs_plot = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs)

    for n in range(n_sources):
        if gs is not None:
            ax_w = fig.add_subplot(gs_plot[n])
        else:
            ax_w = fig.add_subplot(rows, cols, n+1)
        mix = mixing[n]/np.ptp(mixing[n])
        if mea_name == 'Neuronexus-32-Kampff':
            mix = np.insert(mix, [0, 10, -10, 32], np.nan)
            plot_weight(mix, [3, 12], ax=ax_w)
        else:
            plot_weight(mix, mea_dim, ax=ax_w)

    return fig


# def play_mixing(mixing, mea_dim):
#     '''
#
#     Parameters
#     ----------
#     mixing
#     mea_dim
#
#     Returns
#     -------
#
#     '''
#     from neuroplot import plot_weight
#
#     n_sources = len(mixing)
#     cols = int(np.ceil(np.sqrt(n_sources)))
#     rows = int(np.ceil(n_sources / float(cols)))
#     fig_t = plt.figure()
#
#     anim = []
#
#     for n in range(n_sources):
#         ax_w = fig_t.add_subplot(rows, cols, n+1)
#         mix = mixing[n]/np.ptp(mixing[n])
#         im = play_spike(mix, mea_dim, ax=ax_w, fig=fig_t)
#         anim.append(im)
#
#     return fig_t


def plot_templates(templates, mea_pos, mea_pitch, single_figure=True):
    '''

    Parameters
    ----------
    templates
    mea_pos
    mea_pitch

    Returns
    -------

    '''
    from neuroplot import plot_weight

    n_sources = len(templates)
    fig_t = plt.figure()

    if single_figure:
        colors = plt.rcParams['axes.color_cycle']
        ax_t = fig_t.add_subplot(111)

        for n, t in enumerate(templates):
            print 'Plotting spike ', n, ' out of ', n_sources
            if len(t.shape) == 3:
                # plot_mea_recording(w[:5], mea_pos, mea_pitch, colors=colors[np.mod(n, len(colors))], ax=ax_w, lw=0.1)
                plot_mea_recording(t.mean(axis=0), mea_pos, mea_pitch, colors=colors[np.mod(n, len(colors))], ax=ax_t, lw=2)
            else:
                plot_mea_recording(t, mea_pos, mea_pitch, colors=colors[np.mod(n, len(colors))], ax=ax_t, lw=2)

    else:
        cols = int(np.ceil(np.sqrt(n_sources)))
        rows = int(np.ceil(n_sources / float(cols)))

        for n in range(n_sources):
            ax_t = fig_t.add_subplot(rows, cols, n+1)
            plot_mea_recording(templates[n], mea_pos, mea_pitch, ax=ax_t)

    return fig_t



def templates_weights(templates, weights, mea_pos, mea_dim, mea_pitch, pairs=None):
    '''

    Parameters
    ----------
    templates
    weights
    mea_pos
    mea_dim
    mea_pitch
    pairs: (optional)

    Returns
    -------

    '''
    from matplotlib import gridspec
    n_neurons = len(templates)
    gs0 = gridspec.GridSpec(1, 2)

    cols = int(np.ceil(np.sqrt(n_neurons)))
    rows = int(np.ceil(n_neurons / float(cols)))

    gs_t = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs0[0])
    gs_w = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs0[1])
    fig_t = plt.figure()

    min_v = np.min(templates)

    for n in range(n_neurons):
        ax_t = fig_t.add_subplot(gs_t[n])
        ax_w = fig_t.add_subplot(gs_w[n])

        plot_mea_recording(templates[n], mea_pos, mea_pitch, ax=ax_t)
        if pairs is not None:
            if pairs[n, 0] != -1:
                plot_weight(weights[pairs[n, 1]], mea_dim, ax=ax_w)
            else:
                ax_w.axis('off')
        else:
            plot_weight(weights[n], mea_dim, ax=ax_w)

    return fig_t

def plot_ic_waveforms(spiketrains):
    ica_sources = [s.annotations['ica_source'] for s in spiketrains]
    sources = np.unique(ica_sources)
    n_sources = len(sources)
    cols = int(np.ceil(np.sqrt(n_sources)))
    rows = int(np.ceil(n_sources / float(cols)))
    fig_t = plt.figure()
    colors = plt.rcParams['axes.color_cycle']

    for n, s in enumerate(sources):
        ax_t = fig_t.add_subplot(rows, cols, n + 1)

        count = 0
        for st, ic in enumerate(ica_sources):
            if ic == s:
                ax_t.plot(spiketrains[st].annotations['ica_wf'].T, color=colors[count], lw=0.2)
                ax_t.plot(np.mean(spiketrains[st].annotations['ica_wf'], axis=0), color=colors[count], lw=1)
                count += 1

    return fig_t

def plot_waveforms(spiketrains, mea_pos, mea_pitch):
    wf = [s.waveforms for s in spiketrains]
    fig_w = plt.figure()
    ax_w = fig_w.add_subplot(111)
    colors = plt.rcParams['axes.color_cycle']

    for n, w in enumerate(wf):
        print 'Plotting spike ', n, ' out of ', len(wf)
        # plot_mea_recording(w[:5], mea_pos, mea_pitch, colors=colors[np.mod(n, len(colors))], ax=ax_w, lw=0.1)
        plot_mea_recording(w.mean(axis=0), mea_pos, mea_pitch, colors=colors[np.mod(n, len(colors))], ax=ax_w, lw=2)

    return fig_w

def plot_matched_raster(gtst, sst, pairs):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    raster_plots(gtst, color_st=pairs[:, 0], ax=ax1)
    raster_plots(sst, color_st=pairs[:, 1], ax=ax2)

    fig.tight_layout()
    return ax1, ax2

def play_mixing(mixing, mea_dim, time=None, save=False, ax=None, fig=None, file=None, origin='lower'):
    '''

    Parameters
    ----------
    mixing: Nframes x Nchannels
    mea_dim
    time
    save
    ax
    fig
    file

    Returns
    -------

    '''
    #  check if number of mixing is 1
    if len(mixing.shape) == 3:
        print 'Plot one mixing at a time!'
        return
    else:
        if time:
            inter = time
        else:
            inter = 20

    # if save:
    #     plt.switch_backend('agg')
    # else:
    #     plt.switch_backend('qt4agg')
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111)
    ax.axis('off')
    z_min = np.min(mixing)
    z_max = np.max(mixing)

    im0 = ax.imshow(np.zeros((mea_dim[0], mea_dim[1])), vmin=z_min, vmax=z_max, origin=origin)
    if ax is None:
        fig.colorbar(im0)
    ims = []

    if (mea_dim[0] * mea_dim[1]) == mixing.shape[1]:
        for t in range(mixing.shape[0]):
            ims.append([ax.imshow(mixing[t, :].reshape(mea_dim).T,
                                   vmin=z_min, vmax=z_max, origin=origin)])

    im_ani = animation.ArtistAnimation(fig, ims, interval=inter, repeat_delay=2500, blit=True)

    if save:
        plt.switch_backend('agg')
        mywriter = animation.FFMpegWriter(fps=60)
        if file:
            im_ani.save(file, writer=mywriter)
        else:
            im_ani.save('mixing.mp4', writer=mywriter)

    return im_ani


def play_mea_recording(rec, mea_pos, mea_pitch, fs, window=1, step=0.1, colors=None, lw=1, ax=None, fig=None, spacing=None,
                       scalebar=False, time=None, dt=None, vscale=None, spikes=None, repeat=False, interval=10):
    '''

    Parameters
    ----------
    rec
    mea_pos
    mea_pitch
    fs
    window
    step
    colors
    lw
    ax
    spacing
    scalebar
    time
    dt
    vscale

    Returns
    -------

    '''
    n_window = int(fs*window)
    n_step = int(fs*step)
    start = np.arange(0, rec.shape[1], n_step)

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(1,1,1, frameon=False)
        no_tight = False
    else:
        no_tight = True

    if spacing is None:
        spacing = 0.1*np.max(mea_pitch)

    # normalize to min peak
    if vscale is None:
        LFPmin = 1.5 * np.max(np.abs(rec))
        rec_norm = rec / LFPmin * mea_pitch[1]
    else:
        rec_norm = rec / float(vscale) * mea_pitch[1]

    if colors is None:
        if len(rec.shape) > 2:
            colors = plt.rcParams['axes.color_cycle']
        else:
            colors='k'

    number_electrode = mea_pos.shape[0]
    lines = []
    for el in range(number_electrode):
        if len(rec.shape) == 3:  # multiple
            raise Exception('Dimensions should be Nchannels x Nsamples')
        else:
            line, = ax.plot(np.linspace(0, mea_pitch[0] - spacing, n_window) + mea_pos[el, 1],
                            np.zeros(n_window) + mea_pos[el, 2], color=colors, lw=lw)
            lines.append(line)

    text = ax.text(0.7, 0, 'Time: ',
                   color='k', fontsize=15, transform=ax.transAxes)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    def update(i):
        if n_window + i < rec.shape[1]:
            for el in range(number_electrode):
                lines[el].set_ydata(rec_norm[el, i:n_window + i] + mea_pos[el, 2])
        else:
            for el in range(number_electrode):
                lines[el].set_ydata(np.pad(rec_norm[el, i:],
                                           (0, n_window - (rec.shape[1] - i)), 'constant') + mea_pos[el, 2])

        text.set_text('Time: ' + str(round(i/float(fs),1)) + ' s')

        return tuple(lines) + (text,)

    anim = animation.FuncAnimation(fig, update, start, interval=interval, blit=True, repeat=False)
    fig.tight_layout()

    return anim


def play_ica_spiking(mixing, source_idx, mea_dim, gs=None, interval=100, ax=None, fig=None, tstep=None, origin='lower'):
    '''

    Parameters
    ----------
    mixing
    source_idx
    mea_dim
    fs
    interval
    ax
    fig

    Returns
    -------

    '''

    steps = np.arange(len(mixing))

    if fig is None and gs is None:
        fig = plt.figure()
    # if ax is None:
    #     ax = fig.add_subplot(1,1,1, frameon=False)
    #     no_tight = False
    else:
        no_tight = True


    n_sources = mixing[0].shape[0]
    cols = int(np.ceil(np.sqrt(n_sources)))
    rows = int(np.ceil(n_sources / float(cols)))

    if gs is not None:
        from matplotlib import gridspec
        gs_plot = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs)

    images = []
    for ss in range(n_sources):
        if gs is None:
            ax = fig.add_subplot(rows, cols, ss+1)
        else:
            ax = fig.add_subplot(gs_plot[ss])
        ax.axis('off')
        im = ax.imshow(np.zeros(mea_dim).T, animated=True, vmin=np.min(mixing[:, ss]), vmax=np.max(mixing[:, ss]),
                       origin=origin)
        im.set_visible(False)
        images.append(im)

    if tstep is not None:
        # text = fig.suptitle('', color='k', fontsize=15)
        text = ax.text(0, 0, '', color='k', fontsize=15, transform=ax.transAxes)
    else:
        text = fig.suptitle('')

    def update(i, n, images):
        im = images[n]
        if n in source_idx[i]:
            im.set_array(mixing[i, n].reshape(mea_dim).T)
            im.set_visible(True)
        else:
            im.set_visible(False)
        return im,

    def updateText(i):
        if tstep is not None:
            # text = fig.suptitle('Time: ' + str(round(i * tstep, 1)) + ' s')
            text.set_text(str(round(i * tstep, 1)) + ' s')
        return text,

    anim = []
    for n in range(n_sources):
        anim.append(animation.FuncAnimation(fig, update, steps, fargs=[n, images], interval=interval, blit=True))
    anim.append(animation.FuncAnimation(fig, updateText, steps, interval=interval, blit=False))

    print len(anim)

    return anim




