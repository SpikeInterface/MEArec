import numpy as np
import quantities as pq
from quantities import Quantity
import yaml
import json
import neo
import elephant
import time
import shutil
import os
from os.path import join
import MEAutility as mu
import h5py
from pathlib import Path
from distutils.version import StrictVersion

if StrictVersion(yaml.__version__) >= StrictVersion('5.0.0'):
    use_loader = True
else:
    use_loader = False


### GET DEFAULT SETTINGS ###
def get_default_config():
    """
    Returns default_info and mearec_home path.

    Returns
    -------
    default_info : dict
        Default_info from config file
    mearec_path : str
        Mearec home path
    """
    this_dir, this_filename = os.path.split(__file__)
    this_dir = Path(this_dir)
    home = Path(os.path.expanduser("~"))
    mearec_home = home / '.config' / 'mearec'
    if not (mearec_home / 'mearec.conf').is_file():
        mearec_home.mkdir(exist_ok=True)
        shutil.copytree(str(this_dir / 'default_params'), str(mearec_home / 'default_params'))
        shutil.copytree(str(this_dir / 'cell_models'), str(mearec_home / 'cell_models'))
        default_info = {'templates_params': str(mearec_home / 'default_params' / 'templates_params.yaml'),
                        'recordings_params': str(mearec_home / 'default_params' / 'recordings_params.yaml'),
                        'templates_folder': str(mearec_home / 'templates'),
                        'recordings_folder': str(mearec_home / 'recordings'),
                        'cell_models_folder': str(mearec_home /'cell_models' / 'bbp')}
        with (mearec_home / 'mearec.conf').open('w') as f:
            yaml.dump(default_info, f)
    else:
        with (mearec_home / 'mearec.conf').open() as f:
            if use_loader:
                default_info = yaml.load(f, Loader=yaml.FullLoader)
            else:
                default_info = yaml.load(f)
    return default_info, str(mearec_home)


### LOAD FUNCTIONS ###
def load_tmp_eap(templates_folder, celltypes=None, samples_per_cat=None, verbose=False):
    """
    Loads EAP from temporary folder.

    Parameters
    ----------
    templates_folder : str
        Path to temporary folder
    celltypes : list (optional)
        List of celltypes to be loaded
    samples_per_cat : int (optional)
        The number of eap to load per category

    Returns
    -------
    templates : np.array
        Templates (n_eap, n_elec, n_sample)
    locations : np.array
        Locations (n_eap, 3)
    rotations : np.array
        Rotations (n_eap, 3)
    celltypes : np.array
        Cell types (n_eap)

    """
    if verbose:
        print("Loading spike data ...")
    spikelist = [f for f in os.listdir(templates_folder) if f.startswith('eap')]
    loclist = [f for f in os.listdir(templates_folder) if f.startswith('pos')]
    rotlist = [f for f in os.listdir(templates_folder) if f.startswith('rot')]

    spikes_list = []
    loc_list = []
    rot_list = []
    cat_list = []

    spikelist = sorted(spikelist)
    loclist = sorted(loclist)
    rotlist = sorted(rotlist)

    loaded_categories = set()
    ignored_categories = set()

    for idx, f in enumerate(spikelist):
        celltype = f.split('-')[1][:-4]
        if verbose:
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

    if verbose:
        print("Done loading spike data ...")
    return np.array(spikes_list), np.array(loc_list), np.array(rot_list), np.array(cat_list, dtype=str)


def load_templates(templates, return_h5_objects=False, verbose=False):
    """
    Load generated eap templates.

    Parameters
    ----------
    templates : str
        templates file

    Returns
    -------
    tempgen : TemplateGenerator
        TemplateGenerator object

    """
    from MEArec import TemplateGenerator
    if verbose:
        print("Loading templates...")

    temp_dict = {}

    if templates.endswith('h5') or templates.endswith('hdf5'):
        F = h5py.File(templates, 'r')
        info = load_dict_from_hdf5(F, 'info/')
        celltypes = np.array(F.get('celltypes'))
        temp_dict['celltypes'] = np.array([c.decode('utf-8') for c in celltypes])
        if return_h5_objects:
            temp_dict['locations'] = F.get('locations')
        else:
            temp_dict['locations'] = np.array(F.get('locations'))
        if return_h5_objects:
            temp_dict['rotations'] = F.get('rotations')
        else:
            temp_dict['rotations'] = np.array(F.get('rotations'))
        if return_h5_objects:
            temp_dict['templates'] = F.get('templates')
        else:
            temp_dict['templates'] = np.array(F.get('templates'))
    else:
        raise Exception("Recordings must be an hdf5 file (.h5 or .hdf5)")

    if verbose:
        print("Done loading templates...")
    if not return_h5_objects:
        F.close()
    tempgen = TemplateGenerator(temp_dict=temp_dict, info=info)
    return tempgen


def load_recordings(recordings, return_h5_objects=False, verbose=False):
    """
    Load generated recordings.

    Parameters
    ----------
    recordings : str
        recordings file

    Returns
    -------
    recgen : RecordingGenerator
        RecordingGenerator object

    """
    from MEArec import RecordingGenerator
    if verbose:
        print("Loading recordings...")

    rec_dict = {}

    if recordings.endswith('h5') or recordings.endswith('hdf5'):
        F = h5py.File(recordings, 'r')
        info = load_dict_from_hdf5(F, 'info/')
        if F.get('voltage_peaks') is not None:
            if return_h5_objects:
                rec_dict['voltage_peaks'] = F.get('voltage_peaks')
            else:
                rec_dict['voltage_peaks'] = np.array(F.get('voltage_peaks'))
        if F.get('channel_positions') is not None:
            if return_h5_objects:
                rec_dict['channel_positions'] = F.get('channel_positions')
            else:
                rec_dict['channel_positions'] = np.array(F.get('channel_positions'))
        if F.get('recordings') is not None:
            if return_h5_objects:
                rec_dict['recordings'] = F.get('recordings')
            else:
                rec_dict['recordings'] = np.array(F.get('recordings'))
        if F.get('spike_traces') is not None:
            if return_h5_objects:
                rec_dict['spike_traces'] = F.get('spike_traces')
            else:
                rec_dict['spike_traces'] = np.array(F.get('spike_traces'))
        if F.get('templates') is not None:
            if return_h5_objects:
                rec_dict['templates'] = F.get('templates')
            else:
                rec_dict['templates'] = np.array(F.get('templates'))
        if F.get('timestamps') is not None:
            if return_h5_objects:
                rec_dict['timestamps'] = F.get('timestamps')
            else:
                rec_dict['timestamps'] = np.array(F.get('timestamps'))
        if F.get('spiketrains') is not None:
            spiketrains = []
            sorted_units = sorted([int(u) for u in F.get('spiketrains/')])
            for unit in sorted_units:
                unit = str(unit)
                times = np.array(F.get('spiketrains/' + unit + '/times'))
                t_stop = np.array(F.get('spiketrains/' + unit + '/t_stop'))
                if F.get('spiketrains/' + unit + '/waveforms') is not None:
                    waveforms = np.array(F.get('spiketrains/' + unit + '/waveforms'))
                else:
                    waveforms = None
                annotations = load_dict_from_hdf5(F, 'spiketrains/' + unit + '/annotations/')
                st = neo.core.SpikeTrain(
                    times,
                    t_stop=t_stop,
                    waveforms=waveforms,
                    units=pq.s
                )
                st.annotations = annotations
                spiketrains.append(st)
            rec_dict['spiketrains'] = spiketrains
    else:
        raise Exception("Recordings must be an hdf5 file (.h5 or .hdf5)")

    if verbose:
        print("Done loading recordings...")

    if not return_h5_objects:
        F.close()
    recgen = RecordingGenerator(rec_dict=rec_dict, info=info)
    return recgen


def save_template_generator(tempgen, filename=None, verbose=True):
    """
    Save templates to disk.

    Parameters
    ----------
    tempgen : TemplateGenerator
        TemplateGenerator object to be saved
    filename : str
        Path to .h5 file or folder
    verbose : bool
        If True output is verbose
    """
    if filename.endswith('h5') or filename.endswith('hdf5'):
        F = h5py.File(filename, 'w')
        save_dict_to_hdf5(tempgen.info, F, 'info/')
        if len(tempgen.celltypes) > 0:
            celltypes = [str(x).encode('utf-8') for x in tempgen.celltypes]
            F.create_dataset('celltypes', data=celltypes)
        if len(tempgen.locations) > 0:
            F.create_dataset('locations', data=tempgen.locations)
        if len(tempgen.rotations) > 0:
            F.create_dataset('rotations', data=tempgen.rotations)
        if len(tempgen.templates) > 0:
            F.create_dataset('templates', data=tempgen.templates)
        F.close()
        if verbose:
            print('\nSaved template generator templates in', filename, '\n')
    else:
        raise Exception('Provide an .h5 or .hdf5 file name')


def save_recording_generator(recgen, filename=None, verbose=True):
    """
    Save recordings to disk.

    Parameters
    ----------
    recgen : RecordingGenerator
        RecordingGenerator object to be saved
    filename : str
        Path to .h5 file or folder
    verbose : bool
        If True output is verbose
    """
    if filename.endswith('h5') or filename.endswith('hdf5'):
        F = h5py.File(filename, 'w')
        save_dict_to_hdf5(recgen.info, F, 'info/')
        if len(recgen.voltage_peaks) > 0:
            F.create_dataset('voltage_peaks', data=recgen.voltage_peaks)
        if len(recgen.channel_positions) > 0:
            F.create_dataset('channel_positions', data=recgen.channel_positions)
        if len(recgen.recordings) > 0:
            F.create_dataset('recordings', data=recgen.recordings)
        if len(recgen.spike_traces) > 0:
            F.create_dataset('spike_traces', data=recgen.spike_traces)
        if len(recgen.spiketrains) > 0:
            for ii in range(len(recgen.spiketrains)):
                st = recgen.spiketrains[ii]
                F.create_dataset('spiketrains/{}/times'.format(ii), data=st.times.rescale('s').magnitude)
                F.create_dataset('spiketrains/{}/t_stop'.format(ii), data=st.t_stop)
                if st.waveforms is not None:
                    F.create_dataset('spiketrains/{}/waveforms'.format(ii), data=st.waveforms)
                save_dict_to_hdf5(st.annotations, F, 'spiketrains/{}/annotations/'.format(ii))
        if len(recgen.templates) > 0:
            F.create_dataset('templates', data=recgen.templates)
        if len(recgen.timestamps) > 0:
            F.create_dataset('timestamps', data=recgen.timestamps)
        F.close()
        if verbose:
            print('\nSaved recordings in', filename, '\n')
    else:
        raise Exception('Provide an .h5 or .hdf5 file name')


def save_dict_to_hdf5(dic, h5file, path):
    """
    Save dictionary to h5 file.

    Parameters
    ----------
    dic : dict
        Dictionary to be saved
    h5file : file
        Hdf5 file object
    path : str
        Path to the h5 field
    """
    recursively_save_dict_contents_to_group(h5file, path, dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Save dictionary recursively  to h5 file (helper function).

    Parameters
    ----------
    dic : dict
        Dictionary to be saved
    h5file : file
        Hdf5 file object
    path : str
        Path to the h5 field
    """
    for key, item in dic.items():
        if isinstance(item, (int, float, np.int64, np.float64, str, bytes)):
            if isinstance(item, np.str_):
                item = str(item)
            h5file[path + key] = item
        elif isinstance(item, pq.Quantity):
            h5file[path + key] = float(item.magnitude)
        elif isinstance(item, (list, np.ndarray)):
            if len(item) > 0:
                if isinstance(item[0], (str, bytes)):
                    item = [n.encode("ascii", "ignore") for n in item]
                    h5file[path + key] = np.array(item)
                else:
                    h5file[path + key] = np.array(item)
            else:
                item = '[]'
                h5file[path + key] = item
        elif isinstance(item, tuple):
            h5file[path + key] = np.array(item)
        elif item is None:
            h5file[path + key] = 'null'
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            print(key, item)
            raise ValueError('Cannot save %s type' % type(item))


def load_dict_from_hdf5(h5file, path):
    """
    Load h5 object as dict.

    Parameters
    ----------
    h5file :file
        Hdf5 file object
    path : str
        Path to the h5 field
    Returns
    -------
    dictionary : dict
        Loaded dictionary
    """
    return recursively_load_dict_contents_from_group(h5file, path)


def recursively_load_dict_contents_from_group(h5file, path):
    """
    Load h5 object as dict recursively (helper function).

    Parameters
    ----------
    h5file :file
        Hdf5 file object
    path : str
        Path to the h5 field
    Returns
    -------
    dictionary : dict
        Loaded dictionary
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return clean_dict(ans)


def clean_dict(d):
    """
    Clean dictionary loaded from h5 file.

    Parameters
    ----------
    d : dict
        Dictionary to be cleaned.

    Returns
    -------
    d : dict
        Cleaned dictionary
    """
    for key, item in d.items():
        if isinstance(item, dict):
            clean_dict(item)
        elif isinstance(item, str):
            if item == 'null':
                d[key] = None
            elif item == '[]':
                d[key] = np.array([])
        elif isinstance(item, np.ndarray):
            if len(item) > 0:
                if isinstance(item[0], np.bytes_):
                    d[key] = list([n.decode() for n in item])
                else:
                    d[key] = list(item)
    return d


### TEMPLATES INFO ###
def get_binary_cat(celltypes, excit, inhib):
    """
    Returns binary category depending on cell type.

    Parameters
    ----------
    celltypes : np.array
        String array with cell types
    excit : list
        List of substrings for excitatory cell types (e.g. ['PC', 'UTPC'])
    inhib : list
        List of substrings for inhibitory celltypes (e.g. ['BP', 'MC'])

    Returns
    -------
    binary_cat : np.array
        Array with binary cell type (E-I)

    """
    binary_cat = []
    for i, cat in enumerate(celltypes):
        if np.any([ex in str(cat) for ex in excit]):
            binary_cat.append('E')
        elif np.any([inh in str(cat) for inh in inhib]):
            binary_cat.append('I')
    return np.array(binary_cat, dtype=str)


def get_templates_features(templates, feat_list, dt=None, templates_times=None, threshold_detect=0, normalize=False,
                           reference_mode='t0'):
    """
    Computes several templates features.

    Parameters
    ----------
    templates : np.array
        EAP templates
    feat_list : list
        List of features to be computed (amp, width, fwhm, ratio, speed, na, rep)
    dt : float
        Sampling period
    threshold_detect : float
        Threshold to zero out features

    Returns
    -------
    feature_dict : dict
        Dictionary with features (keys: amp, width, fwhm, ratio, speed, na, rep)

    """
    if dt is not None:
        templates_times = np.arange(templates.shape[-1]) * dt
    else:
        if 'width' in feat_list or 'fwhm' in feat_list or 'speed' in feat_list:
            raise NotImplementedError('Please, specify either dt or templates_times.')

    if len(templates.shape) == 1:
        templates = np.reshape(templates, [1, 1, -1])
    elif len(templates.shape) == 2:
        templates = np.reshape(templates, [1, templates.shape[0], templates.shape[1]])
    if len(templates.shape) != 3:
        raise ValueError('Cannot handle templatess with shape', templates.shape)

    if normalize:
        signs = np.sign(np.min(templates.reshape([templates.shape[0], -1]), axis=1))
        norm = np.abs(np.min(templates.reshape([templates.shape[0], -1]), axis=1))
        templates = np.array([templates[i] / n if signs[i] > 0 else templates[i] / n - 2. for i, n in enumerate(norm)])

    features = {}

    amps = np.zeros((templates.shape[0], templates.shape[1]))
    na_peak = np.zeros((templates.shape[0], templates.shape[1]))
    rep_peak = np.zeros((templates.shape[0], templates.shape[1]))
    if 'width' in feat_list:
        features['width'] = np.zeros((templates.shape[0], templates.shape[1]))
    if 'fwhm' in feat_list:
        features['fwhm'] = np.zeros((templates.shape[0], templates.shape[1]))
    if 'ratio' in feat_list:
        features['ratio'] = np.zeros((templates.shape[0], templates.shape[1]))
    if 'speed' in feat_list:
        features['speed'] = np.zeros((templates.shape[0], templates.shape[1]))
    if 'na' in feat_list:
        features['na'] = np.zeros((templates.shape[0], templates.shape[1]))
    if 'rep' in feat_list:
        features['rep'] = np.zeros((templates.shape[0], templates.shape[1]))

    for i in range(templates.shape[0]):
        # For AMP feature
        min_idx = np.array([np.unravel_index(templates[i, e].argmin(), templates[i, e].shape)[0] for e in
                            range(templates.shape[1])])
        max_idx = np.array([np.unravel_index(templates[i, e, min_idx[e]:].argmax(),
                                             templates[i, e, min_idx[e]:].shape)[0]
                            + min_idx[e] for e in range(templates.shape[1])])
        # for na and rep
        min_elid, min_idx_na = np.unravel_index(templates[i].argmin(), templates[i].shape)
        max_idx_rep = templates[i, min_elid, min_idx_na:].argmax() + min_idx_na
        na_peak[i, :] = templates[i, :, min_idx_na]
        rep_peak[i, :] = templates[i, :, max_idx_rep]

        amps[i, :] = np.array([templates[i, e, max_idx[e]] - templates[i, e, min_idx[e]]
                               for e in range(templates.shape[1])])

        too_low = np.where(amps[i, :] < threshold_detect)
        amps[i, too_low] = 0

        if 'ratio' in feat_list:
            min_id_ratio = np.array([np.unravel_index(templates[i, e, min_idx_na:].argmin(),
                                                      templates[i, e, min_idx_na:].shape)[0]
                                     + min_idx_na for e in range(templates.shape[1])])
            max_id_ratio = np.array([np.unravel_index(templates[i, e, min_idx_na:].argmax(),
                                                      templates[i, e, min_idx_na:].shape)[0]
                                     + min_idx_na for e in range(templates.shape[1])])
            features['ratio'][i, :] = np.array([np.abs(templates[i, e, max_id_ratio[e]]) /
                                                np.abs(templates[i, e, min_id_ratio[e]])
                                                for e in range(templates.shape[1])])
            # If below 'detectable threshold, set amp and width to 0
            too_low = np.where(amps[i, :] < threshold_detect)
            features['ratio'][i, too_low] = 1
        if 'speed' in feat_list:
            features['speed'][i, :] = np.array((min_idx - min_idx_na) * dt)
            features['speed'][i, too_low] = min_idx_na * dt

        if 'width' in feat_list:
            features['width'][i, :] = np.abs(templates_times[max_idx] - templates_times[min_idx])
            features['width'][i, too_low] = templates.shape[2] * dt  # templates_times[-1]-templates_times[0]

        if 'fwhm' in feat_list:
            import scipy.signal as ss
            min_peak = np.min(templates[i], axis=1)
            fwhm_ref = np.array([templates[i, e, 0] for e in range(templates.shape[1])])
            fwhm_V = (fwhm_ref + min_peak) / 2.
            id_trough = [np.where(templates[i, e] < fwhm_V[e])[0] for e in range(templates.shape[1])]

            # no linear interpolation
            features['fwhm'][i, :] = [(id_trough[e][-1] - id_trough[e][0]) * dt if len(id_trough[e]) > 1 \
                                          else templates.shape[2] * dt for e in range(templates.shape[1])]
            features['fwhm'][i, too_low] = templates.shape[2] * dt  # EAP_times[-1]-EAP_times[0]

    if 'amp' in feat_list:
        features.update({'amp': amps})
    if 'na' in feat_list:
        features.update({'na': na_peak})
    if 'rep' in feat_list:
        features.update({'rep': rep_peak})

    return features


### TEMPLATES OPERATIONS ###
def is_position_within_boundaries(position, x_lim, y_lim, z_lim):
    """
    Check if position is within given boundaries.

    Parameters
    ----------
    position : np.array
        3D position
    x_lim : list
        Boundaries in x dimension (low, high)
    y_lim : list
        Boundaries in y dimension (low, high)
    z_lim : list
        Boundaries in z dimension (low, high)

    Returns
    -------
    valid_position : bool
        If True the position is within boundaries

    """
    valid_position = True
    if x_lim is not None:
        if position[0] < x_lim[0] or position[0] > x_lim[1]:
            valid_position = False
    if y_lim is not None:
        if position[1] < y_lim[0] or position[1] > y_lim[1]:
            valid_position = False
    if z_lim is not None:
        if position[2] < z_lim[0] or position[2] > z_lim[1]:
            valid_position = False
    return valid_position


def select_templates(loc, templates, bin_cat, n_exc, n_inh, min_dist=25, x_lim=None, y_lim=None, z_lim=None,
                     min_amp=None, max_amp=None, drifting=False, drift_dir=None, preferred_dir=None, angle_tol=15,
                     verbose=False):
    """
    Select templates given specified rules.

    Parameters
    ----------
    loc : np.array
        Array with 3D soma locations
    templates : np.array
        Array with eap templates (n_eap, n_channels, n_samples)
    bin_cat : np.array
        Array with binary category (E-I)
    n_exc : int
        Number of excitatory cells to be selected
    n_inh : int
        Number of inhibitory cells to be selected
    min_dist : float
        Minimum allowed distance between somata (in um)
    x_lim : list
        Boundaries in x dimension (low, high)
    y_lim : list
        Boundaries in y dimension (low, high)
    z_lim : list
        Boundaries in z dimension (low, high)
    min_amp : float
        Minimum amplitude in uV
    max_amp : float
        Maximum amplitude in uV
    drifting : bool
        If True drifting templates are selected
    drift_dir : np.array
        3D array with drift direction for each template
    preferred_dir : np.array
        3D array with preferred
    angle_tol : float
        Tollerance in degrees for selecting final drift position
    verbose : bool
        If True the output is verbose

    Returns
    -------
    selected_idxs : np.array
        Selected template indexes
    selected_cat : list
        Selected templates binary type


    """
    pos_sel = []
    selected_idxs = []
    categories = np.unique(bin_cat)
    if 'E' in categories and 'I' in categories:
        if verbose:
            print('Selecting Excitatory and Inhibitory cells')
        excinh = True
        selected_cat = []
    else:
        if verbose:
            print('Selecting random templates (cell types not specified)')
        excinh = False
        selected_cat = []
    permuted_idxs = np.random.permutation(len(bin_cat))
    permuted_bin_cats = bin_cat[permuted_idxs]

    if min_amp is None:
        min_amp = 0

    if max_amp is None:
        max_amp = np.inf

    if drifting:
        if drift_dir is None or preferred_dir is None:
            raise Exception('For drift selection provide drifting angles and preferred drift direction')

    n_sel = 0
    n_sel_exc = 0
    n_sel_inh = 0
    iter = 0

    for i, (id_cell, bcat) in enumerate(zip(permuted_idxs, permuted_bin_cats)):
        placed = False
        iter += 1
        if n_sel == n_exc + n_inh:
            break
        if excinh:
            if bcat == 'E':
                if n_sel_exc < n_exc:
                    dist = np.array([np.linalg.norm(loc[id_cell] - p) for p in pos_sel])
                    if np.any(dist < min_dist):
                        if verbose:
                            print('Distance violation', dist, iter)
                        pass
                    else:
                        amp = np.max(np.abs(np.min(templates[id_cell])))
                        if not drifting:
                            if is_position_within_boundaries(loc[id_cell], x_lim, y_lim, z_lim) and amp > min_amp and \
                                    amp < max_amp:
                                # save cell
                                pos_sel.append(loc[id_cell])
                                selected_idxs.append(id_cell)
                                n_sel += 1
                                placed = True
                            else:
                                if verbose:
                                    print('Amplitude or boundary violation', amp, loc[id_cell], iter)
                        else:
                            # drifting
                            if is_position_within_boundaries(loc[id_cell, 0], x_lim, y_lim, z_lim) and amp > min_amp and \
                                    amp < max_amp:
                                # save cell
                                drift_angle = np.rad2deg(np.arccos(np.dot(drift_dir[id_cell], preferred_dir)))
                                if drift_angle - angle_tol <= 0:
                                    pos_sel.append(loc[id_cell])
                                    selected_idxs.append(id_cell)
                                    n_sel += 1
                                    placed = True
                                else:
                                    if verbose:
                                        print('Drift violation', loc[id_cell, 0], iter)
                            else:
                                if verbose:
                                    print('Amplitude or boundary violation', amp, loc[id_cell, 0], iter)
                    if placed:
                        n_sel_exc += 1
                        selected_cat.append('E')
            elif bcat == 'I':
                if n_sel_inh < n_inh:
                    dist = np.array([np.linalg.norm(loc[id_cell] - p) for p in pos_sel])
                    if np.any(dist < min_dist):
                        if verbose:
                            print('Distance violation', dist, iter)
                        pass
                    else:
                        amp = np.max(np.abs(np.min(templates[id_cell])))
                        if not drifting:
                            if is_position_within_boundaries(loc[id_cell], x_lim, y_lim, z_lim) and amp > min_amp and \
                                    amp < max_amp:
                                # save cell
                                pos_sel.append(loc[id_cell])
                                selected_idxs.append(id_cell)
                                n_sel += 1
                                placed = True
                            else:
                                if verbose:
                                    print('Amplitude or boundary violation', amp, loc[id_cell], iter)
                        else:
                            # drifting
                            if is_position_within_boundaries(loc[id_cell, 0], x_lim, y_lim, z_lim) and amp > min_amp and \
                                    amp < max_amp:
                                # save cell
                                drift_angle = np.rad2deg(np.arccos(np.dot(drift_dir[id_cell], preferred_dir)))
                                if drift_angle - angle_tol <= 0:
                                    selected_idxs.append(id_cell)
                                    n_sel += 1
                                    placed = True
                                else:
                                    if verbose:
                                        print('Drift violation', loc[id_cell], iter)
                            else:
                                if verbose:
                                    print('Amplitude or boundary violation', amp, loc[id_cell, 0], iter)
                    if placed:
                        n_sel_inh += 1
                        selected_cat.append('I')
        else:
            dist = np.array([np.linalg.norm(loc[id_cell] - p) for p in pos_sel])
            if np.any(dist < min_dist):
                if verbose:
                    print('Distance violation', dist, iter)
                pass
            else:
                amp = np.max(np.abs(np.min(templates[id_cell])))
                if not drifting:
                    if is_position_within_boundaries(loc[id_cell], x_lim, y_lim, z_lim) and amp > min_amp and \
                                    amp < max_amp:
                        # save cell
                        pos_sel.append(loc[id_cell])
                        selected_idxs.append(id_cell)
                        placed = True
                    else:
                        if verbose:
                            print('Amplitude or boundary violation', amp, loc[id_cell], iter)
                else:
                    # drifting
                    if is_position_within_boundaries(loc[id_cell, 0], x_lim, y_lim, z_lim) and amp > min_amp and \
                                    amp < max_amp:
                        # save cell
                        drift_angle = np.rad2deg(np.arccos(np.dot(drift_dir[id_cell], preferred_dir)))
                        if drift_angle - angle_tol <= 0:
                            pos_sel.append(loc[id_cell])
                            selected_idxs.append(id_cell)
                            placed = True
                        else:
                            if verbose:
                                print('Drift violation', loc[id_cell, 0], iter)
                    else:
                        if verbose:
                            print('Amplitude or boundary violation', amp, loc[id_cell, 0], iter)
            if placed:
                n_sel += 1
                selected_cat.append('U')

    if i == len(permuted_idxs) - 1 and n_sel < n_exc + n_inh:
        raise RuntimeError("Templates could not be selected. \n"
                           "Decrease number of spiketrains, decrease 'min-dist', or use more templates.")

    return selected_idxs, selected_cat


def cubic_padding(template, pad_len, fs):
    """
    Cubic spline padding on left and right side to 0. The initial offset of the templates is also removed.

    Parameters
    ----------
    template : np.array
        Templates to be padded (n_elec, n_samples)
    pad_len : list
        Padding in ms before and after the template
    fs : quantity
        Sampling frequency

    Returns
    -------
    padded_template : np.array
        Padded template

    """
    import scipy.interpolate as interp
    n_pre = int(pad_len[0] * fs.rescale('kHz'))
    n_post = int(pad_len[1] * fs.rescale('kHz'))

    padded_template = np.zeros((template.shape[0], int(n_pre) + template.shape[1] + n_post))
    splines = np.zeros((template.shape[0], int(n_pre) + template.shape[1] + n_post))

    for i, sp in enumerate(template):
        # Remove inital offset
        padded_sp = np.zeros(n_pre + len(sp) + n_post)
        padded_t = np.arange(len(padded_sp))
        initial_offset = np.mean(sp[0])
        sp -= initial_offset

        x_pre = float(n_pre)
        x_pre_pad = np.arange(n_pre)
        x_post = float(n_post)
        x_post_pad = np.arange(n_post)[::-1]

        # fill pre and post intervals with linear values from sp[0] - sp[-1] to 0 for better fit
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

    return padded_template


def find_overlapping_templates(templates, thresh=0.7):
    """
    Find spatially overlapping templates.

    Parameters
    ----------
    templates : np.array
        Array with templates (n_templates, n_elec, n_samples)
    thresh : float
        Percent threshold to consider two templates to be overlapping.

    Returns
    -------
    overlapping_pairs : np.array
        Array with overlapping pairs (n_overlapping, 2)

    """
    overlapping_pairs = []

    for i, temp_1 in enumerate(templates):
        if len(templates.shape) == 4: # jitter
            temp_1 = temp_1[0]

        peak_electrode_idx = np.unravel_index(temp_1.argmin(), temp_1.shape)

        for j, temp_2 in enumerate(templates):
            if len(templates.shape) == 4:  # jitter
                temp_2 = temp_2[0]

            if i != j:
                peak_2_on_max = np.abs(np.min(temp_2[peak_electrode_idx]))
                peak_2 = np.abs(np.min(temp_2))

                if peak_2_on_max > thresh * peak_2:
                    if [i, j] not in overlapping_pairs and [j, i] not in overlapping_pairs:
                        overlapping_pairs.append(sorted([i, j]))

    return np.array(overlapping_pairs)


### SPIKETRAIN OPERATIONS ###
def annotate_overlapping_spikes(spiketrains, t_jitt=1 * pq.ms, overlapping_pairs=None, parallel=True, verbose=True):
    """
    Annotate spike trains with temporal and spatio-temporal overlapping labels.
    NO - Non overlap
    O - Temporal overlap
    SO - Spatio-temporal overlap

    Parameters
    ----------
    spiketrains : list
        List of neo spike trains to be annotated
    t_jitt : Quantity
        Time jitter to consider overlapping spikes in time (default 1 ms)
    overlapping_pairs : np.array
        Array with overlapping information between spike trains (n_spiketrains, 2)
    parallel : bool
        If True spike trains are processed in parallel with multiprocessing
    verbose : bool
        If True output is verbose

    """
    nprocesses = len(spiketrains)
    if parallel:
        import multiprocessing
        # t_start = time.time()
        pool = multiprocessing.Pool(nprocesses)
        _ = [pool.apply_async(annotate_parallel(i, st_i, spiketrains, t_jitt, overlapping_pairs, verbose, ))
             for i, st_i in enumerate(spiketrains)]
    else:
        # find overlapping spikes
        for i, st_i in enumerate(spiketrains):
            if verbose:
                print('Annotating overlapping spike train ', i)
            over = np.array(['NO'] * len(st_i))
            for i_sp, t_i in enumerate(st_i):
                for j, st_j in enumerate(spiketrains):
                    if i != j:
                        # find overlapping
                        id_over = np.where((st_j > t_i - t_jitt) & (st_j < t_i + t_jitt))[0]
                        if not np.any(overlapping_pairs):
                            if len(id_over) != 0:
                                over[i_sp] = 'O'
                        else:
                            pair = [i, j]
                            pair_i = [j, i]
                            if np.any([np.all(pair == p) for p in overlapping_pairs]) or \
                                    np.any([np.all(pair_i == p) for p in overlapping_pairs]):
                                if len(id_over) != 0:
                                    over[i_sp] = 'SO'
                            else:
                                if len(id_over) != 0:
                                    over[i_sp] = 'O'
            st_i.annotate(overlap=over)


def annotate_parallel(i, st_i, spiketrains, t_jitt, overlapping_pairs, verbose):
    """
    Helper function to annotate spike trains in parallel.

    Parameters
    ----------
    i : int
        Index of spike train
    st_i : neo.SpikeTrain
        Spike train to be processed
    spiketrains : list
        List of neo spiketrains
    t_jitt : Quantity
        Time jitter to consider overlapping spikes in time (default 1 ms)
    overlapping_pairs : np.array
        Array with overlapping information between spike trains (n_spiketrains, 2)
    verbose : bool
        If True output is verbose

    """
    if verbose:
        print('Annotating overlapping spike train ', i)
    over = np.array(['NO'] * len(st_i))
    for i_sp, t_i in enumerate(st_i):
        for j, st_j in enumerate(spiketrains):
            if i != j:
                # find overlapping
                id_over = np.where((st_j > t_i - t_jitt) & (st_j < t_i + t_jitt))[0]
                if not np.any(overlapping_pairs):
                    if len(id_over) != 0:
                        over[i_sp] = 'O'
                else:
                    pair = [i, j]
                    pair_i = [j, i]
                    if np.any([np.all(pair == p) for p in overlapping_pairs]) or \
                            np.any([np.all(pair_i == p) for p in overlapping_pairs]):
                        if len(id_over) != 0:
                            over[i_sp] = 'SO'
                    else:
                        if len(id_over) != 0:
                            over[i_sp] = 'O'
    st_i.annotate(overlap=over)


def resample_spiketrains(spiketrains, fs=None, T=None):
    """
    Resamples spike trains. Provide either fs or T parameters

    Parameters
    ----------
    spiketrains : list
        List of neo spiketrains to be resampled
    fs : Quantity
        New sampling frequency
    T : Quantity
        New sampling period

    Returns
    -------
    resampled_mat : np.array
        Matrix with resampled binned spike trains

    """
    import elephant.conversion as conv

    resampled_mat = []
    if not fs and not T:
        print('Provide either sampling frequency fs or time period T')
        return
    elif fs:
        if not isinstance(fs, Quantity):
            raise ValueError("fs must be of type pq.Quantity")
        binsize = 1. / fs
        binsize.rescale('ms')
        resampled_mat = []
        for sts in spiketrains:
            spikes = conv.BinnedSpikeTrain(sts, binsize=binsize).to_array()
            resampled_mat.append(np.squeeze(spikes))
    elif T:
        binsize = T
        if not isinstance(T, Quantity):
            raise ValueError("T must be of type pq.Quantity")
        binsize.rescale('ms')
        resampled_mat = []
        for sts in spiketrains:
            spikes = conv.BinnedSpikeTrain(sts, binsize=binsize).to_array()
            resampled_mat.append(np.squeeze(spikes))
    return np.array(resampled_mat)


def compute_sync_rate(st1, st2, time_jitt):
    """
    Compute synchrony rate between two wpike trains.

    Parameters
    ----------
    st1 : neo.SpikeTrain
        Spike train 1
    st2 : neo.SpikeTrain
        Spike train 2
    time_jitt : quantity
        Maximum time jittering between added spikes

    Returns
    -------
    rate : float
        Synchrony rate (0-1)
    """
    count = 0
    times1 = st1.times
    times2 = st2.times
    for t1 in times1:
        if len(np.where(np.abs(times2 - t1) <= time_jitt)[0]) >= 1:
            if len(np.where(np.abs(times2 - t1) <= time_jitt)[0]) > 1:
                print('Len: ', len(np.where(np.abs(times2 - t1) <= time_jitt)[0]))
            count += 1
    rate = count / (len(times1) + len(times2))
    return rate


### CONVOLUTION OPERATIONS ###
def compute_modulation(st, n_el=1, mrand=1, sdrand=0.05, n_spikes=1, exp=0.2, max_burst_duration=100 * pq.ms):
    """
    Computes modulation value for an input spike train.

    Parameters
    ----------
    st : neo.SpikeTrain
        Input spike train
    n_el : int
        Number of electrodes to compute modulation.
        If 1, modulation is computed at the template level.
        If n_elec, modulation is computed at the electrode level.
    mrand : float
        Mean for Gaussian modulation (should be 1)
    sdrand : float
        Standard deviation for Gaussian modulation
    n_spikes : int
        Number of spikes for bursting behavior.
        If 1, no bursting behavior.
        If > 1, up to n_spikes consecutive spike are modulated with an exponentially decaying function.
    exp : float
        Exponent for exponential modulation (default 0.2)
    max_burst_duration : Quantity
        Maximum duration of a bursting event. After this duration, bursting modulation is reset.


    Returns
    -------
    mod : np.array
        Modulation value for each spike in the spike train
    cons : np.array
        Number of consecutive spikes computed for each spike

    """

    import elephant.statistics as stat

    if n_el == 1:
        ISI = stat.isi(st).rescale('ms')
        # max_burst_duration = 2*mean_ISI
        mod = np.zeros(len(st))
        mod[0] = sdrand * np.random.randn() + mrand
        cons = np.zeros(len(st))

        last_burst_event = 0 * pq.s
        for i, isi in enumerate(ISI):
            if n_spikes == 0:
                # no isi-dependent modulation
                mod[i + 1] = sdrand * np.random.randn() + mrand
            elif n_spikes == 1:
                if isi > max_burst_duration:
                    mod[i + 1] = sdrand * np.random.randn() + mrand
                else:
                    mod[i + 1] = isi.magnitude ** exp * (1. / max_burst_duration.magnitude ** exp) \
                                     + sdrand * np.random.randn()
            else:
                if last_burst_event.magnitude == 0:
                    consecutive_idx = np.where((st > st[i] - max_burst_duration) & (st <= st[i]))[0]
                    consecutive = len(consecutive_idx)
                else:
                    consecutive_idx = np.where((st > last_burst_event) & (st <= st[i]))[0]
                    consecutive = len(consecutive_idx)

                # if consecutive >= 1:
                # print('Bursting event duration: ', ISI[i - consecutive], ' with spikes: ', consecutive)

                if consecutive == n_spikes:
                    last_burst_event = st[i + 1]
                if consecutive >= 1:
                    if st[i + 1] - st[consecutive_idx[0]] >= max_burst_duration:
                        last_burst_event = st[i + 1]

                if consecutive == 0:
                    mod[i + 1] = sdrand * np.random.randn() + mrand
                elif consecutive == 1:
                    amp = (isi / float(consecutive)) ** exp * (1. / max_burst_duration.magnitude ** exp)
                    # scale std by amp
                    mod[i + 1] = amp + amp * sdrand * np.random.randn()
                else:
                    if i != len(ISI):
                        isi_mean = np.mean(ISI[i - consecutive + 1:i + 1])
                    else:
                        isi_mean = np.mean(ISI[i - consecutive + 1:])
                    amp = (isi_mean / float(consecutive)) ** exp * (1. / max_burst_duration.magnitude ** exp)
                    # scale std by amp
                    mod[i + 1] = amp + amp * sdrand * np.random.randn()

                cons[i + 1] = consecutive
    else:
        if n_spikes == 0:
            mod = sdrand * np.random.randn(len(st), n_el) + mrand
            cons = []
        else:
            ISI = stat.isi(st).rescale('ms')
            mod = np.zeros((len(st), n_el))
            mod[0] = sdrand * np.random.randn(n_el) + mrand
            cons = np.zeros(len(st))

            last_burst_event = 0 * pq.s
            for i, isi in enumerate(ISI):
                if n_spikes == 1:
                    if isi > max_burst_duration:
                        mod[i + 1] = sdrand * np.random.randn(n_el) + mrand
                    else:
                        mod[i + 1] = isi.magnitude ** exp * (
                                    1. / max_burst_duration.magnitude ** exp) + sdrand * np.random.randn(n_el)
                else:
                    if isi > max_burst_duration:
                        mod[i + 1] = sdrand * np.random.randn(n_el) + mrand
                        consecutive = 0
                    elif last_burst_event.magnitude == 0:
                        consecutive_idx = np.where((st > st[i] - max_burst_duration) & (st <= st[i]))[0]
                        consecutive = len(consecutive_idx)
                    else:
                        consecutive_idx = np.where((st > last_burst_event) & (st <= st[i]))[0]
                        consecutive = len(consecutive_idx)

                    # if consecutive >= 1:
                        # print('Bursting event duration: ', ISI[i - consecutive], ' with spikes: ', consecutive)

                    if consecutive == n_spikes:
                        last_burst_event = st[i + 1]
                    if consecutive >= 1:
                        if st[i + 1] - st[consecutive_idx[0]] >= max_burst_duration:
                            last_burst_event = st[i + 1]

                    if consecutive == 0:
                        mod[i + 1] = sdrand * np.random.randn(n_el) + mrand
                    elif consecutive == 1:
                        amp = (isi.magnitude / float(consecutive)) ** exp * (1. / max_burst_duration.magnitude ** exp)
                        # scale std by amp
                        if amp > 1:
                            raise Exception
                        mod[i + 1] = amp + amp * sdrand * np.random.randn(n_el)
                    else:
                        if i != len(ISI):
                            isi_mean = np.mean(ISI[i - consecutive + 1:i + 1])
                        else:
                            isi_mean = np.mean(ISI[i - consecutive + 1:])
                        amp = (isi_mean / float(consecutive)) ** exp * (1. / max_burst_duration.magnitude ** exp)
                        # scale std by amp
                        mod[i + 1] = amp + amp * sdrand * np.random.randn(n_el)
                    cons[i + 1] = consecutive

    return np.array(mod), cons


def compute_bursting_template(template, mod, wc_mod):
    """
    Compute modulation in shape for a template with low-pass filter.

    Parameters
    ----------
    template : np.array
        Template to be modulated (num_chan, n_samples) or (n_samples)
    mod : int or np.array
        Amplitude modulation for template or single electrodes
    wc_mod : float
        Normalized frequency of low-pass filter

    Returns
    -------
    temp_filt : np.array
        Modulated template

    """
    import scipy.signal as ss
    b, a = ss.butter(3, wc_mod)
    if len(template.shape) == 2:
        temp_filt = ss.filtfilt(b, a, template, axis=1)
        if mod.size > 1:
            temp_filt = np.array([m * np.min(temp) / np.min(temp_f) *
                                  temp_f for (m, temp, temp_f) in zip(mod, template, temp_filt)])
        else:
            temp_filt = (mod * np.min(template) / np.min(temp_filt)) * temp_filt
    else:
        temp_filt = ss.filtfilt(b, a, template)
        temp_filt = (mod * np.min(template) / np.min(temp_filt)) * temp_filt
    return temp_filt


def convolve_single_template(spike_id, spike_bin, template, cut_out=None, modulation=False, mod_array=None,
                             bursting=False, fc=None, fs=None):
    """Convolve single template with spike train. Used to compute 'spike_traces'.

    Parameters
    ----------
    spike_id : int
        Index of spike trains - template.
    spike_bin : np.array
        Binary array with spike times
    template : np.array
        Array with template on single electrode (n_samples)
    cut_out : list
        Number of samples before and after the peak
    modulation : bool
        If True modulation is applied
    mod_array : np.array
        Array with modulation value for each spike
    bursting : bool
        If True templates are modulated in shape
    fc : list
        Min and max frequency for low-pass filter for bursting modulation
    fs : quantity
        Sampling frequency

    Returns
    -------
    spike_trace : np.array
        Trace with convolved signal (n_samples)

    """
    if len(template.shape) == 2:
        njitt = template.shape[0]
        len_spike = template.shape[1]
    if cut_out is None:
        cut_out = [len_spike // 2, len_spike // 2]
    spike_pos = np.where(spike_bin == 1)[0]
    n_samples = len(spike_bin)
    spike_trace = np.zeros(n_samples)

    if len(template.shape) == 2:
        rand_idx = np.random.randint(njitt)
        temp_jitt = template[rand_idx]
        if not modulation:
            for pos, spos in enumerate(spike_pos):
                if spos - cut_out[0] >= 0 and spos - cut_out[0] + len_spike <= n_samples:
                    spike_trace[spos - cut_out[0]:spos - cut_out[0] + len_spike] += temp_jitt
                elif spos - cut_out[0] < 0:
                    diff = -(spos - cut_out[0])
                    spike_trace[:spos - cut_out[0] + len_spike] += temp_jitt[diff:]
                else:
                    diff = n_samples - (spos - cut_out[0])
                    spike_trace[spos - cut_out[0]:] += temp_jitt[:diff]
        else:
            if bursting:
                assert len(fc) == 2 and fs is not None
                fs = fs.rescale('Hz').magnitude
                wc_mod_array = ((fc[1] - fc[0]) / (np.max(mod_array) - np.min(mod_array)) *
                                (mod_array - np.min(mod_array)) + fc[0]) / (fs / 2.)
                wc_mod_mean = wc_mod_array
                for pos, spos in enumerate(spike_pos):
                    if spos - cut_out[0] >= 0 and spos - cut_out[0] + len_spike <= n_samples:
                        spike_trace[spos - cut_out[0]:spos - cut_out[0] + len_spike] += \
                            compute_bursting_template(temp_jitt, mod_array[pos], wc_mod_mean[pos])
                    elif spos - cut_out[0] < 0:
                        diff = -(spos - cut_out[0])
                        temp_filt = compute_bursting_template(temp_jitt, mod_array[pos], wc_mod_mean[pos])
                        spike_trace[:spos - cut_out[0] + len_spike] += temp_filt[diff:]
                    else:
                        diff = n_samples - (spos - cut_out[0])
                        temp_filt = compute_bursting_template(temp_jitt, mod_array[pos], wc_mod_mean[pos])
                        spike_trace[spos - cut_out[0]:] += temp_filt[:diff]
            else:
                for pos, spos in enumerate(spike_pos):
                    if spos - cut_out[0] >= 0 and spos - cut_out[0] + len_spike <= n_samples:
                        spike_trace[spos - cut_out[0]:spos - cut_out[0] + len_spike] += mod_array[pos] * temp_jitt
                    elif spos - cut_out[0] < 0:
                        diff = -(spos - cut_out[0])
                        spike_trace[:spos - cut_out[0] + len_spike] += mod_array[pos] * temp_jitt[diff:]
                    else:
                        diff = n_samples - (spos - cut_out[0])
                        spike_trace[spos - cut_out[0]:] += mod_array[pos] * temp_jitt[:diff]
    else:
        raise Exception('For drifting len(template.shape) should be 2')
    return spike_trace


def convolve_templates_spiketrains(spike_id, spike_bin, template, cut_out=None, modulation=False, mod_array=None,
                                   verbose=False, bursting=False, fc=None, fs=None):
    """
    Convolve template with spike train on all electrodes. Used to compute 'recordings'.

    Parameters
    ----------
    spike_id : int
        Index of spike trains - template.
    spike_bin : np.array
        Binary array with spike times
    template : np.array
        Array with template on single electrode (n_samples)
    cut_out : list
        Number of samples before and after the peak
    modulation : bool
        If True modulation is applied
    mod_array : np.array
        Array with modulation value for each spike
    verbose : bool
        If True output is verbose
    bursting : bool
        If True templates are modulated in shape
    fc : list
        Min and max frequency for low-pass filter for bursting modulation
    fs : quantity
        Sampling frequency

    Returns
    -------
    recordings: np.array
        Trace with convolved signals (n_elec, n_samples)

    """
    if verbose:
        print('Starting convolution with spike ', spike_id)
    if len(template.shape) == 3:
        njitt = template.shape[0]
        n_elec = template.shape[1]
        len_spike = template.shape[2]
    n_samples = len(spike_bin)
    recordings = np.zeros((n_elec, n_samples))
    if cut_out is None:
        cut_out = [len_spike // 2, len_spike // 2]
    if modulation is False:
        # No modulation
        spike_pos = np.where(spike_bin == 1)[0]
        mod_array = np.ones_like(spike_pos)
        if len(template.shape) == 3:
            rand_idx = np.random.randint(njitt)
            temp_jitt = template[rand_idx]
            for pos, spos in enumerate(spike_pos):
                if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                    recordings[:, spos - cut_out[0]:spos + cut_out[1]] += mod_array[pos] * temp_jitt
                elif spos - cut_out[0] < 0:
                    diff = -(spos - cut_out[0])
                    recordings[:, :spos + cut_out[1]] += mod_array[pos] * temp_jitt[:, diff:]
                else:
                    diff = n_samples - (spos - cut_out[0])
                    recordings[:, spos - cut_out[0]:] += mod_array[pos] * temp_jitt[:, :diff]
        else:
            raise Exception('For drifting len(template.shape) should be 3')
    else:
        assert mod_array is not None
        spike_pos = np.where(spike_bin == 1)[0]
        if len(template.shape) == 3:
            rand_idx = np.random.randint(njitt)
            temp_jitt = template[rand_idx]
            if not isinstance(mod_array[0], (list, tuple, np.ndarray)):
                # Template modulation
                if bursting:
                    assert len(fc) == 2 and fs is not None
                    fs = fs.rescale('Hz').magnitude
                    wc_mod_array = ((fc[1] - fc[0]) / (np.max(mod_array) - np.min(mod_array)) *
                                    (mod_array - np.min(mod_array)) + fc[0]) / (fs / 2.)
                    wc_mod_mean = wc_mod_array
                    for pos, spos in enumerate(spike_pos):
                        if spos - cut_out[0] >= 0 and spos - cut_out[0] + len_spike <= n_samples:
                            recordings[:, spos - cut_out[0]:spos + cut_out[1]] += \
                                compute_bursting_template(temp_jitt, mod_array[pos], wc_mod_mean[pos])
                        elif spos - cut_out[0] < 0:
                            diff = -(spos - cut_out[0])
                            temp_filt = compute_bursting_template(temp_jitt, mod_array[pos], wc_mod_mean[pos])
                            recordings[:, :spos + cut_out[1]] += temp_filt[:, diff:]
                        else:
                            diff = n_samples - (spos - cut_out[0])
                            temp_filt = compute_bursting_template(temp_jitt, mod_array[pos], wc_mod_mean[pos])
                            recordings[:, spos - cut_out[0]:] += temp_filt[:, :diff]
                else:
                    for pos, spos in enumerate(spike_pos):
                        if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                            recordings[:, spos - cut_out[0]:spos + cut_out[1]] += mod_array[pos] * temp_jitt
                        elif spos - cut_out[0] < 0:
                            diff = -(spos - cut_out[0])
                            recordings[:, :spos + cut_out[1]] += mod_array[pos] * temp_jitt[:, diff:]
                        else:
                            diff = n_samples - (spos - cut_out[0])
                            recordings[:, spos - cut_out[0]:] += mod_array[pos] * temp_jitt[:, :diff]
            else:
                # Electrode modulation
                if bursting:
                    assert len(fc) == 2 and fs is not None
                    fs = fs.rescale('Hz').magnitude
                    wc_mod_array = ((fc[1] - fc[0]) / (np.max(mod_array) - np.min(mod_array)) *
                                    (mod_array - np.min(mod_array)) + fc[0]) / (fs / 2.)
                    wc_mod_mean = np.mean(wc_mod_array, axis=1)
                    for pos, spos in enumerate(spike_pos):
                        if spos - cut_out[0] >= 0 and spos - cut_out[0] + len_spike <= n_samples:
                            recordings[:, spos - cut_out[0]:spos + cut_out[1]] += \
                                compute_bursting_template(temp_jitt, mod_array[pos], wc_mod_mean[pos])
                        elif spos - cut_out[0] < 0:
                            diff = -(spos - cut_out[0])
                            temp_filt = compute_bursting_template(temp_jitt, mod_array[pos], wc_mod_mean[pos])
                            recordings[:, :spos + cut_out[1]] += temp_filt[:, diff:]
                        else:
                            diff = n_samples - (spos - cut_out[0])
                            temp_filt = compute_bursting_template(temp_jitt, mod_array[pos], wc_mod_mean[pos])
                            recordings[:, spos - cut_out[0]:] += temp_filt[:, :diff]
                else:
                    for pos, spos in enumerate(spike_pos):
                        if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                            recordings[:, spos - cut_out[0]:spos + cut_out[1]] += \
                                [a * t for (a, t) in zip(mod_array[pos], temp_jitt)]
                        elif spos - cut_out[0] < 0:
                            diff = -(spos - cut_out[0])
                            recordings[:, :spos + cut_out[1]] += \
                                [a * t for (a, t) in zip(mod_array[pos], temp_jitt[:, diff:])]
                        else:
                            diff = n_samples - (spos - cut_out[0])
                            recordings[:, spos - cut_out[0]:] += \
                                [a * t for (a, t) in zip(mod_array[pos], temp_jitt[:, :diff])]
        else:
            raise Exception('For drifting len(template.shape) should be 3')
    if verbose:
        print('Done convolution with spike ', spike_id)

    return recordings


def convolve_drifting_templates_spiketrains(spike_id, spike_bin, template, fs, loc, v_drift, t_start_drift,
                                            cut_out=None, modulation=False, mod_array=None, n_step_sec=1,
                                            verbose=False, bursting=False, fc=None):
    """
    Convolve template with spike train on all electrodes. Used to compute 'recordings'.

    Parameters
    ----------
    spike_id : int
        Index of spike trains - template
    spike_bin : np.array
        Binary array with spike times
    template : np.array
        Array with template on single electrode (n_samples)
    fs : Quantity
        Sampling frequency
    loc : np.array
        Locations of drifting templates
    v_drift : float
        Drifting speed in um/s
    t_start_drift : Quantity
        Drifting start time
    cut_out : list
        Number of samples before and after the peak
    modulation : bool
        If True modulation is applied
    mod_array : np.array
        Array with modulation value for each spike
    n_step_sec : int
        Number of drifting steps in a second
    verbose : bool
        If True output is verbose
    bursting : bool
        If True templates are modulated in shape
    fc : list
        Min and max frequency for low-pass filter for bursting modulation


    Returns
    -------
    recordings: np.array
        Trace with convolved signals (n_elec, n_samples)
    final_loc : np.array
        Final 3D location of neuron
    final_idx : int
        Final index among n drift steps
    peaks : np.array
        Drifting peak images (n_steps, n_elec)

    """
    if verbose:
        print('Starting drifting convolution with spike ', spike_id)
    if len(template.shape) == 4:
        njitt = template.shape[1]
        n_elec = template.shape[2]
        len_spike = template.shape[3]
    n_samples = len(spike_bin)
    dur = (n_samples / fs).rescale('s').magnitude
    t_steps = np.arange(0, dur, n_step_sec)
    dt = 1. / fs.magnitude
    if cut_out is None:
        cut_out = [len_spike // 2, len_spike // 2]

    peaks = np.zeros((int(n_samples / float(fs.rescale('Hz').magnitude)), n_elec))
    recordings = np.zeros((n_elec, n_samples))

    # recordings_test = np.zeros((n_elec, n_samples))
    if not modulation:
        # No modulation
        spike_pos = np.where(spike_bin == 1)[0]
        mod_array = np.ones_like(spike_pos)
        if len(template.shape) == 4:
            rand_idx = np.random.randint(njitt)
            for pos, spos in enumerate(spike_pos):
                sp_time = spos / fs
                if sp_time < t_start_drift:
                    temp_idx = 0
                    temp_jitt = template[temp_idx, rand_idx]
                else:
                    # compute current position
                    new_pos = np.array(loc[0] + v_drift * (sp_time - t_start_drift).rescale('s').magnitude)
                    temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc])
                    temp_jitt = template[temp_idx, rand_idx]

                if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                    recordings[:, spos - cut_out[0]:spos + cut_out[1]] += mod_array[pos] * temp_jitt
                elif spos - cut_out[0] < 0:
                    diff = -(spos - cut_out[0])
                    recordings[:, :spos + cut_out[1]] += mod_array[pos] * temp_jitt[:, diff:]
                else:
                    diff = n_samples - (spos - cut_out[0])
                    recordings[:, spos - cut_out[0]:] += mod_array[pos] * temp_jitt[:, :diff]

            for i, t in enumerate(t_steps):
                if t < t_start_drift:
                    temp_idx = 0
                    temp_jitt = template[temp_idx, rand_idx]
                else:
                    # compute current position
                    new_pos = np.array(loc[0] + v_drift * (t - t_start_drift.rescale('s').magnitude))
                    temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc])
                    temp_jitt = template[temp_idx, rand_idx]

                feat = get_templates_features(np.squeeze(temp_jitt), ['na'], dt=dt)
                peaks[i] = -np.squeeze(feat['na'])
        else:
            raise Exception('For drifting len(template.shape) should be 4')
    else:
        assert mod_array is not None
        spike_pos = np.where(spike_bin == 1)[0]
        if len(template.shape) == 4:
            rand_idx = np.random.randint(njitt)
            if not isinstance(mod_array[0], (list, tuple, np.ndarray)):
                # Template modulation
                if bursting:
                    assert len(fc) == 2 and fs is not None
                    fs_mag = fs.rescale('Hz').magnitude
                    wc_mod_array = ((fc[1] - fc[0]) / (np.max(mod_array) - np.min(mod_array)) *
                                    (mod_array - np.min(mod_array)) + fc[0]) / (fs_mag / 2.)
                    wc_mod_mean = wc_mod_array
                    for pos, spos in enumerate(spike_pos):
                        sp_time = spos / fs
                        if sp_time < t_start_drift:
                            temp_idx = 0
                            temp_jitt = template[temp_idx, rand_idx]
                        else:
                            # compute current position
                            new_pos = np.array(loc[0] + v_drift * (sp_time - t_start_drift).rescale('s').magnitude)
                            temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc])
                            temp_jitt = template[temp_idx, rand_idx]
                        if spos - cut_out[0] >= 0 and spos - cut_out[0] + len_spike <= n_samples:
                            recordings[:, spos - cut_out[0]:spos + cut_out[1]] += \
                                compute_bursting_template(temp_jitt, mod_array[pos], wc_mod_mean[pos])
                        elif spos - cut_out[0] < 0:
                            diff = -(spos - cut_out[0])
                            temp_filt = compute_bursting_template(temp_jitt, mod_array[pos], wc_mod_mean[pos])
                            recordings[:, :spos + cut_out[1]] += temp_filt[:, diff:]
                        else:
                            diff = n_samples - (spos - cut_out[0])
                            temp_filt = compute_bursting_template(temp_jitt, mod_array[pos], wc_mod_mean[pos])
                            recordings[:, spos - cut_out[0]:] += temp_filt[:, :diff]
                else:
                    for pos, spos in enumerate(spike_pos):
                        sp_time = spos / fs
                        if sp_time < t_start_drift:
                            temp_idx = 0
                            temp_jitt = template[temp_idx, rand_idx]
                        else:
                            # compute current position
                            new_pos = np.array(loc[0] + v_drift * (sp_time - t_start_drift).rescale('s').magnitude)
                            temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc])
                            temp_jitt = template[temp_idx, rand_idx]
                        if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                            recordings[:, spos - cut_out[0]:spos + cut_out[1]] += mod_array[pos] \
                                                                                  * temp_jitt
                        elif spos - cut_out[0] < 0:
                            diff = -(spos - cut_out[0])
                            recordings[:, :spos + cut_out[1]] += mod_array[pos] * temp_jitt[:, diff:]
                        else:
                            diff = n_samples - (spos - cut_out[0])
                            recordings[:, spos - cut_out[0]:] += mod_array[pos] * temp_jitt[:, :diff]
            else:
                # Electrode modulation
                if bursting:
                    assert len(fc) == 2 and fs is not None
                    fs_mag = fs.rescale('Hz').magnitude
                    wc_mod_array = ((fc[1] - fc[0]) / (np.max(mod_array) - np.min(mod_array)) *
                                    (mod_array - np.min(mod_array)) + fc[0]) / (fs_mag / 2.)
                    wc_mod_mean = np.mean(wc_mod_array, axis=1)
                    for pos, spos in enumerate(spike_pos):
                        sp_time = spos / fs
                        if sp_time < t_start_drift:
                            temp_idx = 0
                            temp_jitt = template[temp_idx, rand_idx]
                            if spos - cut_out[0] >= 0 and spos - cut_out[0] + len_spike <= n_samples:
                                recordings[:, spos - cut_out[0]:spos + cut_out[1]] += \
                                    compute_bursting_template(temp_jitt, mod_array[pos], wc_mod_mean[pos])
                            elif spos - cut_out[0] < 0:
                                diff = -(spos - cut_out[0])
                                temp_filt = compute_bursting_template(temp_jitt, mod_array[pos], wc_mod_mean[pos])
                                recordings[:, :spos + cut_out[1]] += temp_filt[:, diff:]
                            else:
                                diff = n_samples - (spos - cut_out[0])
                                temp_filt = compute_bursting_template(temp_jitt, mod_array[pos], wc_mod_mean[pos])
                                recordings[:, spos - cut_out[0]:] += temp_filt[:, :diff]
                        else:
                            # compute current position
                            new_pos = np.array(loc[0] + v_drift * (sp_time - t_start_drift).rescale('s').magnitude)
                            temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc])
                            new_temp_jitt = template[temp_idx, rand_idx]
                            if spos - cut_out[0] >= 0 and spos - cut_out[0] + len_spike <= n_samples:
                                recordings[:, spos - cut_out[0]:spos + cut_out[1]] += \
                                    compute_bursting_template(new_temp_jitt, mod_array[pos], wc_mod_mean[pos])
                            elif spos - cut_out[0] < 0:
                                diff = -(spos - cut_out[0])
                                temp_filt = compute_bursting_template(new_temp_jitt, mod_array[pos], wc_mod_mean[pos])
                                recordings[:, :spos + cut_out[1]] += temp_filt[:, diff:]
                            else:
                                diff = n_samples - (spos - cut_out[0])
                                temp_filt = compute_bursting_template(new_temp_jitt, mod_array[pos], wc_mod_mean[pos])
                                recordings[:, spos - cut_out[0]:] += temp_filt[:, :diff]

                else:
                    for pos, spos in enumerate(spike_pos):
                        sp_time = spos / fs
                        if sp_time < t_start_drift:
                            temp_idx = 0
                            temp_jitt = template[temp_idx, rand_idx]
                            if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                                recordings[:, spos - cut_out[0]:spos + cut_out[1]] += \
                                    [a * t for (a, t) in zip(mod_array[pos], temp_jitt)]
                            elif spos - cut_out[0] < 0:
                                diff = -(spos - cut_out[0])
                                recordings[:, :spos + cut_out[1]] += \
                                    [a * t for (a, t) in zip(mod_array[pos], temp_jitt[:, diff:])]
                            else:
                                diff = n_samples - (spos - cut_out[0])
                                recordings[:, spos - cut_out[0]:] += \
                                    [a * t for (a, t) in zip(mod_array[pos], temp_jitt[:, :diff])]
                        else:
                            # compute current position
                            new_pos = np.array(loc[0] + v_drift * (sp_time - t_start_drift).rescale('s').magnitude)
                            temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc])
                            new_temp_jitt = template[temp_idx, rand_idx]
                            if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                                recordings[:, spos - cut_out[0]:spos + cut_out[1]] += \
                                    [a * t for (a, t) in zip(mod_array[pos], new_temp_jitt)]
                            elif spos - cut_out[0] < 0:
                                diff = -(spos - cut_out[0])
                                recordings[:, :spos + cut_out[1]] += \
                                    [a * t for (a, t) in zip(mod_array[pos], new_temp_jitt[:, diff:])]
                            else:
                                diff = n_samples - (spos - cut_out[0])
                                recordings[:, spos - cut_out[0]:] += \
                                    [a * t for (a, t) in zip(mod_array[pos], new_temp_jitt[:, :diff])]
                for i, t in enumerate(t_steps):
                    if t < t_start_drift:
                        temp_idx = 0
                        temp_jitt = template[temp_idx, rand_idx]
                    else:
                        # compute current position
                        new_pos = np.array(loc[0] + v_drift * (t - t_start_drift.rescale('s').magnitude))
                        temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc])
                        temp_jitt = template[temp_idx, rand_idx]

                    feat = get_templates_features(np.squeeze(temp_jitt), ['na'], dt=dt)
                    peaks[i] = -np.squeeze(feat['na'])
        else:
            raise Exception('For drifting len(template.shape) should be 4')
    final_loc = loc[temp_idx]
    final_idx = temp_idx

    if verbose:
        print('Done drifting convolution with spike ', spike_id)

    return recordings, final_loc, final_idx, peaks


### RECORDING OPERATION ###
def extract_wf(spiketrains, recordings, fs, pad_len=2 * pq.ms, timestamps=None):
    """
    Extract waveforms from recordings and load it in waveform field of neo spike trains.

    Parameters
    ----------
    spiketrains : list
        List of neo spike trains
    recordings : np.array
        Array with recordings (n_elec, n_samples)
    fs : Quantity
        Sampling frequency
    pad_len : Quantity (single or list)
         Length in ms to cut before and after spike peak. If a single value the cut is symmetrical
    timestamps : Quantity array (optional)
        Array with recordings timestamps
    """
    if not isinstance(pad_len, list):
        n_pad = int(pad_len * fs.rescale('kHz'))
        n_pad = [n_pad, n_pad]
    else:
        n_pad = [int(p * fs.rescale('kHz')) for p in pad_len]

    n_elec, n_samples = recordings.shape
    if timestamps is None:
        timestamps = np.arange(n_samples) / fs
    unit = timestamps[0].rescale('ms').units

    for st in spiketrains:
        sp_rec_wf = []
        sp_amp = []
        for t in st:
            idx = np.where(timestamps >= t)[0][0]
            # find single waveforms crossing thresholds
            if idx - n_pad[0] > 0 and idx + n_pad[1] < n_samples:
                t_spike = timestamps[idx - n_pad[0]:idx + n_pad[1]]
                spike_rec = recordings[:, idx - n_pad[0]:idx + n_pad[1]]
            elif idx - n_pad[0] < 0:
                t_spike = timestamps[:idx + n_pad[1]]
                t_spike = np.pad(t_spike, (np.abs(idx - n_pad[0]), 0), 'constant') * unit
                spike_rec = recordings[:, :idx + n_pad[1]]
                spike_rec = np.pad(spike_rec, ((0, 0), (np.abs(idx - n_pad[0]), 0)), 'constant')
            elif idx + n_pad[1] > n_samples:
                t_spike = timestamps[idx - n_pad[0]:]
                t_spike = np.pad(t_spike, (0, idx + n_pad[1] - n_samples), 'constant') * unit
                spike_rec = recordings[:, idx - n_pad[0]:]
                spike_rec = np.pad(spike_rec, ((0, 0), (0, idx + n_pad[1] - n_samples)), 'constant')
            sp_rec_wf.append(spike_rec)
        st.waveforms = np.array(sp_rec_wf)


def filter_analog_signals(signals, freq, fs, filter_type='bandpass', order=3):
    """
    Filter analog signals with zero-phase Butterworth filter.
    The function raises an Exception if the required filter is not stable.

    Parameters
    ----------
    signals : np.array
        Array of analog signals (n_elec, n_samples)
    freq : list or float
        Cutoff frequency-ies in Hz
    fs : Quantity
        Sampling frequency
    filter_type : string
        Filter type ('lowpass', 'highpass', 'bandpass', 'bandstop')
    order : int
        Filter order

    Returns
    -------
    signals_filt : np.array
        Filtered signals
    """
    from scipy.signal import butter, filtfilt
    fn = fs / 2.
    fn = fn.rescale(pq.Hz)
    freq = freq.rescale(pq.Hz)
    band = freq / fn

    b, a = butter(order, band, btype=filter_type)

    if np.all(np.abs(np.roots(a)) < 1) and np.all(np.abs(np.roots(a)) < 1):
        # print('Filtering signals with ', filter_type, ' filter at ', freq, '...')
        if len(signals.shape) == 2:
            signals_filt = filtfilt(b, a, signals, axis=1)
        elif len(signals.shape) == 1:
            signals_filt = filtfilt(b, a, signals)
        return signals_filt
    else:
        raise ValueError('Filter is not stable')


### PLOTTING ###
def plot_rasters(spiketrains, bintype=False, ax=None, overlap=False, color=None, fs=10,
                 marker='|', mew=2, markersize=5):
    """
    Plot raster for spike trains.

    Parameters
    ----------
    spiketrains : list
        List of neo spike trains
    bintype : bool
        If True and 'bintype' in spike train annotation spike trains are plotted based on their type
    ax : axes
        Plot on the given axes
    overlap : bool
        Plot spike colors based on overlap
    labels : bool
        Plot spike colors based on labels
    color : matplotlib color (single or list)
        Color or color list
    fs : int
        Font size
    marker : matplotlib arg
        Marker type
    mew : matplotlib arg
        Width of marker
    markersize : int
        Marker size

    Returns
    -------
    ax : axis
        Matplotlib axis
    """

    import matplotlib.pylab as plt
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if overlap:
        if 'overlap' not in spiketrains[0].annotations.keys():
            raise Exception()
    for i, spiketrain in enumerate(spiketrains):
        t = spiketrain.rescale(pq.s)
        if bintype:
            if spiketrain.annotations['bintype'] == 'E':
                ax.plot(t, i * np.ones_like(t), 'b', marker=marker, mew=mew, markersize=markersize, ls='')
            elif spiketrain.annotations['bintype'] == 'I':
                ax.plot(t, i * np.ones_like(t), 'r', marker=marker, mew=mew, markersize=markersize, ls='')
        else:
            if not overlap:
                if color is not None:
                    if isinstance(color, list) or isinstance(color, np.ndarray):
                        ax.plot(t, i * np.ones_like(t), color=color[i], marker=marker, mew=mew, markersize=markersize,
                                ls='')
                    else:
                        ax.plot(t, i * np.ones_like(t), color=color, marker=marker, mew=mew, markersize=markersize,
                                ls='')
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

    ax.axis('tight')
    ax.set_xlim([spiketrains[0].t_start.rescale(pq.s), spiketrains[0].t_stop.rescale(pq.s)])
    ax.set_xlabel('Time (s)', fontsize=fs)
    ax.set_ylabel('Spike Train Index', fontsize=fs)
    ax.set_yticks(np.arange(len(spiketrains)))
    ax.set_yticklabels(np.arange(len(spiketrains)))

    return ax


def plot_templates(gen, single_axes=False, max_templates=None):
    """
    Plot templates.

    Parameters
    ----------
    gen : TemplateGenerator or RecordingGenerator
        Generator object containing templates
    single_axes : bool
        If True all templates are plotted on the same axis.
    max_templates: int
        Maximum number of templates to be plotted.

    Returns
    -------
    fig : figure
        Matplotlib figure

    """
    import matplotlib.pylab as plt

    templates = gen.templates
    mea = mu.return_mea(info=gen.info['electrodes'])

    if max_templates is not None:
        if max_templates < len(templates):
            templates = templates[np.random.permutation(len(templates))][:max_templates]

    n_sources = len(templates)
    fig = plt.figure()

    if single_axes:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ax_t = fig.add_subplot(111)

        for n, t in enumerate(templates):
            print('Plotting spike ', n, ' out of ', n_sources)
            if len(t.shape) == 3:
                mu.plot_mea_recording(t.mean(axis=0), mea, colors=colors[np.mod(n, len(colors))], ax=ax_t, lw=2)
            else:
                mu.plot_mea_recording(t, mea, colors=colors[np.mod(n, len(colors))], ax=ax_t, lw=2)

    else:
        cols = int(np.ceil(np.sqrt(n_sources)))
        rows = int(np.ceil(n_sources / float(cols)))

        for n in range(n_sources):
            ax_t = fig.add_subplot(rows, cols, n + 1)
            mu.plot_mea_recording(templates[n], mea, ax=ax_t)

    return fig


def plot_recordings(recgen, ax=None, start_frame=None, end_frame=None, **kwargs):
    """
    Plot recordings.

    Parameters
    ----------
    recgen : RecordingGenerator
        Recording generator object to plot

    Returns
    -------
    ax : axis
        Matplotlib axis

    """
    import matplotlib.pylab as plt

    recordings = recgen.recordings
    mea = mu.return_mea(info=recgen.info['electrodes'])
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = recordings.shape[1]

    mu.plot_mea_recording(recordings[:, start_frame:end_frame], mea, ax=ax, **kwargs)
    return ax


def plot_waveforms(recgen, spiketrain_id=0, ax=None, color_isi=False, color='k', cmap='viridis', electrode=None):
    """
    Plot waveforms of a spike train.

    Parameters
    ----------
    recgen : RecordingGenerator
        Recording generator object to plot spike train waveform from
    spiketrain_id : int
        Indes of spike train
    ax : axis
        Matplotlib  axis
    color_isi : bool
        If True the color is mapped to the isi
    color : matplotlib color
        Color of the waveforms
    cmap : matplotlib colormap
        Colormap if color_isi is True
    electrode : int or 'max'
        Electrode id or 'max'

    Returns
    -------
    ax : axis
        Matplotlib axis

    """
    import matplotlib.pylab as plt
    import matplotlib as mpl

    wf = recgen.spiketrains[spiketrain_id].waveforms

    if wf is None:
        fs = recgen.info['recordings']['fs'] * pq.Hz
        extract_wf([recgen.spiketrains[spiketrain_id]], recgen.recordings, fs)
        wf = recgen.spiketrains[spiketrain_id].waveforms
    mea = mu.return_mea(info=recgen.info['electrodes'])

    if color_isi:
        import elephant.statistics as stat
        isi = stat.isi(recgen.spiketrains[spiketrain_id]).rescale('ms')
        cm = mpl.cm.get_cmap(cmap)
        color = [cm(1)]
        for i in isi:
            color.append(cm(i/np.max(isi)))

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if electrode is None:
        vscale = 1.5 * np.max(np.abs(wf))
        ax = mu.plot_mea_recording(wf, mea, colors=color, ax=ax, lw=0.1, vscale=vscale)
        ax = mu.plot_mea_recording(wf.mean(axis=0), mea, colors=color, ax=ax, lw=2, vscale=vscale)
    else:
        assert isinstance(electrode, (int, np.integer)) or electrode == 'max', "electrode must be int or 'max'"
        if electrode == 'max':
            electrode = np.unravel_index(np.argmin(wf.mean(axis=0)), wf.mean(axis=0).shape)[0]
            print('max electrode: ', electrode)
        if not color_isi:
            ax.plot(wf[:, electrode], color=color, lw=0.1)
            ax.plot(wf[:, electrode].mean(axis=0), color=color, lw=2)
        else:
            for i in range(wf.shape[0]):
                ax.plot(wf[i, electrode], color=color[i], lw=0.5)

    return ax


# if __name__ == '__main__':
