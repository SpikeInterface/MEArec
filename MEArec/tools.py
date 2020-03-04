import numpy as np
import quantities as pq
from quantities import Quantity
import yaml
import neo
import scipy.signal as ss
import shutil
import os
from os.path import join
import MEAutility as mu
import h5py
from pathlib import Path
from copy import deepcopy
from distutils.version import StrictVersion
from joblib import Parallel, delayed

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
        mearec_home.mkdir(exist_ok=True, parents=True)
        shutil.copytree(str(this_dir / 'default_params'), str(mearec_home / 'default_params'))
        shutil.copytree(str(this_dir / 'cell_models'), str(mearec_home / 'cell_models'))
        default_info = {'templates_params': str(mearec_home / 'default_params' / 'templates_params.yaml'),
                        'recordings_params': str(mearec_home / 'default_params' / 'recordings_params.yaml'),
                        'templates_folder': str(mearec_home / 'templates'),
                        'recordings_folder': str(mearec_home / 'recordings'),
                        'cell_models_folder': str(mearec_home / 'cell_models' / 'bbp')}
        with (mearec_home / 'mearec.conf').open('w') as f:
            yaml.dump(default_info, f)
    else:
        with (mearec_home / 'mearec.conf').open() as f:
            if use_loader:
                default_info = yaml.load(f, Loader=yaml.FullLoader)
            else:
                default_info = yaml.load(f)
    return default_info, str(mearec_home)


def get_default_cell_models_folder():
    """
    Returns default cell models folder.

    Returns
    -------
    cell_models_folder : str
        Path to default cell models folder
    """
    default_info, mearec_home = get_default_config()
    cell_models_folder = default_info['cell_models_folder']

    return cell_models_folder


def get_default_templates_params():
    """
    Returns default templates parameters.

    Returns
    -------
    templates_params : dict
        Dictionary with default teplates parameters
    """
    default_info, mearec_home = get_default_config()
    templates_params_file = default_info['templates_params']

    # load template parameters
    with open(templates_params_file, 'r') as f:
        if use_loader:
            templates_params = yaml.load(f, Loader=yaml.FullLoader)
        else:
            templates_params = yaml.load(f)
    return templates_params


def get_default_recordings_params():
    """
    Returns default recordings parameters.

    Returns
    -------
    recordings_params : dict
        Dictionary with default recording parameters
    """
    default_info, mearec_home = get_default_config()
    recordings_params_file = default_info['recordings_params']

    # load template parameters
    with open(recordings_params_file, 'r') as f:
        if use_loader:
            recordings_params = yaml.load(f, Loader=yaml.FullLoader)
        else:
            recordings_params = yaml.load(f)
    return recordings_params


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
        print("Loading eap data ...")
    eaplist = [f for f in os.listdir(templates_folder) if f.startswith('eap')]
    loclist = [f for f in os.listdir(templates_folder) if f.startswith('pos')]
    rotlist = [f for f in os.listdir(templates_folder) if f.startswith('rot')]

    eap_list = []
    loc_list = []
    rot_list = []
    cat_list = []

    eaplist = sorted(eaplist)
    loclist = sorted(loclist)
    rotlist = sorted(rotlist)

    loaded_categories = set()
    ignored_categories = set()

    for idx, f in enumerate(eaplist):
        celltype = f.split('-')[1][:-4]
        if verbose:
            print('loading cell type: ', f)
        if celltypes is not None:
            if celltype in celltypes:
                eaps = np.load(join(templates_folder, f))
                locs = np.load(join(templates_folder, loclist[idx]))
                rots = np.load(join(templates_folder, rotlist[idx]))

                if samples_per_cat is None or samples_per_cat > len(eaps):
                    samples_to_read = len(eaps)
                else:
                    samples_to_read = samples_per_cat

                eap_list.extend(eaps[:samples_to_read])
                rot_list.extend(rots[:samples_to_read])
                loc_list.extend(locs[:samples_to_read])
                cat_list.extend([celltype] * samples_to_read)
                loaded_categories.add(celltype)
            else:
                ignored_categories.add(celltype)
        else:
            eaps = np.load(join(templates_folder, f))
            locs = np.load(join(templates_folder, loclist[idx]))
            rots = np.load(join(templates_folder, rotlist[idx]))

            if samples_per_cat is None or samples_per_cat > len(eaps):
                samples_to_read = len(eaps)
            else:
                samples_to_read = samples_per_cat

            eap_list.extend(eaps[:samples_to_read])
            rot_list.extend(rots[:samples_to_read])
            loc_list.extend(locs[:samples_to_read])
            cat_list.extend([celltype] * samples_to_read)
            loaded_categories.add(celltype)

    if verbose:
        print("Done loading spike data ...")
    return np.array(eap_list), np.array(loc_list), np.array(rot_list), np.array(cat_list, dtype=str)


def load_templates(templates, return_h5_objects=True, verbose=False, check_suffix=True):
    """
    Load generated eap templates.

    Parameters
    ----------
    templates : str or Path object
        templates file
    return_h5_objects : bool
        If True output objects are h5 objects
    verbose : bool
        If True output is verbose
    check_suffix : bool
        If True, hfdf5 suffix is checked


    Returns
    -------
    tempgen : TemplateGenerator
        TemplateGenerator object

    """
    from MEArec import TemplateGenerator
    if verbose:
        print("Loading templates...")

    temp_dict = {}
    templates = Path(templates)
    if (templates.suffix == '.h5' or templates.suffix == '.hdf5') or (not check_suffix):
        F = h5py.File(str(templates), 'r')
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


def load_recordings(recordings, return_h5_objects=True,
                    load=None, load_waveforms=True, check_suffix=True, verbose=False):
    """
    Load generated recordings.

    Parameters
    ----------
    recordings : str or Path object
        Recordings file
    return_h5_objects : bool
        If True output objects are h5 objects
    load : list
        List of fields to be loaded (('recordings', 'channel_positions', 'voltage_peaks', 'spiketrains',
                                       'timestamps', 'spike_traces', 'templates'))
    load_waveforms : bool
        If True waveforms are loaded to spiketrains
    verbose : bool
        If True output is verbose
    check_suffix : bool
        If True, hfdf5 suffix is checked

    Returns
    -------
    recgen : RecordingGenerator
        RecordingGenerator object

    """
    from MEArec import RecordingGenerator
    if verbose:
        print("Loading recordings...")

    if load is None:
        load = ['recordings', 'channel_positions', 'voltage_peaks', 'spiketrains', 'timestamps',
                'spike_traces', 'templates']
    else:
        assert isinstance(load, list), "'load' should be a list with strings of what to be loaded " \
                                       "('recordings', 'channel_positions', 'voltge_peaks', 'spiketrains', " \
                                       "'timestamps', 'spike_traces', 'templates')"

    rec_dict = {}
    recordings = Path(recordings)
    if (recordings.suffix == '.h5' or recordings.suffix == '.hdf5') or (not check_suffix):
        F = h5py.File(str(recordings), 'r')
        info = load_dict_from_hdf5(F, 'info/')
        if F.get('voltage_peaks') is not None and 'voltage_peaks' in load:
            if return_h5_objects:
                rec_dict['voltage_peaks'] = F.get('voltage_peaks')
            else:
                rec_dict['voltage_peaks'] = np.array(F.get('voltage_peaks'))
        if F.get('channel_positions') is not None and 'channel_positions' in load:
            if return_h5_objects:
                rec_dict['channel_positions'] = F.get('channel_positions')
            else:
                rec_dict['channel_positions'] = np.array(F.get('channel_positions'))
        if F.get('recordings') is not None and 'recordings' in load:
            if return_h5_objects:
                rec_dict['recordings'] = F.get('recordings')
            else:
                rec_dict['recordings'] = np.array(F.get('recordings'))
        if F.get('spike_traces') is not None and 'spike_traces' in load:
            if return_h5_objects:
                rec_dict['spike_traces'] = F.get('spike_traces')
            else:
                rec_dict['spike_traces'] = np.array(F.get('spike_traces'))
        if F.get('templates') is not None and 'templates' in load:
            if return_h5_objects:
                rec_dict['templates'] = F.get('templates')
            else:
                rec_dict['templates'] = np.array(F.get('templates'))
            if F.get('template_locations') is not None:
                if return_h5_objects:
                    rec_dict['template_locations'] = F.get('template_locations')
                else:
                    rec_dict['template_locations'] = np.array(F.get('template_locations'))
            if F.get('template_rotations') is not None:
                if return_h5_objects:
                    rec_dict['template_rotations'] = F.get('template_rotations')
                else:
                    rec_dict['template_rotations'] = np.array(F.get('template_rotations'))
            if F.get('template_celltypes') is not None:
                celltypes = np.array(([n.decode() for n in F.get('template_celltypes')]))
                rec_dict['template_celltypes'] = np.array(celltypes)
        if F.get('timestamps') is not None and 'timestamps' in load:
            if return_h5_objects:
                rec_dict['timestamps'] = F.get('timestamps')
            else:
                rec_dict['timestamps'] = np.array(F.get('timestamps')) * pq.s
        if F.get('spiketrains') is not None and 'spiketrains' in load:
            spiketrains = []
            sorted_units = sorted([int(u) for u in F.get('spiketrains/')])
            for unit in sorted_units:
                unit = str(unit)
                times = np.array(F.get('spiketrains/' + unit + '/times'))
                t_stop = np.array(F.get('spiketrains/' + unit + '/t_stop'))
                if F.get('spiketrains/' + unit + '/waveforms') is not None and load_waveforms:
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
    filename = Path(filename)
    if not filename.parent.is_dir():
        os.makedirs(str(filename.parent))
    if filename.suffix == '.h5' or filename.suffix == '.hdf5':
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
            print('\nSaved  templates in', filename, '\n')
    else:
        raise Exception('Provide an .h5 or .hdf5 file name')


def save_recording_generator(recgen, filename=None, verbose=False):
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
    filename = Path(filename)
    if not filename.parent.is_dir():
        os.makedirs(str(filename.parent))
    if filename.suffix == '.h5' or filename.suffix == '.hdf5':
        with h5py.File(filename, 'w') as F:
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
            if len(recgen.template_locations) > 0:
                F.create_dataset('template_locations', data=recgen.template_locations)
            if len(recgen.template_rotations) > 0:
                F.create_dataset('template_rotations', data=recgen.template_rotations)
            if len(recgen.template_celltypes) > 0:
                celltypes = [n.encode("ascii", "ignore") for n in recgen.template_celltypes]
                F.create_dataset('template_celltypes', data=celltypes)
            if len(recgen.timestamps) > 0:
                F.create_dataset('timestamps', data=recgen.timestamps)
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
        if isinstance(item, (int, float, np.integer, np.float, str, bytes, np.bool_)):
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
    sample_type = celltypes[0]
    # Find if bbp or custom models
    if sample_type.startswith('L') and len(sample_type.split('_')) == 4:
        models = 'bbp'
    else:
        models = 'custom'
    for i, cat in enumerate(celltypes):
        if models == 'bbp':
            cell_str = str(cat).split('_')[1]
        else:
            cell_str = str(cat)
        if np.any([ex in cell_str for ex in excit]):
            binary_cat.append('E')
        elif np.any([inh in str(cat) for inh in inhib]):
            binary_cat.append('I')
        else:
            binary_cat.append('U')
    return np.array(binary_cat, dtype=str)


def get_templates_features(templates, feat_list, dt=None, templates_times=None, threshold_detect=0):
    """
    Computes several templates features.

    Parameters
    ----------
    templates : np.array
        EAP templates
    feat_list : list
        List of features to be computed (amp, width, fwhm, ratio, speed, neg, pos)
    dt : float
        Sampling period
    threshold_detect : float
        Threshold to zero out features

    Returns
    -------
    feature_dict : dict
        Dictionary with features (keys: amp, width, fwhm, ratio, speed, neg, pos)

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
    if 'neg' in feat_list:
        features['neg'] = np.zeros((templates.shape[0], templates.shape[1]))
    if 'pos' in feat_list:
        features['pos'] = np.zeros((templates.shape[0], templates.shape[1]))

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
    if 'neg' in feat_list:
        features.update({'neg': na_peak})
    if 'pos' in feat_list:
        features.update({'pos': rep_peak})

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
                     n_overlap_pairs=None, overlap_threshold=0.8, verbose=False):
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
    n_overlap_pairs: int
        Number of spatially overlapping templates to select
    overlap_threshold: float
        Threshold for considering spatially overlapping pairs ([0-1])
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

    if bin_cat is not None and 'E' in categories and 'I' in categories:
        if verbose:
            print('Selecting Excitatory and Inhibitory cells')
        excinh = True
        selected_cat = []
    else:
        if verbose:
            print('Selecting random templates (cell types not specified)')
        excinh = False
        selected_cat = []

    permuted_idxs = np.random.permutation(len(loc))
    if bin_cat is not None:
        permuted_bin_cats = bin_cat[permuted_idxs]
    else:
        permuted_bin_cats = ['U'] * len(loc)

    if verbose:
        print('Min dist: ', min_dist, 'Min amp: ', min_amp)

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
    current_overlapping_pairs = 0

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
                            print('Distance violation', np.min(dist), iter)
                        pass
                    else:
                        amp = np.max(np.abs(np.min(templates[id_cell])))
                        if not drifting:
                            if is_position_within_boundaries(loc[id_cell], x_lim, y_lim, z_lim) \
                                    and min_amp < amp < max_amp:
                                # save cell
                                if n_overlap_pairs is None:
                                    pos_sel.append(loc[id_cell])
                                    selected_idxs.append(id_cell)
                                    n_sel += 1
                                    placed = True
                                else:
                                    if len(selected_idxs) == 0:
                                        # save cell
                                        pos_sel.append(loc[id_cell])
                                        selected_idxs.append(id_cell)
                                        n_sel += 1
                                        placed = True
                                    else:
                                        possible_selected = deepcopy(selected_idxs)
                                        possible_selected.append(id_cell)
                                        possible_overlapping_pairs = len(find_overlapping_templates(
                                            templates[sorted(possible_selected)],
                                            overlap_threshold))
                                        current_overlapping_pairs = len(find_overlapping_templates(
                                            templates[sorted(selected_idxs)],
                                            overlap_threshold))
                                        if current_overlapping_pairs < n_overlap_pairs and \
                                                possible_overlapping_pairs <= n_overlap_pairs:
                                            if possible_overlapping_pairs == current_overlapping_pairs:
                                                continue
                                            else:
                                                pos_sel.append(loc[id_cell])
                                                selected_idxs.append(id_cell)
                                                n_sel += 1
                                                placed = True
                                                if verbose:
                                                    print('Number of overlapping pairs:', possible_overlapping_pairs)
                                        else:
                                            if possible_overlapping_pairs == current_overlapping_pairs:
                                                pos_sel.append(loc[id_cell])
                                                selected_idxs.append(id_cell)
                                                n_sel += 1
                                                placed = True
                                            else:
                                                if verbose:
                                                    print('Overlapping violation:', possible_overlapping_pairs)
                            else:
                                if verbose:
                                    print('Amplitude or boundary violation', amp, loc[id_cell], iter)
                        else:
                            # drifting
                            if is_position_within_boundaries(loc[id_cell, 0], x_lim, y_lim, z_lim) \
                                    and min_amp < amp < max_amp:
                                # save cell
                                drift_angle = np.rad2deg(np.arccos(np.dot(drift_dir[id_cell], preferred_dir)))
                                if drift_angle - angle_tol <= 0:
                                    if n_overlap_pairs is None:
                                        pos_sel.append(loc[id_cell])
                                        selected_idxs.append(id_cell)
                                        n_sel += 1
                                        placed = True
                                    else:
                                        if len(selected_idxs) == 0:
                                            # save cell
                                            pos_sel.append(loc[id_cell])
                                            selected_idxs.append(id_cell)
                                            n_sel += 1
                                            placed = True
                                        else:
                                            possible_selected = deepcopy(selected_idxs)
                                            possible_selected.append(id_cell)
                                            possible_overlapping_pairs = len(find_overlapping_templates(
                                                templates[sorted(possible_selected), 0],
                                                overlap_threshold))
                                            current_overlapping_pairs = len(find_overlapping_templates(
                                                templates[sorted(selected_idxs), 0],
                                                overlap_threshold))
                                            if current_overlapping_pairs < n_overlap_pairs and \
                                                    possible_overlapping_pairs <= n_overlap_pairs:
                                                if possible_overlapping_pairs == current_overlapping_pairs:
                                                    continue
                                                else:
                                                    pos_sel.append(loc[id_cell])
                                                    selected_idxs.append(id_cell)
                                                    n_sel += 1
                                                    placed = True
                                                    if verbose:
                                                        print('Number of overlapping pairs:',
                                                              possible_overlapping_pairs)
                                            else:
                                                if possible_overlapping_pairs == current_overlapping_pairs:
                                                    pos_sel.append(loc[id_cell])
                                                    selected_idxs.append(id_cell)
                                                    n_sel += 1
                                                    placed = True
                                                else:
                                                    if verbose:
                                                        print('Overlapping violation:', possible_overlapping_pairs)
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
                            print('Distance violation', np.min(dist), iter)
                        pass
                    else:
                        amp = np.max(np.abs(np.min(templates[id_cell])))
                        if not drifting:
                            if is_position_within_boundaries(loc[id_cell], x_lim, y_lim, z_lim) \
                                    and min_amp < amp < max_amp:
                                # save cell
                                if n_overlap_pairs is None:
                                    pos_sel.append(loc[id_cell])
                                    selected_idxs.append(id_cell)
                                    n_sel += 1
                                    placed = True
                                else:
                                    if len(selected_idxs) == 0:
                                        # save cell
                                        pos_sel.append(loc[id_cell])
                                        selected_idxs.append(id_cell)
                                        n_sel += 1
                                        placed = True
                                    else:
                                        possible_selected = deepcopy(selected_idxs)
                                        possible_selected.append(id_cell)
                                        possible_overlapping_pairs = len(find_overlapping_templates(
                                            templates[sorted(possible_selected)],
                                            overlap_threshold))
                                        current_overlapping_pairs = len(find_overlapping_templates(
                                            templates[sorted(selected_idxs)],
                                            overlap_threshold))
                                        if current_overlapping_pairs < n_overlap_pairs and \
                                                possible_overlapping_pairs <= n_overlap_pairs:
                                            if possible_overlapping_pairs == current_overlapping_pairs:
                                                continue
                                            else:
                                                pos_sel.append(loc[id_cell])
                                                selected_idxs.append(id_cell)
                                                n_sel += 1
                                                placed = True
                                                if verbose:
                                                    print('Number of overlapping pairs:', possible_overlapping_pairs)
                                        else:
                                            if possible_overlapping_pairs == current_overlapping_pairs:
                                                pos_sel.append(loc[id_cell])
                                                selected_idxs.append(id_cell)
                                                n_sel += 1
                                                placed = True
                                            else:
                                                if verbose:
                                                    print('Overlapping violation:', possible_overlapping_pairs)
                            else:
                                if verbose:
                                    print('Amplitude or boundary violation', amp, loc[id_cell], iter)
                        else:
                            # drifting
                            if is_position_within_boundaries(loc[id_cell, 0], x_lim, y_lim, z_lim) \
                                    and min_amp < amp < max_amp:
                                # save cell
                                drift_angle = np.rad2deg(np.arccos(np.dot(drift_dir[id_cell], preferred_dir)))
                                if drift_angle - angle_tol <= 0:
                                    if n_overlap_pairs is None:
                                        selected_idxs.append(id_cell)
                                        n_sel += 1
                                        placed = True
                                    else:
                                        if len(selected_idxs) == 0:
                                            # save cell
                                            pos_sel.append(loc[id_cell])
                                            selected_idxs.append(id_cell)
                                            n_sel += 1
                                            placed = True
                                        else:
                                            possible_selected = deepcopy(selected_idxs)
                                            possible_selected.append(id_cell)
                                            possible_overlapping_pairs = len(find_overlapping_templates(
                                                templates[sorted(possible_selected), 0],
                                                overlap_threshold))
                                            current_overlapping_pairs = len(find_overlapping_templates(
                                                templates[sorted(selected_idxs), 0],
                                                overlap_threshold))
                                            if current_overlapping_pairs < n_overlap_pairs and \
                                                    possible_overlapping_pairs <= n_overlap_pairs:
                                                if possible_overlapping_pairs == current_overlapping_pairs:
                                                    continue
                                                else:
                                                    pos_sel.append(loc[id_cell])
                                                    selected_idxs.append(id_cell)
                                                    n_sel += 1
                                                    placed = True
                                                    if verbose:
                                                        print('Number of overlapping pairs:',
                                                              possible_overlapping_pairs)
                                            else:
                                                if possible_overlapping_pairs == current_overlapping_pairs:
                                                    pos_sel.append(loc[id_cell])
                                                    selected_idxs.append(id_cell)
                                                    n_sel += 1
                                                    placed = True
                                                else:
                                                    if verbose:
                                                        print('Overlapping violation:', possible_overlapping_pairs)
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
                    print('Distance violation', np.min(dist), iter)
                pass
            else:
                amp = np.max(np.abs(np.min(templates[id_cell])))
                if not drifting:
                    if is_position_within_boundaries(loc[id_cell], x_lim, y_lim, z_lim) and min_amp < amp < max_amp:
                        if n_overlap_pairs is None:
                            # save cell
                            pos_sel.append(loc[id_cell])
                            selected_idxs.append(id_cell)
                            n_sel += 1
                            placed = True
                        else:
                            if len(selected_idxs) == 0:
                                # save cell
                                pos_sel.append(loc[id_cell])
                                selected_idxs.append(id_cell)
                                n_sel += 1
                                placed = True
                            else:
                                possible_selected = deepcopy(selected_idxs)
                                possible_selected.append(id_cell)
                                possible_overlapping_pairs = len(find_overlapping_templates(
                                    templates[np.array(possible_selected)],
                                    overlap_threshold))
                                current_overlapping_pairs = len(find_overlapping_templates(
                                    templates[np.array(selected_idxs)],
                                    overlap_threshold))
                                if current_overlapping_pairs < n_overlap_pairs and \
                                        possible_overlapping_pairs <= n_overlap_pairs:
                                    if possible_overlapping_pairs == current_overlapping_pairs:
                                        continue
                                    else:
                                        pos_sel.append(loc[id_cell])
                                        selected_idxs.append(id_cell)
                                        n_sel += 1
                                        placed = True
                                        if verbose:
                                            print('Number of overlapping pairs:', possible_overlapping_pairs)
                                else:
                                    if possible_overlapping_pairs == current_overlapping_pairs:
                                        pos_sel.append(loc[id_cell])
                                        selected_idxs.append(id_cell)
                                        n_sel += 1
                                        placed = True
                                    else:
                                        if verbose:
                                            print('Overlapping violation:', possible_overlapping_pairs)
                    else:
                        if verbose:
                            print('Amplitude or boundary violation', amp, loc[id_cell], iter)
                else:
                    # drifting
                    if is_position_within_boundaries(loc[id_cell, 0], x_lim, y_lim, z_lim) and min_amp < amp < max_amp:
                        # save cell
                        drift_angle = np.rad2deg(np.arccos(np.dot(drift_dir[id_cell], preferred_dir)))
                        if drift_angle - angle_tol <= 0:
                            if n_overlap_pairs is None:
                                pos_sel.append(loc[id_cell])
                                selected_idxs.append(id_cell)
                                placed = True
                            else:
                                possible_selected = deepcopy(selected_idxs)
                                possible_selected.append(id_cell)
                                overlapping = find_overlapping_templates(templates[np.array(possible_selected), 0],
                                                                         overlap_threshold)
                                possible_overlapping_pairs = len(overlapping)
                                if possible_overlapping_pairs <= n_overlap_pairs:
                                    pos_sel.append(loc[id_cell])
                                    selected_idxs.append(id_cell)
                                    n_sel += 1
                                    placed = True
                                    current_overlapping_pairs = len(overlapping)
                                    if verbose:
                                        print('Number of overlapping pairs:', current_overlapping_pairs)
                                else:
                                    if verbose:
                                        print('Overlapping violation:', current_overlapping_pairs)
                        else:
                            if verbose:
                                print('Drift violation', loc[id_cell, 0], iter)
                    else:
                        if verbose:
                            print('Amplitude or boundary violation', amp, loc[id_cell, 0], iter)
            if placed:
                selected_cat.append('U')

    if i == len(permuted_idxs) - 1 and n_sel < n_exc + n_inh:
        raise RuntimeError("Templates could not be selected. \n"
                           "Decrease number of spiketrains, decrease 'min_dist', or use more templates.")
    return selected_idxs, selected_cat


def resample_templates(templates, n_resample, up, down, drifting, dtype, verbose, n_jobs=None,
                       tmp_file=None, parallel=False):
    """
    Resamples the templates to a specified sampling frequency.
    Parameters
    ----------
    templates : np.array
        Array with templates (n_neurons, n_channels, n_samples)
        or (n_neurons, n_drift, n_channels, n_samples) if drifting
    n_resample : int
        Samples for resampled templates
    up : float
        The original sampling frequency in Hz
    down : float
        The new sampling frequency in Hz
    drifting : bool
        If True templates are assumed to be drifting
    verbose : bool
        If True output is verbose
    n_jobs : int
        Number of jobs for parallel processing. If None half cpus are used
    tmp_file : str
        Path to tmp file to memmap. If None, processing is in memory
    parallel : bool
        If True each template is resampled in parellel
    Returns
    -------
    template_rs : np.array
        Array with resampled templates (n_neurons, n_channels, n_resample)
        or (n_neurons, n_drift, n_channels, n_resample) if drifting
    """
    # create resampled templates
    if not drifting:
        if tmp_file is not None:
            templates_rs = np.memmap(tmp_file, shape=(templates.shape[0], templates.shape[1], n_resample),
                                     dtype=dtype, mode='w+')
        else:
            templates_rs = np.zeros((templates.shape[0], templates.shape[1], n_resample))
    else:
        if tmp_file is not None:
            templates_rs = np.memmap(tmp_file, shape=(templates.shape[0], templates.shape[1],
                                                      templates.shape[2], n_resample),
                                     dtype=dtype, mode='w+')
        else:
            templates_rs = np.zeros(
                (templates.shape[0], templates.shape[1], templates.shape[2], n_resample))

    if parallel:
        if n_jobs is None:
            n_jobs = os.cpu_count() // 2

        if verbose:
            print('Resampling with', n_jobs, 'jobs')

        output_list = Parallel(n_jobs=n_jobs)(
            delayed(_resample_parallel)(i, tem, up, down, drifting,) for i, tem in enumerate(templates))

        for i, tem in enumerate(templates):
            template_rs = output_list[i]
            if template_rs.shape[-1] < templates_rs.shape[-1]:
                if not drifting:
                    templates_rs[i, :, :len(template_rs)] = template_rs
                else:
                    templates_rs[i, :, :, :len(template_rs)] =template_rs
            elif template_rs.shape[-1] > templates_rs.shape[-1]:
                if not drifting:
                    templates_rs[i] = template_rs[:, :templates_rs.shape[-1]]
                else:
                    templates_rs[i] = template_rs[:, :, :templates_rs.shape[-1]]
            else:
                templates_rs[i] = template_rs
    else:
        for i, tem in enumerate(templates):
            template_rs = _resample_parallel(i, tem, up, down, drifting)
            if template_rs.shape[-1] < templates_rs.shape[-1]:
                if not drifting:
                    templates_rs[i, :, :len(template_rs)] = template_rs
                else:
                    templates_rs[i, :, :, :len(template_rs)] =template_rs
            elif template_rs.shape[-1] > templates_rs.shape[-1]:
                if not drifting:
                    templates_rs[i] = template_rs[:, :templates_rs.shape[-1]]
                else:
                    templates_rs[i] = template_rs[:, :, :templates_rs.shape[-1]]
            else:
                templates_rs[i] = template_rs

    return templates_rs


def pad_templates(templates, pad_samples, drifting, dtype, verbose, n_jobs=None,
                  tmp_file=None, parallel=False):
    """
    Pads the templates on both ends.
    Parameters
    ----------
    templates : np.array
        Array with templates (n_neurons, n_channels, n_samples)
        or (n_neurons, n_drift, n_channels, n_samples) if drifting
    pad_samples : list
        List of 2 ints with number of samples for padding before and after
    drifting : bool
        If True templates are assumed to be drifting
    verbose : bool
        If True output is verbose
    n_jobs : int
        Number of jobs for parallel processing. If None half cpus are used
    tmp_file : str
        Path to tmp file to memmap. If None, processing is in memory
    parallel : bool
        If True each template is padded in parellel
    Returns
    -------
    template_pad : np.array
        Array with padded templates (n_neurons, n_channels, n_padded_sample)
        or (n_neurons, n_drift, n_channels, n_padded_sample) if drifting
    """
    padded_template_samples = templates.shape[-1] + np.sum(pad_samples)
    # create padded templates
    if not drifting:
        if tmp_file is not None:
            templates_pad = np.memmap(tmp_file, shape=(templates.shape[0], templates.shape[1],
                                                       padded_template_samples),
                                      dtype=dtype, mode='w+')
        else:
            templates_pad = np.zeros((templates.shape[0], templates.shape[1], padded_template_samples))
    else:
        if tmp_file is not None:
            templates_pad = np.memmap(tmp_file, shape=(templates.shape[0], templates.shape[1],
                                                       templates.shape[2], padded_template_samples),
                                      dtype=dtype, mode='w+')
        else:
            templates_pad = np.zeros(
                (templates.shape[0], templates.shape[1], templates.shape[2], padded_template_samples))
    if parallel:
        if n_jobs is None:
            n_jobs = os.cpu_count() // 2

        if verbose:
            print('Padding with', n_jobs, 'jobs')
        output_list = Parallel(n_jobs=n_jobs)(delayed(_pad_parallel)(i, tem, pad_samples, drifting, verbose,)
                                                      for i, tem in enumerate(templates))

        for i, tem in enumerate(templates_pad):
            templates_pad[i] = output_list[i]
    else:
        for i, tem in enumerate(templates):
            templates_pad[i] = _pad_parallel(i, tem, pad_samples, drifting, verbose)
    return templates_pad


def jitter_templates(templates, upsample, fs, n_jitters, jitter, drifting, dtype, verbose, n_jobs=None,
                     tmp_file=None, parallel=False):
    """
    Adds jittered replicas to the templates.
    Parameters
    ----------
    templates : np.array
        Array with templates (n_neurons, n_channels, n_samples)
        or (n_neurons, n_drift, n_channels, n_samples) if drifting
    upsample : int
        Factor for upsampling the templates
    n_jitters : int
        Number of jittered copies for each template
    jitter : quantity
        Jitter in time for shifting the template
    drifting : bool
        If True templates are assumed to be drifting
    verbose : bool
        If True output is verbose
    n_jobs : int
        Number of jobs for parallel processing. If None half cpus are used
    tmp_file : str
        Path to tmp file to memmap. If None, processing is in memory
    parallel : bool
        If True each template is jittered in parellel
    Returns
    -------
    template_jitt : np.array
        Array with jittered templates (n_neurons, n_jitters, n_channels, n_samples)
        or (n_neurons, n_drift, n_jitters, n_channels, n_samples) if drifting
    """
    # create padded templates
    if not drifting:
        if tmp_file is not None:
            templates_jitter = np.memmap(tmp_file, shape=(templates.shape[0], n_jitters, templates.shape[1],
                                                          templates.shape[2]),
                                         dtype=dtype, mode='w+')
        else:
            templates_jitter = np.zeros((templates.shape[0], n_jitters, templates.shape[1],
                                                          templates.shape[2]))
    else:
        if tmp_file is not None:
            templates_jitter = np.memmap(tmp_file, shape=(templates.shape[0], templates.shape[1], n_jitters,
                                         templates.shape[2], templates.shape[3]),
                                      dtype=dtype, mode='w+')
        else:
            templates_jitter = np.zeros((templates.shape[0], templates.shape[1], n_jitters,
                                      templates.shape[2], templates.shape[3]))
    if parallel:
        if n_jobs is None:
            n_jobs = os.cpu_count() // 2

        if verbose:
            print('Jittering with', n_jobs, 'jobs')

        output_list = Parallel(n_jobs=n_jobs)(delayed(_jitter_parallel)(i, tem, upsample, fs, n_jitters, jitter,
                                                                        drifting, verbose,)
                                                      for i, tem in enumerate(templates))
        for i, tem in enumerate(templates_jitter):
            templates_jitter[i] = output_list[i]
    else:
        for i, tem in enumerate(templates):
            templates_jitter[i] = _jitter_parallel(i, tem, upsample, fs, n_jitters, jitter, drifting, verbose,)

    return templates_jitter


def cubic_padding(template, pad_samples):
    """
    Cubic spline padding on left and right side to 0. The initial offset of the templates is also removed.

    Parameters
    ----------
    template : np.array
        Templates to be padded (n_elec, n_samples)
    pad_samples : list
        Padding samples before and after the template

    Returns
    -------
    padded_template : np.array
        Padded template

    """
    import scipy.interpolate as interp
    n_pre = pad_samples[0]
    n_post = pad_samples[1]

    padded_template = np.zeros((template.shape[0], int(n_pre) + template.shape[1] + n_post))
    splines = np.zeros((template.shape[0], int(n_pre) + template.shape[1] + n_post))

    for i, sp in enumerate(template):
        # Remove inital offset
        sp_copy = deepcopy(sp)
        padded_sp = np.zeros(n_pre + len(sp) + n_post)
        padded_t = np.arange(len(padded_sp))
        initial_offset = np.mean(sp[0])
        sp_copy -= initial_offset

        x_pre = float(n_pre)
        x_pre_pad = np.arange(n_pre)
        x_post = float(n_post)
        x_post_pad = np.arange(n_post)[::-1]

        # fill pre and post intervals with linear values from sp[0] - sp[-1] to 0 for better fit
        m_pre = sp_copy[0] / x_pre
        m_post = sp_copy[-1] / x_post

        padded_sp[:n_pre] = m_pre * x_pre_pad
        padded_sp[n_pre:-n_post] = sp_copy
        padded_sp[-n_post:] = m_post * x_post_pad

        f = interp.interp1d(padded_t, padded_sp, kind='cubic')
        splines[i] = f(np.arange(len(padded_sp)))

        padded_template[i, :n_pre] = f(x_pre_pad)
        padded_template[i, n_pre:-n_post] = sp_copy
        padded_template[i, -n_post:] = f(np.arange(n_pre + len(sp_copy), n_pre + len(sp_copy) + n_post))

    return padded_template


def find_overlapping_templates(templates, thresh=0.8):
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
        if len(templates.shape) == 4:  # jitter
            temp_1 = temp_1[0]

        peak_electrode_idx = np.unravel_index(temp_1.argmin(), temp_1.shape)

        for j, temp_2 in enumerate(templates):
            if len(templates.shape) == 4:  # jitter
                temp_2 = temp_2[0]

            if i != j:
                if are_templates_overlapping([temp_1, temp_2], thresh):
                    if [i, j] not in overlapping_pairs and [j, i] not in overlapping_pairs:
                        overlapping_pairs.append(sorted([i, j]))

    return np.array(overlapping_pairs)


def are_templates_overlapping(templates, thresh):
    """
    Returns true if templates are spatially overlapping

    Parameters
    ----------
    templates : np.array
        Array with 2 templates (2, n_elec, n_samples)
    thresh : float
        Overlapping threshold ([0 - 1])

    Returns
    -------
    overlab : bool
        Whether the templates are spatially overlapping or not
    """
    assert len(templates) == 2
    temp_1 = templates[0]
    temp_2 = templates[1]
    peak_electrode_idx = np.unravel_index(temp_1.argmin(), temp_1.shape)
    peak_2_on_max = np.abs(np.min(temp_2[peak_electrode_idx]))
    peak_2 = np.abs(np.min(temp_2))

    if peak_2_on_max > thresh * peak_2:
        return True
    else:
        return False


### SPIKETRAIN OPERATIONS ###
def annotate_overlapping_spikes(spiketrains, t_jitt=1 * pq.ms, overlapping_pairs=None, parallel=True, verbose=True):
    """
    Annotate spike trains with temporal and spatio-temporal overlapping labels.
    NO - Non overlap
    TO - Temporal overlap
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
    if parallel:
        import multiprocessing
        threads = []
        manager = multiprocessing.Manager()
        return_spiketrains = manager.dict()
        for i, st_i in enumerate(spiketrains):
            p = multiprocessing.Process(target=annotate_parallel, args=(i, st_i, spiketrains, t_jitt,
                                                                        overlapping_pairs, return_spiketrains,
                                                                        verbose,))
            p.start()
            threads.append(p)
        for p in threads:
            p.join()
        # retrieve annotated spiketrains
        for i, st in enumerate(spiketrains):
            spiketrains[i] = return_spiketrains[i]
    else:
        # find overlapping spikes
        for i, st_i in enumerate(spiketrains):
            if verbose:
                print('Annotating overlapping spike train ', i)
            over = np.array(['NONE'] * len(st_i))
            for i_sp, t_i in enumerate(st_i):
                for j, st_j in enumerate(spiketrains):
                    if i != j:
                        # find overlapping
                        id_over = np.where((st_j > t_i - t_jitt) & (st_j < t_i + t_jitt))[0]
                        if not np.any(overlapping_pairs):
                            if len(id_over) != 0:
                                over[i_sp] = 'TO'
                        else:
                            pair = [i, j]
                            pair_i = [j, i]
                            if np.any([np.all(pair == p) for p in overlapping_pairs]) or \
                                    np.any([np.all(pair_i == p) for p in overlapping_pairs]):
                                if len(id_over) != 0:
                                    over[i_sp] = 'STO'
                            else:
                                if len(id_over) != 0:
                                    over[i_sp] = 'TO'
            over[over == 'NONE'] = 'NO'
            st_i.annotate(overlap=over)


def annotate_parallel(i, st_i, spiketrains, t_jitt, overlapping_pairs, return_spiketrains, verbose):
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
    over = np.array(['NONE'] * len(st_i))
    for i_sp, t_i in enumerate(st_i):
        for j, st_j in enumerate(spiketrains):
            if i != j:
                # find overlapping
                id_over = np.where((st_j > t_i - t_jitt) & (st_j < t_i + t_jitt))[0]
                if not np.any(overlapping_pairs):
                    if len(id_over) != 0:
                        over[i_sp] = 'TO'
                else:
                    pair = [i, j]
                    pair_i = [j, i]
                    if np.any([np.all(pair == p) for p in overlapping_pairs]) or \
                            np.any([np.all(pair_i == p) for p in overlapping_pairs]):
                        if len(id_over) != 0:
                            over[i_sp] = 'STO'
                    else:
                        if len(id_over) != 0:
                            over[i_sp] = 'TO'
    over[over == 'NONE'] = 'NO'
    st_i.annotate(overlap=over)
    return_spiketrains[i] = st_i


def resample_spiketrains(spiketrains, fs=None):
    """
    Resamples spike trains. Provide either fs or T parameters

    Parameters
    ----------
    spiketrains : list
        List of neo spiketrains to be resampled
    fs : Quantity
        New sampling frequency

    Returns
    -------
    resampled_mat : np.array
        Matrix with resampled binned spike trains

    """
    import elephant.conversion as conv

    resampled_mat = []
    if not fs:
        raise Exception('Provide either sampling frequency fs or time period T')
    elif fs:
        if not isinstance(fs, Quantity):
            raise ValueError("fs must be of type pq.Quantity")
        binsize = 1. / fs
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

                if consecutive == n_spikes - 1:
                    last_burst_event = st[i + 1]
                if consecutive >= 1:
                    if st[i + 1] - st[consecutive_idx[0]] >= max_burst_duration:
                        last_burst_event = st[i + 1] - 0.001 * pq.ms
                        consecutive = 0

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

                    if consecutive == n_spikes:
                        last_burst_event = st[i + 1]
                    if consecutive >= 1:
                        if st[i + 1] - st[consecutive_idx[0]] >= max_burst_duration:
                            last_burst_event = st[i + 1] - 0.001 * pq.ms
                            consecutive = 0

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


def compute_bursting_template(template, mod, wc_mod, filtfilt=False):
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
    filtfilt: bool
        If True forward-backward filter is used

    Returns
    -------
    temp_filt : np.array
        Modulated template

    """
    import scipy.signal as ss
    b, a = ss.butter(3, wc_mod)
    if len(template.shape) == 2:
        if filtfilt:
            temp_filt = ss.filtfilt(b, a, template, axis=1)
        else:
            temp_filt = ss.lfilter(b, a, template, axis=1)
        if mod.size > 1:
            temp_filt = np.array([m * np.min(temp) / np.min(temp_f) *
                                  temp_f for (m, temp, temp_f) in zip(mod, template, temp_filt)])
        else:
            temp_filt = (mod * np.min(template) / np.min(temp_filt)) * temp_filt
    else:
        if filtfilt:
            temp_filt = ss.filtfilt(b, a, template)
        else:
            temp_filt = ss.lfilter(b, a, template)
        temp_filt = (mod * np.min(template) / np.min(temp_filt)) * temp_filt
    return temp_filt


def sigmoid(x, b=1):
    """
    Compute sigmoid function

    Parameters
    ----------
    x: np.array
        Array to compute sigmoid
    b: float
        Sigmoid slope

    Returns
    -------
    x_sig: np.array
        Output sigmoid array
    """
    return 1 / (1 + np.exp(-b * x)) - 0.5


def compute_stretched_template(template, mod, shape_stretch=30.):
    """
    Compute modulation in shape for a template with low-pass filter.

    Parameters
    ----------
    template : np.array
        Template to be modulated (num_chan, n_samples) or (n_samples)
    mod : int or np.array
        Amplitude modulation for template or single electrodes
    shape_stretch : float
        Sigmoid range to stretch the template

    Returns
    -------
    temp_filt : np.array
        Modulated template
    """
    import scipy.interpolate as interp

    if isinstance(mod, (int, np.integer)):
        mod = np.array(mod)

    if mod.size > 1:
        stretch_factor = np.mean(mod)
        mod_value = np.mean(mod)
    else:
        stretch_factor = mod
        mod_value = mod

    if len(template.shape) == 2:
        min_idx = np.unravel_index(np.argmin(template), template.shape)[1]
        x_centered = np.arange(-min_idx, template.shape[1] - min_idx)
        x_centered = x_centered / float(np.ptp(x_centered))
        x_centered = x_centered * shape_stretch

        if stretch_factor >= 1:
            x_stretch = x_centered
        else:
            x_stretch = sigmoid(x_centered, 1 - stretch_factor)

        x_stretch = x_stretch / float(np.ptp(x_stretch))
        x_stretch *= shape_stretch + (np.min(x_centered) - np.min(x_stretch))
        x_recovered = np.max(x_stretch) / np.max(x_centered) * x_centered
        x_stretch = np.round(x_stretch, 6)
        x_recovered = np.round(x_recovered, 6)
        temp_filt = np.zeros(template.shape)
        for i, t in enumerate(template):
            try:
                f = interp.interp1d(x_stretch, t, kind='cubic')
                temp_filt[i] = f(x_recovered)
            except Exception as e:
                raise Exception("'shape_stretch' is too large. Try reducing it (default = 30)")
        if mod.size > 1:
            temp_filt = np.array([m * np.min(temp) / np.min(temp_f) *
                                  temp_f for (m, temp, temp_f) in zip(mod, template, temp_filt)])
        else:
            temp_filt = (mod * np.min(template) / np.min(temp_filt)) * temp_filt
    else:
        min_idx = np.argmin(template)
        x_centered = np.arange(-min_idx, len(template) - min_idx)
        x_centered = x_centered / float(np.ptp(x_centered))
        x_centered = x_centered * shape_stretch

        if stretch_factor >= 1:
            x_stretch = x_centered
        else:
            x_stretch = sigmoid(x_centered, 1 - stretch_factor)
        x_stretch = x_stretch / float(np.ptp(x_stretch))
        x_stretch *= shape_stretch + (np.min(x_centered) - np.min(x_stretch))
        x_recovered = np.max(x_stretch) / np.max(x_centered) * x_centered
        x_stretch = np.round(x_stretch, 6)
        x_recovered = np.round(x_recovered, 6)
        try:
            f = interp.interp1d(x_stretch, template, kind='cubic')
            temp_filt = f(x_recovered)
        except Exception as e:
            raise Exception("'shape_stretch' is too large. Try reducing it (default = 30)")
        temp_filt = (mod_value * np.min(template) / np.min(temp_filt)) * temp_filt
    return temp_filt


def convolve_single_template(spike_id, st_idx, template, n_samples, cut_out=None, modulation=False, mod_array=None,
                             bursting=False, shape_stretch=None):
    """Convolve single template with spike train. Used to compute 'spike_traces'.

    Parameters
    ----------
    spike_id : int
        Index of spike trains - template.
    st_idx : np.array
        Spike times
    template : np.array
        Array with single template
    n_samples : int
        Number of samples in chunk
    cut_out : list
        Number of samples before and after the peak
    modulation : bool
        If True modulation is applied
    mod_array : np.array
        Array with modulation value for each spike
    bursting : bool
        If True templates are modulated in shape
    shape_stretch : float
        Range of sigmoid transform for bursting shape stretch

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

    spike_trace = np.zeros(n_samples)

    if len(template.shape) == 2:
        rand_idx = np.random.randint(njitt)
        temp_jitt = template[rand_idx]
        for pos, spos in enumerate(st_idx):
            if not modulation:
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
                    if spos - cut_out[0] >= 0 and spos - cut_out[0] + len_spike <= n_samples:
                        spike_trace[spos - cut_out[0]:spos - cut_out[0] + len_spike] += \
                            compute_stretched_template(temp_jitt, mod_array[pos], shape_stretch)
                    elif spos - cut_out[0] < 0:
                        diff = -(spos - cut_out[0])
                        temp_filt = compute_stretched_template(temp_jitt, mod_array[pos], shape_stretch)
                        spike_trace[:spos - cut_out[0] + len_spike] += temp_filt[diff:]
                    else:
                        diff = n_samples - (spos - cut_out[0])
                        temp_filt = compute_stretched_template(temp_jitt, mod_array[pos], shape_stretch)
                        spike_trace[spos - cut_out[0]:] += temp_filt[:diff]
                else:
                    if mod_array[pos].size > 1:
                        mod_value = np.mean(mod_array[pos])
                    else:
                        mod_value = mod_array[pos]
                    if spos - cut_out[0] >= 0 and spos - cut_out[0] + len_spike <= n_samples:
                        spike_trace[spos - cut_out[0]:spos - cut_out[0] + len_spike] += mod_value * temp_jitt
                    elif spos - cut_out[0] < 0:
                        diff = -(spos - cut_out[0])
                        spike_trace[:spos - cut_out[0] + len_spike] += mod_value * temp_jitt[diff:]
                    else:
                        diff = n_samples - (spos - cut_out[0])
                        spike_trace[spos - cut_out[0]:] += mod_value * temp_jitt[:diff]
    else:
        raise Exception('For drifting len(template.shape) should be 2')
    return spike_trace


def convolve_templates_spiketrains(spike_id, st_idx, template, n_samples,cut_out=None, modulation=False, mod_array=None,
                                   verbose=False, bursting=False, shape_stretch=None, template_idxs=None,
                                   max_channels_per_template=None, recordings=None):
    """
    Convolve template with spike train on all electrodes. Used to compute 'recordings'.
    Parameters
    ----------
    spike_id : int
        Index of spike trains - template.
    st_idx : np.array
        Spike times
    template : np.array
        Array with template
    n_samples : int
        Number of samples in chunk
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
    shape_stretch : float
        Range of sigmoid transform for bursting shape stretch
    template_idxs : np.array
        Array of template indices (used for drifting templates)
    max_channels_per_template : np.array
        Maximum number of channels to be convolved
    recordings :  np.arrays
        Array to use for recordings. If None it is created
    Returns
    -------
    recordings: np.array
        Trace with convolved signals (n_elec, n_samples)
    """
    if verbose:
        print('Convolution with spike:', spike_id)
    if len(template.shape) == 3:
        n_jitt = template.shape[0]
        n_elec = template.shape[1]
        len_spike = template.shape[2]
    else:
        assert template_idxs is not None, "For drifting templates, pass the 'template_idxs' argument to select the " \
                                          "right template for each spike"
        n_jitt = template.shape[1]
        n_elec = template.shape[2]
        len_spike = template.shape[3]

    if recordings is None:
        recordings = np.zeros((n_samples, n_elec))
    else:
        assert recordings.shape == (n_samples, n_elec), "'recordings' has the wrong shape"

    if cut_out is None:
        cut_out = [len_spike // 2, len_spike // 2]

    if not modulation:
        # No modulation
        mod_array = np.ones_like(st_idx)
    else:
        assert mod_array is not None, " For 'electrode' and 'template' modulations provide 'mod_array'"

    for pos, spos in enumerate(st_idx):
        rand_idx = np.random.randint(n_jitt)
        if template_idxs is not None:
            temp_jitt = template[template_idxs[pos], rand_idx]
        else:
            temp_jitt = template[rand_idx]

        if max_channels_per_template is None:
            elec_idxs = np.arange(n_elec)
        else:
            # find max channels
            peak_idx = np.unravel_index(np.argmax(np.abs(temp_jitt)), temp_jitt.shape)[1]
            elec_idxs = np.argsort(np.abs(temp_jitt[:, peak_idx]))[::-1][:max_channels_per_template]
            temp_jitt = temp_jitt[elec_idxs]

        if bursting:
            if not isinstance(mod_array[0], (list, tuple, np.ndarray)):
                # template
                if spos - cut_out[0] >= 0 and spos - cut_out[0] + len_spike <= n_samples:
                    recordings[spos - cut_out[0]:spos + cut_out[1], elec_idxs] += \
                        compute_stretched_template(temp_jitt, mod_array[pos], shape_stretch).T
                elif spos - cut_out[0] < 0:
                    diff = -(spos - cut_out[0])
                    temp_filt = compute_stretched_template(temp_jitt, mod_array[pos], shape_stretch)
                    recordings[:spos + cut_out[1], elec_idxs] += temp_filt[:, diff:].T
                else:
                    diff = n_samples - (spos - cut_out[0])
                    temp_filt = compute_stretched_template(temp_jitt, mod_array[pos], shape_stretch)
                    recordings[spos - cut_out[0]:, elec_idxs] += temp_filt[:, :diff].T
            else:
                # electrode
                mod_values = mod_array[pos][elec_idxs]
                if spos - cut_out[0] >= 0 and spos - cut_out[0] + len_spike <= n_samples:
                    recordings[spos - cut_out[0]:spos + cut_out[1], elec_idxs] += \
                        compute_stretched_template(temp_jitt, mod_array[pos], shape_stretch).T
                elif spos - cut_out[0] < 0:
                    diff = -(spos - cut_out[0])
                    temp_filt = compute_stretched_template(temp_jitt, mod_array[pos], shape_stretch)
                    recordings[:spos + cut_out[1], elec_idxs] += temp_filt[:, diff:].T
                else:
                    diff = n_samples - (spos - cut_out[0])
                    temp_filt = compute_stretched_template(temp_jitt, mod_array[pos], shape_stretch)
                    recordings[spos - cut_out[0]:, elec_idxs] += temp_filt[:, :diff].T
        else:
            if not isinstance(mod_array[0], (list, tuple, np.ndarray)):
                # template + none
                if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                    recordings[spos - cut_out[0]:spos + cut_out[1], elec_idxs] += mod_array[pos] * temp_jitt.T
                elif spos - cut_out[0] < 0:
                    diff = -(spos - cut_out[0])
                    recordings[:spos + cut_out[1], elec_idxs] += mod_array[pos] * temp_jitt[:, diff:].T
                else:
                    diff = n_samples - (spos - cut_out[0])
                    recordings[spos - cut_out[0]:, elec_idxs] += mod_array[pos] * temp_jitt[:, :diff].T
            else:
                # electrode
                mod_values = mod_array[pos][elec_idxs]
                if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                    recordings[spos - cut_out[0]:spos + cut_out[1], elec_idxs] += \
                        np.transpose([a * t for (a, t) in zip(mod_array[pos], temp_jitt)])
                elif spos - cut_out[0] < 0:
                    diff = -(spos - cut_out[0])
                    recordings[:spos + cut_out[1], elec_idxs] += \
                        np.transpose([a * t for (a, t) in zip(mod_array[pos], temp_jitt[:, diff:])])
                else:
                    diff = n_samples - (spos - cut_out[0])
                    recordings[spos - cut_out[0]:, elec_idxs] += \
                        np.transpose([a * t for (a, t) in zip(mod_array[pos], temp_jitt[:, :diff])])

    return recordings


def convolve_drifting_templates_spiketrains(spike_id, st_idx, template, n_samples, fs, loc, t_start_drift,
                                            drift_mode='slow', slow_drift_velocity=5, fast_drift_period=10,
                                            fast_drift_min_jump=5, fast_drift_max_jump=20, cut_out=None,
                                            modulation=False, mod_array=None,
                                            verbose=False, bursting=False, shape_stretch=None,
                                            chunk_start=None, recordings=None):
    """
    Convolve template with spike train on all electrodes. Used to compute 'recordings'.

    Parameters
    ----------
    spike_id : int
        Index of spike trains - template
    st_idx : np.array
        Spike times
    template : np.array
        Array with drifting template
    n_samples : int
        Number of samples in chunk
    fs : Quantity
        Sampling frequency
    loc : np.array
        Locations of drifting templates
    t_start_drift : Quantity
        Drifting start time in s
    drift_mode : str
        Drift modality ['slow' | 'fast' | 'slow+fast']
    slow_drift_velocity : 3d array
        Slow drifting velocity vector in um/s
    fast_drift_period : Quantity
        Period between fast drifts in s
    fast_drift_min_jump: float
        Minimum 'jump' for fast drifts in uV
    fast_drift_max_jump: float
        Maximum 'jump' for fast drifts in uV
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
    shape_stretch : float
        Range of sigmoid transform for bursting shape stretch
    chunk_start : quantity
        Chunk start time used to compute drifting position
    recordings :  np.arrays
        Array to use for recordings. If None it is created

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
        print('Drifting convolution with spike:', spike_id)
    assert len(template.shape) == 4, "For drifting len(template.shape) should be 4"

    n_jitt = template.shape[1]
    n_elec = template.shape[2]
    len_spike = template.shape[3]

    if cut_out is None:
        cut_out = [len_spike // 2, len_spike // 2]
    if chunk_start is None:
        chunk_start = 0 * pq.s

    if recordings is None:
        recordings = np.zeros((n_samples, n_elec))
    else:
        assert recordings.shape == (n_samples, n_elec), "'recordings' has the wrong shape"

    # find drift direction and magnitude
    drift_direction = loc[-1] - loc[0]
    drift_direction /= np.linalg.norm(drift_direction)
    v_drift_mag = np.linalg.norm(slow_drift_velocity)

    template_idxs = np.zeros(len(st_idx), dtype='int')

    # set states for fast drifts
    temp_jitt = template[0, 0]
    max_chan = np.unravel_index(np.argmin(temp_jitt), temp_jitt.shape)[0]
    # fix multiple chunks fast drift time
    drift_state = {'old_pos': loc[0], 'old_chan': max_chan,
                   'old_amp': np.min(temp_jitt[max_chan]), 'old_idx': 0, 'last_drift_position': loc[0],
                   'direction_sign': 1, 'last_fast_drift': t_start_drift, 'loc_start': loc[0],
                   't_start_drift': t_start_drift}
    drift_params = {'fast_drift_period': fast_drift_period, 'fast_drift_min_jump': fast_drift_min_jump,
                    'fast_drift_max_jump': fast_drift_max_jump, 'v_drift_mag': v_drift_mag,
                    'drift_direction': drift_direction, 'max_iter': 100}

    if not modulation:
        # No modulation
        mod_array = np.ones_like(st_idx)
    else:
        assert mod_array is not None, " For 'electrode' and 'template' modulations provide 'mod_array'"

    for pos, spos in enumerate(st_idx):
        rand_idx = np.random.randint(n_jitt)
        sp_time = chunk_start + spos / fs
        if sp_time < t_start_drift:
            temp_idx = 0
            temp_jitt = template[temp_idx, rand_idx]
        else:
            temp_jitt, temp_idx, drift_state = _find_new_drift_template(drift_mode, st_idx, sp_time, pos,
                                                                        template, template_idxs, loc,
                                                                        chunk_start, t_start_drift, fs,
                                                                        rand_idx, drift_params, drift_state,
                                                                        verbose)
            template_idxs[pos] = temp_idx

        if bursting:
            if not isinstance(mod_array[0], (list, tuple, np.ndarray)):
                # template
                if spos - cut_out[0] >= 0 and spos - cut_out[0] + len_spike <= n_samples:
                    recordings[spos - cut_out[0]:spos + cut_out[1]] += \
                        compute_stretched_template(temp_jitt, mod_array[pos], shape_stretch).T
                elif spos - cut_out[0] < 0:
                    diff = -(spos - cut_out[0])
                    temp_filt = compute_stretched_template(temp_jitt, mod_array[pos], shape_stretch)
                    recordings[:spos + cut_out[1]] += temp_filt[:, diff:].T
                else:
                    diff = n_samples - (spos - cut_out[0])
                    temp_filt = compute_stretched_template(temp_jitt, mod_array[pos], shape_stretch)
                    recordings[spos - cut_out[0]:] += temp_filt[:, :diff].T
            else:
                # electrode
                if spos - cut_out[0] >= 0 and spos - cut_out[0] + len_spike <= n_samples:
                    recordings[spos - cut_out[0]:spos + cut_out[1]] += \
                        compute_stretched_template(temp_jitt, mod_array[pos], shape_stretch).T
                elif spos - cut_out[0] < 0:
                    diff = -(spos - cut_out[0])
                    temp_filt = compute_stretched_template(temp_jitt, mod_array[pos], shape_stretch)
                    recordings[:spos + cut_out[1]] += temp_filt[:, diff:].T
                else:
                    diff = n_samples - (spos - cut_out[0])
                    temp_filt = compute_stretched_template(temp_jitt, mod_array[pos], shape_stretch)
                    recordings[spos - cut_out[0]:] += temp_filt[:, :diff].T
        else:
            if not isinstance(mod_array[0], (list, tuple, np.ndarray)):
                # template + none
                if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                    recordings[spos - cut_out[0]:spos + cut_out[1]] += mod_array[pos] * temp_jitt.T
                elif spos - cut_out[0] < 0:
                    diff = -(spos - cut_out[0])
                    recordings[:spos + cut_out[1]] += mod_array[pos] * temp_jitt[:, diff:].T
                else:
                    diff = n_samples - (spos - cut_out[0])
                    recordings[spos - cut_out[0]:] += mod_array[pos] * temp_jitt[:, :diff].T
            else:
                # electrode
                if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                    recordings[spos - cut_out[0]:spos + cut_out[1]] += \
                        np.transpose([a * t for (a, t) in zip(mod_array[pos], temp_jitt)])
                elif spos - cut_out[0] < 0:
                    diff = -(spos - cut_out[0])
                    recordings[:spos + cut_out[1]] += \
                        np.transpose([a * t.T for (a, t) in zip(mod_array[pos], temp_jitt[:, diff:])])
                else:
                    diff = n_samples - (spos - cut_out[0])
                    recordings[spos - cut_out[0]:] += \
                        np.transpose([a * t.T for (a, t) in zip(mod_array[pos], temp_jitt[:, :diff])])

    return recordings, template_idxs


### RECORDING OPERATION ###
def extract_wf(spiketrains, recordings, fs, cut_out=2, timestamps=None):
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
    cut_out : float or list
         Length in ms to cut before and after spike peak. If a single value the cut is symmetrical
    timestamps : Quantity array (optional)
        Array with recordings timestamps
    """
    if cut_out is None:
        cut_out = 2
    if not isinstance(cut_out, list):
        n_pad = int(cut_out * pq.ms * fs.rescale('kHz'))
        n_pad = [n_pad, n_pad]
    else:
        n_pad = [int(p * pq.ms * fs.rescale('kHz')) for p in cut_out]

    n_elec, n_samples = recordings.shape
    if timestamps is None:
        timestamps = np.arange(n_samples) / fs.rescale('Hz')
    unit = timestamps[0].rescale('ms').units

    for st in spiketrains:
        sp_rec_wf = []
        sp_amp = []
        for t in st:
            idx = np.where(timestamps >= t)[0]
            if len(idx) > 0:
                idx = idx[0]
            else:
                idx = len(timestamps) - 1
            # find single waveforms crossing thresholds
            if idx - n_pad[0] > 0 and idx + n_pad[1] < n_samples:
                spike_rec = recordings[:, idx - n_pad[0]:idx + n_pad[1]]
            elif idx - n_pad[0] < 0:
                spike_rec = recordings[:, :idx + n_pad[1]]
                spike_rec = np.pad(spike_rec, ((0, 0), (np.abs(idx - n_pad[0]), 0)), 'constant')
            elif idx + n_pad[1] > n_samples:
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
            signals_filt = filtfilt(b, a, signals, axis=0)
        elif len(signals.shape) == 1:
            signals_filt = filtfilt(b, a, signals)
        return signals_filt
    else:
        raise ValueError('Filter is not stable')


### PLOTTING ###
def plot_rasters(spiketrains, cell_type=False, ax=None, overlap=False, color=None, fs=10,
                 marker='|', mew=2, markersize=5):
    """
    Plot raster for spike trains.

    Parameters
    ----------
    spiketrains : list
        List of neo spike trains
    cell_type : bool
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
        if cell_type:
            if 'cell_type' in spiketrain.annotations.keys():
                if spiketrain.annotations['cell_type'] == 'E':
                    ax.plot(t, i * np.ones_like(t), 'b', marker=marker, mew=mew, markersize=markersize, ls='')
                elif spiketrain.annotations['cell_type'] == 'I':
                    ax.plot(t, i * np.ones_like(t), 'r', marker=marker, mew=mew, markersize=markersize, ls='')
            else:
                if color is not None:
                    if isinstance(color, list) or isinstance(color, np.ndarray):
                        ax.plot(t, i * np.ones_like(t), color=color[i], marker=marker, mew=mew, markersize=markersize,
                                ls='')
                    else:
                        ax.plot(t, i * np.ones_like(t), color=color, marker=marker, mew=mew, markersize=markersize,
                                ls='')
                else:
                    ax.plot(t, i * np.ones_like(t), 'k', marker=marker, mew=mew, markersize=markersize, ls='')
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
                    if spiketrain.annotations['overlap'][j] == 'STO':
                        ax.plot(t_sp, i, 'r', marker=marker, mew=mew, markersize=markersize, ls='')
                    elif spiketrain.annotations['overlap'][j] == 'TO':
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


def plot_templates(gen, template_ids=None, single_jitter=True, ax=None, single_axes=False, max_templates=None,
                   drifting=False, cmap=None, ncols=6, **kwargs):
    """
    Plot templates.

    Parameters
    ----------
    gen : TemplateGenerator or RecordingGenerator
        Generator object containing templates
    template_ids : int or list
        The template(s) to plot
    single_axes : bool
        If True all templates are plotted on the same axis
    ax : axis
        Matplotlib  axis
    single_jitter: bool
        If True and jittered templates are present, a single jittered template is plotted
    max_templates: int
        Maximum number of templates to be plotted
    drifting: bool
        If True and templates are drifting, drifting templates are displayed
    cmap : matplotlib colormap
        Colormap to be used
    ncols :  int
        Number of columns for subplots

    Returns
    -------
    ax : ax
        Matplotlib axes

    """
    import matplotlib.pylab as plt
    from matplotlib import gridspec

    templates = gen.templates
    mea = mu.return_mea(info=gen.info['electrodes'])

    if 'params' in gen.info.keys():
        if gen.info['params']['drifting']:
            if not drifting:
                templates = templates[:, 0]
    if 'recordings' in gen.info.keys():
        if gen.info['recordings']['drifting']:
            if single_jitter:
                if not drifting:
                    if len(templates.shape) == 5:
                        templates = templates[:, 0, 0]
                    else:
                        templates = templates[:, 0]
                else:
                    if len(templates.shape) == 5:
                        templates = templates[:, :, 0]
            else:
                if not drifting:
                    if len(templates.shape) == 5:
                        templates = templates[:, 0]
        else:
            if single_jitter:
                if len(templates.shape) == 4:
                    templates = templates[:, 0]

    if drifting:
        assert isinstance(template_ids, (int, np.integer)), "When plotting drifting templates, 'template_ids' should " \
                                                            "be a single index (int)"
        single_axes = True

    if template_ids is not None:
        if isinstance(template_ids, (int, np.integer)):
            template_ids = list(np.array([template_ids]))
        elif isinstance(template_ids, list):
            template_ids = list(np.array(template_ids))
    else:
        template_ids = list(np.arange(templates.shape[0]))

    if max_templates is not None:
        if max_templates < len(templates):
            random_idxs = np.random.permutation(len(templates))
            template_ids = np.arange(templates.shape[0])[random_idxs][:max_templates]
            # templates = templates[random_idxs][:max_templates]

    n_sources = len(template_ids)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    if 'vscale' not in kwargs.keys():
        kwargs['vscale'] = 1.5 * np.max(np.abs(templates[template_ids]))

    if single_axes:
        if cmap is not None:
            cm = plt.get_cmap(cmap)
            colors = [cm(i / len(template_ids)) for i in np.arange(len(template_ids))]
        else:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for n, t in enumerate(templates):
            if n in template_ids:
                if len(t.shape) == 3:
                    if not drifting:
                        mu.plot_mea_recording(t.mean(axis=0), mea, colors=colors[np.mod(n, len(colors))], ax=ax,
                                              **kwargs)
                    else:
                        if cmap is None:
                            cmap = 'Reds'
                        cm = plt.get_cmap(cmap)
                        colors = [cm(i / t.shape[0]) for i in np.arange(t.shape[0])]
                        mu.plot_mea_recording(t, mea, colors=colors, ax=ax,
                                              **kwargs)
                else:
                    mu.plot_mea_recording(t, mea, colors=colors[np.mod(n, len(colors))], ax=ax, **kwargs)
    else:
        if n_sources > ncols:
            nrows = int(np.ceil(len(template_ids) / ncols))
        else:
            nrows = 1
            ncols = n_sources

        if cmap is not None:
            cm = plt.get_cmap(cmap)
            colors = [cm(i / len(template_ids)) for i in np.arange(len(template_ids))]
        else:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        gs = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=ax)

        for i_n, n in enumerate(template_ids):
            r = i_n // ncols
            c = np.mod(i_n, ncols)
            gs_sel = gs[r, c]
            ax_t = fig.add_subplot(gs_sel)
            if cmap is not None:
                mu.plot_mea_recording(templates[n], mea, ax=ax_t, colors=colors[i_n], **kwargs)
            else:
                mu.plot_mea_recording(templates[n], mea, ax=ax_t, colors=colors[np.mod(i_n, len(colors))], **kwargs)
        ax.axis('off')

    return ax


def plot_recordings(recgen, ax=None, start_time=None, end_time=None, overlay_templates=False, n_templates=None,
                    max_channels_per_template=16, cmap=None, templates_lw=1, **kwargs):
    """
    Plot recordings.

    Parameters
    ----------
    recgen : RecordingGenerator
        Recording generator object to plot
    ax : axis
        Matplotlib  axis
    start_time : float
        Start time to plot recordings in s
    end_time : float
        End time to plot recordings in s
    overlay_templates : bool
        If True, templates are overlaid on the recordings
    n_templates : int
        Number of templates to overlay (if overlay_templates is True)
    max_channels_per_template : int
        Number of maximum channels in which the template is overlaid
    cmap : matplotlib colormap
        Colormap to be used
    Returns
    -------
    ax : axis
        Matplotlib axis
    """
    import matplotlib.pylab as plt

    recordings = recgen.recordings
    mea = mu.return_mea(info=recgen.info['electrodes'])
    fs = recgen.info['recordings']['fs']
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if start_time is None:
        start_frame = 0
    else:
        start_frame = int(start_time * fs)
    if end_time is None:
        end_frame = recordings.shape[1]
    else:
        end_frame = int(end_time * fs)

    if max_channels_per_template is None:
        max_channels_per_template = len(recordings)

    mu.plot_mea_recording(recordings[:, start_frame:end_frame], mea, ax=ax, **kwargs)

    if 'vscale' not in kwargs.keys():
        kwargs['vscale'] = 1.5 * np.max(np.abs(recordings))

    if overlay_templates:
        if 'lw' in kwargs.keys():
            kwargs['lw'] = templates_lw
        fs = recgen.info['recordings']['fs'] * pq.Hz
        if n_templates is None:
            template_ids = np.arange(len(recgen.templates))
        else:
            template_ids = np.random.permutation(len(recgen.templates))[:n_templates]

        cut_out_samples = [int((c + p) * fs.rescale('kHz').magnitude)
                           for (c, p) in zip(recgen.info['templates']['cut_out'], recgen.info['templates']['pad_len'])]

        spike_idxs = []
        for st in recgen.spiketrains:
            spike_idxs.append((st.times * fs).magnitude.astype('int'))
        n_samples = recordings.shape[1]

        if cmap is not None:
            cm = plt.get_cmap(cmap)
            colors_t = [cm(i / len(template_ids)) for i in np.arange(len(template_ids))]
        else:
            colors_t = plt.rcParams['axes.prop_cycle'].by_key()['color']
        i_col = 0
        if 'lw' not in kwargs.keys():
            kwargs['lw'] = 1
        if 'colors' in kwargs.keys():
            del kwargs['colors']

        for i, (sp, t) in enumerate(zip(spike_idxs, recgen.templates)):
            if i in template_ids:
                template_idxs = None
                if 'drifting' in recgen.spiketrains[i].annotations.keys():
                    if recgen.spiketrains[i].annotations['drifting']:
                        template_idxs = recgen.spiketrains[i].annotations['template_idxs']

                rec_t = convolve_templates_spiketrains(i, sp, t, n_samples, template_idxs=template_idxs,
                                                       max_channels_per_template=max_channels_per_template,
                                                       cut_out=cut_out_samples).T

                rec_t[np.abs(rec_t) < 1e-4] = np.nan
                mu.plot_mea_recording(rec_t[:, start_frame:end_frame], mea, ax=ax,
                                      colors=colors_t[np.mod(i_col, len(colors_t))], **kwargs)
                i_col += 1
                del rec_t
    return ax


def plot_waveforms(recgen, spiketrain_id=None, ax=None, color=None, cmap=None, electrode=None,
                   max_waveforms=None, ncols=6, cut_out=2):
    """
    Plot waveforms of a spike train.

    Parameters
    ----------
    recgen : RecordingGenerator
        Recording generator object to plot spike train waveform from
    spiketrain_id : int or list
        Indexes of spike trains
    ax : axis
        Matplotlib  axis
    color : matplotlib color
        Color of the waveforms
    cmap : matplotlib colormap
        Colormap to be used
    electrode : int or 'max'
        Electrode id or 'max'
    ncols :  int
        Number of columns for subplots
    cut_out : float or list
        Cut outs in ms for waveforms (if not computed). If float the cut out is symmetrical.

    Returns
    -------
    ax : axis
        Matplotlib axis

    """
    import matplotlib.pylab as plt
    import matplotlib.gridspec as gridspec

    if spiketrain_id is None:
        spiketrain_id = np.arange(len(recgen.spiketrains))
    elif isinstance(spiketrain_id, (int, np.integer)):
        spiketrain_id = [spiketrain_id]

    n_units = len(spiketrain_id)

    waveforms = []
    for sp in spiketrain_id:
        wf = recgen.spiketrains[sp].waveforms
        if wf is None:
            fs = recgen.info['recordings']['fs'] * pq.Hz
            extract_wf([recgen.spiketrains[sp]], recgen.recordings, fs, cut_out=cut_out)
            wf = recgen.spiketrains[sp].waveforms
        waveforms.append(wf)

    mea = mu.return_mea(info=recgen.info['electrodes'])
    if max_waveforms is not None:
        for i, wf in enumerate(waveforms):
            if len(wf) > max_waveforms:
                waveforms[i] = wf[np.random.permutation(len(wf))][:max_waveforms]

    if n_units > 1:
        if color is None:
            if cmap is not None:
                cm = plt.get_cmap(cmap)
                colors = [cm(i / n_units) for i in np.arange(n_units)]
            else:
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        else:
            colors = [color] * n_units
    else:
        colors = 'k'

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    if electrode is None:
        if n_units > ncols:
            nrows = int(np.ceil(n_units / ncols))
        else:
            nrows = 1
            ncols = n_units

        gs = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=ax)

        for i, wf in enumerate(waveforms):
            r = i // ncols
            c = np.mod(i, ncols)
            gs_sel = gs[r, c]
            ax_t = fig.add_subplot(gs_sel)
            vscale = 1.5 * np.max(np.abs(wf))
            ax_t = mu.plot_mea_recording(wf, mea, colors=colors[i], ax=ax_t, lw=0.1, vscale=vscale)
            ax_t = mu.plot_mea_recording(wf.mean(axis=0), mea, colors=colors[i], ax=ax_t, lw=2, vscale=vscale)
    else:
        assert isinstance(electrode, (int, np.integer)) or electrode == 'max', "electrode must be int or 'max'"
        if len(spiketrain_id) > ncols:
            nrows = int(np.ceil(len(spiketrain_id) / ncols))
        else:
            nrows = 1
            ncols = len(spiketrain_id)

        gs = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=ax)

        # find ylim
        min_wf = 0
        max_wf = 0

        for wf in waveforms:
            wf_mean = wf.mean(axis=0)
            if np.min(wf_mean) < min_wf:
                min_wf = np.min(wf_mean)
            if np.max(wf_mean) > max_wf:
                max_wf = np.max(wf_mean)

        ylim = [min_wf - 0.2 * abs(min_wf), max_wf + 0.2 * abs(min_wf)]

        for i, wf in enumerate(waveforms):
            r = i // ncols
            c = np.mod(i, ncols)
            gs_sel = gs[r, c]
            ax_sel = fig.add_subplot(gs_sel)
            if electrode == 'max':
                electrode_idx = np.unravel_index(np.argmin(wf.mean(axis=0)), wf.mean(axis=0).shape)[0]
                print('max electrode: ', electrode_idx)
            else:
                electrode_idx = electrode
            if i == 0:
                ax_sel.set_ylabel('voltage ($\mu$V)', fontsize=15)
            ax_sel.plot(wf[:, electrode_idx].T, color=colors[np.mod(i, len(colors))], lw=0.1)
            ax_sel.plot(wf[:, electrode_idx].mean(axis=0), color='k', lw=1)
            ax_sel.set_title('Unit ' + str(i) + ' - Ch. ' + str(electrode_idx), fontsize=12)
            ax_sel.set_ylim(ylim)
            if c != 0:
                ax_sel.spines['left'].set_visible(False)
                ax_sel.set_yticks([])
            ax_sel.spines['right'].set_visible(False)
            ax_sel.spines['top'].set_visible(False)
    ax.axis('off')

    return ax


def plot_amplitudes(recgen, spiketrain_id=None, electrode=None, ax=None, color=None, cmap=None, single_axes=True,
                    marker='*', ms=5, ncols=6):
    """
    Plot waveform amplitudes over time.

    Parameters
    ----------
    recgen : RecordingGenerator
        Recording generator object to plot spike train waveform from
    spiketrain_id : int or list
        Indexes of spike trains
    electrode : int or 'max'
        Electrode id or 'max'
    ax : axis
        Matplotlib  axis
    color : matplotlib color
        Color of the waveform amplitudes
    cmap : matplotlib colormap
        Colormap to be used
    single_axes : bool
        If True all templates are plotted on the same axis
    marker : str
        Matplotlib marker (default '*')
    ms : int
        Markersize (default 5)
    ncols :  int
        Number of columns for subplots
    
    Returns
    -------
    ax : axis
        Matplotlib axis

    """
    import matplotlib.pylab as plt
    import matplotlib.gridspec as gridspec

    if spiketrain_id is None:
        spiketrain_id = np.arange(len(recgen.spiketrains))
    elif isinstance(spiketrain_id, (int, np.integer)):
        spiketrain_id = [spiketrain_id]

    n_units = len(spiketrain_id)

    if electrode is None:
        electrode = 'max'
    waveforms = []
    for sp in spiketrain_id:
        wf = recgen.spiketrains[sp].waveforms
        if wf is None:
            fs = recgen.info['recordings']['fs'] * pq.Hz
            extract_wf([recgen.spiketrains[sp]], recgen.recordings, fs)
            wf = recgen.spiketrains[sp].waveforms
        waveforms.append(wf)

    if n_units > 1:
        if color is None:
            if cmap is not None:
                cm = plt.get_cmap(cmap)
                colors = [cm(i / n_units) for i in np.arange(n_units)]
            else:
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        else:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    else:
        if color is None:
            colors = 'k'
        else:
            colors = color

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()

    amps = []
    for i, id in enumerate(spiketrain_id):
        st = recgen.spiketrains[id]
        wf = st.waveforms
        mwf = np.mean(wf, axis=0)
        if electrode == 'max':
            max_elec = np.unravel_index(np.argmin(mwf), mwf.shape)[0]
            print('Max electrode', max_elec)
        else:
            assert isinstance(electrode, (int, np.integer)), "'electrode' can be 'max' or type int"
            max_elec = electrode
        amps.append(np.array([np.min(w[max_elec]) for w in wf]))

    if single_axes:
        for i_n, n in enumerate(spiketrain_id):
            amp = amps[i_n]
            st = recgen.spiketrains[n]
            ax.plot(st, amp, marker=marker, ms=ms, color=colors[i_n], ls='')
    else:
        if n_units > ncols:
            nrows = int(np.ceil(len(spiketrain_id) / ncols))
        else:
            nrows = 1
            ncols = n_units

        gs = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=ax)

        for i_n, n in enumerate(spiketrain_id):
            r = i_n // ncols
            c = np.mod(i_n, ncols)
            gs_sel = gs[r, c]
            ax_amp = fig.add_subplot(gs_sel)
            amp = amps[i_n]
            st = recgen.spiketrains[n]
            ax_amp.plot(st, amp, marker=marker, ms=ms, color=colors[i_n], ls='')
        ax.axis('off')

    return ax


def plot_pca_map(recgen, n_pc=2, max_elec=None, cmap='rainbow', cut_out=2, n_units=None, ax=None,
                 whiten=False, pc_comp=None):
    """
    Plots a PCA map of the waveforms.

    Parameters
    ----------
    recgen : RecordingGenerator
        Recording generator object to plot PCA scores of
    ax : axis
        Matplotlib  axis
    n_pc : int
        Number of principal components (default 2)
    max_elec :  int
        Max number of electrodes to plot
    cmap : matplotlib colormap
        Colormap to be used
    cut_out : float or list
        Cut outs in ms for waveforms (if not computed). If float the cut out is symmetrical.    n_units
    whiten :  bool
        If True, PCA scores are whitened
    pc_comp : np.array
        PC component matrix to be used.

    Returns
    -------
    ax : axis
        Matplotlib axis
    pca_scores : list
        List of np.arrays with pca scores for the different units
    pca_component : np.array
        PCA components matrix (n_pc, n_waveform_timepoints)

    """
    try:
        from sklearn.decomposition import PCA
    except:
        raise Exception("'plot_pca_map' requires scikit-learn package")

    import matplotlib.pylab as plt
    import matplotlib.gridspec as gridspec

    waveforms = []
    n_spikes = []

    if n_units is None:
        n_units = len(recgen.spiketrains)

    if recgen.spiketrains[0].waveforms is None:
        print('Computing waveforms')
        recgen.extract_waveforms(cut_out=cut_out)

    for st in recgen.spiketrains:
        wf = st.waveforms
        waveforms.append(wf)
    n_elec = waveforms[0].shape[1]

    if n_pc == 1:
        pc_dims = [0]
    elif n_pc > 1:
        pc_dims = np.arange(n_pc)
    else:
        pc_dims = [0]

    if max_elec is not None and max_elec < n_elec:
        if max_elec == 1:
            elec_dims = [np.random.randint(n_elec)]
        elif max_elec > 1:
            elec_dims = np.random.permutation(np.arange(n_elec))[:max_elec]
        else:
            elec_dims = [np.random.randint(n_elec)]
    else:
        elec_dims = np.arange(n_elec)

    for i_w, wf in enumerate(waveforms):
        # wf_reshaped = wf.reshape((wf.shape[0] * wf.shape[1], wf.shape[2]))
        wf_reshaped = wf.reshape((wf.shape[0] * wf.shape[1], wf.shape[2]))
        n_spikes.append(len(wf) * n_elec)

        if i_w == 0:
            all_waveforms = wf_reshaped
        else:
            all_waveforms = np.vstack((all_waveforms, wf_reshaped))

    if pc_comp is None:
        compute_pca = True
    elif pc_comp.shape == (n_pc, all_waveforms.shape[1]):
        compute_pca = False
    else:
        print("'pc_comp' has wrong dimensions. Recomputing PCA")
        compute_pca = True

    if compute_pca:
        print("Fitting PCA of %d dimensions on %d waveforms" % (n_pc, len(all_waveforms)))

        pca = PCA(n_components=n_pc, whiten=whiten)
        # pca.fit_transform(all_waveforms)
        pca.fit(all_waveforms)
        pc_comp = pca.components_

    pca_scores = []
    for st in recgen.spiketrains:
        pct = np.dot(st.waveforms, pc_comp.T)
        if whiten:
            pct /= np.sqrt(pca.explained_variance_)
        pca_scores.append(pct)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    ax.axis('off')

    if cmap is not None:
        cm = plt.get_cmap(cmap)
        colors = [cm(i / n_units) for i in np.arange(n_units)]
    else:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    nrows = len(pc_dims) * len(elec_dims)
    ncols = nrows
    gs = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=ax)

    for p1 in pc_dims:
        for i1, ch1 in enumerate(elec_dims):
            for p2 in pc_dims:
                for i2, ch2 in enumerate(elec_dims):
                    r = n_pc * i1 + p1
                    c = n_pc * i2 + p2
                    gs_sel = gs[r, c]
                    ax_sel = fig.add_subplot(gs_sel)
                    if c < r:
                        ax_sel.axis('off')
                    else:
                        if r == 0:
                            ax_sel.set_xlabel('Ch.' + str(ch2 + 1) + ':PC' + str(p2 + 1))
                            ax_sel.xaxis.set_label_position('top')

                        ax_sel.set_xticks([])
                        ax_sel.set_yticks([])
                        ax_sel.spines['right'].set_visible(False)
                        ax_sel.spines['top'].set_visible(False)
                        for i, pc in enumerate(pca_scores):
                            if i1 == i2 and p1 == p2:
                                h, b, _ = ax_sel.hist(pc[:, i1, p1], bins=50, alpha=0.6,
                                                      color=colors[i], density=True)
                                ax_sel.set_ylabel('Ch.' + str(ch1 + 1) + ':PC' + str(p1 + 1))
                            else:
                                ax_sel.plot(pc[:, i2, p2], pc[:, i1, p1], marker='o',
                                            ms=1, ls='', alpha=0.5, color=colors[i])

    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    return ax, pca_scores, pc_comp


######### HELPER FUNCTIONS #########

def _find_new_drift_template(drift_mode, st_idx, sp_time, pos, template, template_idxs, loc,
                             chunk_start, t_start_drift, fs, rand_idx, drift_params, drift_state, verbose):
    """Helper function to find drifting template"""
    # extract state
    old_chan = drift_state['old_chan']
    old_amp = drift_state['old_amp']
    old_pos = drift_state['old_pos']
    old_idx = drift_state['old_idx']
    last_drift_position = drift_state['last_drift_position']
    last_fast_drift = drift_state['last_fast_drift']
    direction_sign = drift_state['direction_sign']
    loc_start = drift_state['loc_start']
    t_start_drift = drift_state['t_start_drift']

    # extract params
    fast_drift_period = drift_params['fast_drift_period']
    fast_drift_min_jump = drift_params['fast_drift_min_jump']
    fast_drift_max_jump = drift_params['fast_drift_max_jump']
    v_drift_mag = drift_params['v_drift_mag']
    drift_direction = drift_params['drift_direction']
    max_iter = drift_params['max_iter']

    # compute current position
    if drift_mode == 'slow':
        if pos > 1:
            if template_idxs[pos - 1] == len(loc) - 1 and direction_sign == 1:
                direction_sign = -1
                loc_start = loc[-1]
                t_start_drift = chunk_start + st_idx[pos - 1] / fs
                if verbose:
                    print('Reversing drift direction', direction_sign, sp_time)
            elif template_idxs[pos - 1] == 0 and direction_sign == -1:
                direction_sign = 1
                loc_start = loc[0]
                t_start_drift = chunk_start + st_idx[pos - 1] / fs
                if verbose:
                    print('Reversing drift direction', direction_sign, sp_time)
        else:
            loc_start = loc[0]
        new_pos = np.array(loc_start + drift_direction * direction_sign * v_drift_mag *
                           (sp_time - t_start_drift).rescale('s').magnitude)
    elif drift_mode == 'fast':
        if sp_time - last_fast_drift > fast_drift_period:
            found_position = False
            i = 0
            while not found_position and i < max_iter:
                new_idx = np.random.randint(len(loc))
                new_pos = loc[np.random.randint(len(loc))]
                temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc])
                temp_jitt = template[temp_idx, rand_idx]
                new_chan = np.unravel_index(np.argmin(temp_jitt), temp_jitt.shape)[0]
                # new amp is computed on old chan
                new_amp_old_chan = np.min(temp_jitt[old_chan])
                new_amp_new_chan = np.min(temp_jitt[new_chan])
                if fast_drift_min_jump < np.abs(new_amp_old_chan - old_amp) < fast_drift_max_jump \
                        and new_idx != old_idx:
                    found_position = True
                i += 1
            if i == max_iter:
                print('Max iter reached, couldnt find drifting template')
                new_pos = old_pos
            last_fast_drift = sp_time
            if verbose:
                print('Fast drift time', last_fast_drift, 'Old amplitude', old_amp, 'New amplitude',
                      new_amp_old_chan, 'New amplitude (new)', new_amp_new_chan, 'Old chan', old_chan,
                      'New chan', new_chan)
            old_chan = new_chan
            old_amp = new_amp_new_chan
            old_pos = new_pos
            old_idx = new_idx
        else:
            new_pos = old_pos
    else:  # slow+fast mode
        # slow drift first
        if pos > 1:
            if template_idxs[pos - 1] == len(loc) - 1 and direction_sign == 1:
                direction_sign = -1
                loc_start = loc[-1]
                t_start_drift = chunk_start + st_idx[pos - 1] / fs
                if verbose:
                    print('Reversing drift direction', direction_sign)
            elif template_idxs[pos - 1] == 0 and direction_sign == -1:
                direction_sign = 1
                loc_start = loc[0]
                t_start_drift = chunk_start + st_idx[pos - 1] / fs
                if verbose:
                    print('Reversing drift direction', direction_sign)
            else:
                loc_start = last_drift_position
        else:
            loc_start = old_pos
        new_pos = np.array(loc_start + drift_direction * direction_sign * v_drift_mag *
                           (sp_time - t_start_drift).rescale('s').magnitude)
        # set states for fast drift
        temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc])
        temp_jitt = template[temp_idx, rand_idx]
        old_chan = np.unravel_index(np.argmin(temp_jitt), temp_jitt.shape)[0]
        # new amp is computed on old chan
        old_amp = np.min(temp_jitt[old_chan])
        old_idx = temp_idx
        old_pos = new_pos

        # fast drift now
        if sp_time - last_fast_drift > fast_drift_period:
            # find new drift position
            found_position = False
            i = 0
            while not found_position and i < max_iter:
                new_idx = np.random.randint(len(loc))
                new_pos = loc[np.random.randint(len(loc))]
                temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc])
                temp_jitt = template[temp_idx, rand_idx]
                new_chan = np.unravel_index(np.argmin(temp_jitt), temp_jitt.shape)[0]
                # new amp is computed on old chan
                new_amp_old_chan = np.min(temp_jitt[old_chan])
                new_amp_new_chan = np.min(temp_jitt[new_chan])
                if fast_drift_min_jump < np.abs(new_amp_old_chan - old_amp) < fast_drift_max_jump \
                        and new_idx != old_idx:
                    found_position = True
                i += 1
            if i == max_iter:
                print("Max iter reached, couldn't find drifting template")
                new_pos = old_pos
            last_fast_drift = sp_time
            if verbose:
                print('Fast drift time', last_fast_drift)
            old_pos = new_pos
            old_idx = new_idx
            old_amp = new_amp_new_chan
            old_chan = new_chan
            last_drift_position = loc[temp_idx]
        else:
            new_pos = old_pos

    temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc])
    temp_jitt = template[temp_idx, rand_idx]

    new_drift_state = {'old_pos': old_pos, 'old_chan': old_chan,
                       'old_amp': old_amp, 'old_idx': old_idx, 'last_drift_position': last_drift_position,
                       'direction_sign': direction_sign, 'last_fast_drift': last_fast_drift,
                       'loc_start': loc_start, 't_start_drift': t_start_drift}

    return temp_jitt, temp_idx, new_drift_state

def _resample_parallel(i, template, up, down, drifting):
    """
    Resamples a template to a specified sampling frequency.
    Parameters
    ----------
    template : np.array
        Array with one template (n_channels, n_samples) or (n_drift, n_channels, n_samples) if drifting
    up : float
        The original sampling frequency in Hz
    down : float
        The new sampling frequency in Hz
    drifting : bool
        If True templates are assumed to be drifting
    Returns
    -------
    template_rs : np.array
        Array with resampled template (n_channels, n_resample)
        or (n_drift, n_channels, n_resample)
    """
    if not drifting:
        tem_poly = ss.resample_poly(template, up, down, axis=1)
    else:
        tem_poly = ss.resample_poly(template, up, down, axis=2)
    return tem_poly


def _jitter_parallel(i, template, upsample, fs, n_jitters, jitter, drifting, verbose):
    """
    Adds jittered replicas to one template.
    Parameters
    ----------
    template : np.array
        Array with templates (n_channels, n_samples) or (n_drift, n_channels, n_samples) if drifting
    upsample : int
        Factor for upsampling the templates
    n_jitters : int
        Number of jittered copies for each template
    jitter : quantity
        Jitter in time for shifting the template
    drifting : bool
        If True templates are assumed to be drifting
    verbose : bool
        If True output is verbose
    Returns
    -------
    template_jitt : np.array
        Array with one jittered template (n_jitters, n_channels, n_samples)
        or (n_drift, n_jitters, n_channels, n_samples) if drifting
    """
    if not drifting:
        templates_jitter = np.zeros((n_jitters, template.shape[0], template.shape[1]))
        temp_up = ss.resample_poly(template, upsample, 1, axis=1)
        nsamples_up = temp_up.shape[1]
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
            templates_jitter[n] = temp_down
    else:
        if verbose:
            print('Jittering: neuron ', i)
        templates_jitter = np.zeros((template.shape[0], n_jitters, template.shape[1], template.shape[2]))
        for tp, tem_p in enumerate(template):
            temp_up = ss.resample_poly(tem_p, upsample, 1, axis=1)
            nsamples_up = temp_up.shape[1]
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
                templates_jitter[tp, n] = temp_down
    return templates_jitter


def _pad_parallel(i, template, pad_samples, drifting, verbose):
    """
    Pads one template on both ends.
    Parameters
    ----------
    template : np.array
        Array with templates (n_channels, n_samples) or (n_drift n_channels, n_samples) if drifting
    pad_samples : list
        List of 2 ints with number of samples for padding before and after
    drifting : bool
        If True templates are assumed to be drifting
    verbose : bool
        If True output is verbose
    Returns
    -------
    template_pad : np.array
        Array with padded template (n_channels, n_padded_sample)
        or (n_drift, n_channels, n_padded_sample) if drifting
    """
    if not drifting:
        tem_pad = cubic_padding(template, pad_samples)
    else:
        if verbose:
            print('Padding edges: neuron ', i)
        padded_template_samples = template.shape[-1] + np.sum(pad_samples)
        tem_pad = np.zeros((template.shape[0], template.shape[1], padded_template_samples))
        for tp, tem_p in enumerate(template):
            tem_pad[tp] = cubic_padding(tem_p, pad_samples)
    return tem_pad


def _annotate_parallel(i, st_i, spiketrains, t_jitt, overlapping_pairs, verbose):
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
    over = np.array(['NONE'] * len(st_i))
    for i_sp, t_i in enumerate(st_i):
        for j, st_j in enumerate(spiketrains):
            if i != j:
                # find overlapping
                id_over = np.where((st_j > t_i - t_jitt) & (st_j < t_i + t_jitt))[0]
                if not np.any(overlapping_pairs):
                    if len(id_over) != 0:
                        over[i_sp] = 'TO'
                else:
                    pair = [i, j]
                    pair_i = [j, i]
                    if np.any([np.all(pair == p) for p in overlapping_pairs]) or \
                            np.any([np.all(pair_i == p) for p in overlapping_pairs]):
                        if len(id_over) != 0:
                            over[i_sp] = 'STO'
                    else:
                        if len(id_over) != 0:
                            over[i_sp] = 'TO'
    over[over == 'NONE'] = 'NO'
    st_i.annotate(overlap=over)
    return st_i
