import numpy as np
import quantities as pq
from quantities import Quantity
import yaml
import json
import neo
import elephant
import os
from os.path import join
import MEAutility as MEA
import h5py

### LOAD FUNCTIONS ###

def load_tmp_eap(templates_folder, celltypes=None, samples_per_cat=None, verbose=False):
    '''
    Loads EAP from temporary folder

    Parameters
    ----------
    templates_folder: temporary folder
    celltypes: (optional) list of celltypes to be loaded
    samples_per_cat (optional) number of eap to load per category

    Returns
    -------
    templates, locations, rotations, celltypes

    '''
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


def load_templates(templates, verbose=False):
    '''
    Load generated eap templates (from template_gen.py)

    Parameters
    ----------
    template_folder: templates folder

    Returns
    -------
    templates, locations, rotations, celltypes - np.arrays
    info - dict

    '''
    from MEArec import TemplateGenerator
    if verbose:
        print("Loading templates...")

    temp_dict = {}

    if os.path.isdir(templates):
        templates_folder = templates
        if os.path.isfile(join(templates_folder, 'templates.npy')):
            templates = np.load(join(templates_folder, 'templates.npy'))
            temp_dict.update({'templates': templates})
        if os.path.isfile(join(templates_folder, 'locations.npy')):
            locations = np.load(join(templates_folder, 'locations.npy'))
            temp_dict.update({'locations': locations})
        if os.path.isfile(join(templates_folder, 'rotations.npy')):
            rotations = np.load(join(templates_folder, 'rotations.npy'))
            temp_dict.update({'rotations': rotations})
        if os.path.isfile(join(templates_folder, 'celltypes.npy')):
            celltypes = np.load(join(templates_folder, 'celltypes.npy'))
            temp_dict.update({'celltypes': celltypes})
        with open(join(templates_folder, 'info.yaml'), 'r') as f:
            info = yaml.load(f)
    elif templates.endswith('h5') or templates.endswith('hdf5'):
        with h5py.File(templates, 'r') as F:
            info = json.loads(str(F['info'][()]))
            celltypes = np.array(F.get('celltypes'))
            temp_dict['celltypes'] = np.array([c.decode('utf-8') for c in celltypes])
            temp_dict['locations'] = np.array(F.get('locations'))
            temp_dict['rotations'] = np.array(F.get('rotations'))
            temp_dict['templates'] = np.array(F.get('templates'))

    if verbose:
        print("Done loading templates...")

    tempgen = TemplateGenerator(temp_dict=temp_dict, info=info)

    return tempgen


def load_recordings(recordings, verbose=False):
    '''
    Load generated recordings (from template_gen.py)

    Parameters
    ----------
    recording_folder: recordings folder

    Returns
    -------
    recordings, times, positions, templates, spiketrains, sources, peaks - np.arrays
    info - dict

    '''
    from MEArec import RecordingGenerator
    if verbose:
        print("Loading recordings...")

    rec_dict = {}

    if os.path.isdir(recordings):
        recording_folder = recordings
        if os.path.isfile(join(recording_folder, 'recordings.npy')):
            recordings = np.load(join(recording_folder, 'recordings.npy'))
            rec_dict.update({'recordings': recordings})
        if os.path.isfile(join(recording_folder, 'positions.npy')):
            positions = np.load(join(recording_folder, 'positions.npy'))
            rec_dict.update({'positions': positions})
        if os.path.isfile(join(recording_folder, 'times.npy')):
            times = np.load(join(recording_folder, 'times.npy'))
            rec_dict.update({'times': times})
        if os.path.isfile(join(recording_folder, 'templates.npy')):
            templates = np.load(join(recording_folder, 'templates.npy'))
            rec_dict.update({'templates': templates})
        if os.path.isfile(join(recording_folder, 'spiketrains.npy')):
            spiketrains = np.load(join(recording_folder, 'spiketrains.npy'))
            rec_dict.update({'spiketrains': spiketrains})
        if os.path.isfile(join(recording_folder, 'sources.npy')):
            sources = np.load(join(recording_folder, 'sources.npy'))
            rec_dict.update({'sources': sources})
        if os.path.isfile(join(recording_folder, 'peaks.npy')):
            peaks = np.load(join(recording_folder, 'peaks.npy'))
            rec_dict.update({'peaks': peaks})
        with open(join(recording_folder, 'info.yaml'), 'r') as f:
            info = yaml.load(f)
    elif recordings.endswith('h5') or recordings.endswith('hdf5'):
        with h5py.File(recordings, 'r') as F:
            info = json.loads(str(F['info'][()]))
            rec_dict['peaks'] = np.array(F.get('peaks'))
            rec_dict['positions'] = np.array(F.get('positions'))
            rec_dict['recordings'] = np.array(F.get('recordings'))
            rec_dict['sources'] = np.array(F.get('sources'))
            rec_dict['templates'] = np.array(F.get('templates'))
            rec_dict['times'] = np.array(F.get('times'))
            spiketrains = []
            for ii in range(info['recordings']['n_neurons']):
                times = np.array(F.get('spiketrains/{}/times'.format(ii)))
                t_stop = np.array(F.get('spiketrains/{}/t_stop'.format(ii)))
                annotations_str = str(F.get('spiketrains/{}/annotations'.format(ii))[()])
                annotations = json.loads(annotations_str)
                st = neo.core.SpikeTrain(
                    times,
                    t_stop=t_stop,
                    units=pq.s
                )
                st.annotations = annotations
                spiketrains.append(st)
            rec_dict['spiketrains'] = spiketrains

    if verbose:
        print("Done loading recordings...")

    recgen = RecordingGenerator(rec_dict=rec_dict, info=info)

    return recgen


def save_template_generator(tempgen, filename=None):
    if filename.endswith('h5') or filename.endswith('hdf5'):
        F = h5py.File(filename, 'w')
        F.create_dataset('info', data=json.dumps(tempgen.info))
        celltypes = [str(x).encode('utf-8') for x in tempgen.celltypes]
        F.create_dataset('celltypes', data=celltypes)
        F.create_dataset('locations', data=tempgen.locations)
        F.create_dataset('rotations', data=tempgen.rotations)
        F.create_dataset('templates', data=tempgen.templates)
        F.close()
        print('\nSaved template generator templates in', filename, '\n')
    elif filename is not None:
        save_folder = filename
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        np.save(join(save_folder, 'templates'), tempgen.templates)
        np.save(join(save_folder, 'locations'), tempgen.locations)
        np.save(join(save_folder, 'rotations'), tempgen.rotations)
        np.save(join(save_folder, 'celltypes'), tempgen.celltypes)
        info = tempgen.info
        yaml.dump(info, open(join(save_folder, 'info.yaml'), 'w'), default_flow_style=False)
        print('\nSaved template generator templates in', save_folder, ' folder\n')
    else:
        rot = tempgen.info['params']['rot']
        n = tempgen.info['params']['n']
        probe = tempgen.info['params']['probe']
        fname = 'templates_%d_%s_%s' % (n, probe, time.strftime("%d-%m-%Y"))
        save_folder = join(os.getcwd(), fname)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        np.save(join(save_folder, 'templates'), tempgen.templates)
        np.save(join(save_folder, 'locations'), tempgen.locations)
        np.save(join(save_folder, 'rotations'), tempgen.rotations)
        np.save(join(save_folder, 'celltypes'), tempgen.celltypes)
        info = tempgen.info
        yaml.dump(info, open(join(save_folder, 'info.yaml'), 'w'), default_flow_style=False)
        print('\nSaved template generator templates in', save_folder, '\n')


def save_recording_generator(recgen, filename=None):
    if filename.endswith('h5') or filename.endswith('hdf5'):
        F = h5py.File(filename, 'w')
        F.create_dataset('info', data=json.dumps(recgen.info))
        F.create_dataset('peaks', data=recgen.peaks)
        F.create_dataset('positions', data=recgen.positions)
        F.create_dataset('recordings', data=recgen.recordings)
        F.create_dataset('sources', data=recgen.sources)
        for ii in range(len(recgen.spiketrains)):
            st = recgen.spiketrains[ii]
            F.create_dataset('spiketrains/{}/times'.format(ii), data=st.times.rescale('s').magnitude)
            F.create_dataset('spiketrains/{}/t_stop'.format(ii), data=st.t_stop)
            annotations_no_pq = {}
            for k, v in st.annotations.items():
                if isinstance(v, pq.Quantity):
                    annotations_no_pq[k] = float(v.magnitude)
                elif isinstance(v, np.ndarray):
                    annotations_no_pq[k] = list(v)
                else:
                    annotations_no_pq[k] = str(v)
            annotations_str = json.dumps(annotations_no_pq)
            F.create_dataset('spiketrains/{}/annotations'.format(ii), data=annotations_str)
        F.create_dataset('templates', data=recgen.templates)
        F.create_dataset('times', data=recgen.templates)
        F.close()
        print('\nSaved recordings in', filename, '\n')
    elif filename is not None:
        save_folder = filename
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        np.save(join(save_folder, 'recordings'), recgen.recordings)
        np.save(join(save_folder, 'times'), recgen.times)
        np.save(join(save_folder, 'positions'), recgen.positions)
        np.save(join(save_folder, 'templates'), recgen.templates)
        np.save(join(save_folder, 'spiketrains'), recgen.spiketrains)
        np.save(join(save_folder, 'sources'), recgen.sources)
        np.save(join(save_folder, 'peaks'), recgen.peaks)
        with open(join(save_folder, 'info.yaml'), 'w') as f:
            yaml.dump(recgen.info, f, default_flow_style=False)
        print('\nSaved recordings in', save_folder, ' folder\n')
    else:
        info = recgen.info
        n_neurons = info['recordings']['n_neurons']
        electrode_name = info['recordings']['electrode_name']
        duration = info['recordings']['duration']
        noise_level = info['recordings']['noise_level']
        fname = 'recordings_%dcells_%s_%s_%.1fuV_%s' % (n_neurons, electrode_name, duration,
                                                        noise_level, time.strftime("%d-%m-%Y:%H:%M"))
        save_folder = fname
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        np.save(join(save_folder, 'recordings'), recgen.recordings)
        np.save(join(save_folder, 'times'), recgen.times)
        np.save(join(save_folder, 'positions'), recgen.positions)
        np.save(join(save_folder, 'templates'), recgen.templates)
        np.save(join(save_folder, 'spiketrains'), recgen.spiketrains)
        np.save(join(save_folder, 'sources'), recgen.sources)
        np.save(join(save_folder, 'peaks'), recgen.peaks)
        with open(join(save_folder, 'info.yaml'), 'w') as f:
            yaml.dump(info, f, default_flow_style=False)
        print('\nSaved recordings in', save_folder, ' folder\n')

### H5 TOOLS ###

def hdf5_to_recordings(input_file, output_folder):
  if os.path.exists(output_folder):
    raise Exception('Output folder already exists: ' + output_folder)

  os.mkdir(output_folder)

  with h5py.File(input_file,'r') as F:
    info=json.loads(str(F['info'][()]))
    with open(output_folder+'/info.yaml','w') as f:
      yaml.dump(info,f,default_flow_style=False)

    peaks=np.array(F.get('peaks'))
    np.save(output_folder+'/peaks.npy',peaks)
    positions=np.array(F.get('positions'))
    np.save(output_folder+'/positions.npy',positions)
    recordings=np.array(F.get('recordings'))
    np.save(output_folder+'/recordings.npy',recordings)
    sources=np.array(F.get('sources'))
    np.save(output_folder+'/sources.npy',sources)
    templates=np.array(F.get('templates'))
    np.save(output_folder+'/templates.npy',templates)
    times=np.array(F.get('times'))
    np.save(output_folder+'/times.npy',times)
    spiketrains=[]
    for ii in range(info['recordings']['n_neurons']):
      times=np.array(F.get('spiketrains/{}/times'.format(ii)))
      t_stop=np.array(F.get('spiketrains/{}/t_stop'.format(ii)))
      annotations_str=str(F.get('spiketrains/{}/annotations'.format(ii))[()])
      annotations=json.loads(annotations_str)
      st=neo.core.SpikeTrain(
        times,
        t_stop=t_stop,
        units=pq.s
      )
      st.annotations=annotations
      spiketrains.append(st)
    np.save(output_folder+'/spiketrains.npy',spiketrains)


def hdf5_to_templates(input_file, output_folder):
  if os.path.exists(output_folder):
    raise Exception('Output folder already exists: '+output_folder)

  os.mkdir(output_folder)

  with h5py.File(input_file,'r') as F:
    info=json.loads(str(F['info'][()]))
    with open(output_folder+'/info.yaml','w') as f:
      yaml.dump(info,f,default_flow_style=False)

    celltypes=np.array(F.get('celltypes'))
    np.save(output_folder+'/celltypes.npy',celltypes)
    locations=np.array(F.get('locations'))
    np.save(output_folder+'/locations.npy',locations)
    rotations=np.array(F.get('rotations'))
    np.save(output_folder+'/rotations.npy',rotations)
    templates=np.array(F.get('templates'))
    np.save(output_folder+'/templates.npy',templates)


def recordings_to_hdf5(recording_folder, output_fname):
    F = h5py.File(output_fname, 'w')

    with open(recording_folder + '/info.yaml', 'r') as f:
        info = yaml.load(f)

    F.create_dataset('info', data=json.dumps(info))

    peaks = np.load(recording_folder + '/peaks.npy')
    F.create_dataset('peaks', data=peaks)
    positions = np.load(recording_folder + '/positions.npy')
    F.create_dataset('positions', data=positions)
    recordings = np.load(recording_folder + '/recordings.npy')
    F.create_dataset('recordings', data=recordings)
    sources = np.load(recording_folder + '/sources.npy')
    F.create_dataset('sources', data=sources)
    spiketrains = np.load(recording_folder + '/spiketrains.npy')
    for ii in range(len(spiketrains)):
        st = spiketrains[ii]
        F.create_dataset('spiketrains/{}/times'.format(ii), data=st.times.rescale('s').magnitude)
        F.create_dataset('spiketrains/{}/t_stop'.format(ii), data=st.t_stop)
        annotations_no_pq = {}
        for k, v in st.annotations.items():
            if isinstance(v, pq.Quantity):
                annotations_no_pq[k] = float(v.magnitude)
            elif isinstance(v, np.ndarray):
                annotations_no_pq[k] = list(v)
            else:
                annotations_no_pq[k] = v
        annotations_str = json.dumps(annotations_no_pq)
        F.create_dataset('spiketrains/{}/annotations'.format(ii), data=annotations_str)
    templates = np.load(recording_folder + '/templates.npy')
    F.create_dataset('templates', data=templates)
    templates = np.load(recording_folder + '/times.npy')
    F.create_dataset('times', data=templates)
    F.close()


def templates_to_hdf5(templates_folder, output_fname):
    F = h5py.File(output_fname, 'w')

    with open(templates_folder + '/info.yaml', 'r') as f:
        info = yaml.load(f)

    F.create_dataset('info', data=json.dumps(info))

    celltypes = np.load(templates_folder + '/celltypes.npy')
    celltypes = [str(x).encode('utf-8') for x in celltypes]
    F.create_dataset('celltypes', data=celltypes)
    locations = np.load(templates_folder + '/locations.npy')
    F.create_dataset('locations', data=locations)
    rotations = np.load(templates_folder + '/rotations.npy')
    F.create_dataset('rotations', data=rotations)
    templates = np.load(templates_folder + '/templates.npy')
    F.create_dataset('templates', data=templates)

    F.close()


### TEMPLATES INFO ###

def get_binary_cat(celltypes, excit, inhib):
    '''

    Parameters
    ----------
    celltypes: np.array with celltypes
    excit: list of excitatory celltypes
    inhib: list of inhibitory celltypes

    Returns
    -------
    bin_cat: binary celltype (E-I) - np.array

    '''
    binary_cat = []
    for i, cat in enumerate(celltypes):
        if np.any([ex in str(cat) for ex in excit]):
            binary_cat.append('E')
        elif np.any([inh in str(cat) for inh in inhib]):
            binary_cat.append('I')
    return np.array(binary_cat, dtype=str)


def get_EAP_features(EAP, feat_list, dt=None, EAP_times=None, threshold_detect=0, normalize=False):
    '''

    Parameters
    ----------
    EAP
    feat_list
    dt
    EAP_times
    threshold_detect
    normalize

    Returns
    -------

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


### TEMPLATES OPERATIONS ###

def is_position_within_boundaries(position, x_lim, y_lim, z_lim):
    '''

    Parameters
    ----------
    position
    x_lim
    y_lim
    z_lim

    Returns
    -------

    '''
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


def select_templates(loc, spikes, bin_cat, n_exc, n_inh, min_dist=25, x_lim=None, y_lim=None, z_lim=None,
                     min_amp=None, drift=False, drift_dir_ang=[], preferred_dir=None, ang_tol=30, verbose=False):
    '''

    Parameters
    ----------
    loc
    spikes
    bin_cat
    n_exc
    n_inh
    min_dist
    x_lim
    y_lim
    z_lim
    min_amp
    drift
    drift_dir_ang
    preferred_dir
    ang_tol
    verbose

    Returns
    -------

    '''
    pos_sel = []
    idxs_sel = []
    categories = np.unique(bin_cat)
    if 'E' in categories and 'I' in categories:
        if verbose:
            print('Selecting Excitatory and Inhibitory cells')
        excinh = True
    else:
        if verbose:
            print('Selecting random templates (cell types not specified)')
        excinh = False
    permuted_idxs = np.random.permutation(len(bin_cat))
    permuted_bin_cats = bin_cat[permuted_idxs]

    if not min_amp:
        min_amp = 0

    if drift:
        if len(drift_dir_ang) == 0 or preferred_dir == None:
            raise Exception('For drift selection provide drifting angles and preferred drift direction')

    placed_exc = 0
    placed_inh = 0
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
                        amp = np.max(np.abs(spikes[id_cell]))
                        if not drift:
                            if is_position_within_boundaries(loc[id_cell], x_lim, y_lim, z_lim) and amp > min_amp:
                                    # save cell
                                    pos_sel.append(loc[id_cell])
                                    idxs_sel.append(id_cell)
                                    n_sel += 1
                                    placed = True
                            else:
                                if verbose:
                                    print('Amplitude or boundary violation', amp, loc[id_cell], iter)
                        else:
                            # drift
                            if is_position_within_boundaries(loc[id_cell], x_lim, y_lim, z_lim) and amp > min_amp:
                                    # save cell
                                if np.abs(drift_dir_ang[id_cell] - preferred_dir) < ang_tol:
                                    pos_sel.append(loc[id_cell])
                                    idxs_sel.append(id_cell)
                                    n_sel += 1
                                    placed = True
                                else:
                                    if verbose:
                                        print('Drift violation', loc[id_cell], iter)
                            else:
                                if verbose:
                                    print('Amplitude or boundary violation', amp, loc[id_cell], iter)
                    if placed:
                        n_sel_exc += 1
            elif bcat == 'I':
                if n_sel_inh < n_inh:
                    dist = np.array([np.linalg.norm(loc[id_cell] - p) for p in pos_sel])
                    if np.any(dist < min_dist):
                        if verbose:
                            print('Distance violation', dist, iter)
                        pass
                    else:
                        amp = np.max(np.abs(spikes[id_cell]))
                        if not drift:
                            if is_position_within_boundaries(loc[id_cell], x_lim, y_lim, z_lim) and amp > min_amp:
                                    # save cell
                                    pos_sel.append(loc[id_cell])
                                    idxs_sel.append(id_cell)
                                    n_sel += 1
                                    placed = True
                            else:
                                if verbose:
                                    print('Amplitude or boundary violation', amp, loc[id_cell], iter)
                        else:
                            # drift
                            if is_position_within_boundaries(loc[id_cell], x_lim, y_lim, z_lim) and amp > min_amp:
                                    # save cell
                                if np.abs(drift_dir_ang[id_cell] - preferred_dir) < ang_tol:
                                    pos_sel.append(loc[id_cell])
                                    idxs_sel.append(id_cell)
                                    n_sel += 1
                                    placed = True
                                else:
                                    if verbose:
                                        print('Drift violation', loc[id_cell], iter)
                            else:
                                if verbose:
                                    print('Amplitude or boundary violation', amp, loc[id_cell], iter)
                    if placed:
                        n_sel_inh += 1
        else:
            dist = np.array([np.linalg.norm(loc[id_cell] - p) for p in pos_sel])
            if np.any(dist < min_dist):
                if verbose:
                    print('Distance violation', dist, iter)
                pass
            else:
                amp = np.max(np.abs(spikes[id_cell]))
                if not drift:
                    if is_position_within_boundaries(loc[id_cell], x_lim, y_lim, z_lim) and amp > min_amp:
                        # save cell
                        pos_sel.append(loc[id_cell])
                        idxs_sel.append(id_cell)
                        placed = True
                    else:
                        if verbose:
                            print('Amplitude or boundary violation', amp, loc[id_cell], iter)
                else:
                    # drift
                    if is_position_within_boundaries(loc[id_cell], x_lim, y_lim, z_lim) and amp > min_amp:
                        # save cell
                        if np.abs(drift_dir_ang[id_cell] - preferred_dir) < ang_tol:
                            pos_sel.append(loc[id_cell])
                            idxs_sel.append(id_cell)
                            placed = True
                        else:
                            if verbose:
                                print('Drift violation', loc[id_cell], iter)
                    else:
                        if verbose:
                            print('Amplitude or boundary violation', amp, loc[id_cell], iter)
            if placed:
                n_sel += 1
                print(i, len(permuted_idxs), n_sel)
    if i == len(permuted_idxs)-1 and n_sel < n_exc + n_inh:
        raise RuntimeError("Templates could not be selected. \n"
                           "Decrease number of spiketrains, decrease 'min-dist', or use more templates.")

    return idxs_sel


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


def find_overlapping_templates(templates, thresh=0.7):
    '''

    Parameters
    ----------
    templates
    thresh

    Returns
    -------

    '''
    overlapping_pairs = []

    for i in range(templates.shape[0] - 1):
        temp_1 = templates[i]
        max_ptp = (np.array([np.ptp(t) for t in temp_1]).max())
        max_ptp_idx = (np.array([np.ptp(t) for t in temp_1]).argmax())

        for j in range(i + 1, templates.shape[0]):
            temp_2 = templates[j]
            ptp_on_max = np.ptp(temp_2[max_ptp_idx])

            max_ptp_2 = (np.array([np.ptp(t) for t in temp_2]).max())

            max_peak = np.max([ptp_on_max, max_ptp])
            min_peak = np.min([ptp_on_max, max_ptp])

            if min_peak > thresh * max_peak and ptp_on_max > thresh * max_ptp_2:
                overlapping_pairs.append([i, j])  # , max_ptp_idx, max_ptp, ptp_on_max

    return np.array(overlapping_pairs)


### SPIKETRAIN OPERATIONS ###

def annotate_overlapping_spikes(gtst, t_jitt = 1*pq.ms, overlapping_pairs=None, verbose=False, parallel=True):
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
        results = [pool.apply_async(annotate_parallel(i, st_i, gtst, overlapping_pairs, t_jitt, ))
                   for i, st_i in enumerate(gtst)]
    else:
        # find overlapping spikes
        for i, st_i in enumerate(gtst):
            if verbose:
                print('SPIKETRAIN ', i)
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
                                #     print('found overlap! spike 1: ', i, t_i, ' spike 2: ', j, st_j[id_over])
                        else:
                            pair = [i, j]
                            pair_i = [j, i]
                            if np.any([np.all(pair == p) for p in overlapping_pairs]) or \
                                    np.any([np.all(pair_i == p) for p in overlapping_pairs]):
                                if len(id_over) != 0:
                                    over[i_sp] = 'SO'
                                    # if verbose:
                                    #     print('found spatial overlap! spike 1: ', i, t_i, ' spike 2: ', j, st_j[id_over])
                            else:
                                if len(id_over) != 0:
                                    over[i_sp] = 'O'
                                    # if verbose:
                                    #     print('found overlap! spike 1: ', i, t_i, ' spike 2: ', j, st_j[id_over])
            st_i.annotate(overlap=over)


def annotate_parallel(i, st_i, gtst, overlapping_pairs, t_jitt):
    '''

    Parameters
    ----------
    i
    st_i
    gtst
    overlapping_pairs
    t_jitt

    Returns
    -------

    '''
    print('SPIKETRAIN ', i)
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
                        #     print('found overlap! spike 1: ', i, t_i, ' spike 2: ', j, st_j[id_over])
                else:
                    pair = [i, j]
                    pair_i = [j, i]
                    if np.any([np.all(pair == p) for p in overlapping_pairs]) or \
                            np.any([np.all(pair_i == p) for p in overlapping_pairs]):
                        if len(id_over) != 0:
                            over[i_sp] = 'SO'
                            # if verbose:
                            #     print('found spatial overlap! spike 1: ', i, t_i, ' spike 2: ', j, st_j[id_over])
                    else:
                        if len(id_over) != 0:
                            over[i_sp] = 'O'
                            # if verbose:
                            #     print('found overlap! spike 1: ', i, t_i, ' spike 2: ', j, st_j[id_over])
    st_i.annotate(overlap=over)


def resample_spiketrains(spiketrains, fs=None, T=None):
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
    import elephant.conversion as conv

    resampled_mat = []
    if not fs and not T:
        print('Provide either sampling frequency fs or time period T')
        return
    elif fs:
        if not isinstance(fs, Quantity):
            raise ValueError("fs must be of type pq.Quantity")
        binsize = 1./fs
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


### CONVOLUTION OPERATIONS ###

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

# TODO use cut outs to align spikes!
def convolve_single_template(spike_id, spike_bin, template, cut_out=None, modulation=False, amp_mod=None):
    '''

    Parameters
    ----------
    spike_id
    spike_bin
    template
    modulation
    amp_mod

    Returns
    -------

    '''
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
        # print('rand_idx: ', rand_idx)
        temp_jitt = template[rand_idx]
        if not modulation:
            for pos, spos in enumerate(spike_pos):
                if spos - cut_out[0] >= 0 and spos - cut_out[0] + len_spike <= n_samples:
                    gt_source[spos - cut_out[0]:spos - cut_out[0] + len_spike] +=  temp_jitt
                elif spos - cut_out[0] < 0:
                    diff = -(spos - cut_out[0])
                    gt_source[:spos - cut_out[0] + len_spike] += temp_jitt[diff:]
                else:
                    diff = n_samples - (spos - cut_out[0])
                    gt_source[spos - cut_out[0]:] += temp_jitt[:diff]
        else:
            # print('Template-Electrode modulation')
            for pos, spos in enumerate(spike_pos):
                if spos - cut_out[0] >= 0 and spos - cut_out[0] + len_spike <= n_samples:
                    gt_source[spos - cut_out[0]:spos - cut_out[0] + len_spike] += amp_mod[pos]*temp_jitt
                elif spos - cut_out[0] < 0:
                    diff = -(spos - cut_out[0])
                    gt_source[:spos - cut_out[0] + len_spike] += amp_mod[pos]*temp_jitt[diff:]
                else:
                    diff = n_samples - (spos - cut_out[0])
                    gt_source[spos - cut_out[0]:] += amp_mod[pos]*temp_jitt[:diff]
    else:
        # print('No modulation')
        for pos, spos in enumerate(spike_pos):
            if spos - cut_out[0] >= 0 and spos - cut_out[0] + len_spike <= n_samples:
                gt_source[spos - cut_out[0]:spos - cut_out[0] + len_spike] += template
            elif spos - cut_out[0] < 0:
                diff = -(spos - cut_out[0])
                gt_source[:spos - cut_out[0] + len_spike] += template[diff:]
            else:
                diff = n_samples - (spos - cut_out[0])
                gt_source[spos - cut_out[0]:] += template[:diff]

    return gt_source


def convolve_templates_spiketrains(spike_id, spike_bin, template,  cut_out=None, modulation=False, amp_mod=None, recordings=[]):
    '''

    Parameters
    ----------
    spike_id
    spike_bin
    template
    modulation
    amp_mod
    recordings

    Returns
    -------

    '''
    print('START: convolution with spike ', spike_id)
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
    if cut_out is None:
        cut_out = [len_spike//2, len_spike//2]        

    # recordings_test = np.zeros((n_elec, n_samples))
    if not modulation:
        spike_pos = np.where(spike_bin == 1)[0]
        amp_mod = np.ones_like(spike_pos)
        if len(template.shape) == 3:
            rand_idx = np.random.randint(njitt)
            # print('rand_idx: ', rand_idx)
            temp_jitt = template[rand_idx]
            # print('No modulation')
            for pos, spos in enumerate(spike_pos):
                if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                    recordings[:, spos - cut_out[0]:spos + cut_out[1]] += amp_mod[pos] * temp_jitt
                elif spos - cut_out[0] < 0:
                    diff = -(spos - cut_out[0])
                    recordings[:, :spos + cut_out[1]] += amp_mod[pos] * temp_jitt[:, diff:]
                else:
                    diff = n_samples - (spos - cut_out[0])
                    recordings[:, spos - cut_out[0]:] += amp_mod[pos] * temp_jitt[:, :diff]

            raise Exception()
        else:
            # print('No jitter')
            for pos, spos in enumerate(spike_pos):
                if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                    recordings[:, spos - cut_out[0]:spos + cut_out[1]] += amp_mod[
                                                                                                  pos] * template
                elif spos - cut_out[0] < 0:
                    diff = -(spos - cut_out[0])
                    recordings[:, :spos + cut_out[1]] += amp_mod[pos] * template[:, diff:]
                else:
                    diff = n_samples - (spos - cut_out[0])
                    recordings[:, spos - cut_out[0]:] += amp_mod[pos] * template[:, :diff]
    else:
        assert amp_mod is not None
        spike_pos = np.where(spike_bin == 1)[0]
        if len(template.shape) == 3:
            rand_idx = np.random.randint(njitt)
            # print('rand_idx: ', rand_idx)
            temp_jitt = template[rand_idx]
            if not isinstance(amp_mod[0], (list, tuple, np.ndarray)):
                #print('Template modulation')
                for pos, spos in enumerate(spike_pos):
                    if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                        recordings[:, spos - cut_out[0]:spos + cut_out[1]] +=  amp_mod[pos] * temp_jitt
                    elif spos - cut_out[0] < 0:
                        diff = -(spos - cut_out[0])
                        recordings[:, :spos + cut_out[1]] += amp_mod[pos] * temp_jitt[:, diff:]
                    else:
                        diff = n_samples-(spos - cut_out[0])
                        recordings[:, spos - cut_out[0]:] += amp_mod[pos] * temp_jitt[:, :diff]
            else:
                #print('Electrode modulation')
                for pos, spos in enumerate(spike_pos):
                    if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                        recordings[:, spos - cut_out[0]:spos + cut_out[1]] += \
                            [a * t for (a, t) in zip(amp_mod[pos], temp_jitt)]
                    elif spos - cut_out[0] < 0:
                        diff = -(spos - cut_out[0])
                        recordings[:, :spos + cut_out[1]] += \
                            [a * t for (a, t) in zip(amp_mod[pos], temp_jitt[:, diff:])]
                        # recordings[:, :spos + cut_out[1]] += amp_mod[pos] * template[:, diff:]
                    else:
                        diff = n_samples-(spos - cut_out[0])
                        recordings[:, spos - cut_out[0]:] += \
                            [a * t for (a, t) in zip(amp_mod[pos], temp_jitt[:, :diff])]
        else:
            if not isinstance(amp_mod[0], (list, tuple, np.ndarray)):
                #print('Template modulation')
                for pos, spos in enumerate(spike_pos):
                    if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                        recordings[:, spos - cut_out[0]:spos + cut_out[1]] += amp_mod[
                                                                                                      pos] * template
                    elif spos - cut_out[0] < 0:
                        diff = -(spos - cut_out[0])
                        recordings[:, :spos + cut_out[1]] += amp_mod[pos] * template[:, diff:]
                    else:
                        diff = n_samples - (spos - cut_out[0])
                        recordings[:, spos - cut_out[0]:] += amp_mod[pos] * template[:, :diff]

            else:
                #print('Electrode modulation')
                for pos, spos in enumerate(spike_pos):
                    if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                        recordings[:, spos - cut_out[0]:spos + cut_out[1]] += \
                            [a * t for (a, t) in zip(amp_mod[pos], template)]
                    elif spos - cut_out[0] < 0:
                        diff = -(spos - cut_out[0])
                        recordings[:, : spos + cut_out[1]] += \
                            [a * t for (a, t) in zip(amp_mod[pos], template[:, diff:])]
                        # recordings[:, :spos + cut_out[1]] += amp_mod[pos] * template[:, diff:]
                    else:
                        diff = n_samples-(spos - cut_out[0])
                        recordings[:, spos - cut_out[0]:] += \
                            [a * t for (a, t) in zip(amp_mod[pos], template[:, :diff])]
                        # recordings[:, spos - cut_out[0]:] += amp_mod[pos] * template[:, :diff]


    #print('DONE: convolution with spike ', spike_id)

    return recordings


def convolve_drifting_templates_spiketrains(spike_id, spike_bin, template, fs, loc, v_drift, t_start_drift,
                                            modulation=False, amp_mod=None, recordings=[], n_step_sec=1):
    '''

    Parameters
    ----------
    spike_id
    spike_bin
    template
    fs
    loc
    v_drift
    t_start_drift
    modulation
    amp_mod
    recordings
    n_step_sec

    Returns
    -------

    '''
    print('START: convolution with spike ', spike_id)
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
            print('rand_idx: ', rand_idx)
            print('No modulation')
            for pos, spos in enumerate(spike_pos):
                sp_time = spos / fs
                if sp_time < t_start_drift:
                    print(sp_time, 'No drift', loc[0])
                    temp_idx = 0
                    temp_jitt = template[temp_idx, rand_idx]
                else:
                    # compute current position
                    new_pos = np.array(loc[0, 1:] + v_drift * (sp_time - t_start_drift).rescale('s').magnitude)
                    temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc[:, 1:]])
                    print(sp_time, temp_idx, 'Drifting', new_pos, loc[temp_idx, 1:])
                    temp_jitt = template[temp_idx, rand_idx]

                if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                    recordings[:, spos - cut_out[0]:spos + cut_out[1]] += amp_mod[pos] * temp_jitt
                elif spos - cut_out[0] < 0:
                    diff = -(spos - cut_out[0])
                    recordings[:, :spos + cut_out[1]] += amp_mod[pos] * temp_jitt[:, diff:]
                else:
                    diff = n_samples - (spos - cut_out[0])
                    recordings[:, spos - cut_out[0]:] += amp_mod[pos] * temp_jitt[:, :diff]

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
            print('No jitter')
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
                if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                    recordings[:, spos - cut_out[0]:spos + cut_out[1]] += amp_mod[pos] * temp
                elif spos - cut_out[0] < 0:
                    diff = -(spos - cut_out[0])
                    recordings[:, :spos + cut_out[1]] += amp_mod[pos] * temp[:, diff:]
                else:
                    diff = n_samples - (spos - cut_out[0])
                    recordings[:, spos - cut_out[0]:] += amp_mod[pos] * temp[:, :diff]
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
            print('rand_idx: ', rand_idx)
            if not isinstance(amp_mod[0], (list, tuple, np.ndarray)):
                print('Template modulation')
                for pos, spos in enumerate(spike_pos):
                    sp_time = spos / fs
                    if sp_time < t_start_drift:
                        print(sp_time, 'No drift', loc[0])
                        temp_idx = 0
                        temp_jitt = template[temp_idx, rand_idx]
                    else:
                        # compute current position
                        new_pos = np.array(loc[0, 1:] + v_drift * (sp_time - t_start_drift).rescale('s').magnitude)
                        temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc[:, 1:]])
                        temp_jitt = template[temp_idx, rand_idx]
                        print(sp_time, temp_idx, 'Drifting', new_pos, loc[temp_idx, 1:])
                    if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                        recordings[:, spos - cut_out[0]:spos + cut_out[1]] += amp_mod[pos] \
                                                                                                  * temp_jitt
                    elif spos - cut_out[0] < 0:
                        diff = -(spos - cut_out[0])
                        recordings[:, :spos + cut_out[1]] += amp_mod[pos] * temp_jitt[:, diff:]
                    else:
                        diff = n_samples - (spos - cut_out[0])
                        recordings[:, spos - cut_out[0]:] += amp_mod[pos] * temp_jitt[:, :diff]
            else:
                print('Electrode modulation')
                for pos, spos in enumerate(spike_pos):
                    sp_time = spos / fs
                    if sp_time < t_start_drift:
                        temp_idx = 0
                        temp_jitt = template[temp_idx, rand_idx]
                        print(sp_time, 'No drift', loc[0])
                        if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                            recordings[:, spos - cut_out[0]:spos + cut_out[1]] += \
                                [a * t for (a, t) in zip(amp_mod[pos], temp_jitt)]
                        elif spos - cut_out[0] < 0:
                            diff = -(spos - cut_out[0])
                            recordings[:, :spos + cut_out[1]] += \
                                [a * t for (a, t) in zip(amp_mod[pos], temp_jitt[:, diff:])]
                            # recordings[:, :spos + cut_out[1]] += amp_mod[pos] * template[:, diff:]
                        else:
                            diff = n_samples - (spos - cut_out[0])
                            recordings[:, spos - cut_out[0]:] += \
                                [a * t for (a, t) in zip(amp_mod[pos], temp_jitt[:, :diff])]
                    else:
                        # compute current position
                        new_pos = np.array(loc[0, 1:] + v_drift * (sp_time - t_start_drift).rescale('s').magnitude)
                        temp_idx = np.argmin([np.linalg.norm(p - new_pos) for p in loc[:, 1:]])
                        new_temp_jitt = template[temp_idx, rand_idx]
                        print(sp_time, temp_idx, 'Drifting', new_pos, loc[temp_idx, 1:])
                        if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                            recordings[:, spos - cut_out[0]:spos + cut_out[1]] += \
                                [a * t for (a, t) in zip(amp_mod[pos], new_temp_jitt)]
                        elif spos - cut_out[0] < 0:
                            diff = -(spos - cut_out[0])
                            recordings[:, :spos + cut_out[1]] += \
                                [a * t for (a, t) in zip(amp_mod[pos], new_temp_jitt[:, diff:])]
                            # recordings[:, :spos + cut_out[1]] += amp_mod[pos] * template[:, diff:]
                        else:
                            diff = n_samples - (spos - cut_out[0])
                            recordings[:, spos - cut_out[0]:] += \
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
            print('No jitter')
            if not isinstance(amp_mod[0], (list, tuple, np.ndarray)):
                print('Template modulation')
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
                    if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                        recordings[:, spos - cut_out[0]:spos + cut_out[1]] += amp_mod[pos] * temp
                    elif spos - cut_out[0] < 0:
                        diff = -(spos - cut_out[0])
                        recordings[:, :spos + cut_out[1]] += amp_mod[pos] * temp[:, diff:]
                    else:
                        diff = n_samples - (spos - cut_out[0])
                        recordings[:, spos - cut_out[0]:] += amp_mod[pos] * temp[:, :diff]

            else:
                print('Electrode modulation')
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
                    if spos - cut_out[0] >= 0 and spos + cut_out[1] <= n_samples:
                        recordings[:, spos - cut_out[0]:spos + cut_out[1]] += \
                            [a * t for (a, t) in zip(amp_mod[pos], temp)]
                    elif spos - cut_out[0] < 0:
                        diff = -(spos - cut_out[0])
                        recordings[:, :spos + cut_out[1]] += \
                            [a * t for (a, t) in zip(amp_mod[pos], temp[:, diff:])]
                        # recordings[:, :spos + cut_out[1]] += amp_mod[pos] * template[:, diff:]
                    else:
                        diff = n_samples - (spos - cut_out[0])
                        recordings[:, spos - cut_out[0]:] += \
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
    print('DONE: convolution with spike ', spike_id)

    return recordings, final_pos, mixing

### RECORDING OPERATION ###

def extract_wf(spiketrains, recordings, times, fs, n_pad=2):
    '''

    Parameters
    ----------
    spiketrains
    recordings
    times
    fs
    n_pad

    Returns
    -------

    '''
    n_pad = int(n_pad * pq.ms * fs.rescale('kHz'))
    unit = times[0].rescale('ms').units

    nChs, nPts = recordings.shape

    for st in spiketrains:
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
                first_spike = False

            min_amp = np.min(spike_rec)
            sp_rec_wf.append(spike_rec)
            sp_amp.append(min_amp)

        st.waveforms = np.array(sp_rec_wf)


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
        print('Filtering signals with ', filter_type, ' filter at ', freq, '...')
        if len(anas.shape) == 2:
            anas_filt = filtfilt(b, a, anas, axis=1)
        elif len(anas.shape) == 1:
            anas_filt = filtfilt(b, a, anas)
        return anas_filt
    else:
        raise ValueError('Filter is not stable')


### PLOTTING ###

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
            if spiketrain.annotations['bintype'] == 'E':
                ax.plot(t, i * np.ones_like(t), 'b', marker=marker, mew=mew, markersize=markersize, ls='')
            elif spiketrain.annotations['bintype'] == 'I':
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


def plot_templates(templates, mea, single_figure=True):
    '''

    Parameters
    ----------
    templates
    mea_pos
    mea_pitch

    Returns
    -------

    '''
    import matplotlib.pylab as plt

    n_sources = len(templates)
    fig_t = plt.figure()

    if single_figure:
        colors = plt.rcParams['axes.color_cycle']
        ax_t = fig_t.add_subplot(111)

        for n, t in enumerate(templates):
            print('Plotting spike ', n, ' out of ', n_sources)
            if len(t.shape) == 3:
                MEA.plot_mea_recording(t.mean(axis=0), mea, colors=colors[np.mod(n, len(colors))], ax=ax_t, lw=2)
            else:
                MEA.plot_mea_recording(t, mea, colors=colors[np.mod(n, len(colors))], ax=ax_t, lw=2)

    else:
        cols = int(np.ceil(np.sqrt(n_sources)))
        rows = int(np.ceil(n_sources / float(cols)))

        for n in range(n_sources):
            ax_t = fig_t.add_subplot(rows, cols, n+1)
            MEA.plot_mea_recording(templates[n], mea, ax=ax_t)

    return fig_t


def plot_waveforms(spiketrains, mea):
    '''

    Parameters
    ----------
    spiketrains
    mea

    Returns
    -------

    '''
    import matplotlib.pylab as plt

    wf = [s.waveforms for s in spiketrains]
    fig_w = plt.figure()
    ax_w = fig_w.add_subplot(111)
    colors = plt.rcParams['axes.color_cycle']

    for n, w in enumerate(wf):
        print('Plotting spike ', n, ' out of ', len(wf))
        MEA.plot_mea_recording(w.mean(axis=0), mea, colors=colors[np.mod(n, len(colors))], ax=ax_w, lw=2)

    return fig_w