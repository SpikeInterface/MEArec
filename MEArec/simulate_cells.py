"""
Test implementation using cell models of the Blue Brain Project with LFPy.
The example assumes that cell models available from
https://bbpnmc.epfl.ch/nmc-portal/downloads are unzipped in the folder 'cell_models'

The function compile_all_mechanisms must be run once before any cell simulation
"""

import os
from os.path import join
import sys
from glob import glob
import numpy as np
import MEAutility as mu
import yaml
import time
from distutils.version import StrictVersion

if StrictVersion(yaml.__version__) >= StrictVersion('5.0.0'):
    use_loader = True
else:
    use_loader = False


def get_templatename(f):
    """
    Assess from hoc file the templatename being specified within

    Arguments
    ---------
    f : file, mode 'r'

    Returns
    -------
    templatename : str

    """
    templatename = None
    f = open("template.hoc", 'r')
    for line in f.readlines():
        if 'begintemplate' in line.split():
            templatename = line.split()[-1]
            continue
    return templatename


def compile_all_mechanisms(cell_folder, verbose=False):
    """ Attempt to set up a folder with all unique mechanism *.mod files and 
        compile them all. assumes all cell models are in a folder 'cell_models'
    
    Parameters:
    -----------
    model : string (optional, default='bbp')
        Cell model type ('bbp' - Blue Brain Project, i.e. NMC database)
    """

    if not os.path.isdir(join(cell_folder, 'mods')):
        os.mkdir(join(cell_folder, 'mods'))

    neurons = [join(cell_folder, f) \
               for f in os.listdir(join(cell_folder)) \
               if f != 'mods']
    if verbose:
        print(neurons)

    for nrn in neurons:
        for nmodl in glob(join(nrn, 'mechanisms', '*.mod')):
            while not os.path.isfile(join(cell_folder, 'mods', os.path.split(nmodl)[-1])):
                if verbose:
                    print('cp {} {}'.format(nmodl, join(cell_folder, 'mods')))
                os.system('cp {} {}'.format(nmodl, join(cell_folder, 'mods')))
    starting_dir = os.getcwd()
    os.chdir(join(cell_folder, 'mods'))
    os.system('nrnivmodl')
    os.chdir(starting_dir)


def return_cell(cell_folder, model_type, cell_name, end_T, dt, start_T, verbose=False):
    """ Function to load cell models
    
    Parameters:
    -----------
    cell_folder : string
        Path to folder where cell model is saved.
    model_type : string
        Cell model type (e.g. 'bbp' for Human Brain Project)    
    cell_name : string
        Name of the cell
    end_T : float
        Simulation length [ms]
    dt: float
        Time step of simulation [ms]
    start_T: float
        Simulation start time (recording starts at 0 ms)

    Returns:
    --------
    cell : object
        LFPy cell object
    """
    import mpi4py.MPI
    import LFPy
    import neuron
    neuron.h.load_file("stdrun.hoc")
    neuron.h.load_file("import3d.hoc")

    cwd = os.getcwd()
    os.chdir(cell_folder)
    if verbose:
        print("Simulating ", cell_name)

    if model_type == 'bbp':
        neuron.load_mechanisms('../mods')

        f = open("template.hoc", 'r')
        templatename = get_templatename(f)
        f.close()

        f = open("biophysics.hoc", 'r')
        biophysics = get_templatename(f)
        f.close()

        f = open("morphology.hoc", 'r')
        morphology = get_templatename(f)
        f.close()

        # get synapses template name
        f = open(join("synapses", "synapses.hoc"), 'r')
        synapses = get_templatename(f)
        f.close()

        neuron.h.load_file('constants.hoc')
        if not hasattr(neuron.h, morphology):
            neuron.h.load_file(1, "morphology.hoc")

        if not hasattr(neuron.h, biophysics):
            neuron.h.load_file(1, "biophysics.hoc")

        if not hasattr(neuron.h, synapses):
            # load synapses
            neuron.h.load_file(1, join('synapses', 'synapses.hoc'))

        if not hasattr(neuron.h, templatename):
            neuron.h.load_file(1, "template.hoc")

        morphologyfile = os.listdir('morphology')[0]  # glob('morphology\\*')[0]

        # Instantiate the cell(s) using LFPy
        cell = LFPy.TemplateCell(morphology=join('morphology', morphologyfile),
                                 templatefile=join('template.hoc'),
                                 templatename=templatename,
                                 templateargs=0,
                                 tstop=end_T,
                                 tstart=start_T,
                                 dt=dt,
                                 v_init=-70,
                                 pt3d=True,
                                 delete_sections=True,
                                 verbose=True)

    else:
        raise NotImplementedError('Cell model %s is not implemented' \
                                  % model_type)

    os.chdir(cwd)
    return cell


def return_cell_morphology(cell_name, cell_folder):
    """ Function to load cell models

    Parameters:
    -----------
    cell_name : string
        Name of the cell type.
    cell_folder : string
        Folder containing cell models.

    Returns:
    --------
    cell : object
        LFPy cell object
    """
    import mpi4py.MPI
    import LFPy

    if not os.path.isdir(join(cell_folder, cell_name)):
        raise NotImplementedError('Cell model %s is not found in %s' \
                                  % (cell_name, cell_folder))

    morphologyfile = os.listdir(join(cell_folder, cell_name, 'morphology'))[0]
    morphology = join(cell_folder, cell_name, 'morphology', morphologyfile)

    cell = LFPy.Cell(morphology=morphology, pt3d=True)
    return cell


def find_spike_idxs(v, thresh=-30, find_max=30):
    """ Find spike indices
    
    Parameters:
    -----------
    v: array_like
        Membrane potential
    thresh: float (optional, default = -30)
        Threshold for spike detections
    find_max: int
        Number of sample to find spike maximum after detection crossing

    Returns:
    --------
    spikes : array_like
        Indices of spike peaks in the positive direction, i.e. spikes
    """
    spikes_th = [idx for idx in range(len(v) - 1) if v[idx] < thresh < v[idx + 1]]
    spikes = []
    for sp in spikes_th:
        max_idx = np.argmax(v[sp:sp + find_max])
        spikes.append(sp + max_idx)
    return spikes


def set_input(weight, dt, T, cell, delay, stim_length):
    """ Set current input synapse in soma
    
    Parameters:
    -----------
    weight : float
        Strength of input current [nA]
    dt : float
        Time step of simulation [ms]
    T : float
        Total simulation time [ms]
    cell : object
        Cell object from LFPy
    delay : float
        Delay for input,i.e. when to start the input [ms]
    stim_length: float
        Duration of injected current [ms]
    
    Returns:
    --------
    noiseVec :  NEURON vector
        NEURON vector of input current
    cell : object
        LFPy cell object
    synapse : NEURON synapse
        NEURON synapse object
    """
    import neuron
    tot_ntsteps = int(round(T / dt + 1))

    I = np.ones(tot_ntsteps) * weight
    # I[stim_idxs] = weight
    noiseVec = neuron.h.Vector(I)
    syn = None
    for sec in cell.allseclist:
        if 'soma' in sec.name():
            # syn = neuron.h.ISyn(0.5, sec=sec) 
            syn = neuron.h.IClamp(0.5, sec=sec)
    syn.dur = stim_length
    syn.delay = delay  # cell.tstartms
    noiseVec.play(syn._ref_amp, dt)

    return noiseVec, cell, syn


def run_cell_model(cell_model, sim_folder, seed, verbose, save=True, return_vi=False, **kwargs):
    """ Run simulation and adjust input strength to have a certain number of 
        spikes (target_spikes[0] < num_spikes <= target_spikes[1]
        where target_spikes=[10,30] by default)

    Parameters:
    -----------
    cell_model : string
        Path to folder where cell model is saved.
    model_type : string
        Cell model type (e.g. 'bbp')    
    sim_folder : string
        Data directory for transmembrane currents and membrane potential
        of the neuron.
    cell_model_id: int
        Arbitrary cell id, used to set the numpy.random seed.

    Returns:
    --------
    cell : object
        LFPy cell object
    """
    cell_name = os.path.split(cell_model)[-1]

    if save:
        if not os.path.isdir(sim_folder):
            os.makedirs(sim_folder)

        imem_files = [f for f in os.listdir(sim_folder) if 'imem' in f]
        vmem_files = [f for f in os.listdir(sim_folder) if 'vmem' in f]

        if not (np.any([cell_name in ifile for ifile in imem_files]) and
                np.any([cell_name in vfile for vfile in vmem_files])):

            np.random.seed(seed)
            T = kwargs['sim_time'] * 1000
            dt = kwargs['dt']
            cell = return_cell(cell_model, 'bbp', cell_name, T, dt, 0)

            delay = kwargs['delay']
            stim_length = T - delay
            weights = kwargs['weights']
            weight = weights[0]
            target_spikes = kwargs['target_spikes']
            cuts = kwargs['cut_out']
            cut_out = [cuts[0] / dt, cuts[1] / dt]

            num_spikes = 0

            i = 0
            while not target_spikes[0] < num_spikes <= target_spikes[1]:
                noiseVec, cell, syn = set_input(weight, dt, T, cell, delay, stim_length)
                cell.simulate(rec_imem=True)

                t = cell.tvec
                v = cell.somav
                t = t
                v = v

                spikes = find_spike_idxs(v[int(cut_out[0]):-int(cut_out[1])])
                spikes = list(np.array(spikes) + cut_out[0])
                num_spikes = len(spikes)

                if verbose:
                    print("Input weight: ", weight, " - Num Spikes: ", num_spikes)
                if num_spikes >= target_spikes[1]:
                    weight *= weights[0]
                elif num_spikes <= target_spikes[0]:
                    weight *= weights[1]

                i += 1
                if i >= 10:
                    sys.exit()

            t = t[0:(int(cut_out[0]) + int(cut_out[1]))] - t[int(cut_out[0])]
            # discard first spike
            i_spikes = np.zeros((num_spikes - 1, cell.totnsegs, len(t)))
            v_spikes = np.zeros((num_spikes - 1, len(t)))

            for idx, spike_idx in enumerate(spikes[1:]):
                spike_idx = int(spike_idx)
                v_spike = v[spike_idx - int(cut_out[0]):spike_idx + int(cut_out[1])]
                i_spike = cell.imem[:, spike_idx - int(cut_out[0]):spike_idx + int(cut_out[1])]
                i_spikes[idx, :, :] = i_spike
                v_spikes[idx, :] = v_spike

            if not os.path.isdir(sim_folder):
                os.makedirs(sim_folder)
            np.save(join(sim_folder, 'imem_%d_%s.npy' % (num_spikes - 1, cell_name)), i_spikes)
            np.save(join(sim_folder, 'vmem_%d_%s.npy' % (num_spikes - 1, cell_name)), v_spikes)

        else:
            if verbose:
                print('\n\n\nCell has already be simulated. Using stored membrane currents\n\n\n')
    else:
        np.random.seed(seed)
        T = kwargs['sim_time'] * 1000
        dt = kwargs['dt']
        cell = return_cell(cell_model, 'bbp', cell_name, T, dt, 0)

        delay = kwargs['delay']
        stim_length = T - delay
        weights = kwargs['weights']
        weight = weights[0]
        target_spikes = kwargs['target_spikes']
        cuts = kwargs['cut_out']
        cut_out = [cuts[0] / dt, cuts[1] / dt]

        num_spikes = 0

        i = 0
        while not target_spikes[0] < num_spikes <= target_spikes[1]:
            noiseVec, cell, syn = set_input(weight, dt, T, cell, delay, stim_length)
            cell.simulate(rec_imem=True)

            t = cell.tvec
            v = cell.somav
            t = t
            v = v

            spikes = find_spike_idxs(v[int(cut_out[0]):-int(cut_out[1])])
            spikes = list(np.array(spikes) + cut_out[0])
            num_spikes = len(spikes)

            if verbose:
                print("Input weight: ", weight, " - Num Spikes: ", num_spikes)
            if num_spikes >= target_spikes[1]:
                weight *= weights[0]
            elif num_spikes <= target_spikes[0]:
                weight *= weights[1]

            i += 1
            if i >= 10:
                sys.exit()

        t = t[0:(int(cut_out[0]) + int(cut_out[1]))] - t[int(cut_out[0])]
        # discard first spike
        i_spikes = np.zeros((num_spikes - 1, cell.totnsegs, len(t)))
        v_spikes = np.zeros((num_spikes - 1, len(t)))

        for idx, spike_idx in enumerate(spikes[1:]):
            spike_idx = int(spike_idx)
            v_spike = v[spike_idx - int(cut_out[0]):spike_idx + int(cut_out[1])]
            i_spike = cell.imem[:, spike_idx - int(cut_out[0]):spike_idx + int(cut_out[1])]
            i_spikes[idx, :, :] = i_spike
            v_spikes[idx, :] = v_spike

        if return_vi:
            return cell, v_spikes, i_spikes


def calc_extracellular(cell_model, save_sim_folder, load_sim_folder, seed, verbose=False, position=None, **kwargs):
    """  Loads data from previous cell simulation, and use results to generate
         arbitrary number of spikes above a certain noise level.

    Parameters:
    -----------
    cell_model : string
        Path to folder where cell model is saved.
    model_type : string
        Cell model type (e.g. 'bbp')    
    save_sim_folder : string
        Path to folder where to save EAP data
    load_sim_folder : string
        Path to folder from which  NEURON simulation results (currents, 
        membrane potential) are loaded
    rotation: string
        Type of rotation to apply to neuron morphologies 
        ('Norot','physrot','3drot')
    cell_model_id: int
        Arbitrary cell id, used to set the numpy.random seed.

    Returns:
    --------
        nothing, but saves the result
    """
    cell_name = os.path.split(cell_model)[-1]
    cell_save_name = cell_name
    np.random.seed(seed)

    T = kwargs['sim_time'] * 1000
    dt = kwargs['dt']
    rotation = kwargs['rot']
    nobs = kwargs['n']
    ncontacts = kwargs['ncontacts']
    overhang = kwargs['overhang']
    x_lim = kwargs['xlim']
    y_lim = kwargs['ylim']
    z_lim = kwargs['zlim']
    min_amp = kwargs['min_amp']
    MEAname = kwargs['probe']
    drifting = kwargs['drifting']
    if drifting:
        max_drift = kwargs['max_drift']
        min_drift = kwargs['min_drift']
        drift_steps = kwargs['drift_steps']
        drift_x_lim = kwargs['drift_xlim']
        drift_y_lim = kwargs['drift_ylim']
        drift_z_lim = kwargs['drift_zlim']

    sim_folder = join(save_sim_folder, rotation)
    cell = return_cell(cell_model, 'bbp', cell_name, T, dt, 0)

    # Load data from previous cell simulation
    imem_file = [f for f in os.listdir(load_sim_folder) if cell_name in f and 'imem' in f][0]
    vmem_file = [f for f in os.listdir(load_sim_folder) if cell_name in f and 'vmem' in f][0]
    i_spikes = np.load(join(load_sim_folder, imem_file))
    v_spikes = np.load(join(load_sim_folder, vmem_file))
    cell.tvec = np.arange(i_spikes.shape[-1]) * dt

    save_spikes = []
    save_pos = []
    save_rot = []
    save_offs = []
    target_num_spikes = int(nobs)

    # load MEA info
    elinfo = mu.return_mea_info(electrode_name=MEAname)

    # Create save folder
    save_folder = join(sim_folder, 'tmp_%d_%s' % (target_num_spikes, MEAname))

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    if verbose:
        print('Cell ', cell_save_name, ' extracellular spikes to be simulated')

    mea = mu.return_mea(info=elinfo)
    pos = mea.positions

    elec_x = pos[:, 0]
    elec_y = pos[:, 1]
    elec_z = pos[:, 2]

    N = np.empty((pos.shape[0], 3))
    for i in np.arange(N.shape[0]):
        N[i,] = [1, 0, 0]  # normal vec. of contacts

    # Add square electrodes (instead of circles)
    if ncontacts > 1:
        electrode_parameters = {
            'sigma': 0.3,  # extracellular conductivity
            'x': elec_x,  # x,y,z-coordinates of contact points
            'y': elec_y,
            'z': elec_z,
            'n': ncontacts,
            'r': elinfo['size'],
            'N': N,
            'contact_shape': elinfo['shape']
        }
    else:
        electrode_parameters = {
            'sigma': 0.3,  # extracellular conductivity
            'x': elec_x,  # x,y,z-coordinates of contact points
            'y': elec_y,
            'z': elec_z
        }

    if x_lim is None:
        x_lim = [float(np.min(elec_x) - overhang),
                 float(np.max(elec_x) + overhang)]
    if y_lim is None:
        y_lim = [float(np.min(elec_y) - overhang),
                 float(np.max(elec_y) + overhang)]
    if z_lim is None:
        z_lim = [float(np.min(elec_z) - overhang),
                 float(np.max(elec_z) + overhang)]

    ignored = 0
    saved = 0
    i = 0

    while len(save_spikes) < target_num_spikes:
        if i > 1000 * target_num_spikes:
            if verbose:
                print("Gave up finding spikes above noise level for %s" % cell_name)
            break
        spike_idx = np.random.randint(0, i_spikes.shape[0])  # Each cell has several spikes to choose from
        cell.imem = i_spikes[spike_idx, :, :]
        cell.somav = v_spikes[spike_idx, :]

        if not drifting:
            espikes, pos, rot, offs = return_extracellular_spike(cell=cell, cell_name=cell_name, model_type='bbp',
                                                                 electrode_parameters=electrode_parameters,
                                                                 limits=[x_lim, y_lim, z_lim], rotation=rotation,
                                                                 pos=position)
            # Method of Images for semi-infinite planes
            if elinfo['type'] == 'mea':
                espikes = espikes * 2

            if check_espike(espikes, min_amp):
                save_spikes.append(espikes)
                save_pos.append(pos)
                save_rot.append(rot)
                save_offs.append(offs)
                plot_spike = False
                if verbose:
                    print('Cell: ' + cell_name + ' Progress: [' +
                          str(len(save_spikes)) + '/' + str(target_num_spikes) + ']')
                saved += 1
        else:
            espikes, pos, rot, offs = return_extracellular_spike(cell, cell_name, 'bbp', electrode_parameters,
                                                                 [x_lim, y_lim, z_lim], rotation, pos=position)

            # Method of Images for semi-infinite planes
            if elinfo['type'] == 'mea':
                espikes = espikes * 2

            if pos[0] - drift_x_lim[0] > x_lim[0] and pos[0] - drift_x_lim[1] < x_lim[1] and \
                    pos[1] - drift_y_lim[0] > y_lim[0] and pos[1] - drift_y_lim[1] < y_lim[1] and \
                    pos[2] - drift_z_lim[0] > z_lim[0] and pos[2] - drift_z_lim[1] < z_lim[1]:
                if check_espike(espikes, min_amp):
                    drift_ok = False
                    # fix rotation while drifting
                    cell.set_rotation(rot[0], rot[1], rot[2])
                    max_trials = 100
                    tr = 0
                    while not drift_ok and tr < max_trials:
                        init_pos = pos
                        # find drifting final position within drift limits
                        x_rand = np.random.uniform(init_pos[0] + drift_x_lim[0], init_pos[0] + drift_x_lim[1])
                        y_rand = np.random.uniform(init_pos[1] + drift_y_lim[0], init_pos[1] + drift_y_lim[1])
                        z_rand = np.random.uniform(init_pos[2] + drift_z_lim[0], init_pos[2] + drift_z_lim[1])
                        final_pos = [x_rand, y_rand, z_rand]
                        drift_dist = np.linalg.norm(np.array(init_pos) - np.array(final_pos))

                        # check location and boundaries
                        if drift_dist < max_drift and drift_dist > min_drift and \
                                x_rand > x_lim[0] and x_rand < x_lim[1] and \
                                y_rand > y_lim[0] and y_rand < y_lim[1] and \
                                z_rand > z_lim[0] and z_rand < z_lim[1]:
                            espikes, pos, rot_, offs = return_extracellular_spike(cell, cell_name, 'bbp',
                                                                                  electrode_parameters,
                                                                                  [x_lim, y_lim, z_lim],
                                                                                  rotation=None,
                                                                                  pos=final_pos)
                            # check final position spike amplitude
                            if check_espike(espikes, min_amp):
                                if verbose:
                                    print('Found final drifting position')
                                drift_ok = True
                            else:
                                tr += 1
                                pass
                        else:
                            tr += 1
                            pass

                    # now compute drifting templates
                    if drift_ok:
                        drift_spikes = []
                        drift_pos = []
                        drift_rot = []
                        drift_dist = np.linalg.norm(np.array(init_pos) - np.array(final_pos))
                        drift_dir = np.array(final_pos) - np.array(init_pos)
                        for i, dp in enumerate(np.linspace(0, 1, drift_steps)):
                            pos_drift = init_pos + dp * drift_dir
                            espikes, pos, r_, offs = return_extracellular_spike(cell, cell_name, 'bbp',
                                                                                electrode_parameters,
                                                                                [x_lim, y_lim, z_lim],
                                                                                rotation=None,
                                                                                pos=pos_drift)
                            drift_spikes.append(espikes)
                            drift_pos.append(pos)
                            drift_rot.append(rot)

                        # reverse rotation
                        rev_rot = [-r for r in rot]
                        cell.set_rotation(rev_rot[0], rev_rot[1], rev_rot[2], rotation_order='zyx')

                        drift_spikes = np.array(drift_spikes)
                        drift_pos = np.array(drift_pos)
                        if verbose:
                            print('Drift done from ', init_pos, ' to ', final_pos, ' with ', drift_steps, ' steps')

                        save_spikes.append(drift_spikes)
                        save_pos.append(drift_pos)
                        save_rot.append(rot)
                        save_offs.append(offs)
                        if verbose:
                            print('Cell: ' + cell_name + ' Progress: [' + str(len(save_spikes)) + '/' +
                                  str(target_num_spikes) + ']')
                        saved += 1
                    else:
                        if verbose:
                            print('Discarded for trials')
                else:
                    pass
            else:
                if verbose:
                    print('Discarded position: ', pos)
        i += 1

    save_spikes = np.array(save_spikes)
    save_pos = np.array(save_pos)
    save_rot = np.array(save_rot)

    np.save(join(save_folder, 'eap-%s' % cell_save_name), save_spikes)
    np.save(join(save_folder, 'pos-%s' % cell_save_name), save_pos)
    np.save(join(save_folder, 'rot-%s' % cell_save_name), save_rot)


def check_espike(espikes, min_amp):
    """
    Check extracellular spike amplitude and shape (neg peak > pos peak)

    Parameters
    ----------
    espike: np.array
        EAP (n_elec, n_samples)
    min_amp: float
        Minimum amplitude

    Returns
    -------
    valid: bool
        If True EAP is valid

    """
    valid = True
    if np.max(np.abs(np.min(espikes))) < min_amp:
        valid = False
    if np.abs(np.min(espikes)) < np.abs(np.max(espikes)):
        valid = False
    return valid


def get_physrot_specs(cell_name, model):
    """  Return physrot specifications for cell types
    
    Parameters:
    -----------
    cell_name : string
        The name of the cell.

    Returns:
    --------
    polarlim : array_like
        lower and upper bound for the polar angle
    pref_orient : array_like
        3-dim vetor of preferred orientation 
    """
    if model == 'bbp':
        polarlim = {'BP': [0., 15.],
                    'BTC': None,  # [0.,15.],
                    'ChC': None,  # [0.,15.],
                    'DBC': None,  # [0.,15.],
                    'LBC': None,  # [0.,15.],
                    'MC': [0., 15.],
                    'NBC': None,
                    'NGC': None,
                    'SBC': None,
                    'STPC': [0., 15.],
                    'TTPC1': [0., 15.],
                    'TTPC2': [0., 15.],
                    'UTPC': [0., 15.]}
        # how it's implemented, the NMC y axis points into the pref_orient direction after rotation
        pref_orient = {'BP': [0., 0., 1.],
                       'BTC': None,  # [0.,0.,1.],
                       'ChC': None,  # [0.,0.,1.],
                       'DBC': None,  # [0.,0.,1.],
                       'LBC': None,  # [0.,0.,1.],
                       'MC': [0., 0., 1.],
                       'NBC': None,
                       'NGC': None,
                       'SBC': None,
                       'STPC': [0., 0., 1.],
                       'TTPC1': [0., 0., 1.],
                       'TTPC2': [0., 0., 1.],
                       'UTPC': [0., 0., 1.]}
        return polarlim[cell_name.split('_')[1]], pref_orient[cell_name.split('_')[1]]
    else:
        raise NotImplementedError('Cell model %s is not implemented' \
                                  % model)


def return_extracellular_spike(cell, cell_name, model_type,
                               electrode_parameters, limits, rotation, pos=None):
    """    Calculate extracellular spike on MEA 
           at random position relative to cell

    Parameters:
    -----------
    cell: object
        cell object from LFPy
    cell_name: string
        name of cell model
    electrode_parameters: dict
        parameters to initialize LFPy.RecExtElectrode
    limits: array_like
        boundaries for neuron locations, shape=(3,2)
    rotation: string 
        random rotation to apply to the neuron ('Norot', '3drot', 'physrot')
    pos: array_like, (optional, default None)
        Can be used to set the cell soma to a specific position. If ``None``,
        the random position is used.
    Returns:
    --------
    Extracellular spike for each MEA contact site
    """
    import LFPy

    def get_xyz_angles(R):
        """ Get rotation angles for each axis from rotation matrix
        
        Parameters;
        -----------
        R : matrix
            3x3 rotation matrix

        Returns:
        --------
        R_z : float
        R_y : float
        R_x : float
            Three angles for rotations around axis, defined by R = R_z.R_y.R_x
        """
        rot_x = np.arctan2(R[2, 1], R[2, 2])
        rot_y = np.arcsin(-R[2, 0])
        rot_z = np.arctan2(R[1, 0], R[0, 0])
        return rot_x, rot_y, rot_z

    def get_rnd_rot_Arvo():
        """ Generate uniformly distributed random rotation matrices
        see: 'Fast Random Rotation Matrices' by Arvo (1992)
        
        Returns:
        --------
        R : 3x3 matrix
            random rotation matrix
        """
        gamma = np.random.uniform(0, 2. * np.pi)
        rotation_z = np.matrix([[np.cos(gamma), -np.sin(gamma), 0],
                                [np.sin(gamma), np.cos(gamma), 0],
                                [0, 0, 1]])
        x = np.random.uniform(size=2)
        v = np.array([np.cos(2. * np.pi * x[0]) * np.sqrt(x[1]),
                      np.sin(2. * np.pi * x[0]) * np.sqrt(x[1]),
                      np.sqrt(1 - x[1])])
        H = np.identity(3) - 2. * np.outer(v, v)
        M = -np.dot(H, rotation_z)
        return M

    def check_solidangle(matrix, pre, post, polarlim):
        """ Check whether a matrix rotates the vector 'pre' into a region
            defined by 'polarlim' around the vector 'post'

        Parameters:
        -----------
        matrix : matrix
            3x3 rotation matrix
        pre : array_like
            3-dim vector to be rotated
        post : array_like
            axis of the cones defining the post-rotation region
        polarlim : [float,float]
            Angles specifying the opening of the inner and outer cone 
            (aperture = 2*polarlim),
            i.e. the angle between rotated pre vector and post vector has to ly 
            within these polar limits.  

        Returns:
        --------
        test : bool
            True if the vector np.dot(matrix,pre) lies inside the specified region.
        """
        postest = np.dot(matrix, pre)
        c = np.dot(post / np.linalg.norm(post), postest / np.linalg.norm(postest))
        if np.cos(np.deg2rad(polarlim[1])) <= c <= np.cos(np.deg2rad(polarlim[0])):
            return True
        else:
            return False

    electrodes = LFPy.RecExtElectrode(cell, **electrode_parameters)

    """Rotate neuron"""
    if rotation == 'norot':
        if model_type == 'bbp':
            # orientate cells in z direction
            x_rot_offset = np.pi / 2.
            y_rot_offset = 0
            z_rot_offset = 0
        x_rot = x_rot_offset
        y_rot = y_rot_offset
        z_rot = z_rot_offset
    elif rotation == 'xrot':
        if model_type == 'bbp':
            # orientate cells in z direction
            x_rot_offset = np.pi / 2.
            y_rot_offset = 0
            z_rot_offset = 0
        x_rot, _, _ = get_xyz_angles(np.array(get_rnd_rot_Arvo()))
        x_rot = x_rot + x_rot_offset
        y_rot = y_rot_offset
        z_rot = z_rot_offset
    elif rotation == 'yrot':
        if model_type == 'bbp':
            # orientate cells in z direction
            x_rot_offset = np.pi / 2.
            y_rot_offset = 0
            z_rot_offset = 0
        _, y_rot, _ = get_xyz_angles(np.array(get_rnd_rot_Arvo()))
        x_rot = x_rot_offset
        y_rot = y_rot + y_rot_offset
        z_rot = z_rot_offset
    elif rotation == 'zrot':
        if model_type == 'bbp':
            # orientate cells in z direction
            x_rot_offset = np.pi / 2.
            y_rot_offset = 0
            z_rot_offset = 0
        _, _, z_rot = get_xyz_angles(np.array(get_rnd_rot_Arvo()))
        x_rot = x_rot_offset
        y_rot = y_rot_offset
        z_rot = z_rot + z_rot_offset
    elif rotation == '3drot':
        if model_type == 'bbp':
            x_rot_offset = np.pi / 2.  # align neuron with z axis
            y_rot_offset = 0  # align neuron with z axis
            z_rot_offset = 0  # align neuron with z axis
        x_rot, y_rot, z_rot = get_xyz_angles(np.array(get_rnd_rot_Arvo()))
        x_rot = x_rot + x_rot_offset
        y_rot = y_rot + y_rot_offset
        z_rot = z_rot + z_rot_offset
    elif rotation == 'physrot':
        polarlim, pref_orient = get_physrot_specs(cell_name, model_type)
        if model_type == 'bbp':
            x_rot_offset = np.pi / 2.  # align neuron with z axis
            y_rot_offset = 0  # align neuron with z axis
            z_rot_offset = 0  # align neuron with z axis
        while True:
            R = np.array(get_rnd_rot_Arvo())
            if polarlim is None or pref_orient is None:
                valid = True
            else:
                valid = check_solidangle(R, [0., 0., 1.], pref_orient, polarlim)
            if valid:
                x_rot, y_rot, z_rot = get_xyz_angles(R)
                x_rot = x_rot + x_rot_offset
                y_rot = y_rot + y_rot_offset
                z_rot = z_rot + z_rot_offset
                break
    else:
        rotation = None
        x_rot = 0
        y_rot = 0
        z_rot = 0

    if pos is None:
        """Move neuron randomly"""
        x_rand = np.random.uniform(limits[0][0], limits[0][1])
        y_rand = np.random.uniform(limits[1][0], limits[1][1])
        z_rand = np.random.uniform(limits[2][0], limits[2][1])
        cell.set_pos(x_rand, y_rand, z_rand)
        pos = [x_rand, y_rand, z_rand]
    else:
        cell.set_pos(pos[0], pos[1], pos[2])
    cell.set_rotation(x=x_rot, y=y_rot, z=z_rot)
    rot = [x_rot, y_rot, z_rot]

    electrodes.calc_lfp()

    # Reverse rotation to bring cell back into initial rotation state
    if rotation is not None:
        rev_rot = [-r for r in rot]
        cell.set_rotation(rev_rot[0], rev_rot[1], rev_rot[2], rotation_order='zyx')

    return 1000 * electrodes.LFP, pos, rot, electrodes.offsets


def str2bool(v):
    """ Transform string to bool
    
    Parameters:
    -----------
    v : str
    
    Returns:
    --------
    transformed_v, bool
        If v is any of ("yes", "true", "t", "1") (case insensitive) 
        ``True`` is returned, else ``False``
    """
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':

    if len(sys.argv) == 3 and sys.argv[1] == 'compile':
        cell_folder = sys.argv[2]
        compile_all_mechanisms(cell_folder)
        sys.exit(0)
    elif len(sys.argv) == 5:
        cell_model = sys.argv[1]
        intraonly = str2bool(sys.argv[2])
        params_path = sys.argv[3]
        verbose = str2bool(sys.argv[4])

        with open(params_path, 'r') as f:
            if use_loader:
                params = yaml.load(f, Loader=yaml.FullLoader)
            else:
                params = yaml.load(f)

        sim_folder = params['templates_folder']
        cell_folder = params['cell_models_folder']
        rot = params['rot']

        extra_sim_folder = params['templates_folder']
        vm_im_sim_folder = join(params['templates_folder'], 'intracellular')

        print('Intracellular simulation: ', cell_model)
        run_cell_model(cell_model, vm_im_sim_folder, verbose=verbose, **params)
        if not intraonly:
            print('Extracellular simulation: ', cell_model)
            calc_extracellular(cell_model, extra_sim_folder, vm_im_sim_folder, verbose=verbose, **params)
