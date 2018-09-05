#!/usr/bin/env python
from __future__ import division
from __future__ import print_function


'''
Test implementation using cell models of the Blue Brain Project with LFPy.
The example assumes that cell models available from
https://bbpnmc.epfl.ch/nmc-portal/downloads are unzipped in the folder 'cell_models'

The function compile_all_mechanisms must be run once before any cell simulation
'''

import os
from os.path import join
import sys
from glob import glob
import numpy as np
import LFPy
import neuron
import MEAutility as MEA
import yaml
import time

from defaultconfig import *
if os.path.exists('./config.py'):
    from config import *

def get_templatename(f):
    '''
    Assess from hoc file the templatename being specified within

    Arguments
    ---------
    f : file, mode 'r'

    Returns
    -------
    templatename : str

    '''
    templatename = None
    f = file("template.hoc", 'r')
    for line in f.readlines():
        if 'begintemplate' in line.split():
            templatename = line.split()[-1]
            print('template {} found!'.format(templatename))
            continue
    return templatename


def compile_all_mechanisms(model='bbp'):
    """ Attempt to set up a folder with all unique mechanism *.mod files and 
        compile them all. assumes all cell models are in a folder 'cell_models'
    
    Parameters:
    -----------
    model : string (optional, default='bbp')
        Cell model type ('bbp' - Blue Brain Project, i.e. NMC database)
    """

    if not os.path.isdir(join(root_folder, 'cell_models', model, 'mods')):
        os.mkdir(join(root_folder, 'cell_models', model, 'mods'))

    neurons = [join(root_folder,'cell_models', model, f) \
               for f in os.listdir(join(root_folder, 'cell_models', model)) \
               if f != 'mods']
    print(neurons)

    for nrn in neurons:
        for nmodl in glob(join(nrn, 'mechanisms', '*.mod')):
            print(nmodl)
            while not os.path.isfile(join(root_folder, 'cell_models', model, 'mods', os.path.split(nmodl)[-1])):
                print('cp {} {}'.format(nmodl, join(root_folder, 'cell_models', model, 'mods')))
                os.system('cp {} {}'.format(nmodl, join(root_folder, 'cell_models', model, 'mods')))

    os.chdir(join(root_folder, 'cell_models', model, 'mods'))
    os.system('nrnivmodl')
    os.chdir(root_folder)


def return_cell(cell_folder, model_type, cell_name, end_T, dt, start_T):
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
    import neuron
    neuron.h.load_file("stdrun.hoc")
    neuron.h.load_file("import3d.hoc")

    cwd = os.getcwd()
    os.chdir(cell_folder)
    print("Simulating ", cell_name)


    if model_type == 'bbp':
        neuron.load_mechanisms('../mods')

        f = file("template.hoc", 'r')
        templatename = get_templatename(f)
        f.close()

        f = file("biophysics.hoc", 'r')
        biophysics = get_templatename(f)
        f.close()

        f = file("morphology.hoc", 'r')
        morphology = get_templatename(f)
        f.close()

        #get synapses template name
        f = file(join("synapses", "synapses.hoc"), 'r')
        synapses = get_templatename(f)
        f.close()

        print('Loading constants')
        neuron.h.load_file('constants.hoc')
        print('...done.')
        if not hasattr(neuron.h, morphology):
            print('loading morpho...')
            neuron.h.load_file(1, "morphology.hoc")
            print('done.')

        if not hasattr(neuron.h, biophysics):
            neuron.h.load_file(1, "biophysics.hoc")

        if not hasattr(neuron.h, synapses):
            # load synapses
            neuron.h.load_file(1, join('synapses', 'synapses.hoc'))

        if not hasattr(neuron.h, templatename):
            print('Loading template...')
            neuron.h.load_file(1, "template.hoc")
            print('done.')

        morphologyfile = os.listdir('morphology')[0]#glob('morphology\\*')[0]

        # Instantiate the cell(s) using LFPy
        print('Initialize cell...')
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
        print('...done.')

    else:
        raise NotImplementedError('Cell model %s is not implemented'\
                                  % model_type)

    os.chdir(cwd)
    return cell


def find_spike_idxs(v, thresh=-30):
    """ Find spike indices
    
    Parameters:
    -----------
    v: array_like
        Membrane potential
    thresh: float (optional, default = -30)
        Threshold for spike detections

    Returns:
    --------
    spikes : array_like
        Indices of threshold crossings in the positive direction, i.e. spikes
    """
    spikes = [idx for idx in range(len(v) - 1) if v[idx] < thresh < v[idx + 1]]
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

    tot_ntsteps = int(round(T / dt + 1))

    I = np.ones(tot_ntsteps) * weight
    #I[stim_idxs] = weight
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


def run_cell_model(cell_model, model_type, sim_folder, cell_model_id):
    """ Run simulation and adjust input strength to have a certain number of 
        spikes (num_to_save < num_spikes <= 3*num_to_save 
        where num_to_save=10 by default)

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

    if not os.path.isfile(join(sim_folder, ('i_spikes_%s.npy' % cell_name))) and \
            not os.path.isfile(join(sim_folder, ('v_spikes_%s.npy' % cell_name))):

        np.random.seed(123 * cell_model_id)
        T = 1200
        dt = 2 ** -5
        cell = return_cell(cell_model, model_type, cell_name, T, dt, 0)

        delay = 200
        stim_length = 1000
        weight = 0.23
        # weight = -1.25

        num_spikes = 0
        spikes = []

        cut_out = [2. / dt, 5. / dt]
        num_to_save = 10

        i = 0

        while not num_to_save < num_spikes <= num_to_save * 3:
            noiseVec, cell, syn = set_input(weight, dt, T, cell, delay, stim_length)

            cell.simulate(rec_imem=True)

            t = cell.tvec
            v = cell.somav
            t = t
            v = v

            # ipdb.set_trace()

            spikes = find_spike_idxs(v[int(cut_out[0]):-int(cut_out[1])])
            spikes = list(np.array(spikes) + cut_out[0])
            num_spikes = len(spikes)

            print("Input weight: ", weight, " - Num Spikes: ", num_spikes)
            if num_spikes >= num_to_save * 3:
                weight *= 0.75
            elif num_spikes <= num_to_save:
                weight *= 1.25

            i += 1

            if i >= 10:
                sys.exit()

        t = t[0:(int(cut_out[0]) + int(cut_out[1]))] - t[int(cut_out[0])]
        i_spikes = np.zeros((num_to_save, cell.totnsegs, len(t)))
        v_spikes = np.zeros((num_to_save, len(t)))

        for idx, spike_idx in enumerate(spikes[1:num_to_save+1]):
            spike_idx = int(spike_idx)
            v_spike = v[spike_idx - int(cut_out[0]):spike_idx + int(cut_out[1])]
            i_spike = cell.imem[:, spike_idx - int(cut_out[0]):spike_idx + int(cut_out[1])]
            i_spikes[idx, :, :] = i_spike
            v_spikes[idx, :] = v_spike

        if not os.path.isdir(sim_folder):
            os.makedirs(sim_folder)
        np.save(join(sim_folder, 'i_spikes_%s.npy' % cell_name), i_spikes)
        np.save(join(sim_folder, 'v_spikes_%s.npy' % cell_name), v_spikes)

        return cell

    else:
        print('Cell has already be simulated. Using stored membrane currents')
        np.random.seed(123 * cell_model_id)
        T = 1200
        dt = 2 ** -5
        cell = return_cell(cell_model, model_type, cell_name, T, dt, 0)

        return cell

def calc_extracellular(cell_model, model_type, save_sim_folder, load_sim_folder,\
                       rotation, cell_model_id, elname, nobs, params_path=None, position=None):
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
    sim_folder = join(save_sim_folder, rotation)

    np.random.seed(123 * cell_model_id)
    dt = 2**-5
    T = 1

    cell_name = os.path.split(cell_model)[-1]
    cell_save_name = cell_name

    cell = return_cell(cell_model, model_type, cell_name, T, dt, 0)

    # Load data from previous cell simulation
    i_spikes = np.load(join(load_sim_folder, 'i_spikes_%s.npy' % cell_name))
    v_spikes = np.load(join(load_sim_folder, 'v_spikes_%s.npy' % cell_name))

    cell.tvec = np.arange(i_spikes.shape[-1]) * dt

    save_spikes = []
    save_pos = []
    save_rot = []
    save_offs = []
    target_num_spikes = int(nobs)

    if params_path is None:
        ncontacts = 1
        overhang = 30
        xplane = 0
        x_lim = [10., 80.]
        threshold_detect = 30
    else:
        with open(params_path, 'r') as f:
            params = yaml.load(f)
        ncontacts = params['ncontacts']
        overhang = params['overhang']
        x_plane = params['x_plane']
        x_lim = params['x_lim']
        threshold_detect = params['threshold_detect']

    i = 0
    # specify MEA
    MEAname = elname

    # load MEA info
    elinfo = MEA.return_mea_info(electrode_name=MEAname)
    
    # specify number of points for average EAP on each site
    n = 1  # 10 # 50
    elinfo.update({'n_points': ncontacts})

    # Create save folder
    # Create directory with target_spikes and date
    save_folder = join(sim_folder, 'tmp_%d_%s' % (target_num_spikes, MEAname))

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    # Check if already existing
    if os.path.isfile(join(save_folder, 'eap-%s.npy' % cell_save_name)) and \
        os.path.isfile(join(save_folder, 'pos-%s.npy' % cell_save_name)) and \
        os.path.isfile(join(save_folder, 'rot-%s.npy' % cell_save_name)):
        print('Cell ', cell_save_name, ' extracellular spikes have already been simulated and saved')
    else:
        print('Cell ', cell_save_name, ' extracellular spikes to be simulated')

        x_plane = 0.
        pos, dim, pitch = MEA.return_mea(electrode_name=MEAname, x_plane=x_plane)

        elec_x = pos[:, 0]
        elec_y = pos[:, 1]
        elec_z = pos[:, 2]

        N = np.empty((pos.shape[0], 3))
        for i in xrange(N.shape[0]):
            N[i, ] = [1, 0, 0]  # normal vec. of contacts

        # Add square electrodes (instead of circles)
        if ncontacts > 1:
            electrode_parameters = {
                'sigma': 0.3,  # extracellular conductivity
                'x': elec_x,  # x,y,z-coordinates of contact points
                'y': elec_y,
                'z': elec_z,
                'n': ncontacts,
                'r': elinfo['r'],
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
            
        y_lim = [float(min(elec_y)-elinfo['pitch'][0]/2.-overhang), float(max(elec_y)+elinfo['pitch'][0]/2.+overhang)]
        z_lim = [float(min(elec_z)-elinfo['pitch'][1]/2.-overhang), float(max(elec_z)+elinfo['pitch'][1]/2.+overhang)]
        print(x_lim)

        ignored=0
        saved = 0

        while len(save_spikes) < target_num_spikes:
            if i > 1000 * target_num_spikes:
                print("Gave up finding spikes above noise level for %s" % cell_name)
                break
            spike_idx = np.random.randint(0, i_spikes.shape[0])  # Each cell has several spikes to choose from
            cell.imem = i_spikes[spike_idx, :, :]
            cell.somav = v_spikes[spike_idx, :]

            espikes, pos, rot, offs = return_extracellular_spike(cell, cell_name, model_type, electrode_parameters,
                                                                 [x_lim, y_lim, z_lim], rotation, pos=position)
            if (np.ptp(espikes, axis=1) >= threshold_detect).any():
                save_spikes.append(espikes)
                save_pos.append(pos)
                save_rot.append(rot)
                save_offs.append(offs)
                plot_spike = False
                print('Cell: ' + cell_name + ' Progress: [' + str(len(save_spikes)) + '/' + str(target_num_spikes) + ']')
                saved += 1
            else:
                pass

            i += 1

        save_spikes = np.array(save_spikes)
        save_pos = np.array(save_pos)
        save_rot = np.array(save_rot)
        save_offs = np.array(save_offs)

        np.save(join(save_folder, 'eap-%s' % cell_save_name), save_spikes)
        np.save(join(save_folder, 'pos-%s' % cell_save_name), save_pos)
        np.save(join(save_folder, 'rot-%s' % cell_save_name), save_rot)

        if not os.path.isfile(join(save_folder, 'e_elpts_%d.npy' % target_num_spikes)):
            np.save(join(save_folder, 'e_elpts_%d.npy' % target_num_spikes),
                    save_offs)

        # Log information: (consider xml)
        with open(join(save_folder, 'info_%s.yaml' % cell_save_name),'w') as f:
            # create dictionary for yaml file
            data_yaml = {'General': {'cell name': cell_name, 'target spikes': target_num_spikes, 
                                     'detect threshold': threshold_detect, 'NEURON': neuron.h.nrnversion(1),
                                     'LFPy': LFPy.__version__ , 'dt': dt},
                        'Electrodes': elinfo,
                        'Location': {'z_lim': list(z_lim),'y_lim': list(y_lim), 'x_lim': list(x_lim), 'rotation': rotation}
                        }
            yaml.dump(data_yaml, f, default_flow_style=False)


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
        polarlim = {'BP': [0.,15.],
                    'BTC': None, # [0.,15.],
                    'ChC': None, # [0.,15.],
                    'DBC': None, # [0.,15.],
                    'LBC': None, # [0.,15.],
                    'MC': [0.,15.],
                    'NBC': None,
                    'NGC': None,
                    'SBC': None,
                    'STPC': [0.,15.],
                    'TTPC1': [0.,15.],
                    'TTPC2': [0.,15.],
                    'UTPC': [0.,15.]}
        # how it's implemented, the NMC y axis points into the pref_orient direction after rotation
        pref_orient = {'BP': [0.,0.,1.],
                       'BTC': None, # [0.,0.,1.],
                       'ChC': None, # [0.,0.,1.],
                       'DBC': None, # [0.,0.,1.],
                       'LBC': None, # [0.,0.,1.],
                       'MC': [0.,0.,1.],
                       'NBC': None,
                       'NGC': None,
                       'SBC': None,
                       'STPC': [0.,0.,1.],
                       'TTPC1': [0.,0.,1.],
                       'TTPC2': [0.,0.,1.],
                       'UTPC': [0.,0.,1.]}
        return polarlim[cell_name.split('_')[1]], pref_orient[cell_name.split('_')[1]]
    else:
        raise NotImplementedError('Cell model %s is not implemented'\
                                  % model_type)

def return_extracellular_spike(cell, cell_name, model_type,\
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

    def get_xyz_angles(R):
        ''' Get rotation angles for each axis from rotation matrix
        
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
        '''
        rot_x = np.arctan2(R[2,1],R[2,2])
        rot_y = np.arcsin(-R[2,0])
        rot_z = np.arctan2(R[1,0],R[0,0])
        return rot_x,rot_y,rot_z

    def get_rnd_rot_Arvo():
        """ Generate uniformly distributed random rotation matrices
        see: 'Fast Random Rotation Matrices' by Arvo (1992)
        
        Returns:
        --------
        R : 3x3 matrix
            random rotation matrix
        """
        gamma = np.random.uniform(0,2.*np.pi)
        rotation_z = np.matrix([[np.cos(gamma), -np.sin(gamma), 0],
                                [np.sin(gamma), np.cos(gamma), 0],
                                [0, 0, 1]])
        x = np.random.uniform(size=2)
        v = np.array([np.cos(2.*np.pi*x[0])*np.sqrt(x[1]),
                      np.sin(2.*np.pi*x[0])*np.sqrt(x[1]),
                      np.sqrt(1-x[1])])
        H = np.identity(3)-2.*np.outer(v,v)
        M = -np.dot(H,rotation_z)
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
        postest = np.dot(matrix,pre)
        c=np.dot(post/np.linalg.norm(post),postest/np.linalg.norm(postest))
        if np.cos(np.deg2rad(polarlim[1])) <= c <= np.cos(np.deg2rad(polarlim[0])):
            return True
        else:
            return False

    electrodes = LFPy.RecExtElectrode(cell, **electrode_parameters)

    '''Rotate neuron'''
    if rotation == 'Norot':
        # orientate cells in z direction
        if model_type == 'bbp':
            x_rot = np.pi / 2.
            y_rot = 0
            z_rot = 0
    elif rotation == '3drot':
        if model_type == 'bbp':
            x_rot_offset = np.pi / 2. # align neuron with z axis
            y_rot_offset = 0 # align neuron with z axis
            z_rot_offset = 0 # align neuron with z axis

        x_rot, y_rot, z_rot = get_xyz_angles(np.array(get_rnd_rot_Arvo()))
        x_rot = x_rot + x_rot_offset
        y_rot = y_rot + y_rot_offset
        z_rot = z_rot + z_rot_offset

    elif rotation == 'physrot':
        polarlim, pref_orient  = get_physrot_specs(cell_name, model_type)
        if model_type == 'bbp':
            x_rot_offset = np.pi / 2. # align neuron with z axis
            y_rot_offset = 0 # align neuron with z axis
            z_rot_offset = 0 # align neuron with z axis
        while True:
            R = np.array(get_rnd_rot_Arvo())
            if polarlim is None or pref_orient is None:
                valid = True
            else:
                valid = check_solidangle(R,[0.,0.,1.],pref_orient,polarlim)
            if valid:
                x_rot,y_rot,z_rot = get_xyz_angles(R)
                x_rot = x_rot + x_rot_offset
                y_rot = y_rot + y_rot_offset
                z_rot = z_rot + z_rot_offset
                break
    else:
        x_rot = 0
        y_rot = 0
        z_rot = 0

    '''Move neuron randomly'''
    x_rand = np.random.uniform(limits[0][0], limits[0][1])
    y_rand = np.random.uniform(limits[1][0], limits[1][1])
    z_rand = np.random.uniform(limits[2][0], limits[2][1])


    if pos == None:
        cell.set_pos(x_rand, y_rand, z_rand)
    else:
        cell.set_pos(pos[0], pos[1], pos[2])
    cell.set_rotation(x=x_rot, y=y_rot, z=z_rot)
    pos = [x_rand, y_rand, z_rand]
    rot = [x_rot, y_rot, z_rot]

    electrodes.calc_lfp()

    # Reverse rotation to bring cell back into initial rotation state
    rev_rot = [-rot[e] for e in range(len(rot))]
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

    if len(sys.argv) == 2 and sys.argv[1] == 'compile':
            compile_all_mechanisms()
            sys.exit(0)
    elif len(sys.argv) == 8:
        cell_folder, model, numb, only_intracellular, rotation, probe, nobs = sys.argv[1:]
        only_intracellular = str2bool(only_intracellular)
        params_path = None
    elif len(sys.argv) == 9:
        cell_folder, model, numb, only_intracellular, rotation, probe, nobs, params_path = sys.argv[1:]
        only_intracellular = str2bool(only_intracellular)
    else: 
        raise RuntimeError("Wrong usage. Give arguments: \n \
        \t 'compile' to compile mechanisms \n \
        \n\t or for cell simulation: \n \
        \t <cell name> \t path to cell model files\n \
        \t <model_type> \t cell model type (here 'bbp')\n \
        \t <cell id> \t arbitrary cell id, used to set the numpy.random seed\n \
        \t <only_intra> \t (bool) whether to simulate only the intracellular potential \n \
        \t <rotation> \t specifies neuron-MEA alignment ('Norot','physrot','3drot')\n \
        \t <probe> \t MEA probe name (corresponding to the json file in electrodes directory)\n \
        \t <nobs> \t number of EAP observations to simulate")

    data_dir = os.path.dirname(os.path.abspath(__file__))

    extra_sim_folder = join(data_dir, 'templates', model)
    vm_im_sim_folder = join(data_dir, 'templates', model, 'Vm_Im')

    cell = run_cell_model(cell_folder, model, vm_im_sim_folder, int(numb))

    if not only_intracellular:
        calc_extracellular(cell_folder, model, extra_sim_folder, vm_im_sim_folder, rotation, int(numb), probe, nobs,
                           params_path)


