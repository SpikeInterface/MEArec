# coding: utf-8
import os
from os.path import join
import time
import MEAutility as MEA
import click
import numpy as np
import yaml
import shutil
from .tools import *
from .generators import SpikeTrainGenerator
from .generators import RecordingGenerator
from .utils.h5tools import *


@click.group()
def cli():
    """MEArec: Fast and customizable simulation of extracellular recordings on Multi-Electrode-Arrays """
    pass


@cli.command()
@click.option('--params', '-prm', default=None,
              help='path to params.yaml (otherwise default params are used and some of the parameters'
                   'can be overwritten with the following options)')
@click.option('--default', is_flag=True,
              help='shows default values for simulation')
@click.option('--fname', '-fn', default=None,
              help='template filename')
@click.option('--folder', '-fol', default=None,
              help='templates output base folder')
@click.option('--cellfolder', '-cf', default=None,
              help='folder containing bbp cell models')
@click.option('--rot', '-r', default=None,
              help='possible rotation arguments: Norot-physrot-3drot (default=physrot)')
@click.option('--probe', '-p', default=None,
              help='probe name from available electrodes (default=None)')
@click.option('--intraonly', '-i', default=False, type=bool, is_flag=True,
              help='if True it only simulate intracellular (default=False)')
@click.option('--n', '-n', default=None, type=int,
              help='number of observations per cell type (default=1000)')
@click.option('--ncontacts', '-nc', default=None, type=int,
              help='number of contacts per electrode (default=1)')
@click.option('--overhang', '-ov', default=None, type=float,
              help='extension (um) beyond MEA boundaries for neuron locations (default=30.)')
@click.option('--xplane', '-xp', default=None, type=float,
              help='x_plane (um) coordinate for MEA (default=0)')
@click.option('--xlim', '-xl', default=None,  nargs=2, type=float,
              help='limits ( low high ) for neuron locations in the x-axis (depth) (default=[10.,80.])')
@click.option('--det-thresh', '-dt', default=None, type=float,
              help='detection threshold for EAPs (default=30)')
@click.option('--intraonly', '-io', is_flag=True,
              help='only run intracellular simulations')
def gen_templates(params, **kwargs):
    """Generates EAP templates on multi-electrode arrays using biophyical NEURON simulations and LFPy"""
    # Retrieve params file
    if params is None:
        with open(join('params/template_params.yaml'), 'r') as pf:
            params_dict = yaml.load(pf)
    else:
        with open(params, 'r') as pf:
            params_dict = yaml.load(pf)

    if kwargs['default'] is True:
        print(params_dict)
        MEA.return_mea()
        return

    if kwargs['cellfolder'] is not None:
        model_folder = kwargs['cellfolder']
    else:
        model_folder = params_dict['cell_folder']
    cell_models = [f for f in os.listdir(join(model_folder)) if 'mods' not in f]

    if kwargs['folder'] is not None:
        params_dict['template_folder'] = kwargs['folder']
    template_folder = params_dict['template_folder']
    intraonly = kwargs['intraonly']

    if kwargs['rot'] is not None:
        params_dict['rot'] = kwargs['rot']
    if kwargs['n'] is not None:
        params_dict['n'] = kwargs['n']
    if kwargs['ncontacts'] is not None:
        params_dict['ncontacts'] = kwargs['ncontacts']
    if kwargs['overhang'] is not None:
        params_dict['overhang'] = kwargs['overhang']
    if kwargs['xplane'] is not None:
        params_dict['xplane'] = kwargs['xplane']
    if kwargs['det_thresh'] is not None:
        params_dict['det_thresh'] = kwargs['det_thresh']
    if kwargs['probe'] is not None:
        params_dict['probe'] = kwargs['probe']
    else:
        intraonly = True

    with open('tmp_params.yaml', 'w') as tf:
        yaml.dump(params_dict, tf)
    params = 'tmp_params.yaml'

    # Compile NEURON models (nrnivmodl)
    if not os.path.isdir(join(model_folder, 'mods')):
        print('Compiling NEURON models')
        os.system('python simulate_cells.py compile %s' % model_folder)

    # Simulate neurons and EAP for different cell models sparately
    for numb, cell_model in enumerate(cell_models):
        print('\n\n', cell_model, numb + 1, '/', len(cell_models), '\n\n')
        os.system('python simulate_cells.py %s %s %s'\
                  % (join(model_folder, cell_model), intraonly, params))

    if os.path.isfile('tmp_params.yaml'):
        os.remove('tmp_params.yaml')

    # Merge simulated data and cleanup
    if not intraonly:
        rot = params_dict['rot']
        n = params_dict['n']
        probe = params_dict['probe']
        if kwargs['fname'] is None:
            fname = 'templates_%d_%s_%s' % (n, probe, time.strftime("%d-%m-%Y"))
        else:
            fname = kwargs['fname']
        tmp_folder = join(template_folder, rot, 'tmp_%d_%s' % (n, probe))
        templates, locations, rotations, celltypes, info = load_tmp_eap(tmp_folder)
        save_folder = join(template_folder, rot, fname)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        np.save(join(save_folder, 'templates'), templates)
        np.save(join(save_folder, 'locations'), locations)
        np.save(join(save_folder, 'rotations'), rotations)
        np.save(join(save_folder, 'celltypes'), celltypes)
        info.update({'Params': params_dict})
        yaml.dump(info, open(join(save_folder, 'info.yaml'), 'w'), default_flow_style=False)
        shutil.rmtree(tmp_folder)
        print('\nSaved eap templates in', save_folder, '\n')


@cli.command()
@click.option('--params', '-prm', default=None,
              help='path to params.yaml (otherwise default params are used)')
@click.option('--default', is_flag=True,
              help='shows default values for simulation')
@click.option('--fname', '-fn', default=None,
              help='spike train filename')
@click.option('--folder', '-fol', default=None,
              help='spike train output base folder')
@click.option('--n-exc', '-ne', default=None, type=int,
              help='number of excitatory cells (default=15)')
@click.option('--n-inh', '-ni', default=None, type=int,
              help='number of inhibitory cells (default=5)')
@click.option('--f-exc', '-fe', default=None, type=float,
              help='average firing rate of excitatory cells in Hz (default=5)')
@click.option('--f-inh', '-fi', default=None, type=float,
              help='average firing rate of inhibitory cells in Hz (default=15)')
@click.option('--st-exc', '-se', default=None, type=float,
              help='firing rate standard deviation of excitatory cells in Hz (default=1)')
@click.option('--st-inh', '-si', default=None, type=float,
              help='firing rate standard deviation of inhibitory cells in Hz (default=3)')
@click.option('--min-rate', '-mr', default=None, type=float,
              help='minimum firing rate (default=0.5)')
@click.option('--ref-per', '-rp', default=None, type=float,
              help='refractory period in ms (default=2)')
@click.option('--process', '-p', default='poisson', type=click.Choice(['poisson', 'gamma']),
              help='process for generating spike trains (default=poisson)')
@click.option('--tstart', default=None, type=float,
              help='start time in s (default=0)')
@click.option('--duration', '-d', default=None, type=float,
              help='duration in s (default=10)')
def gen_spiketrains(params, **kwargs):
    """Generates spike trains for recordings"""
    # Retrieve params file
    if params is None:
        with open(join('params/spiketrain_params.yaml'), 'r') as pf:
            params_dict = yaml.load(pf)
    else:
        with open(params, 'r') as pf:
            params_dict = yaml.load(pf)

    if kwargs['default'] is True:
        print(params_dict)
        return

    if kwargs['folder'] is not None:
        params_dict['spiketrain_folder'] = kwargs['folder']
    spiketrain_folder = params_dict['spiketrain_folder']

    if kwargs['n_exc'] is not None:
        params_dict['n_exc'] = kwargs['n_exc']
    if kwargs['n_inh'] is not None:
        params_dict['n_inh'] = kwargs['n_inh']
    if kwargs['f_exc'] is not None:
        params_dict['f_exc'] = kwargs['f_exc']
    if kwargs['f_inh'] is not None:
        params_dict['f_inh'] = kwargs['f_inh']
    if kwargs['st_exc'] is not None:
        params_dict['st_exc'] = kwargs['st_exc']
    if kwargs['st_inh'] is not None:
        params_dict['st_inh'] = kwargs['st_inh']
    if kwargs['min_rate'] is not None:
        params_dict['min_rate'] = kwargs['min_rate']
    if kwargs['ref_per'] is not None:
        params_dict['ref_per'] = kwargs['ref_per']
    if kwargs['process'] is not None:
        params_dict['process'] = kwargs['process']
    if kwargs['min_rate'] is not None:
        params_dict['min_rate'] = kwargs['min_rate']
    if kwargs['tstart'] is not None:
        params_dict['t_start'] = kwargs['tstart']
    if kwargs['duration'] is not None:
        params_dict['duration'] = kwargs['duration']

    info = {'Params': params_dict}
    yaml.dump(info, open('tmp_info.yaml', 'w'), default_flow_style=False)

    spgen = SpikeTrainGenerator(params_dict)
    spgen.generate_spikes()

    spiketrains = spgen.all_spiketrains
    n_neurons = len(spiketrains)

    if kwargs['fname'] is None:
        fname = 'spiketrains_%d_%s' % (n_neurons, time.strftime("%d-%m-%Y:%H:%M"))
    else:
        fname = kwargs['fname']
    save_folder = join(spiketrain_folder, fname)

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    np.save(join(save_folder, 'gtst'), spiketrains)
    shutil.move('tmp_info.yaml', join(save_folder, 'info.yaml'))
    print('\nSaved spike trains in', save_folder, '\n')


@cli.command()
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
@click.option('--folder', '-fol', default=None,
              help='recording output base folder')
@click.option('--fs', default=None, type=float,
              help='sampling frequency in kHz (default from templates sampling frequency)')
@click.option('--min-dist', '-md', default=None, type=int,
              help='minumum distance between neuron in um (default=25)')
@click.option('--min-amp', '-ma', default=None, type=int,
              help='minumum eap amplitude in uV (default=50)')
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
@click.option('--overlap', is_flag=True,
              help='if True it annotates overlapping spikes')
def gen_recordings(params, **kwargs):
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

    if kwargs['folder'] is not None:
        params_dict['recording_folder'] = kwargs['folder']
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
    if kwargs['min_amp'] is not None:
        params_dict['min_amp'] = kwargs['min_amp']
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

    overlap = kwargs['overlap']

    recgen = RecordingGenerator(template_folder, spiketrain_folder, params_dict, overlap)
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
    np.save(join(rec_path, 'times'), recgen.times)
    np.save(join(rec_path, 'positions'), recgen.positions)
    np.save(join(rec_path, 'templates'), recgen.templates)
    np.save(join(rec_path, 'spiketrains'), recgen.spiketrains)
    np.save(join(rec_path, 'sources'), recgen.sources)
    np.save(join(rec_path, 'peaks'), recgen.peaks)

    with open(join(rec_path, 'info.yaml'), 'w') as f:
        yaml.dump(info, f, default_flow_style=False)

    print('\nSaved recordings in', rec_path, '\n')


@cli.command()
@click.option('--type', '-t', type=click.Choice(['t', 's', 'r']),
              help='template (t), spike train (s), or recording (r)')
@click.option('--folder', '-f', default=None,
              help='template/spiketrain/recording folder')
@click.option('--fname', '-fn', default=None,
              help='output filename path')
def tohdf5(type, folder, fname):
    """Convert templates spike trains, and recordings to hdf5"""
    pass

@cli.command()
@click.option('--type', '-t', type=click.Choice(['t', 's', 'r']),
              help='template (t), spike train (s), or recording (r)')
@click.option('--h5file', '-h', default=None,
              help='template/spiketrain/recording hdf5/h5 folder')
@click.option('--fname', '-fn', default=None,
              help='output folder path')
def fromhdf5(type, h5file, fname):
    """Convert templates spike trains, and recordings from hdf5"""
    pass
