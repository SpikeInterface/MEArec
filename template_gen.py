from __future__ import print_function

"""
Generates EAP templates from Blue Brain Project cells in the cell_models folder in random positions and rotations.
"""


import os
from os.path import join
import time
import MEAutility as MEA
import click
import numpy as np
import yaml
import shutil
from tools import load_tmp_eap


@click.command()
@click.option('--params', '-prm', default=None,
              help='path to params.yaml (otherwise default params are used and some of the parameters'
                   'can be overwritten with the following options)')
@click.option('--default', is_flag=True,
              help='shows default values for simulation')
@click.option('--fname', '-fn', default=None,
              help='template filename')
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
def run(params, **kwargs):
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

    model_folder = params_dict['cell_folder']
    cell_models = [f for f in os.listdir(join(model_folder)) if 'mods' not in f]
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
        os.system('python simulate_cells.py compile')

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
        templates, locations, rotations, celltypes, loaded_cat, info = load_tmp_eap(tmp_folder)
        save_folder = join(template_folder, rot, fname)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        np.save(join(save_folder, 'templates'), templates)
        np.save(join(save_folder, 'locations'), locations)
        np.save(join(save_folder, 'rotations'), rotations)
        np.save(join(save_folder, 'celltypes'), celltypes)
        info.update({'Params': params_dict})
        yaml.dump(info, open(join(save_folder, 'info.yaml'), 'w'))
        shutil.rmtree(tmp_folder)


if __name__ == '__main__':
    run()