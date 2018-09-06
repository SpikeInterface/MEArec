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

only_intracellular = True
model_folder = 'cell_models'
template_folder = 'templates'
cells = 'all'

@click.command()
@click.option('--model', '-m', default='bbp',
              help='cell model name (default=bbp)')
@click.option('--rot', '-r', default='physrot',
              help='possible rotation arguments: Norot-physrot-3drot (default=physrot)')
@click.option('--probe', '-p', default=None,
              help='probe name from available electrodes (default=None)')
@click.option('--intraonly', '-i', default=False, type=bool,
              help='if True it only simulate intracellular (default=False)')
@click.option('--n', '-n', default=1000,
              help='number of observations per cell type (default=1000)')
@click.option('--target-spikes', '-ts', default=[10, 50], nargs=2, type=int,
              help='target spikes in 1 second - number of intracellular spikes per cell model is the first value '
                   '(default=[10,50])')
@click.option('--ncontacts', '-nc', default=1,
              help='number of contacts per electrode (default=1)')
@click.option('--overhang', '-ov', default=30.,
              help='extension (um) beyopnd MEA boundaries for neuron locations (default=30.)')
@click.option('--xplane', '-xp', default=0,
              help='x_plane (um) coordinate for MEA (default=0)')
@click.option('--xlim', '-xl', default=[10,80],  nargs=2, type=float,
              help='limits ( low high ) for neuron locations in the x-axis (depth) (default=[10.,80.])')
@click.option('--det-thresh', '-dt', default=30.,
              help='detection threshold for EAPs (default=30)')
@click.option('--params', '-prm', default=None,
              help='path to params.yaml (otherwise default params are used)')
def run(model, rot, probe, intraonly, n, ncontacts, overhang, xplane, xlim, det_thresh, target_spikes, params):
    """Generates EAP templates on multi-electrode arrays using biophyical NEURON simulations and LFPy"""
    if model == 'bbp':
        cell_models = [f for f in os.listdir(join(model_folder, model)) if 'mods' not in f]
    else:
        raise Exception('Only Blue Brain Project cells can be used')

    if probe==None:
        MEA.return_mea()
        return

    # Compile NEURON models (nrnivmodl)
    if not os.path.isdir(join(model_folder, model, 'mods')):
        print('Compiling NEURON models')
        os.system('python simulate_cells.py compile')

    # Retrieve params file
    if params is None:
        with open('tmp_params.yaml', 'w') as tf:
            params_dict = {'ncontacts': ncontacts, 'overhang': overhang, 'x_plane': xplane, 'x_lim': xlim,
                           'threshold_detect': det_thresh, 'target_spikes': target_spikes}
            yaml.dump(params_dict, tf)
        params = 'tmp_params.yaml'

    # Simulate neurons and EAP for different cell models sparately
    for numb, cell_model in enumerate(cell_models):
        print('\n\n', cell_model, numb + 1, '/', len(cell_models), '\n\n')
        os.system('python simulate_cells.py %s %s %d %r %s %s %d %s'\
                  % (join(model_folder, model, cell_model), model, numb, intraonly, rot, probe, n, params))

    if os.path.isfile('tmp_params.yaml'):
        os.remove('tmp_params.yaml')

    # Merge simulated data and cleanup
    tmp_folder = join(template_folder, model, rot, 'tmp_%d_%s' % (n, probe))
    templates, locations, rotations, celltypes, loaded_cat, info = load_tmp_eap(tmp_folder)
    save_folder = join(template_folder, model, rot, 'templates_%d_%s_%s' % (n, probe, time.strftime("%d-%m-%Y")))
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    np.save(join(save_folder, 'templates'), templates)
    np.save(join(save_folder, 'locations'), locations)
    np.save(join(save_folder, 'rotations'), rotations)
    np.save(join(save_folder, 'celltypes'), celltypes)
    yaml.dump(info, open(join(save_folder, 'info.yaml'), 'w'))
    shutil.rmtree(tmp_folder)


if __name__ == '__main__':
    run()
