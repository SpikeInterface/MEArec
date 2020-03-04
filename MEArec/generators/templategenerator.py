import numpy as np
import time
from copy import deepcopy
from MEArec.tools import *
import MEAutility as mu
import shutil
import yaml
import os
from os.path import join
from distutils.version import StrictVersion
from pathlib import Path
from joblib import Parallel, delayed

if StrictVersion(yaml.__version__) >= StrictVersion('5.0.0'):
    use_loader = True
else:
    use_loader = False


def simulate_cell_templates(i, simulate_script, tot, cell_model,
                            model_folder, intraonly, params, verbose):
    print("Starting ", i + 1)
    print('\n\n', cell_model, i + 1, '/', tot, '\n\n')
    os.system('python %s %s %s %s %s' \
              % (simulate_script, join(model_folder, cell_model), intraonly, params, verbose))
    print("Exiting ", i + 1)


class TemplateGenerator:
    """
    Class for generation of templates called by the gen_templates function.
    The list of parameters is in default_params/templates_params.yaml.

    Parameters
    ----------
    cell_models_folder : str
        Path to folder containing Blue Brain Project cell models
    templates_folder : str
        Path to output template folder (if not in params)
    temp_dict :  dict
        Dictionary to instantiate TemplateGenerator with existing data. It contains the following fields:
          - templates : float (n_templates, n_electrodes, n_timepoints)
          - locations : float (n_templates, 3)
          - rotations : float (n_templates, 3)
          - celltypes : str (n_templates)
    info :  dict
        Info dictionary to instantiate TemplateGenerator with existing data. It contains the following fields:
          - params : dict with template generation parameters
          - electrodes : dict with probe info (from MEAutility.return_mea_info('probe-name'))
    params : dict
        Dictionary with parameters to simulate templates. Default values can be retrieved with
        mr.get_default_template_params()
    intraonly : bool
        If True, only intracellular simulations are performed
    parallel : bool
        If True, cell models are run in parallel
    recompile: bool
        If True, cell models are recompiled (suggested if new models are added)
    n_jobs: int
        If None, all cpus are used
    delete_tmp : bool
        If True, temporary files are removed
    verbose : bool
        If True, output is verbose
    """

    def __init__(self, cell_models_folder=None, templates_folder=None, temp_dict=None, info=None,
                 params=None, intraonly=False, parallel=True, recompile=False, n_jobs=None, delete_tmp=True,
                 verbose=False):
        self._verbose = verbose
        if temp_dict is not None and info is not None:
            if 'templates' in temp_dict.keys():
                self.templates = temp_dict['templates']
            if 'locations' in temp_dict.keys():
                self.locations = temp_dict['locations']
            if 'rotations' in temp_dict.keys():
                self.rotations = temp_dict['rotations']
            if 'celltypes' in temp_dict.keys():
                self.celltypes = temp_dict['celltypes']
            self.info = info
            self.params = deepcopy(info)
        else:
            if cell_models_folder is None:
                raise AttributeError("Specify cell folder!")
            if params is None:
                if self._verbose:
                    print("Using default parameters")
                self.params = {}
            else:
                self.params = deepcopy(params)
            self.cell_model_folder = cell_models_folder
            self.n_jobs = n_jobs
            self.templates_folder = templates_folder
            self.simulation_params = {'intraonly': intraonly, 'parallel': parallel, 'delete_tmp': delete_tmp,
                                      'recompile': recompile}

    def generate_templates(self):
        """
        Generate templates.
        """
        cell_models_folder = self.cell_model_folder
        templates_folder = self.templates_folder
        intraonly = self.simulation_params['intraonly']
        parallel = self.simulation_params['parallel']
        recompile = self.simulation_params['recompile']
        delete_tmp = self.simulation_params['delete_tmp']

        if os.path.isdir(cell_models_folder):
            cell_models = [f for f in os.listdir(join(cell_models_folder)) if 'mods' not in f]
            if len(cell_models) == 0:
                raise AttributeError(cell_models_folder, ' contains no cell models!')
        else:
            raise NotADirectoryError('Cell models folder: does not exist!')

        this_dir, this_filename = os.path.split(__file__)
        simulate_script = str(Path(this_dir).parent / 'simulate_cells.py')
        self.params['cell_models_folder'] = cell_models_folder
        self.params['templates_folder'] = templates_folder

        # Compile NEURON models (nrnivmodl)
        if not os.path.isdir(join(cell_models_folder, 'mods')) or recompile:
            if self._verbose:
                print('Compiling NEURON models')
            os.system('python %s compile %s' % (simulate_script, cell_models_folder))

        if 'sim_time' not in self.params.keys():
            self.params['sim_time'] = 1
        if 'target_spikes' not in self.params.keys():
            self.params['target_spikes'] = [3, 50]
        if 'cut_out' not in self.params.keys():
            self.params['cut_out'] = [2, 5]
        if 'dt' not in self.params.keys():
            self.params['dt'] = 2 ** -5
        if 'delay' not in self.params.keys():
            self.params['delay'] = 10
        if 'weights' not in self.params.keys():
            self.params['weights'] = [0.25, 1.75]

        if 'rot' not in self.params.keys():
            self.params['rot'] = 'physrot'
        if 'probe' not in self.params.keys():
            available_mea = mu.return_mea_list()
            probe = available_mea[np.random.randint(len(available_mea))]
            if self._verbose:
                print("Probe randomly set to: %s" % probe)
            self.params['probe'] = probe
        if 'ncontacts' not in self.params.keys():
            self.params['ncontacts'] = 1
        if 'overhang' not in self.params.keys():
            self.params['overhang'] = 1
        if 'xlim' not in self.params.keys():
            self.params['xlim'] = [10, 80]
        if 'ylim' not in self.params.keys():
            self.params['ylim'] = None
        if 'zlim' not in self.params.keys():
            self.params['zlim'] = None
        if 'offset' not in self.params.keys():
            self.params['offset'] = 0
        if 'det_thresh' not in self.params.keys():
            self.params['det_thresh'] = 30
        if 'n' not in self.params.keys():
            self.params['n'] = 50
        if 'min_amp' not in self.params.keys():
            self.params['min_amp'] = 30
        if 'seed' not in self.params.keys():
            self.params['seed'] = np.random.randint(1, 10000)
        elif self.params['seed'] is None:
            self.params['seed'] = np.random.randint(1, 10000)
        if templates_folder is None:
            self.params['templates_folder'] = os.getcwd()
            templates_folder = self.params['templates_folder']
        else:
            self.params['templates_folder'] = templates_folder
        if 'drifting' not in self.params.keys():
            self.params['drifting'] = False
        if 'max_drift' not in self.params.keys():
            self.params['max_drift'] = 100
        if 'min_drift' not in self.params.keys():
            self.params['min_drift'] = 30
        if 'drift_steps' not in self.params.keys():
            self.params['drift_steps'] = 10
        if 'drift_xlim' not in self.params.keys():
            self.params['drift_xlim'] = [-10, 10]
        if 'drift_ylim' not in self.params.keys():
            self.params['drift_ylim'] = [-10, 10]
        if 'drift_zlim' not in self.params.keys():
            self.params['drift_zlim'] = [20, 80]

        rot = self.params['rot']
        n = self.params['n']
        probe = self.params['probe']

        tmp_params_path = 'tmp_params_path'
        with open(tmp_params_path, 'w') as f:
            yaml.dump(self.params, f)

        # Simulate neurons and EAP for different cell models separately
        if parallel and self.n_jobs not in (0, 1):
            start_time = time.time()
            tot = len(cell_models)
            if self.n_jobs is None:
                n_jobs = tot
            else:
                n_jobs = self.n_jobs

            if self._verbose:
                print('Running with', n_jobs, 'jobs')

            Parallel(n_jobs=n_jobs)(delayed(simulate_cell_templates)(i, simulate_script, tot, cell_model,
                                                                     cell_models_folder, intraonly, tmp_params_path,
                                                                     self._verbose, )
                                    for i, cell_model in enumerate(cell_models))
            print('\n\n\nSimulation time: ', time.time() - start_time, '\n\n\n')
        else:
            # TODO try without subprocess
            start_time = time.time()
            for numb, cell_model in enumerate(cell_models):
                if self._verbose:
                    print('\n\n', cell_model, numb + 1, '/', len(cell_models), '\n\n')
                os.system('python %s %s %s %s %s' \
                          % (simulate_script, join(cell_models_folder, cell_model), intraonly, tmp_params_path,
                             self._verbose))
            print('\n\n\nSimulation time: ', time.time() - start_time, '\n\n\n')

        tmp_folder = join(templates_folder, rot, 'tmp_%d_%s' % (n, probe))

        if not Path(tmp_folder).is_dir():
            raise FileNotFoundError(f'{tmp_folder} not found. Something went wrong in the template generation phase.')

        templates, locations, rotations, celltypes = load_tmp_eap(tmp_folder)
        if delete_tmp:
            shutil.rmtree(tmp_folder)
            os.remove(tmp_params_path)

        self.info = {}

        self.templates = templates
        self.locations = locations
        self.rotations = rotations
        self.celltypes = celltypes
        self.info['params'] = self.params
        self.info['electrodes'] = mu.return_mea_info(probe)