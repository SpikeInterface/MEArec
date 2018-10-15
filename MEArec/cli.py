import os
from os.path import join
import time
import MEAutility as MEA
import click
import numpy as np
import yaml
import shutil
import MEArec.generators as generators
from MEArec import recordings_to_hdf5, templates_to_hdf5, hdf5_to_recordings, hdf5_to_templates
import pprint
import time


def getDefaultConfig():
    this_dir, this_filename = os.path.split(__file__)
    home = os.path.expanduser("~")
    mearec_home = join(home, '.config', 'mearec')
    if not os.path.isdir(mearec_home):
        os.makedirs(mearec_home)
        shutil.copytree(join(this_dir, 'default_params'), join(mearec_home, 'default_params'))
        default_info = {'templates_params': join(mearec_home, 'default_params', 'templates_params.yaml'),
                        'recordings_params': join(mearec_home, 'default_params', 'recordings_params.yaml'),
                        'templates_folder': join(mearec_home, 'templates'),
                        'recordings_folder': join(mearec_home, 'recordings'),
                        'cell_models_folder': ''}
        with open(join(mearec_home, 'mearec.conf'), 'w') as f:
            yaml.dump(default_info, f)
    else:
        with open(join(mearec_home, 'mearec.conf'), 'r') as f:
            default_info = yaml.load(f)
    return default_info, mearec_home

@click.group()
def cli():
    """MEArec: Fast and customizable simulation of extracellular recordings on Multi-Electrode-Arrays """
    pass


@cli.command()
@click.option('--params', '-prm', default=None,
              help='path to default_params.yaml (otherwise default default_params are used and some of the parameters'
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
@click.option('--offset', '-off', default=None, type=float,
              help='x_plane (um) coordinate for MEA (default=0)')
@click.option('--xlim', '-xl', default=None,  nargs=2, type=float,
              help='limits ( low high ) for neuron locations in the x-axis (depth) (default=[10.,80.])')
@click.option('--ylim', '-yl', default=None,  nargs=2, type=float,
              help='limits ( low high ) for neuron locations in the y-axis (default=None)')
@click.option('--xlim', '-xl', default=None,  nargs=2, type=float,
              help='limits ( low high ) for neuron locations in the z-axis (default=None)')
@click.option('--det-thresh', '-dt', default=None, type=float,
              help='detection threshold for EAPs (default=30)')
@click.option('--seed', '-s', default=None, type=int,
              help='random seed for template generation (int)')
@click.option('--intraonly', '-io', is_flag=True,
              help='only run intracellular simulations')
@click.option('--parallel', '-par', is_flag=True,
              help='run with multiprocessing tool')
def gen_templates(params, **kwargs):
    """Generates EAP templates on multi-electrode arrays using biophyical NEURON simulations and LFPy"""
    this_dir, this_filename = os.path.split(__file__)
    info, config_folder = getDefaultConfig()

    if params is None:
        with open(info['templates_params'], 'r') as pf:
            params_dict = yaml.load(pf)
    else:
        with open(params, 'r') as pf:
            params_dict = yaml.load(pf)

    if kwargs['default'] is True:
        pprint.pprint(params_dict)
        MEA.return_mea()
        return

    if kwargs['cellfolder'] is not None:
        model_folder = kwargs['cellfolder']
    else:
        model_folder = info['cell_models_folder']
    params_dict['cell_models_folder'] = model_folder

    if kwargs['seed'] is not None:
        seed = kwargs['seed']
    else:
        seed = np.random.randint(1, 10000)
    params_dict['seed'] = seed


    if kwargs['folder'] is not None:
        params_dict['templates_folder'] = kwargs['folder']
    else:
        templates_folder = info['templates_folder']
    intraonly = kwargs['intraonly']

    if kwargs['rot'] is not None:
        params_dict['rot'] = kwargs['rot']
    if kwargs['n'] is not None:
        params_dict['n'] = kwargs['n']
    if kwargs['ncontacts'] is not None:
        params_dict['ncontacts'] = kwargs['ncontacts']
    if kwargs['overhang'] is not None:
        params_dict['overhang'] = kwargs['overhang']
    if kwargs['offset'] is not None:
        params_dict['offset'] = kwargs['offset']
    if kwargs['det_thresh'] is not None:
        params_dict['det_thresh'] = kwargs['det_thresh']
    if kwargs['probe'] is not None:
        params_dict['probe'] = kwargs['probe']
    else:
        intraonly = True
    parallel = kwargs['parallel']
    params_dict['templates_folder'] = templates_folder

    # Compile NEURON models (nrnivmodl)
    if not os.path.isdir(join(model_folder, 'mods')):
        print('Compiling NEURON models')
        os.system('python %s compile %s' % (simulate_script, model_folder))

    tempgen = generators.gen_templates(model_folder, params_dict, intraonly, parallel)

    # Merge simulated data and cleanup
    if not intraonly:
        rot = params_dict['rot']
        n = params_dict['n']
        probe = params_dict['probe']
        if kwargs['fname'] is None:
            fname = 'templates_%d_%s_%s' % (n, probe, time.strftime("%d-%m-%Y"))
        else:
            fname = kwargs['fname']
        save_folder = join(templates_folder, rot, fname)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        np.save(join(save_folder, 'templates'), tempgen.templates)
        np.save(join(save_folder, 'locations'), tempgen.locations)
        np.save(join(save_folder, 'rotations'), tempgen.rotations)
        np.save(join(save_folder, 'celltypes'), tempgen.celltypes)
        info = tempgen.info
        yaml.dump(info, open(join(save_folder, 'info.yaml'), 'w'), default_flow_style=False)
        print('\nSaved eap templates in', save_folder, '\n')



@cli.command()
@click.option('--templates', '-t', default=None,
              help='eap templates path')
@click.option('--params', '-prm', default=None,
              help='path to default_params.yaml (otherwise default default_params are used and some of the parameters can be overwritten with the following options)')
@click.option('--default', is_flag=True,
              help='shows default values for simulation')
@click.option('--fname', '-fn', default=None,
              help='recording filename')
@click.option('--folder', '-fol', default=None,
              help='recording output base folder')
@click.option('--duration', '-d', default=None, type=float,
              help='duration in s (default=10)')
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
@click.option('--min-dist', '-md', default=None, type=int,
              help='minumum distance between neuron in um (default=25)')
@click.option('--min-amp', '-ma', default=None, type=int,
              help='minumum eap amplitude in uV (default=50)')
@click.option('--fs', default=None, type=float,
              help='sampling frequency in kHz (default from templates sampling frequency)')
@click.option('--noise-lev', '-nl', default=None, type=int,
              help='noise level in uV (default=10)')
@click.option('--modulation', '-m', default=None, type=click.Choice(['none', 'template', 'electrode']),
              help='modulation type')
@click.option('--chunk', '-ch', default=None, type=float,
              help='chunk duration in s for chunk processing (default 0)')
@click.option('--noise-seed', '-nseed', default=None, type=int,
              help='random seed for noise')
@click.option('--st-seed', '-stseed', default=None, type=int,
              help='random seed for spike trains')
@click.option('--temp-seed', '-tseed', default=None, type=int,
              help='random seed for template selection')
@click.option('--no-filt', is_flag=True,
              help='if True no filter is applied')
@click.option('--overlap', is_flag=True,
              help='if True it annotates overlapping spikes')
def gen_recordings(params, **kwargs):
    """Generates recordings from TEMPLATES and SPIKETRAINS"""
    # Retrieve default_params file
    this_dir, this_filename = os.path.split(__file__)
    info, config_folder = getDefaultConfig()

    if params is None:
        with open(info['recordings_params'], 'r') as pf:
            params_dict = yaml.load(pf)
    else:
        with open(params, 'r') as pf:
            params_dict = yaml.load(pf)

    if kwargs['default'] is True:
        pprint.pprint(params_dict)
        return

    if kwargs['folder'] is not None:
        params_dict['recordings'].update({'recordings_folder': kwargs['folder']})
    else:
        params_dict['recordings'].update({'recordings_folder': info['recordings_folder']})
    recordings_folder = params_dict['recordings']['recordings_folder']

    if kwargs['templates'] is None:
        print('Provide eap templates path')
        return
    else:
        if os.path.isdir(kwargs['templates']):
            params_dict['templates'].update({'templates_folder': kwargs['templates']})
        else:
            raise AttributeError("'templates' is not a folder")

    if kwargs['n_exc'] is not None:
        params_dict['spiketrains']['n_exc'] = kwargs['n_exc']
    if kwargs['n_inh'] is not None:
        params_dict['spiketrains']['n_inh'] = kwargs['n_inh']
    if kwargs['f_exc'] is not None:
        params_dict['spiketrains']['f_exc'] = kwargs['f_exc']
    if kwargs['f_inh'] is not None:
        params_dict['spiketrains']['f_inh'] = kwargs['f_inh']
    if kwargs['st_exc'] is not None:
        params_dict['spiketrains']['st_exc'] = kwargs['st_exc']
    if kwargs['st_inh'] is not None:
        params_dict['spiketrains']['st_inh'] = kwargs['st_inh']
    if kwargs['min_rate'] is not None:
        params_dict['spiketrains']['min_rate'] = kwargs['min_rate']
    if kwargs['ref_per'] is not None:
        params_dict['spiketrains']['ref_per'] = kwargs['ref_per']
    if kwargs['process'] is not None:
        params_dict['spiketrains']['process'] = kwargs['process']
    if kwargs['min_rate'] is not None:
        params_dict['spiketrains']['min_rate'] = kwargs['min_rate']
    if kwargs['duration'] is not None:
        params_dict['spiketrains']['duration'] = kwargs['duration']
    if kwargs['tstart'] is not None:
        params_dict['spiketrains']['t_start'] = kwargs['t_start']
    if kwargs['st_seed'] is not None:
        params_dict['spiketrains']['seed'] = kwargs['seed']

    if kwargs['min_dist'] is not None:
        params_dict['templates']['min_dist'] = kwargs['min_dist']
    if kwargs['min_amp'] is not None:
        params_dict['templates']['min_amp'] = kwargs['min_amp']
    if kwargs['temp_seed'] is not None:
        params_dict['templates']['seed'] = kwargs['seed']

    if kwargs['noise_lev'] is not None:
        params_dict['recordings']['noise_lev'] = kwargs['noise_lev']
    if kwargs['modulation'] is not None:
        params_dict['recordings']['modulation'] = kwargs['modulation']
    if kwargs['chunk'] is not None:
        params_dict['recordings']['chunk_duration'] = kwargs['chunk']
    if kwargs['no_filt'] is True:
        params_dict['recordings']['filter'] = False
    if kwargs['fs'] is not None:
        params_dict['recordings']['fs'] = kwargs['fs']
    else:
        params_dict['recordings']['fs'] = None
    if kwargs['noise_seed'] is not None:
        params_dict['recordings']['seed'] = kwargs['seed']

    recgen = generators.gen_recordings(templates=kwargs['templates'], params=params_dict)
    info = recgen.info

    n_neurons = info['recordings']['n_neurons']
    electrode_name = info['recordings']['electrode_name']
    duration = info['recordings']['duration']
    noise_level = info['recordings']['noise_level']

    if kwargs['fname'] is None:
        fname = 'recordings_%dcells_%s_%s_%.1fuV_%s' % (n_neurons, electrode_name, duration,
                                                        noise_level, time.strftime("%d-%m-%Y:%H:%M"))
    else:
        fname = kwargs['fname']

    rec_path = join(recordings_folder, fname)
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
@click.argument('foldername')
@click.argument('h5file')
def temptohdf5(foldername, h5file):
    """Convert templates to hdf5"""
    if not h5file.endswith('.h5') and not h5file.endswith('.hdf5'):
        h5file = h5file + '.h5'
    templates_to_hdf5(foldername, h5file)
    print("Saved: ", h5file)

@cli.command()
@click.argument('h5file')
@click.argument('foldername')
def tempfromhdf5(h5file, foldername):
    """Convert templates from hdf5"""
    if not h5file.endswith('.h5') and not h5file.endswith('.hdf5'):
        raise AttributeError("'h5file' is not an hdf5 file")
    hdf5_to_templates(h5file, foldername)
    print("Saved: ", foldername)

@cli.command()
@click.argument('foldername')
@click.argument('h5file')
def rectohdf5(foldername, h5file):
    """Convert recordings to hdf5"""
    if not h5file.endswith('.h5') and not h5file.endswith('.hdf5'):
        h5file = h5file + '.h5'
    recordings_to_hdf5(foldername, h5file)
    print("Saved: ", h5file)

@cli.command()
@click.argument('h5file')
@click.argument('foldername')
def recfromhdf5(h5file, foldername):
    """Convert recordings from hdf5"""
    if not h5file.endswith('.h5') and not h5file.endswith('.hdf5'):
        raise AttributeError("'h5file' is not an hdf5 file")
    hdf5_to_recordings(h5file, foldername)
    print("Saved: ", foldername)

@cli.command()
def default_config():
    """Print default configurations"""
    info, config = getDefaultConfig()
    import pprint
    pprint.pprint(info)

@cli.command()
@click.argument('cell-models-folder')
def set_cell_models_folder(cell_models_folder):
    """Set default cell_models folder"""
    info, config = getDefaultConfig()
    if os.path.isdir(cell_models_folder):
        info['cell_models_folder'] = os.path.abspath(cell_models_folder)
        with open(join(config, 'mearec.conf'), 'w') as f:
            yaml.dump(info, f)
        print('Set default cell_models_folder to: ', cell_models_folder)
    else:
        print(cell_models_folder, ' is not a folder!')

@cli.command()
@click.argument('templates-folder')
@click.option('--create', is_flag=True, help='if True it creates the directory')
def set_templates_folder(templates_folder, create):
    """Set default templates output folder"""
    info, config = getDefaultConfig()
    templates_folder = os.path.abspath(templates_folder)
    if os.path.isdir(templates_folder):
        info['templates_folder'] = templates_folder
        with open(join(config, 'mearec.conf'), 'w') as f:
            yaml.dump(info, f)
        print('Set default templates_folder to: ', templates_folder)
    elif create:
        os.makedirs(templates_folder)
        info['templates_folder'] = templates_folder
        with open(join(config, 'mearec.conf'), 'w') as f:
            yaml.dump(info, f)
        print('Set default templates_folder to: ', templates_folder)
    else:
        print(templates_folder, ' is not a folder!')

@cli.command()
@click.argument('recordings-folder')
@click.option('--create', is_flag=True, help='if True it creates the directory')
def set_recordings_folder(recordings_folder, create):
    """Set default recordings output folder"""
    info, config = getDefaultConfig()
    recordings_folder = os.path.abspath(recordings_folder)
    if os.path.isdir(recordings_folder):
        info['recordings_folder'] = recordings_folder
        with open(join(config, 'mearec.conf'), 'w') as f:
            yaml.dump(info, f)
        print('Set default recordings_folder to: ', recordings_folder)
    elif create:
        os.makedirs(recordings_folder)
        info['recordings_folder'] = recordings_folder
        with open(join(config, 'mearec.conf'), 'w') as f:
            yaml.dump(info, f)
        print('Set default recordings_folder to: ', recordings_folder)
    else:
        print(recordings_folder, ' is not a folder!')


@cli.command()
@click.argument('templates-params')
def set_templates_params(templates_params):
    """Set default templates output folder"""
    info, config = getDefaultConfig()
    templates_params = os.path.abspath(templates_params)
    if os.path.isfile(templates_params) and (templates_params.endswith('yaml') or templates_params.endswith('yml')):
        info['templates_params'] = templates_params
        with open(join(config, 'mearec.conf'), 'w') as f:
            yaml.dump(info, f)
        print('Set default templates_params to: ', templates_params)
    else:
        print(templates_params, ' is not a yaml!')


@cli.command()
@click.argument('recordings-params')
def set_recordings_params(recordings_params):
    """Set default templates output folder"""
    info, config = getDefaultConfig()
    recordings_params = os.path.abspath(recordings_params)
    if os.path.isfile(recordings_params) and (recordings_params.endswith('yaml') or recordings_params.endswith('yml')):
        info['recordings_params'] = recordings_params
        with open(join(config, 'mearec.conf'), 'w') as f:
            yaml.dump(info, f)
        print('Set default recordings_params to: ', recordings_params)
    else:
        print(recordings_params, ' is not a yaml!')

