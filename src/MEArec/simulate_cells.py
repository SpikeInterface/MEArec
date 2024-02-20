"""
Test implementation using cell models of the Blue Brain Project with LFPy.
The example assumes that cell models available from
https://bbpnmc.epfl.ch/nmc-portal/downloads are unzipped in the folder 'cell_models'

The function compile_all_mechanisms must be run once before any cell simulation
"""

import os
import sys
import time
from pathlib import Path

import MEAutility as mu
import numpy as np
from packaging.version import parse

from MEArec.tools import safe_yaml_load


def import_LFPy_neuron():
    try:
        import LFPy
    except:
        raise ModuleNotFoundError("LFPy is not installed. Install it with 'pip install LFPy'")

    try:
        # disable DISPLAY for subprocess
        # print("Disabling display")
        # os.environ["DISPLAY"] = ""
        import neuron
    except:
        raise ModuleNotFoundError(
            "NEURON is not installed. Install it from https://www.neuron.yale.edu/neuron/download"
        )

    if parse(LFPy.__version__) < parse("2.2"):
        raise ImportError("LFPy version must be >= 2.2. To use a previous LFPy version, downgrade MEArec to <= 1.4.1")

    return LFPy, neuron


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
    f = open("template.hoc", "r")
    for line in f.readlines():
        if "begintemplate" in line.split():
            templatename = line.split()[-1]
            continue
    return templatename


def compile_all_mechanisms(cell_folder, verbose=False):
    """Attempt to set up a folder with all unique mechanism *.mod files and
        compile them all. assumes all cell models are in a folder 'cell_models'

    Parameters
    -----------
    cell_folder : str
        Path to cell folder
    """
    cell_folder = Path(cell_folder)
    mod_folder = cell_folder / "mods"
    mod_folder.mkdir(exist_ok=True, parents=True)

    neurons = [f for f in cell_folder.iterdir() if "mods" not in str(f) and not f.name.startswith(".")]

    if verbose >= 1:
        print(neurons)

    for neuron in neurons:
        for nmodl in (neuron / "mechanisms").iterdir():
            if nmodl.suffix == ".mod":
                while not (cell_folder / "mods" / nmodl.parts[-1]).is_file():
                    if sys.platform == "win32":
                        _command = "copy"
                    else:
                        _command = "cp"
                    if verbose >= 1:
                        print(f"{_command} {nmodl} {cell_folder / 'mods'}")
                    os.system(f"{_command} {nmodl} {cell_folder / 'mods'}")
    starting_dir = os.getcwd()
    os.chdir(str(cell_folder / "mods"))
    os.system("nrnivmodl")
    os.chdir(starting_dir)


def return_bbp_cell(cell_folder, end_T, dt, start_T, verbose=0):
    """Function to load cell models

    Parameters
    ----------
    cell_folder : string
        Path to folder with the BBP cell model
    end_T : float
        Simulation length [ms]
    dt: float
        Time step of simulation [ms]
    start_T: float
        Simulation start time (recording starts at 0 ms)

    Returns
    -------
    cell : object
        LFPy cell object
    """
    LFPy, neuron = import_LFPy_neuron()
    neuron.h.load_file("stdrun.hoc")
    neuron.h.load_file("import3d.hoc")

    cwd = os.getcwd()
    os.chdir(cell_folder)
    if verbose >= 1:
        print(f"Simulating {cell_folder}")

    neuron.load_mechanisms(str(Path(cell_folder).parent / "mods"))

    f = open("template.hoc", "r")
    templatename = get_templatename(f)
    f.close()

    f = open("biophysics.hoc", "r")
    biophysics = get_templatename(f)
    f.close()

    f = open("morphology.hoc", "r")
    morphology = get_templatename(f)
    f.close()

    # get synapses template name
    synapses_file = str(Path("synapses") / "synapses.hoc")
    f = open(synapses_file, "r")
    synapses = get_templatename(f)
    f.close()

    neuron.h.load_file("constants.hoc")
    if not hasattr(neuron.h, morphology):
        neuron.h.load_file(1, "morphology.hoc")

    if not hasattr(neuron.h, biophysics):
        neuron.h.load_file(1, "biophysics.hoc")

    if not hasattr(neuron.h, synapses):
        # load synapses
        neuron.h.load_file(1, synapses_file)

    if not hasattr(neuron.h, templatename):
        neuron.h.load_file(1, "template.hoc")

    morphologyfile = [f for f in Path("morphology").iterdir()][0]

    # Instantiate the cell(s) using LFPy
    cell = LFPy.TemplateCell(
        morphology=str(morphologyfile),
        templatefile=str(Path("template.hoc").absolute()),
        templatename=templatename,
        templateargs=0,
        tstop=end_T,
        tstart=start_T,
        dt=dt,
        v_init=-70,
        pt3d=True,
        delete_sections=True,
        verbose=True,
    )

    os.chdir(cwd)
    return cell


def return_bbp_cell_morphology(cell_name, cell_folder, pt3d=False):
    """Function to load cell models

    Parameters
    ----------
    cell_name : string
        Name of the cell type.
    cell_folder : string
        Folder containing cell models.
    pt3d : bool
        If True detailed 3d morphology is used

    Returns
    -------
    cell : object
        LFPy cell object
    """
    LFPy, neuron = import_LFPy_neuron()
    cell_folder = Path(cell_folder)

    if not (cell_folder / cell_name).is_dir():
        raise NotImplementedError(f"Cell model {cell_name} is not found in {cell_folder}")

    morphology_files = [f for f in (cell_folder / cell_name / "morphology").iterdir()]
    if len(morphology_files) > 1:
        raise Exception(f"More than 1 morphology file found for cell {cell_name}")
    morphology = morphology_files[0]

    cell = LFPy.Cell(morphology=str(morphology), pt3d=pt3d)
    return cell


def find_spike_idxs(v, thresh=-30, find_max=30):
    """Find spike indices

    Parameters
    ----------
    v: array_like
        Membrane potential
    thresh: float (optional, default = -30)
        Threshold for spike detections
    find_max: int
        Number of sample to find spike maximum after detection crossing

    Returns
    -------
    spikes : array_like
        Indices of spike peaks in the positive direction, i.e. spikes
    """
    spikes_th = [idx for idx in range(len(v) - 1) if v[idx] < thresh < v[idx + 1]]
    spikes = []
    for sp in spikes_th:
        max_idx = np.argmax(v[sp : sp + find_max])
        spikes.append(sp + max_idx)
    return spikes


def set_input(weight, dt, T, cell, delay, stim_length):
    """Set current input synapse in soma

    Parameters
    ----------
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

    Returns
    -------
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
        if "soma" in sec.name():
            # syn = neuron.h.ISyn(0.5, sec=sec)
            syn = neuron.h.IClamp(0.5, sec=sec)
    syn.dur = stim_length
    syn.delay = delay  # cell.tstartms
    noiseVec.play(syn._ref_amp, dt)

    return noiseVec, cell, syn


def run_cell_model(
    cell_model_folder, verbose=False, sim_folder=None, save=True, custom_return_cell_function=None, **kwargs
):
    """
    Run simulation and adjust input strength to have a certain number of
    spikes (target_spikes[0] < num_spikes <= target_spikes[1]
    where target_spikes=[10,30] by default)

    Parameters
    ----------
    cell_model_folder : string
        Path to folder where cell model is saved.
    verbose : int
        If 1, output is verbose
    save : bool
        If True, currents and membrane potentials are saved in 'sim_folder'. If False the function returns the simulated
        cell, the soma potentials of the spikes, and the transmembrane currents of the spikes
    sim_folder : string
        Data directory for transmembrane currents and membrane potential of the neuron.
    custom_return_cell_function : function
        Python function to to return an LFPy cell from the cell_model_folder
    **kwargs : keyword arguments
        Kwargs must include: 'sim_time', 'dt', 'delay', 'weights', 'target_spikes', 'cut_out', 'seed'

    Returns
    -------
    cell : object
        LFPy cell object (if save is False)
    v : np.array
        Array (N_spikes x t) with soma membrane potential (if save is False)
    i : np.array
        Array (N_spikes x N_compartments x t) with transmembrane currents (if save is False)

    """
    cell_name = Path(cell_model_folder).parts[-1]
    if sim_folder is not None:
        sim_folder = Path(sim_folder)

    if custom_return_cell_function is None:
        return_function = return_bbp_cell
    else:
        return_function = custom_return_cell_function

    intra_params = kwargs

    if save:
        assert sim_folder is not None, "Specify 'save_sim_folder' argument!"
        sim_folder.mkdir(exist_ok=True, parents=True)

        imem_names = [f.name for f in sim_folder.iterdir() if "imem" in f.name]
        vmem_names = [f.name for f in sim_folder.iterdir() if "vmem" in f.name]
    else:
        imem_names = []
        vmem_names = []

    if not (
        np.any([cell_name in iname for iname in imem_names]) and np.any([cell_name in vname for vname in vmem_names])
    ):
        np.random.seed(intra_params["seed"])
        T = intra_params["sim_time"] * 1000
        dt = intra_params["dt"]
        cell = return_function(cell_model_folder, end_T=T, dt=dt, start_T=0)

        delay = intra_params["delay"]
        stim_length = T - delay
        weights = intra_params["weights"]
        weight = weights[0]
        target_spikes = intra_params["target_spikes"]
        cuts = intra_params["cut_out"]
        cut_out = [int(cuts[0] / dt), int(cuts[1] / dt)]

        num_spikes = 0

        i = 0
        while not target_spikes[0] < num_spikes <= target_spikes[1]:
            noiseVec, cell, syn = set_input(weight, dt, T, cell, delay, stim_length)
            cell.simulate(rec_imem=True)

            t = cell.tvec
            v = cell.somav
            t = t
            v = v

            spikes = find_spike_idxs(v[cut_out[0] : -cut_out[1]])
            spikes = list(np.array(spikes) + cut_out[0])
            num_spikes = len(spikes)

            if verbose >= 1:
                print(f"Input weight: {weight} - Num Spikes: {num_spikes}")
            if num_spikes >= target_spikes[1]:
                weight *= weights[0]
            elif num_spikes <= target_spikes[0]:
                weight *= weights[1]

            i += 1
            if i >= 10:
                sys.exit()

        t = t[0 : (cut_out[0] + cut_out[1])] - t[cut_out[0]]
        # discard first spike
        i_spikes = np.zeros((num_spikes - 1, cell.totnsegs, len(t)))
        v_spikes = np.zeros((num_spikes - 1, len(t)))

        for idx, spike_idx in enumerate(spikes[1:]):
            spike_idx = int(spike_idx)
            v_spike = v[spike_idx - cut_out[0] : spike_idx + cut_out[1]]
            i_spike = cell.imem[:, spike_idx - cut_out[0] : spike_idx + cut_out[1]]
            i_spikes[idx, :, :] = i_spike
            v_spikes[idx, :] = v_spike
        if save:
            np.save(str(sim_folder / f"imem_{num_spikes - 1}_{cell_name}.npy"), i_spikes)
            np.save(str(sim_folder / f"vmem_{num_spikes - 1}_{cell_name}.npy"), v_spikes)
        else:
            return cell, v_spikes, i_spikes
    else:
        if verbose >= 1:
            print("\n\n\nCell has already be simulated. Using stored membrane currents\n\n\n")


def calculate_extracellular_potential(cell, mea, ncontacts=10, position=None, rotation=None):
    """
    Calculates extracellular signal in uV on MEA object.

    Parameters
    ----------
    cell : LFPy Cell
        The simulated cell
    mea : MEA, str, or dict
        Mea object from MEAutility, string with probe name, or dict with probe info

    Returns
    -------
    v_ext : np.array
        Extracellular potential computed on the electrodes (n_elec, n_timestamps)
    """
    LFPy, neuron = import_LFPy_neuron()

    if isinstance(mea, str):
        mea_obj = mu.return_mea(mea)
    elif isinstance(mea, dict):
        mea_obj = mu.return_mea(info=mea)
    elif isinstance(mea, mu.core.MEA):
        mea_obj = mea
    else:
        raise Exception("")

    if ncontacts > 1:
        electrodes = LFPy.RecExtElectrode(cell, probe=mea_obj, n=ncontacts)
    else:
        electrodes = LFPy.RecExtElectrode(cell, probe=mea_obj)

    if position is not None:
        assert len(position) == 3, "'position' should be a 3d array"
        cell.set_pos(position[0], position[1], position[2])

    if rotation is not None:
        assert len(rotation) == 3, "'rotation' should be a 3d array"
        cell.set_rotation(x=rotation[0], y=rotation[1], z=rotation[2])

    lfp = electrodes.get_transformation_matrix() @ cell.imem

    # Reverse rotation to bring cell back into initial rotation state
    if rotation is not None:
        rev_rot = [-r for r in rotation]
        cell.set_rotation(rev_rot[0], rev_rot[1], rev_rot[2], rotation_order="zyx")

    return 1000 * lfp


def calc_extracellular(
    i,
    cell_model_folder,
    load_sim_folder,
    save_sim_folder=None,
    seed=0,
    verbose=0,
    position=None,
    custom_return_cell_function=None,
    cell_locations=None,
    cell_rotations=None,
    save=True,
    max_iterations=1000,
    timeout=300,
    **kwargs,
):
    """
    Loads data from previous cell simulation, and use results to generate
    arbitrary number of spikes above a certain noise level.

    Parameters
    ----------
    i: int
        Index of cell model
    cell_model_folder : string
        Path to folder where cell model is saved.
    model_type : string
        Cell model type (e.g. 'bbp')
    load_sim_folder : string
        Path to folder from which NEURON simulation results (currents, membrane potential) are loaded
    save_sim_folder : string
        Path to folder where to save EAP data
    position : array
        3D position of the soma (optional, default is None and the cell is randomly located within specified limits)
    custom_return_cell_function : function
        Python function to to return an LFPy cell from the cell_model_folder
    save : bool
        If True eaps are saved in the 'save_sim_folder'. If False eaps, positions, and rotations are returned as arrays
    cell_locations: np.array or None
        If passed, the passed locations are used (instead of random generation)
    cell_rotations: np.array or None
        If passed, the passed rotations are used (instead of random generation). Must be set if cell_locations is not
        None
    timeout : int
        Set the timeout in seconds for finding spikes over min_amp. Default 300
    max_iterations : int
        Set the maximum number of iterations for finding spikes over min_amp. Default 1000
    **kwargs: keyword arguments
        Template generation parameters (use mr.get_default_template_parameters() to retrieve the arguments)

    """
    LFPy, neuron = import_LFPy_neuron()

    for sec in neuron.h.allsec():
        neuron.h("%s{delete_section()}" % sec.name())

    cell_model_folder = Path(cell_model_folder)
    cell_name = cell_model_folder.parts[-1]
    cell_save_name = cell_name
    load_sim_folder = Path(load_sim_folder)
    if verbose >= 1:
        print(f"Seed = {seed + i}")
    np.random.seed(seed + i)

    T = kwargs["sim_time"] * 1000
    dt = kwargs["dt"]
    rotation = kwargs["rot"]
    nobs = kwargs["n"]
    ncontacts = kwargs["ncontacts"]
    overhang = kwargs["overhang"]
    x_lim = kwargs["xlim"]
    y_lim = kwargs["ylim"]
    z_lim = kwargs["zlim"]
    x_distr = kwargs["x_distr"]
    beta_distr_params = kwargs["beta_distr_params"]
    min_amp = kwargs["min_amp"]
    check_shape = kwargs["check_eap_shape"]
    MEAname = kwargs["probe"]
    drifting = kwargs["drifting"]

    if drifting:
        max_drift = kwargs["max_drift"]
        min_drift = kwargs["min_drift"]
        drift_steps = kwargs["drift_steps"]
        drift_x_lim = kwargs["drift_xlim"]
        drift_y_lim = kwargs["drift_ylim"]
        drift_z_lim = kwargs["drift_zlim"]
        check_for_drift_amp = kwargs["check_for_drift_amp"]
        drift_within_bounds = kwargs["drift_within_bounds"]

    if "timeout" in kwargs:
        timeout = kwargs.get("timeout")
    if "max_iterations" in kwargs:
        max_iterations = kwargs.get("max_iterations")

    if custom_return_cell_function is None:
        return_function = return_bbp_cell
        model_type = "bbp"
    else:
        return_function = custom_return_cell_function
        model_type = "custom"

    cuts = kwargs["cut_out"]
    cut_out = [int(cuts[0] / dt), int(cuts[1] / dt)]

    cell = return_function(cell_model_folder, end_T=T, dt=dt, start_T=0)

    # Load data from previous cell simulation
    imem_file = [f for f in load_sim_folder.iterdir() if cell_name in f.name and "imem" in f.name][0]
    vmem_file = [f for f in load_sim_folder.iterdir() if cell_name in f.name and "vmem" in f.name][0]
    i_spikes = np.load(str(imem_file))
    v_spikes = np.load(str(vmem_file))
    cell.tvec = np.arange(i_spikes.shape[-1]) * dt

    saved_eaps = []
    saved_positions = []
    saved_rotations = []
    target_num_spikes = int(nobs)

    # load MEA info
    elinfo = mu.return_mea_info(electrode_name=MEAname)

    # Create save folder
    if save:
        assert save_sim_folder is not None, "Specify 'save_sim_folder' argument!"
        save_sim_folder = Path(save_sim_folder)
        sim_folder = save_sim_folder / rotation
        save_folder = sim_folder / f"tmp_{target_num_spikes}_{MEAname}"
        save_folder.mkdir(exist_ok=True, parents=True)

    if verbose >= 1:
        print(f"Cell {cell_save_name} extracellular spikes to be simulated")

    mea = mu.return_mea(MEAname)
    if ncontacts > 1:
        electrodes = LFPy.RecExtElectrode(cell, probe=mea, n=ncontacts)
    else:
        electrodes = LFPy.RecExtElectrode(cell, probe=mea)
    pos = mea.positions
    elec_x = pos[:, 0]
    elec_y = pos[:, 1]
    elec_z = pos[:, 2]

    if x_lim is None:
        x_lim = [float(np.min(elec_x) - overhang), float(np.max(elec_x) + overhang)]
    if y_lim is None:
        y_lim = [float(np.min(elec_y) - overhang), float(np.max(elec_y) + overhang)]
    if z_lim is None:
        z_lim = [float(np.min(elec_z) - overhang), float(np.max(elec_z) + overhang)]

    # pre-simulate positions
    n_rand_positions = int(target_num_spikes * 1e5)
    print(f"Pre-generating {n_rand_positions} random positions")

    if x_distr == "uniform":
        x_rands = np.random.uniform(x_lim[0], x_lim[1], n_rand_positions)
    else:
        x_rands = x_lim[0] + np.random.beta(beta_distr_params[0], beta_distr_params[1], n_rand_positions) * (
            x_lim[1] - x_lim[0]
        )
    y_rands = np.random.uniform(y_lim[0], y_lim[1], n_rand_positions)
    z_rands = np.random.uniform(z_lim[0], z_lim[1], n_rand_positions)

    saved = 0
    i = 0
    tested_loc_idx = 0
    saved_amplitudes = []

    if cell_locations is not None:
        assert cell_rotations is not None, "If 'cell_locations' is not None, 'cell_rotations' should be given"

    if cell_locations is None:
        start_time = time.time()
        while len(saved_eaps) < target_num_spikes and tested_loc_idx < n_rand_positions:
            if i > max_iterations * target_num_spikes:
                if verbose >= 1:
                    print(f"Gave up finding spikes above noise level for {cell_name}")
                break
            # Each cell has several spikes to choose from
            spike_idx = np.random.randint(0, i_spikes.shape[0])
            cell.imem = i_spikes[spike_idx, :, :]
            cell.somav = v_spikes[spike_idx, :]
            tested_pos = [x_rands[tested_loc_idx], y_rands[tested_loc_idx], z_rands[tested_loc_idx]]

            espikes, pos, rot, found_position = return_extracellular_spike(
                cell=cell,
                cell_name=cell_name,
                model_type=model_type,
                electrodes=electrodes,
                limits=[x_lim, y_lim, z_lim],
                rotation=rotation,
                saved_pos=saved_positions,
                pos=tested_pos,
                verbose=False,
            )
            tested_loc_idx += 1
            if not found_position:
                continue

            # Method of Images for semi-infinite planes
            if elinfo["type"] == "mea":
                espikes = espikes * 2

            if not drifting:
                if check_espike(espikes, min_amp, check_shape):
                    skip = skip_duplicate(pos, saved_positions, drifting, verbose)
                    if skip:
                        continue

                    espikes = center_espike(espikes, cut_out)
                    saved_eaps.append(espikes)
                    saved_positions.append(pos)
                    saved_rotations.append(rot)
                    if verbose >= 1:
                        print(f"Cell: {cell_name} Progress: [{len(saved_eaps)}/{target_num_spikes}]")
                    saved += 1
            else:
                if check_espike(espikes, min_amp, check_shape):
                    if verbose >= 2:
                        print("template amplitude:", np.round(np.abs(np.min(espikes)), 1))

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
                        if drift_within_bounds:
                            if not (
                                x_lim[0] < x_rand < x_lim[1]
                                and y_lim[0] < y_rand < y_lim[1]
                                and z_lim[0] < z_rand < z_lim[1]
                            ):
                                if verbose == 2:
                                    print(f"Discarded for final drift position {cell_name}")
                                tr += 1
                                continue
                        if max_drift >= drift_dist >= min_drift:
                            if check_for_drift_amp:
                                # check final position spike amplitude
                                espikes, pos, rot_, found_position = return_extracellular_spike(
                                    cell=cell,
                                    cell_name=cell_name,
                                    model_type=model_type,
                                    electrodes=electrodes,
                                    limits=[x_lim, y_lim, z_lim],
                                    rotation=None,
                                    saved_pos=saved_positions,
                                    pos=final_pos,
                                )
                                if not found_position:
                                    continue

                                # Method of Images for semi-infinite planes
                                if elinfo["type"] == "mea":
                                    espikes = espikes * 2

                                if check_espike(espikes, min_amp, check_shape):
                                    if verbose == 2:
                                        print("Found final drifting position")
                                    drift_ok = True
                                else:
                                    tr += 1
                                    if verbose == 2:
                                        print(f"Discarded for final drift amplitude {cell_name}")
                                    continue
                            else:
                                drift_ok = True
                        else:
                            tr += 1
                            if verbose == 2:
                                print(f"Discarded for drift distance {cell_name}")

                    # now compute drifting templates
                    if drift_ok:
                        drift_spikes = []
                        drift_pos = []
                        drift_dir = np.array(final_pos) - np.array(init_pos)
                        for i, dp in enumerate(np.linspace(0, 1, drift_steps)):
                            pos_drift = init_pos + dp * drift_dir
                            espikes, pos, r_, found_position = return_extracellular_spike(
                                cell=cell,
                                cell_name=cell_name,
                                model_type=model_type,
                                electrodes=electrodes,
                                limits=[x_lim, y_lim, z_lim],
                                rotation=None,
                                saved_pos=saved_positions,
                                pos=pos_drift,
                            )
                            if not found_position:
                                continue

                            # Method of Images for semi-infinite planes
                            if elinfo["type"] == "mea":
                                espikes = espikes * 2
                            espikes = center_espike(espikes, cut_out)
                            drift_spikes.append(espikes)
                            drift_pos.append(pos)

                        # reverse rotation
                        rev_rot = [-r for r in rot]
                        cell.set_rotation(rev_rot[0], rev_rot[1], rev_rot[2], rotation_order="zyx")

                        drift_spikes = np.array(drift_spikes)
                        drift_pos = np.array(drift_pos)
                        if verbose == 2:
                            print(
                                f"Drift done from {np.round(init_pos, 1)} to {np.round(final_pos, 1)} um"
                                f" with {drift_steps} steps"
                            )

                        amp = np.round(np.max(np.abs(drift_spikes[0])), 3)

                        saved_amplitudes.append(amp)
                        saved_eaps.append(drift_spikes)
                        saved_positions.append(drift_pos)
                        saved_rotations.append(rot)
                        if verbose >= 1:
                            print(f"Cell: {cell_name} Progress: [{len(saved_eaps)}/{target_num_spikes}]")
                        saved += 1
                    else:
                        if verbose == 2:
                            print(f"Discarded for trials {cell_name}")
                else:
                    if verbose == 2:
                        print(f"Discarded for minimum amp {cell_name}")
                    pass
            i += 1

            if timeout is not None:
                if time.time() - start_time > timeout:
                    if verbose >= 1:
                        print(f"Timeout finding spikes above noise level for " f"{cell_name}, more than {timeout}")
                    break

    else:
        target_num_spikes = len(cell_locations)
        for loc, rot in zip(cell_locations, cell_rotations):
            # Each cell has several spikes to choose from
            spike_idx = np.random.randint(0, i_spikes.shape[0])
            cell.imem = i_spikes[spike_idx, :, :]
            cell.somav = v_spikes[spike_idx, :]
            cell.set_rotation(rot[0], rot[1], rot[2])

            if not drifting:
                if loc.ndim == 1:
                    pos = loc
                elif loc.ndim == 1:
                    # take first drifting location
                    pos = loc[0]

                espikes, pos_, rot_, found_position = return_extracellular_spike(
                    cell=cell,
                    cell_name=cell_name,
                    model_type=model_type,
                    electrodes=electrodes,
                    limits=[x_lim, y_lim, z_lim],
                    rotation=None,
                    saved_pos=saved_positions,
                    pos=pos,
                )
                if not found_position:
                    continue

                # Method of Images for semi-infinite planes
                if elinfo["type"] == "mea":
                    espikes = espikes * 2

                espikes = center_espike(espikes, cut_out)
                saved_eaps.append(espikes)
                saved_positions.append(pos)
                saved_rotations.append(rot)
            else:
                assert loc.ndim == 2
                drift_spikes = []
                for pos_drift in loc:
                    espikes, pos, r_, found_position = return_extracellular_spike(
                        cell=cell,
                        cell_name=cell_name,
                        model_type=model_type,
                        electrodes=electrodes,
                        limits=[x_lim, y_lim, z_lim],
                        rotation=None,
                        saved_pos=saved_positions,
                        pos=pos_drift,
                    )
                    if not found_position:
                        continue

                    # Method of Images for semi-infinite planes
                    if elinfo["type"] == "mea":
                        espikes = espikes * 2
                    espikes = center_espike(espikes, cut_out)
                    drift_spikes.append(espikes)

                saved_eaps.append(drift_spikes)
                saved_positions.append(loc)
                saved_rotations.append(rot)

            # reverse rotation
            rev_rot = [-r for r in rot]
            cell.set_rotation(rev_rot[0], rev_rot[1], rev_rot[2], rotation_order="zyx")

            if verbose >= 1:
                print(f"Cell: {cell_name} Progress: [{len(saved_eaps)}/{target_num_spikes}]")
            saved += 1
        else:
            pass

    if verbose >= 1:
        print(f"Done generating EAPs for {cell_name}")

    saved_eaps = np.array(saved_eaps, dtype=np.float32)
    saved_positions = np.array(saved_positions)
    saved_rotations = np.array(saved_rotations)

    if save:
        np.save(str(save_folder / f"eap-{cell_save_name}"), saved_eaps)
        np.save(str(save_folder / f"pos-{cell_save_name}"), saved_positions)
        np.save(str(save_folder / f"rot-{cell_save_name}"), saved_rotations)
    else:
        return saved_eaps, saved_positions, saved_rotations


def check_espike(espikes, min_amp, check_shape=True):
    """
    Check extracellular spike amplitude and shape (neg peak > pos peak)

    Parameters
    ----------
    espike: np.array
        EAP (n_elec, n_samples)
    min_amp: float
        Minimum amplitude
    check_shape: bool
        If True, it checks that the minimum peak is larger than the maximum peak

    Returns
    -------
    valid: bool
        If True EAP is valid
    """
    if np.abs(np.min(espikes)) < min_amp:
        return False
    elif check_shape and np.abs(np.min(espikes)) < np.abs(np.max(espikes)):
        return False
    else:
        return True


def center_espike(espike, cut_out_samples, tol=1):
    """
    Centers extracellular spike if the peak is not aligned.

    Parameters
    ----------
    espike : np.array
        EAP (n_elec, n_samples)
    cut_out_samples : list
        Samples before and after the peak
    tol : int
        Tolerance in number of samples

    Returns
    -------
    espike_centered: np.array
        Centered EAP (n_elec, n_samples)
    """
    expexted_peak = cut_out_samples[0]
    peak_idx = np.unravel_index(np.argmin(espike), espike.shape)[1]

    if np.abs(expexted_peak - peak_idx) >= tol:
        if expexted_peak - peak_idx < 0:
            diff = peak_idx - expexted_peak
            cent_espike = np.zeros_like(espike)
            cent_espike[:, :-diff] = espike[:, diff:]
            cent_espike[:, -diff:] = np.tile(espike[:, -1, np.newaxis], [1, diff])
        elif expexted_peak - peak_idx > 0:
            diff = expexted_peak - peak_idx
            cent_espike = np.zeros_like(espike)
            cent_espike[:, diff:] = espike[:, :-diff]
            cent_espike[:, :diff] = np.tile(espike[:, 0, np.newaxis], [1, diff])
    else:
        cent_espike = espike

    cent_peak_idx = np.unravel_index(np.argmin(cent_espike), cent_espike.shape)[1]
    assert np.abs(expexted_peak - cent_peak_idx) < tol, "Something went wrong in centering the spike"

    return cent_espike


def skip_duplicate(pos, saved_positions, drifting, verbose=False):
    """
    Checks if a position has to be skipped because already used.

    Parameters
    ----------
    pos: 3d array
        The position to be tested
    saved_positions: list
        The list of 3d positions already saved
    drifting: bool
        Whether templates are drifting or not
    verbose: bool
        If True, the output is verbose

    Returns
    -------
    skip_duplicate: bool
        If True, the position should be skipped because it's a duplicate
    """
    skip_pos = False
    if len(saved_positions) > 0:
        for pos_s in saved_positions:
            if not drifting:
                test_pos = pos_s
            else:
                test_pos = pos_s[0]
            if np.all(np.round(test_pos, 2) == np.round(pos, 2)):
                if verbose >= 2:
                    print(f"Duplicated position: {np.round(pos, 2)} -- {np.round(pos_s, 2)}. Skipping")
                skip_pos = True
    return skip_pos


def get_physrot_specs(cell_name, model):
    """Return physrot specifications for cell types

    Parameters
    -----------
    cell_name : string
        The name of the cell.
    Returns
    --------
    polarlim : array_like
        lower and upper bound for the polar angle
    pref_orient : array_like
        3-dim vetor of preferred orientation
    """
    if model == "bbp":
        polarlim = {
            "BP": [0.0, 15.0],
            "AC": None,
            "BTC": None,  # [0.,15.],
            "ChC": None,  # [0.,15.],
            "DBC": None,  # [0.,15.],
            "LBC": None,  # [0.,15.],
            "MC": [0.0, 15.0],
            "NBC": None,
            "NGC": None,
            "SBC": None,
            "PC": [0.0, 15.0],
            "SS": [0.0, 15.0],
            "SP": [0.0, 15.0],
        }
        # how it's implemented, the NMC y axis points into the pref_orient direction after rotation
        pref_orient = {
            "BP": [0.0, 0.0, 1.0],
            "AC": None,
            "BTC": None,  # [0.,0.,1.],
            "ChC": None,  # [0.,0.,1.],
            "DBC": None,  # [0.,0.,1.],
            "LBC": None,  # [0.,0.,1.],
            "MC": [0.0, 0.0, 1.0],
            "NBC": None,
            "NGC": None,
            "SBC": None,
            "PC": [0.0, 0.0, 1.0],
            "SS": [0.0, 0.0, 1.0],
            "SP": [0.0, 0.0, 1.0],
        }
        polar = None
        orient = None
        for k in polarlim.keys():
            if k in cell_name.split("_")[1]:
                polar = polarlim[k]
                orient = pref_orient[k]
                break
        return polar, orient
    else:
        raise NotImplementedError("Cell model %s is not implemented" % model)


def return_extracellular_spike(
    cell, cell_name, model_type, electrodes, limits, rotation, saved_pos, pos=None, max_iter=1000, verbose=False
):
    """
    Calculate extracellular spike on MEA at random position relative to cell

    Parameters
    ----------
    cell: LFPy.Cell
        cell object from LFPy
    cell_name: string
        name of cell model
    electrodes: LFPyRecExtElectrode
        The LFPy electrode object
    limits: array_like
        boundaries for neuron locations, shape=(3,2)
    rotation: string
        random rotation to apply to the neuron ('Norot', '3drot', 'physrot')
    saved_pos: np.array
        List of positions already used to be skipped
    pos: array_like, (optional, default None)
        Can be used to set the cell soma to a specific position. If ``None``,
        the random position is used.
    max_iter: int
        Max number of iterations to find a position that hasn't been used yet

    Returns
    -------
    Extracellular spike for each MEA contact site
    """

    def get_xyz_angles(R):
        """Get rotation angles for each axis from rotation matrix

        Parameters;
        -----------
        R : matrix
            3x3 rotation matrix

        Returns
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
        """Generate uniformly distributed random rotation matrices
        see: 'Fast Random Rotation Matrices' by Arvo (1992)

        Returns
        --------
        R : 3x3 matrix
            random rotation matrix
        """
        gamma = np.random.uniform(0, 2.0 * np.pi)
        rotation_z = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
        x = np.random.uniform(size=2)
        v = np.array(
            [np.cos(2.0 * np.pi * x[0]) * np.sqrt(x[1]), np.sin(2.0 * np.pi * x[0]) * np.sqrt(x[1]), np.sqrt(1 - x[1])]
        )
        H = np.identity(3) - 2.0 * np.outer(v, v)
        M = -np.dot(H, rotation_z)
        return M

    def check_solidangle(matrix, pre, post, polarlim):
        """Check whether a matrix rotates the vector 'pre' into a region
            defined by 'polarlim' around the vector 'post'

        Parameters
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

        Returns
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

    # rotate neuron
    if rotation == "norot":
        if model_type == "bbp":
            # orientate cells in z direction
            x_rot_offset = np.pi / 2.0
            y_rot_offset = 0
            z_rot_offset = 0
        else:
            x_rot_offset = 0
            y_rot_offset = 0
            z_rot_offset = 0
        x_rot = x_rot_offset
        y_rot = y_rot_offset
        z_rot = z_rot_offset
    elif rotation == "xrot":
        if model_type == "bbp":
            # orientate cells in z direction
            x_rot_offset = np.pi / 2.0
            y_rot_offset = 0
            z_rot_offset = 0
        else:
            x_rot_offset = 0
            y_rot_offset = 0
            z_rot_offset = 0
        x_rot, _, _ = get_xyz_angles(np.array(get_rnd_rot_Arvo()))
        x_rot = x_rot + x_rot_offset
        y_rot = y_rot_offset
        z_rot = z_rot_offset
    elif rotation == "yrot":
        if model_type == "bbp":
            # orientate cells in z direction
            x_rot_offset = np.pi / 2.0
            y_rot_offset = 0
            z_rot_offset = 0
        else:
            x_rot_offset = 0
            y_rot_offset = 0
            z_rot_offset = 0
        _, y_rot, _ = get_xyz_angles(np.array(get_rnd_rot_Arvo()))
        x_rot = x_rot_offset
        y_rot = y_rot + y_rot_offset
        z_rot = z_rot_offset
    elif rotation == "zrot":
        if model_type == "bbp":
            # orientate cells in z direction
            x_rot_offset = np.pi / 2.0
            y_rot_offset = 0
            z_rot_offset = 0
        else:
            x_rot_offset = 0
            y_rot_offset = 0
            z_rot_offset = 0
        _, _, z_rot = get_xyz_angles(np.array(get_rnd_rot_Arvo()))
        x_rot = x_rot_offset
        y_rot = y_rot_offset
        z_rot = z_rot + z_rot_offset
    elif rotation == "3drot":
        if model_type == "bbp":
            x_rot_offset = np.pi / 2.0  # align neuron with z axis
            y_rot_offset = 0  # align neuron with z axis
            z_rot_offset = 0  # align neuron with z axis
        else:
            x_rot_offset = 0
            y_rot_offset = 0
            z_rot_offset = 0
        x_rot, y_rot, z_rot = get_xyz_angles(np.array(get_rnd_rot_Arvo()))
        x_rot = x_rot + x_rot_offset
        y_rot = y_rot + y_rot_offset
        z_rot = z_rot + z_rot_offset
    elif rotation == "physrot":
        polarlim, pref_orient = get_physrot_specs(cell_name, model_type)
        if model_type == "bbp":
            x_rot_offset = np.pi / 2.0  # align neuron with z axis
            y_rot_offset = 0  # align neuron with z axis
            z_rot_offset = 0  # align neuron with z axis
        else:
            raise NotImplementedError("'physrot' rotation is only available with BBP cells")
        while True:
            R = np.array(get_rnd_rot_Arvo())
            if polarlim is None or pref_orient is None:
                valid = True
            else:
                valid = check_solidangle(R, [0.0, 0.0, 1.0], pref_orient, polarlim)
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
        skip_dup = True
        iter = 0
        x_rands = np.random.uniform(limits[0][0], limits[0][1], 1000)
        y_rands = np.random.uniform(limits[1][0], limits[1][1], 1000)
        z_rands = np.random.uniform(limits[2][0], limits[2][1], 1000)
        while skip_dup and iter < max_iter:
            x_rand = x_rands[iter]
            y_rand = y_rands[iter]
            z_rand = z_rands[iter]
            pos = [x_rand, y_rand, z_rand]
            skip_dup = skip_duplicate(pos, saved_pos, drifting=False, verbose=verbose)
            iter += 1
        if iter == max_iter:
            found_position = False
            return None, None, None, found_position
        else:
            found_position = True
        cell.set_pos(x_rand, y_rand, z_rand)
    else:
        cell.set_pos(pos[0], pos[1], pos[2])
        found_position = True

    cell.set_rotation(x=x_rot, y=y_rot, z=z_rot)
    rot = [x_rot, y_rot, z_rot]

    lfp = np.array(electrodes.get_transformation_matrix() @ cell.imem, dtype=np.float32)

    # Reverse rotation to bring cell back into initial rotation state
    if rotation is not None:
        rev_rot = [-r for r in rot]
        cell.set_rotation(rev_rot[0], rev_rot[1], rev_rot[2], rotation_order="zyx")

    return 1e3 * lfp, pos, rot, found_position


def str2bool(v):
    """Transform string to bool

    Parameters
    -----------
    v : str

    Returns
    --------
    transformed_v, bool
        If v is any of ("yes", "true", "t", "1") (case insensitive)
        ``True`` is returned, else ``False``
    """
    return v.lower() in ("yes", "true", "t", "1")


def simulate_templates_one_cell(cell_model, intra_save_folder, params, verbose, custom_return_cell_function=None):
    """

    Parameters
    ----------
    cell_model
    intra_save_folder
    params
    verbose
    custom_return_cell_function

    Returns
    -------

    """
    run_cell_model(
        cell_model,
        save=True,
        sim_folder=intra_save_folder,
        verbose=verbose,
        custom_return_cell_function=custom_return_cell_function,
        **params,
    )
    print(f"Extracellular simulation: {cell_model}")
    eaps, locs, rots = calc_extracellular(
        0,
        cell_model,
        intra_save_folder,
        verbose=verbose,
        save=False,
        custom_return_cell_function=custom_return_cell_function,
        **params,
    )

    return eaps, locs, rots


def compile_models(cell_folder):
    compile_all_mechanisms(cell_folder)
    print(f"Compiled all cell models in {cell_folder}")


def compute_eap_for_cell_model(i, cell_model, params_path, intraonly=False, verbose=False):
    params = safe_yaml_load(params_path)

    extra_sim_folder = Path(params["templates_folder"])
    vm_im_sim_folder = Path(params["templates_folder"]) / "intracellular"

    print(f"Intracellular simulation: {cell_model}")
    run_cell_model(cell_model_folder=cell_model, save=True, sim_folder=vm_im_sim_folder, verbose=verbose, **params)
    if not intraonly:
        print(f"Extracellular simulation: {cell_model}")
        calc_extracellular(
            i,
            cell_model_folder=cell_model,
            save_sim_folder=extra_sim_folder,
            load_sim_folder=vm_im_sim_folder,
            verbose=verbose,
            **params,
        )


def compute_eap_based_on_tempgen(cell_folder, params_path, tempgen, intraonly=False, verbose=False):
    params = safe_yaml_load(params_path)

    extra_sim_folder = params["templates_folder"]
    vm_im_sim_folder = str(Path(params["templates_folder"]) / "intracellular")

    celltypes = np.unique(tempgen.celltypes)

    for celltype in celltypes:
        celltype_idxs = np.where(tempgen.celltypes == celltype)
        if np.any(np.diff(celltype_idxs) != 1):
            raise NotImplementedError("Cell types in the template generator must be contiguous.")

    # print(f'Intracellular simulation: {cell_model}')
    for i, celltype in enumerate(celltypes):
        cell_model = cell_folder / celltype
        run_cell_model(cell_model_folder=cell_model, sim_folder=vm_im_sim_folder, verbose=verbose, **params)
        celltype_idxs = np.where(tempgen.celltypes == celltype)
        cell_locations = tempgen.locations[celltype_idxs]
        cell_rotations = tempgen.rotations[celltype_idxs]
        if not intraonly:
            print(f"Extracellular simulation: {cell_model}")
            calc_extracellular(
                i,
                cell_model_folder=cell_model,
                save_sim_folder=extra_sim_folder,
                load_sim_folder=vm_im_sim_folder,
                verbose=verbose,
                cell_locations=cell_locations,
                cell_rotations=cell_rotations,
                **params,
            )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "compile":
        cell_folder = sys.argv[2]
        compile_all_mechanisms(cell_folder)
        sys.exit(0)
    elif len(sys.argv) == 6:
        i = int(sys.argv[1])
        cell_model = sys.argv[2]
        intraonly = str2bool(sys.argv[3])
        params_path = sys.argv[4]
        verbose = int(sys.argv[5])
        compute_eap_for_cell_model(i, cell_model, params_path, intraonly=intraonly, verbose=verbose)
