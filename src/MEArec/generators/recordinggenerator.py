# don't enter here without a good guide! (only one person in the world)

import os
import random
import shutil
import string
import tempfile
import time
from copy import deepcopy
from distutils.log import DEBUG
from pathlib import Path
from warnings import warn

import MEAutility as mu
import numpy as np
import quantities as pq
import yaml
from joblib import Parallel, delayed
from packaging.version import parse

from ..drift_tools import generate_drift_dict_from_params
from ..tools import (annotate_overlapping_spikes, compute_modulation,
                     extract_wf, find_overlapping_templates, get_binary_cat,
                     get_templates_features, jitter_templates, pad_templates,
                     resample_templates, select_templates, sigmoid)
from .recgensteps import (chunk_apply_filter, chunk_convolution,
                          chunk_distance_correlated_noise,
                          chunk_uncorrelated_noise)

DEBUG = False

if DEBUG:
    import matplotlib.pyplot as plt

    plt.ion()
    plt.show()


class RecordingGenerator:
    """
    Class for generation of recordings called by the gen_recordings function.
    The list of parameters is in default_params/recordings_params.yaml.

    Parameters
    ----------
    spgen : SpikeTrainGenerator
        SpikeTrainGenerator object containing spike trains
    tempgen : TemplateGenerator
        TemplateGenerator object containing templates
    params : dict
        Dictionary with parameters to simulate recordings. Default values can be retrieved with
        mr.get_default_recording_params()
    rec_dict :  dict
        Dictionary to instantiate RecordingGenerator with existing data. It contains the following fields:
          - recordings : float (n_electrodes, n_samples)
          - spiketrains : list of neo.SpikeTrains (n_spiketrains)
          - templates : float (n_spiketrains, 3)
          - template_locations : float (n_spiketrains, 3)
          - template_rotations : float (n_spiketrains, 3)
          - template_celltypes : str (n_spiketrains)
          - channel_positions : float (n_electrodes, 3)
          - timestamps : float (n_samples)
          - voltage_peaks : float (n_spiketrains, n_electrodes)
          - spike_traces : float (n_spiketrains, n_samples)
    info :  dict
        Info dictionary to instantiate RecordingGenerator with existing data. Same fields as 'params'
    """

    def __init__(self, spgen=None, tempgen=None, params=None, rec_dict=None, info=None):
        from . import SpikeTrainGenerator

        self._verbose = False
        self._verbose_1 = False
        self._verbose_2 = False

        if rec_dict is not None and info is not None:
            if "recordings" in rec_dict.keys():
                self.recordings = rec_dict["recordings"]
            else:
                self.recordings = np.array([])
            if "spiketrains" in rec_dict.keys():
                self.spiketrains = deepcopy(rec_dict["spiketrains"])
            else:
                self.spiketrains = np.array([])
            if "templates" in rec_dict.keys():
                self.templates = rec_dict["templates"]
            else:
                self.templates = np.array([])
            if "original_templates" in rec_dict.keys():
                self.original_templates = rec_dict["original_templates"]
            else:
                self.original_templates = np.array([])
            if "template_locations" in rec_dict.keys():
                self.template_locations = rec_dict["template_locations"]
            else:
                self.template_locations = np.array([])
            if "template_rotations" in rec_dict.keys():
                self.template_rotations = rec_dict["template_rotations"]
            else:
                self.template_rotations = np.array([])
            if "template_celltypes" in rec_dict.keys():
                self.template_celltypes = rec_dict["template_celltypes"]
            else:
                self.template_celltypes = np.array([])
            if "channel_positions" in rec_dict.keys():
                self.channel_positions = rec_dict["channel_positions"]
            else:
                self.channel_positions = np.array([])
            if "timestamps" in rec_dict.keys():
                self.timestamps = rec_dict["timestamps"]
            else:
                self.timestamps = np.array([])
            if "voltage_peaks" in rec_dict.keys():
                self.voltage_peaks = rec_dict["voltage_peaks"]
            else:
                self.voltage_peaks = np.array([])
            if "spike_traces" in rec_dict.keys():
                self.spike_traces = rec_dict["spike_traces"]
            else:
                self.spike_traces = np.array([])
            if "template_ids" in rec_dict.keys():
                self.template_ids = rec_dict["template_ids"]
            else:
                self.template_ids = None
            if "drift_list" in rec_dict.keys():
                self.drift_list = rec_dict["drift_list"]
            else:
                self.drift_list = None

            self.info = deepcopy(info)
            self.params = deepcopy(info)
            if len(self.spiketrains) > 0:
                if "spiketrains" in self.info:
                    self.spgen = SpikeTrainGenerator(spiketrains=self.spiketrains, params=self.info["spiketrains"])
                else:
                    self.spgen = SpikeTrainGenerator(spiketrains=self.spiketrains, params={"custom": True})
            self.tempgen = None
            if isinstance(self.recordings, np.memmap):
                self.tmp_mode = "memmap"
            else:
                self.tmp_mode = None

        else:
            if spgen is None or tempgen is None:
                raise AttributeError("Specify SpikeTrainGenerator and TemplateGenerator objects!")
            if params is None:
                params = {"spiketrains": {}, "celltypes": {}, "templates": {}, "recordings": {}, "seeds": {}}
            self.params = deepcopy(params)
            self.spgen = spgen
            self.tempgen = tempgen
            self.tmp_mode = None
            self.template_ids = None

        self.overlapping = []
        # temp file that should remove on delete
        self._to_remove_on_delete = []
        self.tmp_folder = None
        self.n_jobs = None
        self._is_tmp_folder_local = False

    def __del__(self):
        if not self._is_tmp_folder_local:
            if self.tmp_folder is not None:
                try:
                    shutil.rmtree(self.tmp_folder)
                    if self._verbose >= 1:
                        print("Deleted", self.tmp_folder)
                except Exception as e:
                    if self._verbose >= 1:
                        print("Impossible to delete temp file:", self.tmp_folder, "Error", e)
        else:
            for fname in self._to_remove_on_delete:
                try:
                    os.remove(fname)
                    if self._verbose >= 1:
                        print("Deleted", fname)
                except Exception as e:
                    if self._verbose >= 1:
                        print("Impossible to delete temp file:", fname, "Error", e)
        self.recordings = None
        self.spike_traces = None

    def generate_recordings(
        self, tmp_mode=None, tmp_folder=None, n_jobs=0, template_ids=None, drift_dicts=None, verbose=None
    ):
        """
        Generates recordings

        Parameters
        ----------
        tmp_mode : None, 'memmap'
            Use temporary file h5 memmap or None
            None is no temporary file and then use memory.
        tmp_folder: str or Path
            In case of tmp files, you can specify the folder.
            If None, then it is automatic using tempfile.mkdtemp()
        n_jobs: int
            if >1 then use joblib to execute chunk in parallel else in loop
        template_ids: list or None
            If None, templates are selected randomly based on selection rules. If a list of indices is provided, the
            indices are used to select templates (template selection is bypassed)
        drift_dicts: list or None
            List of drift dictionaries to construct multiple drift vectors. (see `MEArec.get_default_drift_dict()`)
        verbose: bool or int
            Determines the level of verbose. If 1 or True, low-level, if 2 high level, if False, not verbose
        """

        self.tmp_mode = tmp_mode
        self.tmp_folder = tmp_folder
        self.template_ids = template_ids
        self.n_jobs = n_jobs

        if tmp_mode is not None:
            tmp_prefix = "".join([random.choice(string.ascii_letters) for i in range(5)]) + "_"
        if self.tmp_mode is not None:
            if self.tmp_folder is None:
                self.tmp_folder = Path(tempfile.mkdtemp())
                self._is_tmp_folder_local = False
            else:
                self.tmp_folder = Path(self.tmp_folder)
                self.tmp_folder.mkdir(exist_ok=True, parents=True)
                self._is_tmp_folder_local = True
        else:
            self._is_tmp_folder_local = False

        self._verbose = verbose
        if self._verbose is not None and isinstance(self._verbose, bool) or isinstance(self._verbose, int):
            verbose_1 = self._verbose >= 1
            verbose_2 = self._verbose >= 2
        elif isinstance(verbose, bool):
            if self._verbose:
                verbose_1 = True
                verbose_2 = False
            else:
                verbose_1 = False
                verbose_2 = False
        else:  # None
            verbose_1 = False
            verbose_2 = False
        self._verbose_1 = verbose_1
        self._verbose_2 = verbose_2

        params = deepcopy(self.params)
        temp_params = self.params["templates"]
        rec_params = self.params["recordings"]
        st_params = self.params["spiketrains"]
        seeds = self.params["seeds"]

        if "chunk_duration" not in rec_params.keys():
            rec_params["chunk_duration"] = None
        if rec_params["chunk_duration"] is None:
            rec_params["chunk_duration"] = 0

        if self.n_jobs > 1 and rec_params["chunk_duration"] == 0:
            warn(f"For n_jobs {self.n_jobs} you should set chunk_duration > 0. Setting chunk_duration to 5s")
            rec_params["chunk_duration"] = 5

        if "cell_types" in self.params.keys():
            celltype_params = self.params["cell_types"]
        else:
            celltype_params = {}

        tempgen = self.tempgen
        spgen = self.spgen
        if tempgen is not None:
            eaps = tempgen.templates
            locs = tempgen.locations
            rots = tempgen.rotations
            celltypes = tempgen.celltypes
            temp_info = tempgen.info
            cut_outs = temp_info["params"]["cut_out"]
        else:
            temp_info = None
            cut_outs = self.params["templates"]["cut_out"]

        spiketrains = spgen.spiketrains
        n_neurons = len(spiketrains)

        if len(spiketrains) > 0:
            duration = spiketrains[0].t_stop - spiketrains[0].t_start
            only_noise = False
        else:
            if verbose_1:
                print("No spike trains provided: only simulating noise")
            only_noise = True
            duration = st_params["duration"] * pq.s

        if "fs" not in rec_params.keys() and temp_info is not None:
            # when computed from templates fs is in kHz
            params["recordings"]["fs"] = 1.0 / temp_info["params"]["dt"]
            fs = (params["recordings"]["fs"] * pq.kHz).rescale("Hz")
            spike_fs = fs
        elif params["recordings"]["fs"] is None and temp_info is not None:
            params["recordings"]["fs"] = 1.0 / temp_info["params"]["dt"]
            fs = (params["recordings"]["fs"] * pq.kHz).rescale("Hz")
            spike_fs = fs
        else:
            # In the rec_params fs is in Hz
            fs = params["recordings"]["fs"] * pq.Hz
            if temp_info is not None:
                spike_fs = (1.0 / temp_info["params"]["dt"] * pq.kHz).rescale("Hz")
            else:
                spike_fs = fs

        if "dtype" not in rec_params.keys():
            params["recordings"]["dtype"] = "float32"
        elif rec_params["dtype"] is None:
            params["recordings"]["dtype"] = "float32"
        else:
            params["recordings"]["dtype"] = rec_params["dtype"]
        dtype = params["recordings"]["dtype"]

        assert np.dtype(dtype).kind in ("i", "f"), "Only integers and float dtypes are supported"

        params["recordings"]["adc_bit_depth"] = rec_params.get("adc_bit_depth", None)
        adc_bit_depth = params["recordings"]["adc_bit_depth"]
        params["recordings"]["lsb"] = rec_params.get("lsb", None)
        lsb = params["recordings"]["lsb"]
        if lsb is None and np.dtype(dtype).kind == "i":
            lsb = 1
        params["recordings"]["gain"] = rec_params.get("gain", None)
        gain = params["recordings"]["gain"]

        if verbose_1:
            print(f"dtype {dtype}")
            if np.dtype(dtype).kind == "i":
                print(f"ADC bit depth: {adc_bit_depth} -- LSB: {lsb}")

        if "noise_mode" not in rec_params.keys():
            params["recordings"]["noise_mode"] = "uncorrelated"
        noise_mode = params["recordings"]["noise_mode"]

        assert noise_mode in [
            "uncorrelated",
            "distance-correlated",
            "far-neurons",
        ], "'noise_mode can be: 'uncorrelated', 'distance-correlated', or 'far-neurons'"

        if "noise_color" not in rec_params.keys():
            params["recordings"]["noise_color"] = False
        noise_color = params["recordings"]["noise_color"]

        if "sync_rate" not in rec_params.keys():
            params["recordings"]["sync_rate"] = None
        sync_rate = params["recordings"]["sync_rate"]

        if "sync_jitt" not in rec_params.keys():
            params["recordings"]["sync_jitt"] = 1
        sync_jitt = params["recordings"]["sync_jitt"] * pq.ms

        if noise_mode == "distance-correlated":
            if "noise_half_distance" not in rec_params.keys():
                params["recordings"]["noise_half_distance"] = 30
            half_dist = params["recordings"]["noise_half_distance"]

        if noise_mode == "far-neurons":
            if "far_neurons_n" not in rec_params.keys():
                params["recordings"]["far_neurons_n"] = 300
            far_neurons_n = params["recordings"]["far_neurons_n"]
            if "far_neurons_max_amp" not in rec_params.keys():
                params["recordings"]["far_neurons_max_amp"] = 20
            far_neurons_max_amp = params["recordings"]["far_neurons_max_amp"]
            if "far_neurons_noise_floor" not in rec_params.keys():
                params["recordings"]["far_neurons_noise_floor"] = 0.5
            far_neurons_noise_floor = params["recordings"]["far_neurons_noise_floor"]
            if "far_neurons_exc_inh_ratio" not in rec_params.keys():
                params["recordings"]["far_neurons_exc_inh_ratio"] = 0.8
            far_neurons_exc_inh_ratio = params["recordings"]["far_neurons_exc_inh_ratio"]

        if noise_color:
            if "color_peak" not in rec_params.keys():
                params["recordings"]["color_peak"] = 500
            color_peak = params["recordings"]["color_peak"]
            if "color_q" not in rec_params.keys():
                params["recordings"]["color_q"] = 1
            color_q = params["recordings"]["color_q"]
            if "color_noise_floor" not in rec_params.keys():
                params["recordings"]["color_noise_floor"] = 1
            color_noise_floor = params["recordings"]["color_noise_floor"]
        else:
            color_peak, color_q, color_noise_floor = None, None, None

        if "noise_level" not in rec_params.keys():
            params["recordings"]["noise_level"] = 10
        noise_level = params["recordings"]["noise_level"]

        if verbose_1:
            print("Noise Level ", noise_level)

        if "filter" not in rec_params.keys():
            params["recordings"]["filter"] = True
        filter = params["recordings"]["filter"]

        if "filter_cutoff" not in rec_params.keys():
            params["recordings"]["filter_cutoff"] = [300.0, 6000.0]
        cutoff = params["recordings"]["filter_cutoff"] * pq.Hz

        if "filter_order" not in rec_params.keys():
            params["recordings"]["filter_order"] = 3
        filter_order = params["recordings"]["filter_order"]

        if "filter_mode" not in rec_params.keys():
            params["recordings"]["filter_mode"] = "filtfilt"
        filter_mode = params["recordings"]["filter_mode"]

        if "modulation" not in rec_params.keys():
            params["recordings"]["modulation"] = "electrode"
        elif params["recordings"]["modulation"] not in ["none", "electrode", "template"]:
            raise Exception("'modulation' can be 'none', 'template', or 'electrode'")
        modulation = params["recordings"]["modulation"]

        if "bursting" not in rec_params.keys():
            params["recordings"]["bursting"] = False
        bursting = params["recordings"]["bursting"]

        if "shape_mod" not in rec_params.keys():
            params["recordings"]["shape_mod"] = False
        shape_mod = params["recordings"]["shape_mod"]

        if bursting:
            if "bursting_units" not in rec_params.keys():
                rec_params["bursting_units"] = None

            if rec_params["bursting_units"] is not None:
                assert np.all(
                    [b < n_neurons for b in rec_params["bursting_units"]]
                ), "'bursting_units' ids should be lower than the number of neurons"
                n_bursting = len(rec_params["bursting_units"])
                bursting_units = rec_params["bursting_units"]
            else:
                if rec_params["n_bursting"] is None:
                    n_bursting = n_neurons
                    bursting_units = np.arange(n_neurons)
                else:
                    n_bursting = rec_params["n_bursting"]
                    bursting_units = np.random.permutation(n_neurons)[:n_bursting]

            if "exp_decay" not in rec_params.keys():
                params["recordings"]["exp_decay"] = [0.2] * n_bursting
            else:
                if not isinstance(rec_params["exp_decay"], list):
                    assert isinstance(rec_params["exp_decay"], float), "'exp_decay' can be list or float"
                    params["recordings"]["exp_decay"] = [rec_params["exp_decay"]] * n_bursting
                else:
                    assert len(rec_params["exp_decay"]) == n_bursting, (
                        "'exp_decay' should have the same length as " "the number of bursting units"
                    )
                    params["recordings"]["exp_decay"] = rec_params["exp_decay"]
            exp_decay = params["recordings"]["exp_decay"]

            if "n_burst_spikes" not in rec_params.keys():
                params["recordings"]["n_burst_spikes"] = [10] * n_bursting
            else:
                if not isinstance(rec_params["n_burst_spikes"], list):
                    assert isinstance(rec_params["n_burst_spikes"], int), "'n_burst_spikes' can be list or int"
                    params["recordings"]["n_burst_spikes"] = [rec_params["n_burst_spikes"]] * n_bursting
                else:
                    assert len(rec_params["n_burst_spikes"]) == n_bursting, (
                        "'n_burst_spikes' should have the same " "length as the number of bursting units"
                    )
                    params["recordings"]["n_burst_spikes"] = rec_params["n_burst_spikes"]
            n_burst_spikes = params["recordings"]["n_burst_spikes"]

            if "max_burst_duration" not in rec_params.keys():
                params["recordings"]["max_burst_duration"] = [100] * n_bursting
            else:
                if not isinstance(rec_params["max_burst_duration"], list):
                    assert isinstance(
                        rec_params["max_burst_duration"], (float, int, np.integer)
                    ), "'max_burst_duration' can be list or scalar"
                    params["recordings"]["max_burst_duration"] = [rec_params["max_burst_duration"]] * n_bursting
                else:
                    assert (
                        len(rec_params["max_burst_duration"]) == n_bursting
                    ), "'max_burst_duration' should have the same length as the number of bursting units"
                    params["recordings"]["max_burst_duration"] = rec_params["max_burst_duration"]
            max_burst_duration = [m * pq.ms for m in params["recordings"]["max_burst_duration"]]

            if shape_mod:
                if "shape_stretch" not in rec_params.keys():
                    params["recordings"]["shape_stretch"] = 30
                shape_stretch = params["recordings"]["shape_stretch"]
                if verbose_1:
                    print("Bursting with modulation sigmoid: ", shape_stretch)
            else:
                shape_stretch = None
        else:
            exp_decay = None
            n_burst_spikes = None
            max_burst_duration = None
            shape_stretch = None
            bursting_units = []

        chunk_duration = params["recordings"].get("chunk_duration", 0) * pq.s
        if chunk_duration == 0 * pq.s:
            chunk_duration = duration

        if "mrand" not in rec_params.keys():
            params["recordings"]["mrand"] = 1
        mrand = params["recordings"]["mrand"]

        if "sdrand" not in rec_params.keys():
            params["recordings"]["sdrand"] = 0.05
        sdrand = params["recordings"]["sdrand"]

        if "overlap" not in rec_params.keys():
            params["recordings"]["overlap"] = False
        overlap = params["recordings"]["overlap"]

        if "extract_waveforms" not in rec_params.keys():
            params["recordings"]["extract_waveforms"] = False
        extract_waveforms = params["recordings"]["extract_waveforms"]

        if "xlim" not in temp_params.keys():
            params["templates"]["xlim"] = None
        x_lim = params["templates"]["xlim"]

        if "ylim" not in temp_params.keys():
            params["templates"]["ylim"] = None
        y_lim = params["templates"]["ylim"]

        if "zlim" not in temp_params.keys():
            params["templates"]["zlim"] = None
        z_lim = params["templates"]["zlim"]

        if "n_overlap_pairs" not in temp_params.keys():
            params["templates"]["n_overlap_pairs"] = None
        n_overlap_pairs = params["templates"]["n_overlap_pairs"]

        if "min_amp" not in temp_params.keys():
            params["templates"]["min_amp"] = 50
        min_amp = params["templates"]["min_amp"]

        if "max_amp" not in temp_params.keys():
            params["templates"]["max_amp"] = np.inf
        max_amp = params["templates"]["max_amp"]

        if "min_dist" not in temp_params.keys():
            params["templates"]["min_dist"] = 25
        min_dist = params["templates"]["min_dist"]

        if "overlap_threshold" not in temp_params.keys():
            params["templates"]["overlap_threshold"] = 0.8
        overlap_threshold = params["templates"]["overlap_threshold"]

        if "pad_len" not in temp_params.keys():
            params["templates"]["pad_len"] = [3.0, 3.0]
        pad_len = params["templates"]["pad_len"]

        if "smooth_percent" not in temp_params.keys():
            params["templates"]["smooth_percent"] = 0.5
        smooth_percent = params["templates"]["smooth_percent"]

        if "smooth_strength" not in temp_params.keys():
            params["templates"]["smooth_strength"] = 1
        smooth_strength = params["templates"]["smooth_strength"]

        if "n_jitters" not in temp_params.keys():
            params["templates"]["n_jitters"] = 10
        n_jitters = params["templates"]["n_jitters"]

        if "upsample" not in temp_params.keys():
            params["templates"]["upsample"] = 8
        upsample = params["templates"]["upsample"]

        if "drifting" not in rec_params.keys():
            params["recordings"]["drifting"] = False
        drifting = params["recordings"]["drifting"]

        # set seeds
        if "templates" not in seeds.keys():
            temp_seed = np.random.randint(1, 1000)
        elif seeds["templates"] is None:
            temp_seed = np.random.randint(1, 1000)
        else:
            temp_seed = seeds["templates"]

        if "convolution" not in seeds.keys():
            conv_seed = np.random.randint(1, 1000)
        elif seeds["convolution"] is None:
            conv_seed = np.random.randint(1, 1000)
        else:
            conv_seed = seeds["convolution"]

        if "noise" not in seeds.keys():
            noise_seed = np.random.randint(1, 1000)
        elif seeds["noise"] is None:
            noise_seed = np.random.randint(1, 1000)
        else:
            noise_seed = seeds["noise"]

        n_samples = int(duration.rescale("s").magnitude * fs.rescale("Hz").magnitude)

        if drifting:
            if temp_info is not None:
                assert temp_info["params"]["drifting"], "For generating drifting recordings, templates must be drifting"
            else:
                if params["n_jitters"] == 1:
                    assert len(self.templates.shape) == 4
                else:
                    assert len(self.templates.shape) == 5
            preferred_dir = np.array(rec_params["preferred_dir"])
            preferred_dir = preferred_dir / np.linalg.norm(preferred_dir)
            angle_tol = rec_params["angle_tol"]
            if rec_params["n_drifting"] is None:
                n_drifting = n_neurons
            else:
                n_drifting = rec_params["n_drifting"]

            drift_keys = (
                "drift_fs",
                "t_start_drift",
                "t_end_drift",
                "drift_mode_probe",
                "drift_mode_speed",
                "non_rigid_gradient_mode",
                "non_rigid_linear_direction",
                "non_rigid_linear_min_factor",
                "non_rigid_step_depth_boundary",
                "non_rigid_step_factors",
                "slow_drift_velocity",
                "slow_drift_amplitude",
                "slow_drift_waveform",
                "fast_drift_period",
                "fast_drift_max_jump",
                "fast_drift_min_jump",
            )

            if drift_dicts is None:
                drift_params = {k: rec_params[k] for k in drift_keys}
                drift_dicts = [drift_params]
            else:
                if verbose:
                    print(f"Using {len(drift_dicts)} custom drift signals")
                drift_keys += (
                    "external_drift_vector_um",
                    "external_drift_times",
                    "external_drift_factors",
                )
                for drift_params in drift_dicts:
                    for k in drift_params:
                        assert k in drift_keys, f"Wrong drift key {k}"
                    # assert np.all([k in drift_params for k in drift_keys]), "'drift_dicts have some missing keys!"
        else:
            # if drifting templates, but not recordings, consider initial template
            if temp_info is not None:
                if temp_info["params"]["drifting"]:
                    eaps = eaps[:, 0]
                    locs = locs[:, 0]
            elif len(self.templates.shape) == 5:
                self.templates = self.templates[:, 0]
                self.template_locs = self.template_locs[:, 0]
            preferred_dir = None
            angle_tol = None
            drift_list = None
            n_drifting = None

        # load MEA info
        if temp_info is not None:
            mea = mu.return_mea(info=temp_info["electrodes"])
            params["electrodes"] = temp_info["electrodes"]
        else:
            mea = mu.return_mea(info=self.params["electrodes"])
            params["electrodes"] = self.params["electrodes"]
        mea_pos = mea.positions
        n_elec = mea_pos.shape[0]
        # ~ n_samples = int(duration.rescale('s').magnitude * fs.rescale('Hz').magnitude)

        params["recordings"].update(
            {"duration": float(duration.magnitude), "fs": float(fs.rescale("Hz").magnitude), "n_neurons": n_neurons}
        )
        params["templates"].update({"cut_out": cut_outs})

        # create buffer h5/memmap/memmory
        if self.tmp_mode == "memmap":
            tmp_path_0 = self.tmp_folder / (tmp_prefix + "mearec_tmp_file_recordings.raw")
            recordings = np.memmap(tmp_path_0, shape=(n_samples, n_elec), dtype=dtype, mode="w+")
            recordings[:] = 0
            # recordings = recordings.transpose()
            tmp_path_1 = self.tmp_folder / (tmp_prefix + "mearec_tmp_file_spike_traces.raw")
            if not only_noise:
                spike_traces = np.memmap(tmp_path_1, shape=(n_samples, n_neurons), dtype=dtype, mode="w+")
                spike_traces[:] = 0
                # spike_traces = spike_traces.transpose()
            # file names for templates
            tmp_templates_pad = self.tmp_folder / (tmp_prefix + "templates_pad.raw")
            tmp_templates_rs = self.tmp_folder / (tmp_prefix + "templates_resample.raw")
            tmp_templates_jit = self.tmp_folder / (tmp_prefix + "templates_jitter.raw")
            self._to_remove_on_delete.extend(
                [tmp_path_0, tmp_path_1, tmp_templates_pad, tmp_templates_jit]
            )
        else:
            recordings = np.zeros((n_samples, n_elec), dtype=dtype)
            spike_traces = np.zeros((n_samples, n_neurons), dtype=dtype)
            tmp_templates_pad = None
            tmp_templates_rs = None
            tmp_templates_jit = None

        timestamps = np.arange(recordings.shape[0]) / fs

        #######################
        # Step 1: convolution #
        #######################
        if only_noise:
            spiketrains = np.array([])
            voltage_peaks = np.array([])
            spike_traces = np.array([])
            templates = np.array([])
            template_locs = np.array([])
            template_rots = np.array([])
            template_celltypes = np.array([])
            overlapping = np.array([])

            # compute gain
            gain_to_int = None
            if np.dtype(dtype).kind == "i":
                if gain is None:
                    if adc_bit_depth is not None:
                        signal_range = lsb * (2**adc_bit_depth)
                        dtype_depth = np.dtype(dtype).itemsize * 8
                        assert signal_range <= 2**dtype_depth, (
                            f"ADC bit depth and LSB exceed the range of the "
                            f"selected dtype {dtype}. Try reducing them or using "
                            f"a larger dtype"
                        )
                        # in this case we use 4 times the noise level (times 2 for positive-negative)
                        max_noise = 2 * (4 * noise_level)
                        gain_to_int = signal_range / max_noise
                        gain = 1.0 / gain_to_int
                else:
                    gain_to_int = 1.0 / gain
        else:
            if tempgen is not None:
                if celltype_params is not None:
                    if "excitatory" in celltype_params.keys() and "inhibitory" in celltype_params.keys():
                        exc_categories = celltype_params["excitatory"]
                        inh_categories = celltype_params["inhibitory"]
                        bin_cat = get_binary_cat(celltypes, exc_categories, inh_categories)
                    else:
                        bin_cat = np.array(["U"] * len(celltypes))
                else:
                    bin_cat = np.array(["U"] * len(celltypes))

                if "cell_type" in spiketrains[0].annotations.keys():
                    n_exc = [st.annotations["cell_type"] for st in spiketrains].count("E")
                    n_inh = n_neurons - n_exc
                    st_types = np.array([st.annotations["cell_type"] for st in spiketrains])
                elif "rates" in st_params.keys():
                    assert st_params["types"] is not None, (
                        "If 'rates' are provided as spiketrains parameters, "
                        "corresponding 'types' ('E'-'I') must be provided"
                    )
                    n_exc = st_params["types"].count("E")
                    n_inh = st_params["types"].count("I")
                    st_types = np.array(st_params["types"])
                else:
                    if self._verbose:
                        print("Setting random number of excitatory and inhibitory neurons as cell_type info is missing")
                    n_exc = np.random.randint(n_neurons)
                    n_inh = n_neurons - n_exc
                    st_types = np.array(["E"] * n_exc + ["I"] * n_inh)

                e_idx = np.where(st_types == "E")
                i_idx = np.where(st_types == "I")
                if len(e_idx) > 0 and len(i_idx) > 0:
                    if not np.all([[e < i for e in e_idx[0]] for i in i_idx[0]]):
                        if verbose_1:
                            print("Re-arranging spike trains: Excitatory first, Inhibitory last")
                        order = np.argsort(st_types)
                        new_spiketrains = []
                        for idx in order:
                            new_spiketrains.append(spiketrains[idx])
                        spgen.spiketrains = new_spiketrains
                        spiketrains = new_spiketrains

                if verbose_1:
                    print("Templates selection seed: ", temp_seed)
                np.random.seed(temp_seed)

                if drifting:
                    drift_directions = np.array([(p[-1] - p[0]) / np.linalg.norm(p[-1] - p[0]) for p in locs])
                    n_elec = eaps.shape[2]
                else:
                    drift_directions = None
                    n_elec = eaps.shape[1]

                if n_neurons > 100 or drifting:
                    parallel_templates = True
                else:
                    parallel_templates = False

                if verbose_1:
                    print("Selecting cells")

                if self.template_ids is None:
                    idxs_cells, selected_cat = select_templates(
                        locs,
                        eaps,
                        bin_cat,
                        n_exc,
                        n_inh,
                        x_lim=x_lim,
                        y_lim=y_lim,
                        z_lim=z_lim,
                        min_amp=min_amp,
                        max_amp=max_amp,
                        min_dist=min_dist,
                        drifting=drifting,
                        drift_dir=drift_directions,
                        preferred_dir=preferred_dir,
                        angle_tol=angle_tol,
                        n_overlap_pairs=n_overlap_pairs,
                        overlap_threshold=overlap_threshold,
                        verbose=verbose_2,
                    )

                    if not np.any("U" in selected_cat):
                        assert selected_cat.count("E") == n_exc and selected_cat.count("I") == n_inh
                        # Reorder templates according to E-I types
                        reordered_idx_cells = np.array(idxs_cells)[np.argsort(selected_cat)]
                    else:
                        reordered_idx_cells = idxs_cells

                    template_celltypes = celltypes[reordered_idx_cells]
                    template_locs = np.array(locs)[reordered_idx_cells]
                    template_rots = np.array(rots)[reordered_idx_cells]
                    template_bin = np.array(bin_cat)[reordered_idx_cells]
                    templates = np.empty((len(reordered_idx_cells), *eaps.shape[1:]), dtype=eaps.dtype)
                    for i, reordered_idx in enumerate(reordered_idx_cells):
                        templates[i] = eaps[reordered_idx]
                    self.template_ids = reordered_idx_cells
                else:
                    print(f"Using provided template ids: {self.template_ids}")
                    ordered_ids = np.all(np.diff(self.template_ids) > 0)
                    if ordered_ids:
                        template_celltypes = celltypes[self.template_ids]
                        template_locs = np.array(locs[self.template_ids])
                        template_rots = np.array(rots[self.template_ids])
                        template_bin = np.array(bin_cat[self.template_ids])
                        templates = np.array(eaps[self.template_ids])
                    else:
                        order = np.argsort(self.template_ids)
                        order_back = np.argsort(order)
                        sorted_ids = self.template_ids[order]
                        template_celltypes = celltypes[sorted_ids][order_back]
                        template_locs = np.array(locs[sorted_ids])[order_back]
                        template_rots = np.array(rots[sorted_ids])[order_back]
                        template_bin = np.array(bin_cat[sorted_ids])[order_back]
                        templates = np.array(eaps[sorted_ids])[order_back]

                # compute gain
                gain_to_int = None
                if np.dtype(dtype).kind == "i":
                    if gain is None:
                        if adc_bit_depth is not None:
                            signal_range = lsb * (2**adc_bit_depth)
                            dtype_depth = np.dtype(dtype).itemsize * 8
                            assert signal_range <= 2**dtype_depth, (
                                f"ADC bit depth and LSB exceed the range of the "
                                f"selected dtype {dtype}. Try reducing them or "
                                f"using a larger dtype"
                            )
                            # in this case we add 3 times the noise level to the amplitude to allow some margin
                            max_template_noise = np.max(np.abs(templates)) + 3 * noise_level
                            templates_noise_range = 2 * (max_template_noise)
                            gain_to_int = signal_range / templates_noise_range
                            gain = 1.0 / gain_to_int
                    else:
                        gain_to_int = 1.0 / gain

                if gain_to_int is not None:
                    if verbose_1:
                        print(f"Templates and noise scaled by gain: {gain_to_int}")
                    templates *= gain_to_int
                    noise_level *= gain_to_int

                self.original_templates = templates

                if drifting:
                    drift_list = []
                    for drift_params in drift_dicts:
                        drift_signal = generate_drift_dict_from_params(
                            duration=duration,
                            template_locations=template_locs,
                            preferred_dir=preferred_dir,
                            **drift_params,
                        )
                        drift_list.append(drift_signal)
                    if verbose_1:
                        print(
                            f"Num. of drift vectors shape {len(drift_list)} with "
                            f"{[len(d['drift_vector_idxs']) for d in drift_list]} samples"
                        )

                self.drift_list = drift_list

                # find overlapping templates
                overlapping = find_overlapping_templates(templates, thresh=overlap_threshold)

                # peak images
                voltage_peaks = []
                for tem in templates:
                    dt = 1.0 / fs.magnitude
                    if not drifting:
                        feat = get_templates_features(tem, ["neg"], dt=dt)
                    else:
                        feat = get_templates_features(tem[0], ["neg"], dt=dt)
                    voltage_peaks.append(-np.squeeze(feat["neg"]))
                voltage_peaks = np.array(voltage_peaks)

                # pad templates
                pad_samples = [int((pp * spike_fs.rescale("kHz")).magnitude) for pp in pad_len]
                if verbose_1:
                    print("Padding template edges")
                t_pad = time.time()
                templates_pad = pad_templates(
                    templates,
                    pad_samples,
                    drifting,
                    np.float32,
                    verbose_2,
                    tmp_file=tmp_templates_pad,
                    parallel=parallel_templates,
                )

                if verbose_1:
                    print("Elapsed pad time:", time.time() - t_pad)

                # resample spikes
                t_rs = time.time()
                f_up = fs
                f_down = spike_fs
                spike_duration_pad = templates_pad.shape[-1]
                if f_up != f_down:
                    if verbose_1:
                        print(f"Resampling templates at {f_up}")
                    n_resample = int(spike_duration_pad * (f_up / f_down))
                    templates_rs = resample_templates(
                        templates_pad,
                        n_resample,
                        f_up,
                        f_down,
                        drifting,
                        np.float32,
                        verbose_2,
                        tmp_file=tmp_templates_rs,
                        parallel=parallel_templates,
                    )
                    # adjust pad_samples to account for resampling
                    pad_samples = [int((pp * fs.rescale("kHz")).magnitude) for pp in pad_len]
                    if verbose_1:
                        print("Elapsed resample time:", time.time() - t_rs)
                    self._to_remove_on_delete.extend(
                        [tmp_templates_rs]
                    )
                else:
                    templates_rs = templates_pad

                if verbose_1:
                    print("Creating time jittering")
                jitter = 1.0 / fs
                t_j = time.time()
                templates = jitter_templates(
                    templates_rs,
                    upsample,
                    fs,
                    n_jitters,
                    jitter,
                    drifting,
                    np.float32,
                    verbose_2,
                    tmp_file=tmp_templates_jit,
                    parallel=parallel_templates,
                )
                if verbose_1:
                    print("Elapsed jitter time:", time.time() - t_j)

                # find cut out samples for convolution after padding and resampling
                pre_peak_fraction = (pad_len[0] + cut_outs[0]) / (np.sum(pad_len) + np.sum(cut_outs))
                samples_pre_peak = int(pre_peak_fraction * templates.shape[-1])
                samples_post_peak = templates.shape[-1] - samples_pre_peak
                cut_outs_samples = [samples_pre_peak, samples_post_peak]

                # smooth edges
                if smooth_percent > 0:
                    sigmoid_samples = int(smooth_percent * pad_samples[0]) // 2 * 2
                    sigmoid_x = np.arange(-sigmoid_samples // 2, sigmoid_samples // 2)
                    b = smooth_strength
                    sig = sigmoid(sigmoid_x, b) + 0.5
                    window = np.ones(templates.shape[-1])
                    window[:sigmoid_samples] = sig
                    window[-sigmoid_samples:] = sig[::-1]

                    if verbose_1:
                        print("Smoothing templates")
                    templates *= window

                # delete temporary preprocessed templates
                del templates_rs, templates_pad
            else:
                gain_to_int = None
                templates = self.templates
                pre_peak_fraction = (pad_len[0] + cut_outs[0]) / (np.sum(pad_len) + np.sum(cut_outs))
                samples_pre_peak = int(pre_peak_fraction * templates.shape[-1])
                samples_post_peak = templates.shape[-1] - samples_pre_peak
                cut_outs_samples = [samples_pre_peak, samples_post_peak]
                template_locs = self.template_locations
                template_rots = self.template_rotations
                template_celltypes = self.template_celltypes
                if celltype_params is not None:
                    if "excitatory" in celltype_params.keys() and "inhibitory" in celltype_params.keys():
                        exc_categories = celltype_params["excitatory"]
                        inh_categories = celltype_params["inhibitory"]
                        template_bin = get_binary_cat(template_celltypes, exc_categories, inh_categories)
                    else:
                        template_bin = np.array(["U"] * len(celltypes))
                else:
                    template_bin = np.array(["U"] * len(celltypes))
                voltage_peaks = self.voltage_peaks
                overlapping = np.array([])
                # ~ if not drifting:
                # ~ velocity_vector = None
                # ~ else:
                # ~ drift_velocity_ums = drift_velocity / 60.
                # ~ velocity_vector = drift_velocity_ums * preferred_dir
                # ~ if verbose_1:
                # ~ print('Drift velocity vector: ', velocity_vector)

            if sync_rate is not None:
                if verbose_1:
                    print("Modifying synchrony of spatially overlapping spikes")
                if verbose_1:
                    print("Overlapping templates: ", overlapping)
                for over in overlapping:
                    if verbose_1:
                        print("Overlapping pair: ", over)
                    spgen.add_synchrony(over, rate=sync_rate, verbose=verbose_2, time_jitt=sync_jitt)
                    # annotate new firing rates
                    fr1 = len(spgen.spiketrains[over[0]].times) / spgen.spiketrains[over[0]].t_stop
                    fr2 = len(spgen.spiketrains[over[1]].times) / spgen.spiketrains[over[1]].t_stop
                    spgen.spiketrains[over[0]].annotate(fr=fr1)
                    spgen.spiketrains[over[1]].annotate(fr=fr2)
            self.overlapping = overlapping

            # find SNR and annotate
            if verbose_1:
                print("Computing spike train SNR")

            for t_i, temp in enumerate(templates):
                min_peak = np.min(temp)
                snr = np.abs(min_peak / float(noise_level))
                spiketrains[t_i].annotate(snr=snr)

            if verbose_1:
                print("Adding spiketrain annotations")
            for i, st in enumerate(spiketrains):
                st.annotate(bintype=template_bin[i], mtype=template_celltypes[i], soma_position=template_locs[i])

            if overlap:
                annotate_overlapping_spikes(spiketrains, overlapping_pairs=overlapping, verbose=verbose_2)

            if verbose_1:
                print("Convolution seed: ", conv_seed)
            np.random.seed(conv_seed)

            amp_mod = []
            cons_spikes = []

            # modulated convolution
            if drifting:
                drifting_units = np.random.permutation(n_neurons)[:n_drifting]
            else:
                drifting_units = []

            if modulation == "template":
                if verbose_1:
                    print("Template modulation")
                for i_s, st in enumerate(spiketrains):
                    if bursting and i_s in bursting_units:
                        if verbose_1:
                            print("Bursting unit: ", i_s)
                        bursting_idx = list(bursting_units).index(i_s)
                        amp, cons = compute_modulation(
                            st,
                            sdrand=sdrand,
                            n_spikes=n_burst_spikes[bursting_idx],
                            exp=exp_decay[bursting_idx],
                            max_burst_duration=max_burst_duration[bursting_idx],
                        )

                        amp_mod.append(amp)
                        cons_spikes.append(cons)
                        st.annotate(
                            bursting=True,
                            exp_decay=exp_decay[bursting_idx],
                            max_spikes_per_burst=n_burst_spikes[bursting_idx],
                            max_burst_duration=max_burst_duration[bursting_idx],
                        )
                    else:
                        amp, cons = compute_modulation(st, mrand=mrand, sdrand=sdrand, n_spikes=0)
                        amp_mod.append(amp)
                        cons_spikes.append(cons)
                        st.annotate(bursting=False, exp_decay=None, max_spikes_per_burst=None, max_burst_duration=None)

            elif modulation == "electrode":
                if verbose_1:
                    print("Electrode modulaton")
                for i_s, st in enumerate(spiketrains):
                    if bursting and i_s in bursting_units:
                        if verbose_1:
                            print("Bursting unit: ", i_s)
                        bursting_idx = list(bursting_units).index(i_s)
                        amp, cons = compute_modulation(
                            st,
                            n_el=n_elec,
                            mrand=mrand,
                            sdrand=sdrand,
                            n_spikes=n_burst_spikes[bursting_idx],
                            exp=exp_decay[bursting_idx],
                            max_burst_duration=max_burst_duration[bursting_idx],
                        )
                        amp_mod.append(amp)

                        cons_spikes.append(cons)
                        st.annotate(
                            bursting=True,
                            exp_decay=exp_decay[bursting_idx],
                            max_spikes_per_burst=n_burst_spikes[bursting_idx],
                            max_burst_duration=max_burst_duration[bursting_idx],
                        )
                    else:
                        amp, cons = compute_modulation(st, n_el=n_elec, mrand=mrand, sdrand=sdrand, n_spikes=0)
                        amp_mod.append(amp)
                        cons_spikes.append(cons)
                        st.annotate(bursting=False, exp_decay=None, max_spikes_per_burst=None, max_burst_duration=None)

            spike_idxs = []
            for st in spiketrains:
                spike_idxs.append((st.times * fs).magnitude.astype("int"))

            # divide in chunks
            chunk_indexes = make_chunk_indexes(duration, chunk_duration, fs)
            seed_list_conv = [np.random.randint(1000) for i in np.arange(len(chunk_indexes))]

            pad_samples_conv = templates.shape[-1]
            # call the loop on chunks
            args = (
                spike_idxs,
                pad_samples_conv,
                modulation,
                drifting,
                drifting_units,
                templates,
                cut_outs_samples,
                self.drift_list,
                verbose_2,
                amp_mod,
                bursting_units,
                shape_mod,
                shape_stretch,
                True,
                voltage_peaks,
                dtype,
                seed_list_conv,
            )

            assignment_dict = {"recordings": recordings, "spike_traces": spike_traces}
            output_list = run_several_chunks(
                chunk_convolution, chunk_indexes, fs, lsb, args, self.n_jobs, self.tmp_mode, assignment_dict
            )

            # if drift then propagate annoations to spiketrains
            for st in np.arange(n_neurons):
                if drifting and st in drifting_units:
                    spiketrains[st].annotate(drifting=True)
                    # ~ template_idxs = np.array([], dtype='int')
                    # ~ for out in output_list:
                    # ~ template_idxs = np.concatenate((template_idxs, out['template_idxs'][st]))
                    # ~ assert len(template_idxs) == len(spiketrains[st])
                    # ~ spiketrains[st].annotate(template_idxs=template_idxs)

                    # TODO find correct drift index based on different sampling frequencies
                    # if drift_vectors.ndim ==1:
                    #     drift_vector = drift_vectors
                    # else:
                    #     drift_vector = drift_vectors[:, st]
                    # drift_index = drift_vector[spike_idxs[st]]
                    # spiketrains[st].annotate(drift_index=drift_index)

        #################
        # Step 2: noise #
        #################
        if verbose_1:
            print("Adding noise")
            print("Noise seed: ", noise_seed)

        np.random.seed(noise_seed)

        if noise_level == 0:
            if verbose_1:
                print("Noise level is set to 0")
        else:
            # divide in chunks
            chunk_indexes = make_chunk_indexes(duration, chunk_duration, fs)
            seed_list_noise = [np.random.randint(1000) for i in np.arange(len(chunk_indexes))]

            if self.tmp_mode == "memmap":
                tmp_path_noise = self.tmp_folder / (tmp_prefix + "mearec_tmp_noise_file.raw")
                additive_noise = np.memmap(tmp_path_noise, shape=(n_samples, n_elec), dtype=dtype, mode="w+")
                # additive_noise = additive_noise.transpose()
                self._to_remove_on_delete.append(tmp_path_noise)
            else:
                additive_noise = np.zeros((n_samples, n_elec), dtype=dtype)

            if noise_mode == "uncorrelated":
                func = chunk_uncorrelated_noise
                args = (
                    n_elec,
                    noise_level,
                    noise_color,
                    color_peak,
                    color_q,
                    color_noise_floor,
                    dtype,
                    seed_list_noise,
                )
                assignment_dict = {"additive_noise": additive_noise}

                run_several_chunks(func, chunk_indexes, fs, lsb, args, self.n_jobs, self.tmp_mode, assignment_dict)

            elif noise_mode == "distance-correlated":
                cov_dist = np.zeros((n_elec, n_elec))
                for i, el in enumerate(mea.positions):
                    for j, p in enumerate(mea.positions):
                        if i != j:
                            cov_dist[i, j] = (0.5 * half_dist) / np.linalg.norm(el - p)
                        else:
                            cov_dist[i, j] = 1

                func = chunk_distance_correlated_noise
                args = (
                    noise_level,
                    cov_dist,
                    n_elec,
                    noise_color,
                    color_peak,
                    color_q,
                    color_noise_floor,
                    dtype,
                    seed_list_noise,
                )
                assignment_dict = {"additive_noise": additive_noise}

                run_several_chunks(func, chunk_indexes, fs, lsb, args, self.n_jobs, self.tmp_mode, assignment_dict)

            elif noise_mode == "far-neurons":
                from . import SpikeTrainGenerator

                if self.tmp_mode == "memmap":
                    # file names for templates
                    tmp_templates_noise_pad = self.tmp_folder / (tmp_prefix + "templates_noise_pad.raw")
                    tmp_templates_noise_rs = self.tmp_folder / (tmp_prefix + "templates_noise_resample.raw")
                    self._to_remove_on_delete.extend([tmp_templates_noise_pad, tmp_templates_noise_rs])
                else:
                    tmp_templates_noise_pad = None
                    tmp_templates_noise_rs = None
                idxs_cells, selected_cat = select_templates(
                    locs,
                    eaps,
                    bin_cat=None,
                    n_exc=far_neurons_n,
                    n_inh=0,
                    x_lim=x_lim,
                    y_lim=y_lim,
                    z_lim=z_lim,
                    min_amp=0,
                    max_amp=far_neurons_max_amp,
                    min_dist=1,
                    verbose=False,
                )
                idxs_cells = sorted(idxs_cells)
                templates_noise = eaps[idxs_cells]
                # Alessio :
                # TODO handle drift for far neurons ?????
                # template_noise_locs = locs[idxs_cells]
                if drifting:
                    templates_noise = templates_noise[:, 0]
                if gain_to_int is not None:
                    template_noise *= gain_to_int

                # pad spikes
                pad_samples = [int((pp * spike_fs.rescale("kHz")).magnitude) for pp in pad_len]
                if verbose_1:
                    print("Padding noisy template edges")
                t_pad = time.time()
                templates_noise_pad = pad_templates(
                    templates_noise,
                    pad_samples,
                    drifting,
                    dtype,
                    verbose_2,
                    tmp_file=tmp_templates_noise_pad,
                    parallel=True,
                )
                if verbose_1:
                    print("Elapsed pad time:", time.time() - t_pad)

                # resample templates
                t_rs = time.time()
                f_up = fs
                f_down = spike_fs
                spike_duration_pad = templates_noise_pad.shape[-1]
                if f_up != f_down:
                    n_resample = int(spike_duration_pad * (f_up / f_down))
                    templates_noise = resample_templates(
                        templates_noise_pad,
                        n_resample,
                        f_up,
                        f_down,
                        drifting,
                        dtype,
                        verbose_2,
                        tmp_file=tmp_templates_noise_rs,
                    )
                    # adjust pad_samples to account for resampling
                    pad_samples = [int((pp * fs.rescale("kHz")).magnitude) for pp in pad_len]
                    if verbose_1:
                        print("Elapsed resample time:", time.time() - t_rs)
                else:
                    templates_noise = templates_noise_pad

                # find cut out samples for convolution after padding and resampling
                pre_peak_fraction = (pad_len[0] + cut_outs[0]) / (np.sum(pad_len) + np.sum(cut_outs))
                samples_pre_peak = int(pre_peak_fraction * templates_noise.shape[-1])
                samples_post_peak = templates_noise.shape[-1] - samples_pre_peak
                cut_outs_samples = [samples_pre_peak, samples_post_peak]

                del templates_noise_pad

                # create noisy spiketrains
                if verbose_1:
                    print("Generating noisy spike trains")
                noisy_spiketrains_params = params["spiketrains"]
                noisy_spiketrains_params["n_exc"] = int(far_neurons_n * far_neurons_exc_inh_ratio)
                noisy_spiketrains_params["n_inh"] = far_neurons_n - noisy_spiketrains_params["n_exc"]
                noisy_spiketrains_params["seed"] = noise_seed
                spgen_noise = SpikeTrainGenerator(params=noisy_spiketrains_params)
                spgen_noise.generate_spikes()
                spiketrains_noise = spgen_noise.spiketrains

                spike_idxs_noise = []
                for st in spiketrains_noise:
                    spike_idxs_noise.append((st.times * fs).magnitude.astype("int"))

                if verbose_1:
                    print("Convolving noisy spike trains")
                templates_noise = templates_noise.reshape(
                    (templates_noise.shape[0], 1, templates_noise.shape[1], templates_noise.shape[2])
                )

                # call the loop on chunks
                args = (
                    spike_idxs_noise,
                    0,
                    "none",
                    False,
                    # None,
                    None,
                    templates_noise,
                    cut_outs_samples,
                    None,
                    # template_noise_locs, None, None, None, None, None, None,
                    verbose_2,
                    None,
                    None,
                    False,
                    None,
                    False,
                    None,
                    dtype,
                    seed_list_noise,
                )
                assignment_dict = {"recordings": additive_noise}
                run_several_chunks(
                    chunk_convolution, chunk_indexes, fs, lsb, args, self.n_jobs, self.tmp_mode, assignment_dict
                )

                # removing mean
                for i, m in enumerate(np.mean(additive_noise, axis=0)):
                    additive_noise[:, i] -= m
                # adding noise floor
                for i, s in enumerate(np.std(additive_noise, axis=0)):
                    additive_noise[:, i] += far_neurons_noise_floor * s * np.random.randn(additive_noise.shape[0])
                # scaling noise
                noise_scale = noise_level / np.std(additive_noise, axis=0)
                if verbose_1:
                    print("Scaling to reach desired level")
                for i, n in enumerate(noise_scale):
                    additive_noise[:, i] *= n

            # Add it to recordings
            recordings += additive_noise

        ##################
        # Step 3: filter #
        ##################
        if filter:
            if verbose_1:
                print("Filtering")
                if cutoff.size == 1:
                    print("High-pass cutoff", cutoff)
                elif cutoff.size == 2:
                    print("Band-pass cutoff", cutoff)

            chunk_indexes = make_chunk_indexes(duration, chunk_duration, fs)

            # compute pad samples as 3 times the low-cutoff period
            if cutoff.size == 1:
                pad_samples_filt = 3 * int((1.0 / cutoff * fs).magnitude)
            elif cutoff.size == 2:
                pad_samples_filt = 3 * int((1.0 / cutoff[0] * fs).magnitude)

            # call the loop on chunks
            args = (
                recordings,
                pad_samples_filt,
                cutoff,
                filter_order,
                filter_mode,
                dtype,
            )
            assignment_dict = {
                "filtered_chunk": recordings,
            }
            # Done in loop (as before) : this cannot be done in parralel because of bug transpose in joblib!!!!!!!!!!!!!
            run_several_chunks(
                chunk_apply_filter, chunk_indexes, fs, lsb, args, self.n_jobs, self.tmp_mode, assignment_dict
            )

        if gain is not None:
            gain_to_uV = gain
        else:
            gain_to_uV = 1.0

        # assign class variables
        params["templates"]["overlapping"] = np.array(overlapping)
        self.recordings = recordings
        self.timestamps = timestamps
        self.gain_to_uV = gain_to_uV
        self.channel_positions = mea_pos
        self.templates = np.squeeze(templates)
        self.template_locations = template_locs
        self.template_rotations = template_rots
        self.template_celltypes = template_celltypes
        self.spiketrains = spiketrains
        self.voltage_peaks = voltage_peaks
        self.spike_traces = spike_traces
        self.info = params
        self.fs = fs

        #############################
        # Step 4: extract waveforms #
        #############################
        if not only_noise:
            if extract_waveforms:
                if verbose_1:
                    print("Extracting spike waveforms")
                self.extract_waveforms()

    def annotate_overlapping_spikes(self, parallel=True):
        """
        Annnotate spike trains with overlapping information.

        parallel : bool
            If True, spike trains are annotated in parallel
        """
        if self.info["templates"]["overlapping"] is None or len(self.info["templates"]["overlapping"]) == 0:
            if self._verbose_1:
                print("Finding overlapping spikes")
            if len(self.templates.shape) == 3:
                templates = self.templates
            elif len(self.templates.shape) == 4:
                # drifting + no jitt or no drifting + jitt
                templates = self.templates[:, 0]
            elif len(self.templates.shape) == 5:
                # drifting + jitt
                templates = self.templates[:, 0, 0]
            self.overlapping = find_overlapping_templates(templates, thresh=self.info["templates"]["overlap_threshold"])
            print("Overlapping templates: ", self.overlapping)
            self.info["templates"]["overlapping"] = self.overlapping
        annotate_overlapping_spikes(self.spiketrains, overlapping_pairs=self.overlapping, parallel=parallel)

    def extract_waveforms(self, cut_out=[0.5, 2]):
        """
        Extract waveforms from spike trains and recordings.

        Parameters
        ----------
        cut_out : float or list
            Ms before and after peak to cut out. If float the cut is symmetric.
        """
        fs = self.info["recordings"]["fs"] * pq.Hz
        extract_wf(self.spiketrains, self.recordings, fs=fs, cut_out=cut_out)

    def extract_templates(self, cut_out=[0.5, 2], recompute=False):
        """
        Extract templates from spike trains.

        Parameters
        ----------
        cut_out : float or list
            Ms before and after peak to cut out. If float the cut is symmetric.
        recompute :  bool
            If True, templates are recomputed from extracted waveforms
        """
        fs = self.info["recordings"]["fs"] * pq.Hz

        if not len(self.spiketrains) == 0:
            if self.spiketrains[0].waveforms is None:
                extract_wf(self.spiketrains, self.recordings, fs=fs, cut_out=cut_out)

        if self.tempgen is None or len(self.templates) == 0 or not recompute:
            wfs = [st.waveforms for st in self.spiketrains]
            templates = np.array([np.mean(wf, axis=0) for wf in wfs])
            if np.array(cut_out).size == 1:
                cut_out = [cut_out, cut_out]
            self.info["templates"]["cut_out"] = cut_out
            self.info["templates"]["pad_len"] = [0, 0]
            self.templates = templates[:, np.newaxis]
        else:
            raise Exception(
                "templates are already computed. Use the 'recompute' argument to compute them from "
                "extracted waveforms"
            )


def make_chunk_indexes(total_duration, chunk_duration, fs):
    """
    Construct chunks list.
    Return a list of (start, stop) indexes.
    """
    fs_float = fs.rescale("Hz").magnitude
    chunk_size = int(chunk_duration.rescale("s").magnitude * fs_float)
    total_length = int(total_duration.rescale("s").magnitude * fs_float)

    if chunk_size == 0:
        chunk_indexes = [
            (0, total_length),
        ]
    else:
        n = int(np.floor(total_length / chunk_size))
        chunk_indexes = [(i * chunk_size, (i + 1) * chunk_size) for i in range(n)]
        if (total_length % chunk_size) > 0:
            chunk_indexes.append((n * chunk_size, total_length))

    return chunk_indexes


def run_several_chunks(func, chunk_indexes, fs, lsb, args, n_jobs, tmp_mode, assignment_dict):
    """
    Run a function on a list of chunks.

    this can be done in loop if n_jobs=1 (or 0)
    or in paralell if n_jobs>1

    The function can return
    """

    # create task list
    arg_tasks = []
    karg_tasks = []

    for ch, (i_start, i_stop) in enumerate(chunk_indexes):
        fs_Hz = fs.rescale("Hz").magnitude

        arg_task = (ch, i_start, i_stop, fs_Hz, lsb) + args
        arg_tasks.append(arg_task)

        karg_task = dict(assignment_dict=assignment_dict, tmp_mode=tmp_mode)
        karg_tasks.append(karg_task)

    # run chunks
    if n_jobs in (0, 1):
        # simple loop
        output_list = []
        for ch, (i_start, i_stop) in enumerate(chunk_indexes):
            out = func(*arg_tasks[ch], **karg_tasks[ch])
            output_list.append(out)

            if tmp_mode is None:
                for key, full_arr in assignment_dict.items():
                    out_chunk = out[key]
                    full_arr[i_start:i_stop] += out_chunk
            elif tmp_mode == "memmap":
                pass
                # Nothing to do here because done inside the func with FuncThenAddChunk

    else:
        # parallel
        output_list = Parallel(n_jobs=n_jobs)(
            delayed(func)(*arg_task, **karg_task) for arg_task, karg_task in zip(arg_tasks, karg_tasks)
        )

        if tmp_mode == "memmap":
            pass
            # Nothing to do here because done inside the func
        else:
            # This case is very unefficient because it double the memory usage!!!!!!!
            for ch, (i_start, i_stop) in enumerate(chunk_indexes):
                for key, full_arr in assignment_dict.items():
                    full_arr[i_start:i_stop] += output_list[ch][key]

    return output_list
