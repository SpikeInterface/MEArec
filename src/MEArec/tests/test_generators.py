import shutil
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from click.testing import CliRunner
from packaging.version import parse

import MEArec as mr
from MEArec.cli import cli

DEBUG = False


if DEBUG:
    import matplotlib.pyplot as plt

    plt.ion()
    plt.show()


class TestGenerators(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        info, info_folder = mr.get_default_config()
        cell_models_folder = mr.get_default_cell_models_folder()
        self.num_cells = len(
            [f for f in Path(cell_models_folder).iterdir() if "mods" not in f.name and not f.name.startswith(".")]
        )
        self.n = 10
        self.n_drift = 5

        templates_params = mr.get_default_templates_params()
        rec_params = mr.get_default_recordings_params()

        # Set seed
        np.random.seed(2308)

        if not DEBUG:
            self.test_dir = Path(tempfile.mkdtemp())
        else:
            self.test_dir = Path("./tmp").absolute()

        if not (self.test_dir / "templates.h5").is_file() and not (self.test_dir / "templates_drift.h5").is_file():
            templates_params["n"] = self.n
            templates_params["ncontacts"] = 1
            templates_params["probe"] = "Neuronexus-32"
            templates_folder = info["templates_folder"]
            print("Generating non-drifting templates")
            templates_params["min_amp"] = 10
            print(templates_params)
            self.tempgen = mr.gen_templates(
                cell_models_folder,
                templates_tmp_folder=templates_folder,
                params=templates_params,
                parallel=True,
                delete_tmp=True,
                verbose=False,
            )
            self.templates_params = deepcopy(templates_params)
            self.num_templates, self.num_chan, self.num_samples = self.tempgen.templates.shape

            templates_params["drifting"] = True
            templates_params["n"] = self.n_drift
            templates_params["drift_steps"] = 10
            templates_params["rot"] = "norot"
            templates_params["min_amp"] = 10
            print("Generating drifting templates")
            self.tempgen_drift = mr.gen_templates(
                cell_models_folder,
                templates_tmp_folder=templates_folder,
                params=templates_params,
                parallel=True,
                delete_tmp=True,
                verbose=False,
            )
            self.templates_params_drift = deepcopy(templates_params)
            self.num_steps_drift = self.tempgen_drift.templates.shape[1]

            print("Making test recordings to test load functions")
            mr.save_template_generator(self.tempgen, self.test_dir / "templates.h5")
            mr.save_template_generator(self.tempgen_drift, self.test_dir / "templates_drift.h5")
        else:
            print(self.test_dir / "templates.h5")
            self.tempgen = mr.load_templates(self.test_dir / "templates.h5", return_h5_objects=False)
            self.tempgen_drift = mr.load_templates(self.test_dir / "templates_drift.h5")
            self.templates_params = self.tempgen.info["params"]
            self.templates_params_drift = self.tempgen_drift.info["params"]
            self.num_steps_drift = self.tempgen_drift.templates.shape[1]
            self.num_templates, self.num_chan, self.num_samples = self.tempgen.templates.shape

        if not (self.test_dir / "recordings.h5").is_file():
            ne = 2
            ni = 1
            rec_params["spiketrains"]["n_exc"] = ne
            rec_params["spiketrains"]["n_inh"] = ni
            rec_params["recordings"]["modulation"] = "none"
            rec_params["recordings"]["filter"] = False
            rec_params["recordings"]["extract_waveforms"] = True
            rec_params["recordings"]["overlap"] = True
            rec_params["spiketrains"]["duration"] = 2
            rec_params["templates"]["min_dist"] = 1

            self.recgen = mr.gen_recordings(params=rec_params, tempgen=self.tempgen, verbose=False)
            self.recgen.annotate_overlapping_spikes(parallel=False)
            mr.save_recording_generator(self.recgen, self.test_dir / "recordings.h5")
        else:
            self.recgen = mr.load_recordings(self.test_dir / "recordings.h5", return_h5_objects=False)

    @classmethod
    def tearDownClass(self):
        # Remove the directory after the test
        if not DEBUG:
            shutil.rmtree(self.test_dir)

    def test_gen_templates(self):
        print("Test templates generation")
        n = self.n
        num_cells = self.num_cells
        templates_params = self.templates_params

        assert self.tempgen.templates.shape[0] == (n * num_cells)
        assert len(self.tempgen.locations) == (n * num_cells)
        assert len(self.tempgen.rotations) == (n * num_cells)
        assert len(self.tempgen.celltypes) == (n * num_cells)
        assert len(np.unique(self.tempgen.celltypes)) == num_cells
        assert (
            np.min(self.tempgen.locations[:, 0]) > templates_params["xlim"][0]
            and np.max(self.tempgen.locations[:, 0]) < templates_params["xlim"][1]
        )

    def test_gen_templates_drift(self):
        print("Test drifting templates generation")
        n = self.n_drift
        num_cells = self.num_cells
        n_steps = self.num_steps_drift

        assert self.tempgen_drift.templates.shape[0] == (n * num_cells)
        assert self.tempgen_drift.locations.shape == (n * num_cells, n_steps, 3)
        assert len(self.tempgen_drift.rotations) == (n * num_cells)
        assert len(self.tempgen_drift.celltypes) == (n * num_cells)
        assert len(np.unique(self.tempgen_drift.celltypes)) == num_cells
        assert self.tempgen_drift.templates.shape[1] == self.num_steps_drift

    def test_gen_templates_from_tempgen(self):
        print("Test templates generation from existing template generator")
        cell_models_folder = mr.get_default_cell_models_folder()

        # no drift
        params = self.templates_params
        params["probe"] = "Neuropixels-24"
        tempgen2 = mr.gen_templates(cell_models_folder=cell_models_folder, params=params, tempgen=self.tempgen)

        assert tempgen2.templates.shape[0] == self.tempgen.templates.shape[0]
        # assert that all locations are the same
        for loc_new, loc_old in zip(tempgen2.locations, self.tempgen.locations):
            assert np.allclose(loc_new, loc_old)

        # drift
        params = self.templates_params_drift
        params["probe"] = "Neuropixels-24"
        tempgen_drift2 = mr.gen_templates(
            cell_models_folder=cell_models_folder, params=params, tempgen=self.tempgen_drift
        )

        assert tempgen_drift2.templates.shape[0] == self.tempgen_drift.templates.shape[0]
        # assert that all locations are the same
        for loc_new, loc_old in zip(tempgen2.locations, self.tempgen.locations):
            for loc_new_d, loc_old_d in zip(loc_new, loc_old):
                assert np.allclose(loc_new_d, loc_old_d)

    def test_gen_templates_beta_distr(self):
        print("Test templates generation from beta distributions")
        cell_models_folder = mr.get_default_cell_models_folder()

        # no drift
        params = self.templates_params
        params["probe"] = "Neuronexus-32"

        # beta distr
        params["x_distr"] = "beta"
        tempgen_beta = mr.gen_templates(cell_models_folder=cell_models_folder, params=params)

        assert tempgen_beta.templates.shape[0] == self.tempgen.templates.shape[0]

        # drift
        params = self.templates_params_drift
        params["probe"] = "Neuronexus-32"
        params["x_distr"] = "beta"
        params["beta_distr_params"] = [2, 5]

        tempgen_drift_beta = mr.gen_templates(cell_models_folder=cell_models_folder, params=params)

        assert tempgen_drift_beta.templates.shape[0] == self.tempgen_drift.templates.shape[0]

    def test_gen_spiketrains(self):
        import elephant.statistics as stat
        
        print("Test spike train generation")
        rec_params = mr.get_default_recordings_params()
        sp_params = rec_params["spiketrains"]
        spgen = mr.SpikeTrainGenerator(sp_params, seed=0)
        spgen.generate_spikes()

        # check ref period
        for st in spgen.spiketrains:
            isi = stat.isi(st).rescale("ms")
            assert np.all(isi.magnitude > sp_params["ref_per"])
            assert (1 / np.mean(isi.rescale("s"))) > sp_params["min_rate"]

        sp_params["process"] = "gamma"
        spgen = mr.SpikeTrainGenerator(sp_params, seed=0)
        spgen.generate_spikes()
        for st in spgen.spiketrains:
            isi = stat.isi(st).rescale("ms")
            assert np.all(isi.magnitude > sp_params["ref_per"])
            assert (1 / np.mean(isi.rescale("s"))) > sp_params["min_rate"]

        spgen = mr.gen_spiketrains(sp_params, seed=0)
        spiketrains = spgen.spiketrains
        spgen_st = mr.gen_spiketrains(spiketrains=spiketrains)
        for st, st_ in zip(spgen.spiketrains, spgen_st.spiketrains):
            assert np.allclose(st.times.magnitude, st_.times.magnitude)

    def test_gen_recordings_with_spiketrains(self):
        print("Test recording generation - from spiketrains")
        rec_params = mr.get_default_recordings_params()
        sp_params = rec_params["spiketrains"]
        sp_params["n_exc"] = 2
        sp_params["n_inh"] = 1
        sp_params["duration"] = 5
        spgen = mr.gen_spiketrains(sp_params)
        recgen = mr.gen_recordings(params=rec_params, spgen=spgen, tempgen=self.tempgen, verbose=False)

        for st, st_ in zip(spgen.spiketrains, recgen.spiketrains):
            assert np.allclose(st.times.magnitude, st_.times.magnitude)
        del recgen

    def test_gen_recordings_modulations(self):
        print("Test recording generation - modulation")
        ne = 1
        ni = 2
        fe = 30
        fi = 50
        num_chan = self.num_chan
        n_neurons = ne + ni

        rec_params = mr.get_default_recordings_params()

        rec_params["spiketrains"]["n_exc"] = ne
        rec_params["spiketrains"]["n_inh"] = ni
        rec_params["spiketrains"]["f_exc"] = fe
        rec_params["spiketrains"]["f_inh"] = fi
        rec_params["spiketrains"]["duration"] = 3
        n_jitter = [1, 5]
        modulations = ["none", "electrode", "template"]
        chunk_rec = [0, 2]

        rec_params["templates"]["min_dist"] = 1
        bursting = [False, True]

        for mod in modulations:
            for b in bursting:
                for j in n_jitter:
                    for ch in chunk_rec:
                        print("Modulation: modulation", mod, "bursting", b, "jitter", j, "chunk", ch)
                        rec_params["templates"]["n_jitters"] = j
                        rec_params["recordings"]["modulation"] = mod
                        rec_params["recordings"]["bursting"] = b
                        rec_params["recordings"]["chunk_duration"] = ch

                        if mod == "electrode" and b is True and j == 5:
                            rec_params["cell_types"] = None

                        recgen_mod = mr.gen_recordings(params=rec_params, tempgen=self.tempgen, verbose=False)

                        assert recgen_mod.recordings.shape[1] == num_chan
                        assert len(recgen_mod.spiketrains) == n_neurons
                        assert recgen_mod.channel_positions.shape == (num_chan, 3)
                        if j == 1:
                            assert recgen_mod.templates.shape[:2] == (n_neurons, num_chan)
                        else:
                            assert recgen_mod.templates.shape[:3] == (n_neurons, j, num_chan)
                        assert recgen_mod.voltage_peaks.shape == (n_neurons, num_chan)
                        assert recgen_mod.spike_traces.shape[1] == n_neurons
                        del recgen_mod

    def test_gen_recordings_bursting(self):
        print("Test recording generation - shape_mod")
        rates = [30, 50]
        types = ["E", "I"]
        num_chan = self.num_chan
        n_neurons = len(rates)

        rec_params = mr.get_default_recordings_params()

        rec_params["spiketrains"]["rates"] = rates
        rec_params["spiketrains"]["types"] = types
        rec_params["spiketrains"]["duration"] = 3
        n_jitter = 4
        rec_params["templates"]["n_jitters"] = n_jitter
        rec_params["recordings"]["modulation"] = "electrode"
        rec_params["recordings"]["bursting"] = True
        rec_params["recordings"]["shape_mod"] = True
        rec_params["recordings"]["sync_rate"] = 0.2
        rec_params["recordings"]["overlap"] = True
        rec_params["recordings"]["extract_waveforms"] = True
        rec_params["templates"]["min_dist"] = 1
        rec_params["templates"]["min_amp"] = 0

        recgen_burst = mr.gen_recordings(params=rec_params, tempgen=self.tempgen, verbose=False)
        recgen_burst.extract_waveforms()

        assert recgen_burst.recordings.shape[1] == num_chan
        assert len(recgen_burst.spiketrains) == n_neurons
        assert recgen_burst.channel_positions.shape == (num_chan, 3)
        assert recgen_burst.templates.shape[:3] == (n_neurons, n_jitter, num_chan)
        assert recgen_burst.voltage_peaks.shape == (n_neurons, num_chan)
        assert recgen_burst.spike_traces.shape[1] == n_neurons
        del recgen_burst

        rec_params["recordings"]["modulation"] = "template"
        rec_params["recordings"]["n_bursting"] = 2

        recgen_burst = mr.gen_recordings(params=rec_params, tempgen=self.tempgen, verbose=False)
        recgen_burst.extract_waveforms()

        assert recgen_burst.recordings.shape[1] == num_chan
        assert len(recgen_burst.spiketrains) == n_neurons
        assert recgen_burst.channel_positions.shape == (num_chan, 3)
        assert recgen_burst.templates.shape[:3] == (n_neurons, n_jitter, num_chan)
        assert recgen_burst.voltage_peaks.shape == (n_neurons, num_chan)
        assert recgen_burst.spike_traces.shape[1] == n_neurons
        del recgen_burst

    def test_gen_recordings_multiple_bursting_modulation(self):
        print("Test recording generation - multiple bursting modulation")
        rates = [10, 20, 30]
        types = ["E", "E", "I"]
        num_chan = self.num_chan
        n_neurons = len(rates)

        rec_params = mr.get_default_recordings_params()

        rec_params["spiketrains"]["rates"] = rates
        rec_params["spiketrains"]["types"] = types
        rec_params["spiketrains"]["duration"] = 3
        n_jitter = 4
        rec_params["templates"]["n_jitters"] = n_jitter
        rec_params["recordings"]["modulation"] = "electrode"
        rec_params["recordings"]["bursting"] = True
        rec_params["recordings"]["shape_mod"] = True
        rec_params["recordings"]["sync_rate"] = 0.2
        rec_params["recordings"]["overlap"] = True
        rec_params["recordings"]["extract_waveforms"] = True
        rec_params["templates"]["min_dist"] = 1
        rec_params["templates"]["min_amp"] = 0

        rec_params["recordings"]["exp_decay"] = [0.1, 0.15, 0.2]
        rec_params["recordings"]["max_burst_duration"] = [50, 100, 150]
        rec_params["recordings"]["n_burst_spikes"] = [5, 10, 15]
        rec_params["recordings"]["shape_stretch"] = 10

        recgen_burst = mr.gen_recordings(params=rec_params, tempgen=self.tempgen, verbose=False)
        recgen_burst.extract_waveforms()

        assert recgen_burst.recordings.shape[1] == num_chan
        assert len(recgen_burst.spiketrains) == n_neurons
        assert recgen_burst.channel_positions.shape == (num_chan, 3)
        assert recgen_burst.templates.shape[:3] == (n_neurons, n_jitter, num_chan)
        assert recgen_burst.voltage_peaks.shape == (n_neurons, num_chan)
        assert recgen_burst.spike_traces.shape[1] == n_neurons

        for st in recgen_burst.spiketrains:
            assert st.annotations["bursting"]
        del recgen_burst

        rec_params["recordings"]["modulation"] = "template"
        rec_params["recordings"]["bursting_units"] = [1, 2]
        rec_params["recordings"]["exp_decay"] = [0.1, 0.15]
        rec_params["recordings"]["max_burst_duration"] = [50, 100]
        rec_params["recordings"]["n_burst_spikes"] = [10, 15]
        rec_params["recordings"]["shape_stretch"] = 20

        recgen_burst = mr.gen_recordings(params=rec_params, tempgen=self.tempgen, verbose=False)
        recgen_burst.extract_waveforms()

        assert recgen_burst.recordings.shape[1] == num_chan
        assert len(recgen_burst.spiketrains) == n_neurons
        assert recgen_burst.channel_positions.shape == (num_chan, 3)
        assert recgen_burst.templates.shape[:3] == (n_neurons, n_jitter, num_chan)
        assert recgen_burst.voltage_peaks.shape == (n_neurons, num_chan)
        assert recgen_burst.spike_traces.shape[1] == n_neurons
        for i, st in enumerate(recgen_burst.spiketrains):
            if i in rec_params["recordings"]["bursting_units"]:
                assert st.annotations["bursting"]
            else:
                assert not st.annotations["bursting"]
        del recgen_burst

    def test_gen_recordings_noise(self):
        print("Test recording generation - noise")
        ne = 0
        ni = 0
        num_chan = self.num_chan
        n_neurons = ne + ni

        rec_params = mr.get_default_recordings_params()

        rec_params["spiketrains"]["n_exc"] = ne
        rec_params["spiketrains"]["n_inh"] = ni
        rec_params["spiketrains"]["duration"] = 3
        n_jitter = 3
        rec_params["templates"]["n_jitters"] = n_jitter
        rec_params["templates"]["min_dist"] = 1
        rec_params["recordings"]["modulation"] = "none"
        noise_modes = ["uncorrelated", "distance-correlated"]
        chunk_noise = [0, 2]
        noise_color = [True, False]
        noise_level = 10
        rec_params["recordings"]["noise_level"] = noise_level
        rec_params["recordings"]["filter"] = False

        for mode in noise_modes:
            for ch in chunk_noise:
                for color in noise_color:
                    print("Noise: mode", mode, "chunks", ch, "color", color)
                    rec_params["recordings"]["noise_mode"] = mode
                    rec_params["recordings"]["chunk_duration"] = ch
                    rec_params["recordings"]["noise_color"] = color
                    recgen_noise = mr.gen_recordings(params=rec_params, tempgen=self.tempgen, verbose=False)

                    if mode == "uncorrelated" and ch == 0 and not noise_color:
                        rec_params["recordings"]["fs"] = 30000
                        rec_params["recordings"]["chunk_duration"] = 1
                    if noise_color:
                        rec_params["recordings"]["filter_cutoff"] = 500

                    assert recgen_noise.recordings.shape[1] == num_chan
                    assert len(recgen_noise.spiketrains) == n_neurons
                    assert recgen_noise.channel_positions.shape == (num_chan, 3)
                    assert len(recgen_noise.spike_traces) == n_neurons
                    assert np.isclose(np.std(recgen_noise.recordings), noise_level, atol=1)
                    del recgen_noise

    def test_gen_recordings_far_neurons(self):
        print("Test recording generation - far neurons")
        ne = 0
        ni = 0
        num_chan = self.num_chan
        n_neurons = ne + ni

        rec_params = mr.get_default_recordings_params()

        rec_params["spiketrains"]["n_exc"] = ne
        rec_params["spiketrains"]["n_inh"] = ni
        rec_params["spiketrains"]["duration"] = 3
        n_jitter = 3
        rec_params["templates"]["n_jitters"] = n_jitter
        rec_params["templates"]["min_dist"] = 1
        rec_params["recordings"]["modulation"] = "none"
        rec_params["recordings"]["noise_mode"] = "far-neurons"
        rec_params["recordings"]["far_neurons_n"] = 10
        rec_params["recordings"]["far_neurons_max_amp"] = 100
        rec_params["recordings"]["far_neurons_exc_inh_ratio"] = 0.8
        noise_level = 20
        rec_params["recordings"]["noise_level"] = noise_level
        rec_params["recordings"]["filter"] = False
        recgen_noise = mr.gen_recordings(params=rec_params, tempgen=self.tempgen, verbose=False)

        assert recgen_noise.recordings.shape[1] == num_chan
        assert recgen_noise.channel_positions.shape == (num_chan, 3)
        assert len(recgen_noise.spiketrains) == n_neurons
        assert len(recgen_noise.spike_traces) == n_neurons
        assert np.isclose(np.std(recgen_noise.recordings), noise_level, atol=1)
        del recgen_noise

        rec_params["recordings"]["chunk_duration"] = 1
        recgen_noise = mr.gen_recordings(params=rec_params, tempgen=self.tempgen, verbose=False)

        assert recgen_noise.recordings.shape[1] == num_chan
        assert recgen_noise.channel_positions.shape == (num_chan, 3)
        assert len(recgen_noise.spiketrains) == n_neurons
        assert len(recgen_noise.spike_traces) == n_neurons
        assert np.isclose(np.std(recgen_noise.recordings), noise_level, atol=1)
        del recgen_noise

    def test_gen_recordings_filters(self):
        print("Test recording generation - filters")
        ne = 1
        ni = 1
        num_chan = self.num_chan
        n_neurons = ne + ni

        rec_params = mr.get_default_recordings_params()

        filter_modes = ["filtfilt", "lfilter"]
        filter_orders = [1, 3, 5]
        filter_cutoffs = [300, 500, [300, 3000], [300, 6000]]
        chunk_rec = [0, 2]

        rec_params["spiketrains"]["n_exc"] = ne
        rec_params["spiketrains"]["n_inh"] = ni
        rec_params["spiketrains"]["duration"] = 5
        rec_params["recordings"]["filter"] = True
        rec_params["templates"]["min_dist"] = 1

        for mode in filter_modes:
            for order in filter_orders:
                for cutoff in filter_cutoffs:
                    for ch in chunk_rec:
                        print(f"Filter: mode {mode} order {order} cutoff {cutoff} chunk dur {ch}")
                        rec_params["recordings"]["chunk_duration"] = ch
                        rec_params["recordings"]["filter_mode"] = mode
                        rec_params["recordings"]["filter_order"] = order
                        rec_params["recordings"]["filter_cutoff"] = cutoff

                        recgen_filt = mr.gen_recordings(params=rec_params, tempgen=self.tempgen, verbose=False)
                        assert recgen_filt.recordings.shape[1] == num_chan
                        assert len(recgen_filt.spiketrains) == n_neurons
                        assert recgen_filt.channel_positions.shape == (num_chan, 3)
                        assert recgen_filt.spike_traces.shape[1] == n_neurons
                        del recgen_filt

    def test_gen_recordings_drift(self):
        print("Test recording generation - drift")
        ne = 1
        ni = 1
        num_chan = self.num_chan
        n_neurons = ne + ni

        rec_params = mr.get_default_recordings_params()

        rec_params["spiketrains"]["n_exc"] = ne
        rec_params["spiketrains"]["n_inh"] = ni
        rec_params["spiketrains"]["duration"] = 5
        n_jitter = [1, 3]
        rec_params["recordings"]["drifting"] = True
        rec_params["recordings"]["slow_drift_velocity"] = 500
        rec_params["templates"]["min_dist"] = 1
        chunk_rec = [0, 2]

        modulations = ["none", "template", "electrode"]
        bursting = [False, True]
        drift_mode_speeds = ["slow", "fast"]
        drift_mode_probes = ["rigid", "non-rigid"]

        for i, mod in enumerate(modulations):
            for dms in drift_mode_speeds:
                for dmp in drift_mode_probes:
                    for b in bursting:
                        for j in n_jitter:
                            for ch in chunk_rec:
                                print(
                                    "Drifting: modulation",
                                    mod,
                                    "bursting",
                                    b,
                                    "jitter",
                                    j,
                                    "drift mode speed",
                                    dms,
                                    "drift mode probe",
                                    dmp,
                                    "chunk",
                                    ch,
                                )
                                rec_params["templates"]["n_jitters"] = j
                                rec_params["recordings"]["modulation"] = mod
                                rec_params["recordings"]["bursting"] = b
                                rec_params["recordings"]["chunk_duration"] = ch
                                rec_params["recordings"]["drift_mode_speed"] = dms
                                rec_params["recordings"]["drift_mode_probe"] = dmp

                                if i == len(modulations) - 1:
                                    rec_params["recordings"]["fs"] = 30000
                                    rec_params["recordings"]["n_drifting"] = 1
                                if mod == "electrode" and b is True and j == 5:
                                    rec_params["cell_types"] = None
                                    rec_params["recordings"]["shape_mod"] = True
                                recgen_drift = mr.gen_recordings(
                                    params=rec_params, tempgen=self.tempgen_drift, verbose=False
                                )
                                assert recgen_drift.recordings.shape[1] == num_chan
                                assert len(recgen_drift.spiketrains) == n_neurons
                                assert recgen_drift.channel_positions.shape == (num_chan, 3)
                                if j == 1:
                                    assert recgen_drift.templates.shape[0] == n_neurons
                                    assert recgen_drift.templates.shape[2] == num_chan
                                else:
                                    assert recgen_drift.templates.shape[0] == n_neurons
                                    assert recgen_drift.templates.shape[2] == j
                                    assert recgen_drift.templates.shape[3] == num_chan
                                assert recgen_drift.spike_traces.shape[1] == n_neurons
                                del recgen_drift

    def test_recording_custom_drifts(self):
        print("Test recording generation - drift")
        ne = 2
        ni = 1
        num_chan = self.num_chan
        n_neurons = ne + ni

        rec_params = mr.get_default_recordings_params()

        rec_params["spiketrains"]["n_exc"] = ne
        rec_params["spiketrains"]["n_inh"] = ni
        rec_params["spiketrains"]["duration"] = 10
        rec_params["recordings"]["drifting"] = True
        chunk_rec = [0, 2]

        # generate drift signals
        drift_dict1 = mr.get_default_drift_dict()
        drift_dict2 = mr.get_default_drift_dict()
        drift_dict3 = mr.get_default_drift_dict()

        drift_dict1["drift_mode_probe"] = "rigid"
        drift_dict1["drift_fs"] = 5
        drift_dict1["slow_drift_velocity"] = 10
        drift_dict1["slow_drift_amplitude"] = 50

        drift_dict2["drift_mode_speed"] = "fast"
        drift_dict2["fast_drift_period"] = 2
        drift_dict2["fast_drift_max_jump"] = 15

        drift_dict3["drift_mode_probe"] = "non-rigid"
        drift_dict3["drift_mode_speed"] = "slow"
        drift_dict3["slow_drift_waveform"] = "sine"
        drift_dict3["slow_drift_velocity"] = 80
        drift_dict3["slow_drift_amplitude"] = 10

        drift_dicts = [drift_dict1, drift_dict2, drift_dict3]

        for ch in chunk_rec:
            print("Drifting mixed: chunk", ch)
            rec_params["recordings"]["chunk_duration"] = ch

            recgen_drift = mr.gen_recordings(
                params=rec_params, tempgen=self.tempgen_drift, verbose=False, drift_dicts=drift_dicts
            )
            assert recgen_drift.recordings.shape[1] == num_chan
            assert len(recgen_drift.spiketrains) == n_neurons
            assert recgen_drift.channel_positions.shape == (num_chan, 3)
            assert recgen_drift.spike_traces.shape[1] == n_neurons
            assert len(recgen_drift.drift_list) == 3

            recgen_mixed_file = self.test_dir / "recordings_drift_mixed.h5"
            mr.save_recording_generator(recgen_drift, filename=recgen_mixed_file)
            recgen_drift_loaded = mr.load_recordings(recgen_mixed_file)
            assert len(recgen_drift_loaded.drift_list) == 3

            # test plotting drift
            _ = mr.plot_cell_drifts(recgen_drift_loaded)

            del recgen_drift_loaded
            del recgen_drift
            recgen_mixed_file.unlink()

    def test_save_load_templates(self):
        tempgen = mr.load_templates(self.test_dir / "templates.h5", verbose=True)
        tempgen_drift = mr.load_templates(self.test_dir / "templates_drift.h5")
        tempgen_drift_f = mr.load_templates(self.test_dir / "templates_drift.h5", return_h5_objects=True)

        assert np.allclose(tempgen.templates, self.tempgen.templates)
        assert np.allclose(tempgen.locations, self.tempgen.locations)
        assert np.allclose(tempgen.rotations, self.tempgen.rotations)
        assert np.allclose(tempgen_drift.templates, self.tempgen_drift.templates)
        assert np.allclose(tempgen_drift.locations, self.tempgen_drift.locations)
        assert np.allclose(tempgen_drift.rotations, self.tempgen_drift.rotations)
        assert np.allclose(tempgen_drift.templates, np.array(tempgen_drift_f.templates))

    def test_save_load_recordings(self):
        recgen_loaded = mr.load_recordings(
            self.test_dir / "recordings.h5",
            return_h5_objects=False,
            verbose=True,
        )
        recgen_loaded_f = mr.load_recordings(self.test_dir / "recordings.h5", return_h5_objects=False, verbose=True)
        recgen_loaded_r = mr.load_recordings(
            self.test_dir / "recordings.h5", load=["recordings"], return_h5_objects=True, verbose=True
        )

        assert np.allclose(recgen_loaded.templates, self.recgen.templates)
        assert np.allclose(recgen_loaded.recordings, self.recgen.recordings)
        assert np.allclose(recgen_loaded.spike_traces, self.recgen.spike_traces)
        assert np.allclose(recgen_loaded.voltage_peaks, self.recgen.voltage_peaks)
        assert np.allclose(recgen_loaded.channel_positions, self.recgen.channel_positions)
        assert np.allclose(recgen_loaded.timestamps.magnitude, self.recgen.timestamps.magnitude)
        assert np.allclose(recgen_loaded.recordings, np.array(recgen_loaded_f.recordings))
        assert np.allclose(recgen_loaded.recordings, np.array(recgen_loaded_r.recordings))

    def test_plots(self):
        _ = mr.plot_rasters(self.recgen.spiketrains)
        _ = mr.plot_rasters(self.recgen.spiketrains, overlap=True)
        _ = mr.plot_rasters(self.recgen.spiketrains, cell_type=True)
        _ = mr.plot_rasters(self.recgen.spiketrains, color="g")
        _ = mr.plot_rasters(self.recgen.spiketrains, color=["g"] * len(self.recgen.spiketrains))
        _ = mr.plot_recordings(self.recgen)
        _ = mr.plot_recordings(self.recgen, overlay_templates=True)
        _ = mr.plot_recordings(self.recgen, overlay_templates=True, max_channels_per_template=3)
        _ = mr.plot_templates(self.recgen)
        _ = mr.plot_templates(self.recgen, single_axes=True)
        _ = mr.plot_waveforms(self.recgen)
        _ = mr.plot_waveforms(self.recgen, electrode=0)
        _ = mr.plot_waveforms(self.recgen, electrode="max")
        _ = mr.plot_waveforms(self.recgen, electrode="max", max_waveforms=2)
        self.recgen.spiketrains[0].waveforms = None
        _ = mr.plot_waveforms(self.recgen, electrode="max")

    def test_extract_features(self):
        feat_t0 = mr.get_templates_features(
            self.tempgen.templates,
            feat_list=["neg", "pos", "amp", "width", "fwhm", "ratio", "speed"],
            dt=self.tempgen.info["params"]["dt"],
        )
        assert feat_t0["neg"].shape == (self.tempgen.templates.shape[0], self.tempgen.templates.shape[1])
        assert feat_t0["pos"].shape == (self.tempgen.templates.shape[0], self.tempgen.templates.shape[1])
        assert feat_t0["amp"].shape == (self.tempgen.templates.shape[0], self.tempgen.templates.shape[1])
        assert feat_t0["width"].shape == (self.tempgen.templates.shape[0], self.tempgen.templates.shape[1])
        assert feat_t0["fwhm"].shape == (self.tempgen.templates.shape[0], self.tempgen.templates.shape[1])
        assert feat_t0["ratio"].shape == (self.tempgen.templates.shape[0], self.tempgen.templates.shape[1])
        assert feat_t0["speed"].shape == (self.tempgen.templates.shape[0], self.tempgen.templates.shape[1])

    def test_recordings_resample(self):
        print("Test recording generation - resampling")
        ne = 2
        ni = 1
        num_chan = self.num_chan
        n_neurons = ne + ni
        fs = 10000
        duration = 5

        rec_params = mr.get_default_recordings_params()

        rec_params["spiketrains"]["n_exc"] = ne
        rec_params["spiketrains"]["n_inh"] = ni
        rec_params["spiketrains"]["duration"] = duration
        n_jitter = 3
        rec_params["templates"]["n_jitters"] = n_jitter
        rec_params["recordings"]["modulation"] = "none"
        rec_params["recordings"]["fs"] = fs
        recgen_rs = mr.gen_recordings(params=rec_params, tempgen=self.tempgen, verbose=False)

        assert recgen_rs.recordings.shape[0] == int(duration * fs)
        assert recgen_rs.recordings.shape[1] == num_chan
        assert recgen_rs.channel_positions.shape == (num_chan, 3)
        assert len(recgen_rs.spiketrains) == n_neurons
        assert len(recgen_rs.spiketrains) == n_neurons
        assert len(recgen_rs.spiketrains) == n_neurons
        assert recgen_rs.spike_traces.shape[1] == n_neurons
        del recgen_rs

    def test_recordings_backend(self):
        print("Test recording generation - backend")
        ne = 2
        ni = 1
        duration = 3

        rec_params = mr.get_default_recordings_params()

        rec_params["spiketrains"]["n_exc"] = ne
        rec_params["spiketrains"]["n_inh"] = ni
        rec_params["spiketrains"]["duration"] = duration
        n_jitter = 10
        rec_params["templates"]["n_jitters"] = n_jitter
        rec_params["recordings"]["modulation"] = "none"
        rec_params["recordings"]["filter"] = False

        rec_params["seeds"]["templates"] = 0
        rec_params["seeds"]["spiketrains"] = 0
        rec_params["seeds"]["convolution"] = 0
        rec_params["seeds"]["noise"] = 0

        n_jobs = [1, 2]
        chunk_durations = [0, 1]

        for n in n_jobs:
            for ch in chunk_durations:
                print("Test recording backend with", n, "jobs - chunk", ch)
                rec_params["recordings"]["chunk_duration"] = ch

                recgen_memmap = mr.gen_recordings(
                    params=rec_params, tempgen=self.tempgen, tmp_mode="memmap", verbose=False, n_jobs=n
                )
                recgen_np = mr.gen_recordings(
                    params=rec_params, tempgen=self.tempgen, tmp_mode=None, verbose=False, n_jobs=n
                )
                assert np.allclose(np.array(recgen_np.recordings), recgen_memmap.recordings.copy(), atol=1e-4)
                del recgen_memmap, recgen_np

    def test_recordings_seeds(self):
        print("Test recording generation - seeds")
        ne = 2
        ni = 1
        duration = 3

        rec_params = mr.get_default_recordings_params()

        rec_params["spiketrains"]["n_exc"] = ne
        rec_params["spiketrains"]["n_inh"] = ni
        rec_params["spiketrains"]["duration"] = duration
        n_jitter = 2
        rec_params["templates"]["n_jitters"] = n_jitter
        rec_params["recordings"]["modulation"] = "none"

        rec_params["seeds"]["templates"] = 0
        rec_params["seeds"]["spiketrains"] = 0
        rec_params["seeds"]["convolution"] = 0
        rec_params["seeds"]["noise"] = 0

        n_jobs = [1, 2]
        chunk_durations = [0, 1]

        for n in n_jobs:
            for ch in chunk_durations:
                print("Test recording seeds with", n, "jobs - chunk", ch)
                rec_params["chunk_duration"] = n

                print("memmap")
                recgen1 = mr.gen_recordings(
                    params=rec_params, tempgen=self.tempgen, tmp_mode="memmap", verbose=False, n_jobs=n
                )
                recgen2 = mr.gen_recordings(
                    params=rec_params, tempgen=self.tempgen, tmp_mode="memmap", verbose=False, n_jobs=n
                )

                assert np.allclose(np.array(recgen1.recordings), np.array(recgen2.recordings), atol=1e-4)
                del recgen1, recgen2

                print("memory")
                recgen1 = mr.gen_recordings(
                    params=rec_params, tempgen=self.tempgen, tmp_mode=None, verbose=False, n_jobs=n
                )
                recgen2 = mr.gen_recordings(
                    params=rec_params, tempgen=self.tempgen, tmp_mode=None, verbose=False, n_jobs=n
                )

                assert np.allclose(np.array(recgen1.recordings), recgen2.recordings, atol=1e-4)
                del recgen1, recgen2

    def test_recordings_dtype(self):
        print("Test recording generation - dtype")
        ne = 2
        ni = 1
        duration = 1

        dtypes = ["int16", "int32", "float16", "float32", "float64"]
        modulations = ["none", "template", "electrode"]

        rec_params = mr.get_default_recordings_params()
        rec_params["spiketrains"]["n_exc"] = ne
        rec_params["spiketrains"]["n_inh"] = ni
        rec_params["spiketrains"]["duration"] = duration
        n_jitter = 3
        rec_params["templates"]["n_jitters"] = n_jitter

        for i, dt in enumerate(dtypes):
            for mod in modulations:
                rec_params["recordings"]["modulation"] = mod

                print("Dtype:", dt, "modulation", mod)
                rec_params["recordings"]["dtype"] = dt
                recgen_dt = mr.gen_recordings(params=rec_params, tempgen=self.tempgen, verbose=False)

                assert recgen_dt.recordings[0, 0].dtype == dt
                del recgen_dt

    def test_adc_bit_depth_lsb_gain(self):
        print("Test recording generation - adc depth and lsb")
        ne = 2
        ni = 1
        duration = 1

        bit_depths = [10, 11, 12]
        lsbs = [1, 2, 4]
        gains = [0.1, 0.2, 0.3]

        rec_params = mr.get_default_recordings_params()
        rec_params["spiketrains"]["n_exc"] = ne
        rec_params["spiketrains"]["n_inh"] = ni
        rec_params["spiketrains"]["duration"] = duration
        n_jitter = 3
        rec_params["templates"]["n_jitters"] = n_jitter
        rec_params["recordings"]["dtype"] = "int16"
        rec_params["recordings"]["filter"] = False

        for bd in bit_depths:
            for lsb in lsbs:
                rec_params["recordings"]["adc_bit_depth"] = bd
                rec_params["recordings"]["lsb"] = lsb

                print("ADC bit depth:", bd, "lsb", lsb)
                recgen_adc_lsb = mr.gen_recordings(params=rec_params, tempgen=self.tempgen, verbose=False)

                assert np.ptp(recgen_adc_lsb.recordings) / lsb <= 2**bd
                lsb_rec = np.min(np.diff(np.sort(np.unique(recgen_adc_lsb.recordings.ravel()))))
                assert lsb_rec == lsb

                # save and reload gain
                save_path = self.test_dir / f"{bd}_{lsb}.h5"
                mr.save_recording_generator(recgen_adc_lsb, save_path)
                recgen_loaded = mr.load_recordings(save_path)
                assert recgen_loaded.gain_to_uV == recgen_adc_lsb.gain_to_uV

                del recgen_adc_lsb
                save_path.unlink()

        for gain in gains:
            for lsb in lsbs:
                rec_params["recordings"]["gain"] = gain
                rec_params["recordings"]["lsb"] = lsb

                print("Gain:", gain, "lsb", lsb)
                recgen_gain_lsb = mr.gen_recordings(params=rec_params, tempgen=self.tempgen, verbose=False)

                assert recgen_gain_lsb.gain_to_uV == gain
                lsb_rec = np.min(np.diff(np.sort(np.unique(recgen_gain_lsb.recordings.ravel()))))
                assert lsb_rec == lsb

                # save and reload gain
                save_path = self.test_dir / f"{gain}_{lsb}.h5"
                mr.save_recording_generator(recgen_gain_lsb, save_path)
                recgen_loaded = mr.load_recordings(save_path)
                assert recgen_loaded.gain_to_uV == recgen_gain_lsb.gain_to_uV

                del recgen_gain_lsb
                save_path.unlink()

    def test_default_params(self):
        print("Test default params")
        info, info_folder = mr.get_default_config()
        cell_models_folder = info["cell_models_folder"]
        tempgen = mr.gen_templates(cell_models_folder, params={"n": 2}, templates_tmp_folder=info["templates_folder"])
        recgen = mr.gen_recordings(templates=self.test_dir / "templates.h5", verbose=False)
        recgen.params["recordings"]["noise_level"] = 0
        recgen.generate_recordings()
        recgen_loaded = mr.load_recordings(self.test_dir / "recordings.h5", verbose=True)
        recgen_loaded.params["recordings"]["noise_level"] = 0
        recgen_loaded.generate_recordings()
        recgen_empty = mr.RecordingGenerator(rec_dict={}, info={})

        n = 2
        num_cells = self.num_cells
        templates_params = self.templates_params

        assert tempgen.templates.shape[0] == (n * num_cells)
        assert len(tempgen.locations) == (n * num_cells)
        assert len(tempgen.rotations) == (n * num_cells)
        assert len(tempgen.celltypes) == (n * num_cells)
        assert len(np.unique(tempgen.celltypes)) == num_cells
        assert (
            np.min(tempgen.locations[:, 0]) > templates_params["xlim"][0]
            and np.max(tempgen.locations[:, 0]) < templates_params["xlim"][1]
        )

        assert recgen.recordings.shape[1] == self.num_chan
        assert recgen.channel_positions.shape == (self.num_chan, 3)
        assert recgen_loaded.recordings.shape[1] == self.num_chan
        assert recgen_loaded.channel_positions.shape == (self.num_chan, 3)
        assert len(recgen_empty.recordings) == 0
        del recgen, recgen_empty

    def test_cli(self):
        default_config, mearec_home = mr.get_default_config()

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["default-config"])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["available-probes"])
        assert result.exit_code == 0
        result = runner.invoke(
            cli,
            [
                "gen-templates",
                "-n",
                "2",
                "--no-parallel",
                "--recompile",
                "-r",
                "3drot",
                "-nc",
                "2",
                "-ov",
                "20",
                "-s",
                "1",
                "-mind",
                "10",
                "-maxd",
                "100",
                "-drst",
                "10",
                "-v",
            ],
        )
        assert result.exit_code == 0
        result = runner.invoke(
            cli,
            [
                "gen-recordings",
                "-t",
                str(self.test_dir / "templates.h5"),
                "-ne",
                "2",
                "-ni",
                "1",
                "-fe",
                "5",
                "-fi",
                "15",
                "-se",
                "1",
                "-si",
                "1",
                "-mr",
                "0.2",
                "-rp",
                "2",
                "-p",
                "poisson",
                "-md",
                "1",
                "-mina",
                "10",
                "-maxa",
                "1000",
                "--fs",
                "32000",
                "-sr",
                "0",
                "-sj",
                "1",
                "-nl",
                "10",
                "-m",
                "none",
                "-chd",
                "0",
                "-nseed",
                "10",
                "-hd",
                "30",
                "-cn",
                "-cp",
                "500",
                "-cq",
                "1",
                "-rnf",
                "1",
                "-stseed",
                "100",
                "-tseed",
                "10",
                "--filter",
                "-fc",
                "500",
                "-fo",
                "3",
                "--overlap",
                "-ot",
                "0.8",
                "--extract-wf",
                "-angt",
                "15",
                "-drvel",
                "10",
                "-tsd",
                "1",
                "-v",
            ],
        )
        print(result.output)
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-templates-params", "."])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-templates-params", default_config["templates_params"]])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-recordings-params", "."])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-recordings-params", default_config["recordings_params"]])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-cell-models-folder", "./cell_models"])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-cell-models-folder", default_config["cell_models_folder"]])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-templates-folder", "."])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-recordings-folder", "."])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-templates-folder", "./templates", "--create"])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-recordings-folder", "./recordings", "--create"])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-templates-folder", default_config["templates_folder"]])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-recordings-folder", default_config["recordings_folder"]])
        assert result.exit_code == 0

    def test_simulate_cell(self):
        cell_folder = Path(mr.get_default_cell_models_folder())
        params = mr.get_default_templates_params()

        target_spikes = [3, 50]
        params["target_spikes"] = target_spikes
        cell_path = [c for c in cell_folder.iterdir() if "TTPC1" in c.name][0]
        cell_name = cell_path.parts[-1]

        cell, v, i = mr.run_cell_model(
            cell_model_folder=str(cell_path), sim_folder=None, verbose=True, save=False, return_vi=True, **params
        )
        c = mr.return_bbp_cell_morphology(str(cell_name), cell_folder)
        assert target_spikes[0] <= len(v) <= target_spikes[1]
        assert target_spikes[0] <= len(i) <= target_spikes[1]
        assert len(c.x) == len(c.y) and len(c.x) == len(c.z)


if __name__ == "__main__":
    test = TestGenerators()
    test.setUpClass()
    # TestGenerators().test_gen_recordings_drift()
    test.test_gen_recordings_filters()
    # test.test_simulate_cell()
