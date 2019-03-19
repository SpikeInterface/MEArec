import pytest
import numpy as np
import unittest
import MEArec as mr
import tempfile
import shutil
import yaml
import os
import elephant.statistics as stat
import quantities as pq
from distutils.version import StrictVersion
import tempfile

if StrictVersion(yaml.__version__) >= StrictVersion('5.0.0'):
    use_loader = True
else:
    use_loader = False


class TestGenerators(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        info, info_folder = mr.get_default_config()
        cell_models_folder = info['cell_models_folder']
        self.num_cells = len([f for f in os.listdir(cell_models_folder) if 'mods' not in f])
        self.n = 5
        with open(info['templates_params']) as f:
            if use_loader:
                templates_params = yaml.load(f, Loader=yaml.FullLoader)
            else:
                templates_params = yaml.load(f)

        with open(info['recordings_params']) as f:
            if use_loader:
                rec_params = yaml.load(f, Loader=yaml.FullLoader)
            else:
                rec_params = yaml.load(f)

        self.test_dir = tempfile.mkdtemp()

        templates_params['n'] = self.n
        templates_params['probe'] = 'Neuronexus-32'
        templates_folder = info['templates_folder']
        print('Generating non-drifting templates')
        self.tempgen = mr.gen_templates(cell_models_folder, templates_folder=templates_folder,
                                        params=templates_params, parallel=True, delete_tmp=True, verbose=False)
        self.templates_params = templates_params
        self.num_templates, self.num_chan, self.num_samples = self.tempgen.templates.shape

        templates_params['drifting'] = True
        templates_params['drift_steps'] = 5
        print('Generating drifting templates')
        self.tempgen_drift = mr.gen_templates(cell_models_folder, templates_folder=templates_folder,
                                              params=templates_params, parallel=True, delete_tmp=True, verbose=False)
        self.templates_params_drift = templates_params
        self.num_steps_drift = self.tempgen_drift.templates.shape[1]

        print('Making test recordings to test load functions')
        mr.save_template_generator(self.tempgen, self.test_dir + '/templates.h5')
        mr.save_template_generator(self.tempgen_drift, self.test_dir + '/templates_drift.h5')

        ne = 2
        ni = 1
        rec_params['spiketrains']['n_exc'] = ne
        rec_params['spiketrains']['n_inh'] = ni
        rec_params['recordings']['modulation'] = 'none'
        rec_params['recordings']['filter'] = False
        rec_params['spiketrains']['duration'] = 2
        rec_params['templates']['min_dist'] = 1

        self.recgen = mr.gen_recordings(params=rec_params, tempgen=self.tempgen)
        print(self.recgen.templates.shape)
        mr.save_recording_generator(self.recgen, self.test_dir + '/recordings.h5')
        recgen_loaded = mr.load_recordings(self.test_dir + '/recordings.h5')
        print(recgen_loaded.templates.shape)

    def test_gen_templates(self):
        print('Test templates generation')
        n = self.n
        num_cells = self.num_cells
        templates_params = self.templates_params

        assert self.tempgen.templates.shape[0] == (n * num_cells)
        assert len(self.tempgen.locations) == (n * num_cells)
        assert len(self.tempgen.rotations) == (n * num_cells)
        assert len(self.tempgen.celltypes) == (n * num_cells)
        assert len(np.unique(self.tempgen.celltypes)) == num_cells
        assert np.max(self.tempgen.locations[:, 0]) > templates_params['xlim'][0] \
               and np.max(self.tempgen.locations[:, 0]) < templates_params['xlim'][1]

    def test_gen_templates_drift(self):
        print('Test drifting templates generation')
        n = self.n
        num_cells = self.num_cells
        n_steps = self.num_steps_drift
        templates_params = self.templates_params_drift

        assert self.tempgen_drift.templates.shape[0] == (n * num_cells)
        assert self.tempgen_drift.locations.shape == (n * num_cells, n_steps, 3)
        assert len(self.tempgen_drift.rotations) == (n * num_cells)
        assert len(self.tempgen_drift.celltypes) == (n * num_cells)
        assert len(np.unique(self.tempgen_drift.celltypes)) == num_cells
        assert np.max(self.tempgen_drift.locations[:, :, 0]) > templates_params['xlim'][0] \
               and np.max(self.tempgen_drift.locations[:, :, 0]) < templates_params['xlim'][1]
        assert self.tempgen_drift.templates.shape[1] == self.num_steps_drift

    def test_gen_spiketrains(self):
        print('Test spike train generation')
        info, info_folder = mr.get_default_config()
        with open(info['recordings_params']) as f:
            if use_loader:
                rec_params = yaml.load(f, Loader=yaml.FullLoader)
            else:
                rec_params = yaml.load(f)
        sp_params = rec_params['spiketrains']
        spgen = mr.SpikeTrainGenerator(sp_params)
        spgen.generate_spikes()

        #check ref period
        for st in spgen.all_spiketrains:
            isi = stat.isi(st).rescale('ms')
            assert np.all(isi.magnitude > sp_params['ref_per'])
            assert (1 / np.mean(isi.rescale('s'))) > sp_params['min_rate']

        sp_params['process'] = 'gamma'
        spgen = mr.SpikeTrainGenerator(sp_params)
        spgen.generate_spikes()
        for st in spgen.all_spiketrains:
            isi = stat.isi(st).rescale('ms')
            assert np.all(isi.magnitude > sp_params['ref_per'])
            assert (1 / np.mean(isi.rescale('s'))) > sp_params['min_rate']

    def test_gen_recordings_mod_none(self):
        print('Test recording generation - modulation None')
        info, info_folder = mr.get_default_config()
        ne = 3
        ni = 1
        num_chan = self.num_chan
        n_neurons = ne + ni

        with open(info['recordings_params']) as f:
            if use_loader:
                rec_params = yaml.load(f, Loader=yaml.FullLoader)
            else:
                rec_params = yaml.load(f)

        rec_params['spiketrains']['n_exc'] = ne
        rec_params['spiketrains']['n_inh'] = ni
        n_jitter = rec_params['templates']['n_jitters']

        rec_params['recordings']['modulation'] = 'none'
        rec_params['recordings']['filter'] = False
        rec_params['spiketrains']['duration'] = 2
        rec_params['templates']['min_dist'] = 1

        recgen_none = mr.gen_recordings(params=rec_params, tempgen=self.tempgen)

        assert recgen_none.recordings.shape[0] == num_chan
        assert len(recgen_none.spiketrains) == n_neurons
        assert recgen_none.channel_positions.shape == (num_chan, 3)
        assert recgen_none.templates.shape[:3] == (n_neurons, n_jitter, num_chan)
        assert recgen_none.voltage_peaks.shape == (n_neurons, num_chan)
        assert len(recgen_none.spike_traces) == n_neurons

    def test_gen_recordings_mod_temp(self):
        print('Test recording generation - modulation template')
        info, info_folder = mr.get_default_config()
        ne = 2
        ni = 2
        num_chan = self.num_chan
        n_neurons = ne + ni

        with open(info['recordings_params']) as f:
            if use_loader:
                rec_params = yaml.load(f, Loader=yaml.FullLoader)
            else:
                rec_params = yaml.load(f)

        rec_params['spiketrains']['n_exc'] = ne
        rec_params['spiketrains']['n_inh'] = ni
        n_jitter = 6
        rec_params['templates']['n_jitters'] = n_jitter
        rec_params['recordings']['modulation'] = 'template'
        rec_params['templates']['min_dist'] = 1

        recgen_temp = mr.gen_recordings(params=rec_params, tempgen=self.tempgen)

        assert recgen_temp.recordings.shape[0] == num_chan
        assert len(recgen_temp.spiketrains) == n_neurons
        assert recgen_temp.channel_positions.shape == (num_chan, 3)
        assert recgen_temp.templates.shape[:3] == (n_neurons, n_jitter, num_chan)
        assert recgen_temp.voltage_peaks.shape == (n_neurons, num_chan)
        assert len(recgen_temp.spike_traces) == n_neurons

    def test_gen_recordings_mod_elec(self):
        print('Test recording generation - modulation electrode')
        info, info_folder = mr.get_default_config()
        ne = 1
        ni = 3
        num_chan = self.num_chan
        n_neurons = ne + ni

        with open(info['recordings_params']) as f:
            if use_loader:
                rec_params = yaml.load(f, Loader=yaml.FullLoader)
            else:
                rec_params = yaml.load(f)

        rec_params['spiketrains']['n_exc'] = ne
        rec_params['spiketrains']['n_inh'] = ni
        n_jitter = 15
        rec_params['templates']['n_jitters'] = n_jitter
        rec_params['recordings']['modulation'] = 'electrode'
        rec_params['templates']['min_dist'] = 1

        recgen_elec = mr.gen_recordings(params=rec_params, tempgen=self.tempgen)

        assert recgen_elec.recordings.shape[0] == num_chan
        assert len(recgen_elec.spiketrains) == n_neurons
        assert recgen_elec.channel_positions.shape == (num_chan, 3)
        assert recgen_elec.templates.shape[:3] == (n_neurons, n_jitter, num_chan)
        assert recgen_elec.voltage_peaks.shape == (n_neurons, num_chan)
        assert len(recgen_elec.spike_traces) == n_neurons

    def test_gen_recordings_mod_temp_isi(self):
        print('Test recording generation - modulation template-isi')
        info, info_folder = mr.get_default_config()
        ne = 2
        ni = 2
        num_chan = self.num_chan
        n_neurons = ne + ni

        with open(info['recordings_params']) as f:
            if use_loader:
                rec_params = yaml.load(f, Loader=yaml.FullLoader)
            else:
                rec_params = yaml.load(f)

        rec_params['spiketrains']['n_exc'] = ne
        rec_params['spiketrains']['n_inh'] = ni
        n_jitter = 3
        rec_params['templates']['n_jitters'] = n_jitter
        rec_params['recordings']['modulation'] = 'template-isi'
        rec_params['templates']['min_dist'] = 1

        recgen_temp = mr.gen_recordings(params=rec_params, tempgen=self.tempgen)

        assert recgen_temp.recordings.shape[0] == num_chan
        assert len(recgen_temp.spiketrains) == n_neurons
        assert recgen_temp.channel_positions.shape == (num_chan, 3)
        assert recgen_temp.templates.shape[:3] == (n_neurons, n_jitter, num_chan)
        assert recgen_temp.voltage_peaks.shape == (n_neurons, num_chan)
        assert len(recgen_temp.spike_traces) == n_neurons

    def test_gen_recordings_mod_elec_isi(self):
        print('Test recording generation - modulation electrode-isi')
        info, info_folder = mr.get_default_config()
        ne = 3
        ni = 1
        num_chan = self.num_chan
        n_neurons = ne + ni

        with open(info['recordings_params']) as f:
            if use_loader:
                rec_params = yaml.load(f, Loader=yaml.FullLoader)
            else:
                rec_params = yaml.load(f)

        rec_params['spiketrains']['n_exc'] = ne
        rec_params['spiketrains']['n_inh'] = ni
        n_jitter = 4
        rec_params['templates']['n_jitters'] = n_jitter
        rec_params['recordings']['modulation'] = 'electrode-isi'
        rec_params['templates']['min_dist'] = 1

        recgen_elec = mr.gen_recordings(params=rec_params, tempgen=self.tempgen)

        assert recgen_elec.recordings.shape[0] == num_chan
        assert len(recgen_elec.spiketrains) == n_neurons
        assert recgen_elec.channel_positions.shape == (num_chan, 3)
        assert recgen_elec.templates.shape[:3] == (n_neurons, n_jitter, num_chan)
        assert recgen_elec.voltage_peaks.shape == (n_neurons, num_chan)
        assert len(recgen_elec.spike_traces) == n_neurons

    def test_gen_recordings_mod_bursting(self):
        print('Test recording generation - bursting')
        info, info_folder = mr.get_default_config()
        ne = 3
        ni = 1
        num_chan = self.num_chan
        n_neurons = ne + ni

        with open(info['recordings_params']) as f:
            if use_loader:
                rec_params = yaml.load(f, Loader=yaml.FullLoader)
            else:
                rec_params = yaml.load(f)

        rec_params['spiketrains']['n_exc'] = ne
        rec_params['spiketrains']['n_inh'] = ni
        n_jitter = 4
        rec_params['templates']['n_jitters'] = n_jitter
        rec_params['recordings']['modulation'] = 'electrode-isi'
        rec_params['recordings']['bursting'] = True
        rec_params['templates']['min_dist'] = 1

        recgen_elec = mr.gen_recordings(params=rec_params, tempgen=self.tempgen)

        assert recgen_elec.recordings.shape[0] == num_chan
        assert len(recgen_elec.spiketrains) == n_neurons
        assert recgen_elec.channel_positions.shape == (num_chan, 3)
        assert recgen_elec.templates.shape[:3] == (n_neurons, n_jitter, num_chan)
        assert recgen_elec.voltage_peaks.shape == (n_neurons, num_chan)
        assert len(recgen_elec.spike_traces) == n_neurons

    def test_gen_recordings_dist_corr(self):
        print('Test recording generation - noise distance-correlated')
        info, info_folder = mr.get_default_config()
        ne = 1
        ni = 1
        num_chan = self.num_chan
        n_neurons = ne + ni

        with open(info['recordings_params']) as f:
            if use_loader:
                rec_params = yaml.load(f, Loader=yaml.FullLoader)
            else:
                rec_params = yaml.load(f)

        rec_params['spiketrains']['n_exc'] = ne
        rec_params['spiketrains']['n_inh'] = ni
        n_jitter = 3
        rec_params['templates']['n_jitters'] = n_jitter
        rec_params['templates']['min_dist'] = 1
        rec_params['recordings']['modulation'] = 'none'
        rec_params['recordings']['noise_mode'] = 'distance-correlated'

        recgen_elec = mr.gen_recordings(params=rec_params, tempgen=self.tempgen)

        assert recgen_elec.recordings.shape[0] == num_chan
        assert len(recgen_elec.spiketrains) == n_neurons
        assert recgen_elec.channel_positions.shape == (num_chan, 3)
        assert recgen_elec.templates.shape[:3] == (n_neurons, n_jitter, num_chan)
        assert recgen_elec.voltage_peaks.shape == (n_neurons, num_chan)
        assert len(recgen_elec.spike_traces) == n_neurons

    def test_gen_recordings_color_noise(self):
        print('Test recording generation - noise distance-correlated - color')
        info, info_folder = mr.get_default_config()
        ne = 1
        ni = 1
        num_chan = self.num_chan
        n_neurons = ne + ni

        with open(info['recordings_params']) as f:
            if use_loader:
                rec_params = yaml.load(f, Loader=yaml.FullLoader)
            else:
                rec_params = yaml.load(f)

        rec_params['spiketrains']['n_exc'] = ne
        rec_params['spiketrains']['n_inh'] = ni
        n_jitter = 3
        rec_params['templates']['n_jitters'] = n_jitter
        rec_params['templates']['min_dist'] = 1
        rec_params['recordings']['modulation'] = 'none'
        rec_params['recordings']['noise_mode'] = 'distance-correlated'
        rec_params['recordings']['noise_color'] = True
        recgen_elec = mr.gen_recordings(params=rec_params, tempgen=self.tempgen)

        assert recgen_elec.recordings.shape[0] == num_chan
        assert len(recgen_elec.spiketrains) == n_neurons
        assert recgen_elec.channel_positions.shape == (num_chan, 3)
        assert recgen_elec.templates.shape[:3] == (n_neurons, n_jitter, num_chan)
        assert recgen_elec.voltage_peaks.shape == (n_neurons, num_chan)
        assert len(recgen_elec.spike_traces) == n_neurons

    def test_gen_recordings_drift(self):
        print('Test recording generation - drift')
        info, info_folder = mr.get_default_config()
        ne = 1
        ni = 1
        num_chan = self.num_chan
        n_steps = self.num_steps_drift
        n_neurons = ne + ni

        with open(info['recordings_params']) as f:
            if use_loader:
                rec_params = yaml.load(f, Loader=yaml.FullLoader)
            else:
                rec_params = yaml.load(f)

        rec_params['spiketrains']['n_exc'] = ne
        rec_params['spiketrains']['n_inh'] = ni
        n_jitter = 3
        rec_params['templates']['n_jitters'] = n_jitter
        rec_params['recordings']['modulation'] = 'none'
        rec_params['recordings']['drifting'] = True
        rec_params['templates']['min_dist'] = 1

        recgen_drift = mr.gen_recordings(params=rec_params, tempgen=self.tempgen_drift)

        assert recgen_drift.recordings.shape[0] == num_chan
        assert len(recgen_drift.spiketrains) == n_neurons
        assert recgen_drift.channel_positions.shape == (num_chan, 3)
        assert recgen_drift.templates.shape[:4] == (n_neurons, n_steps, n_jitter, num_chan)
        assert len(recgen_drift.spike_traces) == n_neurons

    def test_gen_recordings_drift_burst(self):
        print('Test recording generation - drift bursting')
        info, info_folder = mr.get_default_config()
        ne = 1
        ni = 1
        num_chan = self.num_chan
        n_steps = self.num_steps_drift
        n_neurons = ne + ni

        with open(info['recordings_params']) as f:
            if use_loader:
                rec_params = yaml.load(f, Loader=yaml.FullLoader)
            else:
                rec_params = yaml.load(f)

        rec_params['spiketrains']['n_exc'] = ne
        rec_params['spiketrains']['n_inh'] = ni
        n_jitter = 3
        rec_params['templates']['n_jitters'] = n_jitter
        rec_params['recordings']['modulation'] = 'none'
        rec_params['recordings']['drifting'] = True
        rec_params['recordings']['bursting'] = True
        rec_params['templates']['min_dist'] = 1

        recgen_drift = mr.gen_recordings(params=rec_params, tempgen=self.tempgen_drift)

        assert recgen_drift.recordings.shape[0] == num_chan
        assert len(recgen_drift.spiketrains) == n_neurons
        assert recgen_drift.channel_positions.shape == (num_chan, 3)
        assert recgen_drift.templates.shape[:4] == (n_neurons, n_steps, n_jitter, num_chan)
        assert len(recgen_drift.spike_traces) == n_neurons

    def test_save_load_templates(self):
        tempgen = mr.load_templates(self.test_dir + '/templates.h5')
        tempgen_drift = mr.load_templates(self.test_dir + '/templates_drift.h5')

        assert np.allclose(tempgen.templates, self.tempgen.templates)
        assert np.allclose(tempgen.locations, self.tempgen.locations)
        assert np.allclose(tempgen.rotations, self.tempgen.rotations)
        assert np.allclose(tempgen_drift.templates, self.tempgen_drift.templates)
        assert np.allclose(tempgen_drift.locations, self.tempgen_drift.locations)
        assert np.allclose(tempgen_drift.rotations, self.tempgen_drift.rotations)

    def test_save_load_recordings(self):
        recgen_loaded = mr.load_recordings(self.test_dir + '/recordings.h5')

        assert np.allclose(recgen_loaded.templates, self.recgen.templates)
        assert np.allclose(recgen_loaded.recordings, self.recgen.recordings)
        assert np.allclose(recgen_loaded.spike_traces, self.recgen.spike_traces)
        assert np.allclose(recgen_loaded.voltage_peaks, self.recgen.voltage_peaks)
        assert np.allclose(recgen_loaded.channel_positions, self.recgen.channel_positions)
        assert np.allclose(recgen_loaded.timestamps, self.recgen.timestamps.magnitude)


if __name__ == '__main__':
    unittest.main()
