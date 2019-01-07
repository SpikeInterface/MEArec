import pytest
import numpy as np
import unittest
import MEArec as mr
import tempfile
import shutil
import yaml
import os


class TestGenerators(unittest.TestCase):
    def setUp(self):
        info, info_folder = mr.get_default_config()
        cell_models_folder = info['cell_models_folder']
        self.num_cells = len([f for f in os.listdir(cell_models_folder) if 'mods' not in f])
        self.n = 2
        with open(info['templates_params']) as f:
            templates_params = yaml.load(f)

        templates_params['n'] = self.n
        templates_params['probe'] = 'Neuronexus-32'
        self.tempgen = mr.gen_templates(cell_models_folder, params=templates_params, parallel=True, delete_tmp=True)
        self.templates_params = templates_params
        self.num_chan = self.tempgen.templates.shape[1]

    def test_gen_templates(self):
        n = self.n
        num_cells = self.num_cells
        templates_params = self.templates_params

        assert self.tempgen.templates.shape[0] == (n * num_cells)
        assert len(self.tempgen.locations) == (n * num_cells)
        assert len(self.tempgen.rotations) == (n * num_cells)
        assert len(self.tempgen.celltypes) == (n * num_cells)
        assert len(np.unique(self.tempgen.celltypes)) == num_cells
        assert np.max(self.tempgen.locations[:, 0]) > templates_params['xlim'][0] and \
               np.max(self.tempgen.locations[:, 0]) < templates_params['xlim'][1]

    def test_gen_recordings_mod_none(self):
        info, info_folder = mr.get_default_config()
        ne = 2
        ni = 1
        num_chan = self.num_chan
        n_neurons = ne + ni

        with open(info['recordings_params']) as f:
            rec_params = yaml.load(f)

        rec_params['spiketrains']['n_exc'] = ne
        rec_params['spiketrains']['n_inh'] = ni
        n_jitter = rec_params['templates']['n_jitters']

        rec_params['recordings']['modulation'] = 'none'
        rec_params['recordings']['filter'] = False
        rec_params['spiketrains']['duration'] = 2

        recgen_none = mr.gen_recordings(params=rec_params, tempgen=self.tempgen)

        assert recgen_none.recordings.shape[0] == num_chan
        assert len(recgen_none.spiketrains) == n_neurons
        assert recgen_none.channel_positions.shape == (num_chan, 3)
        assert recgen_none.templates.shape[:3] == (n_neurons, n_jitter, num_chan)
        assert recgen_none.voltage_peaks.shape == (n_neurons, num_chan)
        assert len(recgen_none.spike_traces) == n_neurons

    def test_gen_recordings_mod_temp(self):
        info, info_folder = mr.get_default_config()
        ne = 2
        ni = 2
        num_chan = self.num_chan
        n_neurons = ne + ni

        with open(info['recordings_params']) as f:
            rec_params = yaml.load(f)

        rec_params['spiketrains']['n_exc'] = ne
        rec_params['spiketrains']['n_inh'] = ni
        n_jitter = 5
        rec_params['templates']['n_jitters'] = n_jitter
        rec_params['recordings']['modulation'] = 'template'

        recgen_temp = mr.gen_recordings(params=rec_params, tempgen=self.tempgen)

        assert recgen_temp.recordings.shape[0] == num_chan
        assert len(recgen_temp.spiketrains) == n_neurons
        assert recgen_temp.channel_positions.shape == (num_chan, 3)
        assert recgen_temp.templates.shape[:3] == (n_neurons, n_jitter, num_chan)
        assert recgen_temp.voltage_peaks.shape == (n_neurons, num_chan)
        assert len(recgen_temp.spike_traces) == n_neurons

    def test_gen_recordings_mod_elec(self):
        info, info_folder = mr.get_default_config()
        ne = 3
        ni = 1
        num_chan = self.num_chan
        n_neurons = ne + ni

        with open(info['recordings_params']) as f:
            rec_params = yaml.load(f)

        rec_params['spiketrains']['n_exc'] = ne
        rec_params['spiketrains']['n_inh'] = ni
        n_jitter = 15
        rec_params['templates']['n_jitters'] = n_jitter
        rec_params['recordings']['modulation'] = 'template'
        rec_params['recordings']['modulation'] = 'electrode'
        recgen_elec = mr.gen_recordings(params=rec_params, tempgen=self.tempgen)

        assert recgen_elec.recordings.shape[0] == num_chan
        assert len(recgen_elec.spiketrains) == n_neurons
        assert recgen_elec.channel_positions.shape == (num_chan, 3)
        assert recgen_elec.templates.shape[:3] == (n_neurons, n_jitter, num_chan)
        assert recgen_elec.voltage_peaks.shape == (n_neurons, num_chan)
        assert len(recgen_elec.spike_traces) == n_neurons


if __name__ == '__main__':
    unittest.main()
