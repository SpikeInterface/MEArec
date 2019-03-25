import pytest
import os
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
from click.testing import CliRunner

from MEArec.cli import cli, default_config, set_cell_models_folder, set_recordings_folder, set_recordings_params, \
    set_templates_folder, set_templates_params, gen_templates, gen_recordings

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
                                        params=templates_params, parallel=True, delete_tmp=True, verbose=True)
        self.templates_params = templates_params
        self.num_templates, self.num_chan, self.num_samples = self.tempgen.templates.shape

        templates_params['drifting'] = True
        templates_params['drift_steps'] = 5
        templates_params['rot'] = 'norot'
        print('Generating drifting templates')
        self.tempgen_drift = mr.gen_templates(cell_models_folder, templates_folder=templates_folder,
                                              params=templates_params, parallel=True, delete_tmp=True, verbose=True)
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
        rec_params['recordings']['extract_waveforms'] = True
        rec_params['recordings']['overlap'] = True
        rec_params['spiketrains']['duration'] = 2
        rec_params['templates']['min_dist'] = 1

        self.recgen = mr.gen_recordings(params=rec_params, tempgen=self.tempgen)
        self.recgen.annotate_overlapping_spike(parallel=False)
        mr.save_recording_generator(self.recgen, self.test_dir + '/recordings.h5')

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

    def test_gen_recordings_modulations(self):
        print('Test recording generation - modulation')
        info, info_folder = mr.get_default_config()
        ne = 1
        ni = 2
        fe = 30
        fi = 50
        num_chan = self.num_chan
        n_neurons = ne + ni

        with open(info['recordings_params']) as f:
            if use_loader:
                rec_params = yaml.load(f, Loader=yaml.FullLoader)
            else:
                rec_params = yaml.load(f)

        rec_params['spiketrains']['n_exc'] = ne
        rec_params['spiketrains']['n_inh'] = ni
        rec_params['spiketrains']['f_exc'] = fe
        rec_params['spiketrains']['f_inh'] = fi
        rec_params['spiketrains']['duration'] = 3
        n_jitter = [1, 5]
        modulations = ['none', 'electrode', 'template', 'template-isi', 'electrode-isi']
        rec_params['templates']['min_dist'] = 1
        bursting = [False, True]

        for mod in modulations:
            for b in bursting:
                for j in n_jitter:
                    print('Modulation: modulation', mod, 'bursting', b, 'jitter', j)
                    rec_params['templates']['n_jitters'] =j
                    rec_params['recordings']['modulation'] = mod
                    rec_params['recordings']['bursting'] = b

                    recgen_mod = mr.gen_recordings(params=rec_params, tempgen=self.tempgen)

                    assert recgen_mod.recordings.shape[0] == num_chan
                    assert len(recgen_mod.spiketrains) == n_neurons
                    assert recgen_mod.channel_positions.shape == (num_chan, 3)
                    if j == 1:
                        assert recgen_mod.templates.shape[:2] == (n_neurons, num_chan)
                    else:
                        assert recgen_mod.templates.shape[:3] == (n_neurons, j, num_chan)
                    assert recgen_mod.voltage_peaks.shape == (n_neurons, num_chan)
                    assert len(recgen_mod.spike_traces) == n_neurons
                    del recgen_mod

    def test_gen_recordings_mod_bursting_sync(self):
        print('Test recording generation - bursting')
        info, info_folder = mr.get_default_config()
        ne = 10
        ni = 5
        fe = 30
        fi = 50
        num_chan = self.num_chan
        n_neurons = ne + ni

        with open(info['recordings_params']) as f:
            if use_loader:
                rec_params = yaml.load(f, Loader=yaml.FullLoader)
            else:
                rec_params = yaml.load(f)

        rec_params['spiketrains']['n_exc'] = ne
        rec_params['spiketrains']['n_inh'] = ni
        rec_params['spiketrains']['f_exc'] = fe
        rec_params['spiketrains']['f_inh'] = fi
        rec_params['spiketrains']['duration'] = 3
        n_jitter = 4
        rec_params['templates']['n_jitters'] = n_jitter
        rec_params['recordings']['modulation'] = 'electrode-isi'
        rec_params['recordings']['bursting'] = True
        rec_params['recordings']['sync_rate'] = 0.2
        rec_params['recordings']['overlap'] = False
        rec_params['recordings']['extract_waveforms'] = True
        rec_params['templates']['min_dist'] = 1
        rec_params['templates']['min_amp'] = 30

        recgen_burst = mr.gen_recordings(params=rec_params, tempgen=self.tempgen)

        assert recgen_burst.recordings.shape[0] == num_chan
        assert len(recgen_burst.spiketrains) == n_neurons
        assert recgen_burst.channel_positions.shape == (num_chan, 3)
        assert recgen_burst.templates.shape[:3] == (n_neurons, n_jitter, num_chan)
        assert recgen_burst.voltage_peaks.shape == (n_neurons, num_chan)
        assert len(recgen_burst.spike_traces) == n_neurons

    def test_gen_recordings_noise(self):
        print('Test recording generation - noise')
        info, info_folder = mr.get_default_config()
        rates = [30, 50]
        types = ['E', 'I']
        num_chan = self.num_chan
        n_neurons = len(rates)

        with open(info['recordings_params']) as f:
            if use_loader:
                rec_params = yaml.load(f, Loader=yaml.FullLoader)
            else:
                rec_params = yaml.load(f)

        rec_params['spiketrains']['rates'] = rates
        rec_params['spiketrains']['types'] = types
        rec_params['spiketrains']['duration'] = 3
        n_jitter = 3
        rec_params['templates']['n_jitters'] = n_jitter
        rec_params['templates']['min_dist'] = 1
        rec_params['recordings']['modulation'] = 'none'
        noise_modes = ['uncorrelated', 'distance-correlated']
        chunk_noise = [0, 2]
        noise_color = [True, False]

        for mode in noise_modes:
            for ch in chunk_noise:
                for color in noise_color:
                    print('Noise: mode', mode, 'chunks', ch, 'color', color)
                    rec_params['recordings']['noise_mode'] = mode
                    rec_params['recordings']['chunk_noise_duration'] = ch
                    rec_params['recordings']['noise_color'] = color
                    recgen_noise = mr.gen_recordings(params=rec_params, tempgen=self.tempgen)

                    assert recgen_noise.recordings.shape[0] == num_chan
                    assert len(recgen_noise.spiketrains) == n_neurons
                    assert recgen_noise.channel_positions.shape == (num_chan, 3)
                    assert recgen_noise.templates.shape[:3] == (n_neurons, n_jitter, num_chan)
                    assert recgen_noise.voltage_peaks.shape == (n_neurons, num_chan)
                    assert len(recgen_noise.spike_traces) == n_neurons
                    del recgen_noise

    def test_gen_recordings_only_noise(self):
        print('Test recording generation - only noise')
        info, info_folder = mr.get_default_config()
        ne = 0
        ni = 0
        num_chan = self.num_chan
        n_neurons = ne + ni

        with open(info['recordings_params']) as f:
            if use_loader:
                rec_params = yaml.load(f, Loader=yaml.FullLoader)
            else:
                rec_params = yaml.load(f)

        rec_params['spiketrains']['n_exc'] = ne
        rec_params['spiketrains']['n_inh'] = ni
        rec_params['spiketrains']['duration'] = 3
        rec_params['cell_types'] = None
        n_jitter = 3
        rec_params['templates']['n_jitters'] = n_jitter
        rec_params['templates']['min_dist'] = 1
        rec_params['recordings']['modulation'] = 'none'
        rec_params['recordings']['noise_mode'] = 'distance-correlated'
        rec_params['recordings']['noise_color'] = True
        recgen_noise = mr.gen_recordings(params=rec_params, tempgen=self.tempgen)

        assert recgen_noise.recordings.shape[0] == num_chan
        assert recgen_noise.channel_positions.shape == (num_chan, 3)
        assert len(recgen_noise.spiketrains) == n_neurons
        assert len(recgen_noise.spiketrains) == n_neurons
        assert len(recgen_noise.spiketrains) == n_neurons
        assert len(recgen_noise.spike_traces) == n_neurons

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
        rec_params['spiketrains']['duration'] = 3
        n_jitter = [1, 3]
        rec_params['recordings']['modulation'] = 'none'
        rec_params['recordings']['drifting'] = True
        rec_params['recordings']['drift_velocity'] = 300
        rec_params['templates']['min_dist'] = 1

        modulations = ['none', 'template', 'electrode']
        bursting = [False, True]

        for i, mod in enumerate(modulations):
            for b in bursting:
                for j in n_jitter:
                    print('Drifting: modulation', mod, 'bursting', b, 'jitter', j)
                    rec_params['templates']['n_jitters'] = j
                    rec_params['recordings']['modulation'] = mod
                    rec_params['recordings']['bursting'] = b
                    if i == len(modulations) - 1:
                        rec_params['recordings']['fs'] = 30
                        rec_params['recordings']['n_drifting'] = 1
                    recgen_drift = mr.gen_recordings(params=rec_params, tempgen=self.tempgen_drift)
                    assert recgen_drift.recordings.shape[0] == num_chan
                    assert len(recgen_drift.spiketrains) == n_neurons
                    assert recgen_drift.channel_positions.shape == (num_chan, 3)
                    if j == 1:
                        assert recgen_drift.templates.shape[0] == n_neurons
                        assert recgen_drift.templates.shape[2] == num_chan
                    else:
                        assert recgen_drift.templates.shape[0] == n_neurons
                        assert recgen_drift.templates.shape[2] == j
                        assert recgen_drift.templates.shape[3] == num_chan
                    assert len(recgen_drift.spike_traces) == n_neurons
                    del recgen_drift

    def test_save_load_templates(self):
        tempgen = mr.load_templates(self.test_dir + '/templates.h5', verbose=True)
        tempgen_drift = mr.load_templates(self.test_dir + '/templates_drift.h5')
        tempgen_drift_f = mr.load_templates(self.test_dir + '/templates_drift.h5', return_h5_objects=True)

        assert np.allclose(tempgen.templates, self.tempgen.templates)
        assert np.allclose(tempgen.locations, self.tempgen.locations)
        assert np.allclose(tempgen.rotations, self.tempgen.rotations)
        assert np.allclose(tempgen_drift.templates, self.tempgen_drift.templates)
        assert np.allclose(tempgen_drift.locations, self.tempgen_drift.locations)
        assert np.allclose(tempgen_drift.rotations, self.tempgen_drift.rotations)
        assert np.allclose(tempgen_drift.templates, np.array(tempgen_drift_f.templates))

    def test_save_load_recordings(self):
        recgen_loaded = mr.load_recordings(self.test_dir + '/recordings.h5', verbose=True)
        recgen_loaded_f = mr.load_recordings(self.test_dir + '/recordings.h5', return_h5_objects=True, verbose=True)

        assert np.allclose(recgen_loaded.templates, self.recgen.templates)
        assert np.allclose(recgen_loaded.recordings, self.recgen.recordings)
        assert np.allclose(recgen_loaded.spike_traces, self.recgen.spike_traces)
        assert np.allclose(recgen_loaded.voltage_peaks, self.recgen.voltage_peaks)
        assert np.allclose(recgen_loaded.channel_positions, self.recgen.channel_positions)
        assert np.allclose(recgen_loaded.timestamps, self.recgen.timestamps.magnitude)
        assert np.allclose(recgen_loaded.recordings, np.array(recgen_loaded_f.recordings))

    def test_plots(self):
        _ = mr.plot_rasters(self.recgen.spiketrains)
        _ = mr.plot_rasters(self.recgen.spiketrains, overlap=True)
        _ = mr.plot_rasters(self.recgen.spiketrains, bintype=True)
        _ = mr.plot_rasters(self.recgen.spiketrains, color='g')
        _ = mr.plot_rasters(self.recgen.spiketrains, color=['g']*len(self.recgen.spiketrains))
        _ = mr.plot_recordings(self.recgen)
        _ = mr.plot_templates(self.recgen)
        _ = mr.plot_templates(self.recgen, single_axes=True)
        _ = mr.plot_waveforms(self.recgen)
        _ = mr.plot_waveforms(self.recgen, electrode=0)
        _ = mr.plot_waveforms(self.recgen, electrode='max')
        self.recgen.spiketrains[0].waveforms = None
        _ = mr.plot_waveforms(self.recgen, electrode='max', color_isi=True)

    def test_extract_features(self):
        feat_t0 = mr.get_templates_features(self.tempgen.templates, feat_list=['na', 'rep', 'amp', 'width', 'fwhm',
                                                                               'ratio', 'speed'],
                                            dt=self.tempgen.info['params']['dt'])
        assert feat_t0['na'].shape == (self.tempgen.templates.shape[0], self.tempgen.templates.shape[1])
        assert feat_t0['rep'].shape == (self.tempgen.templates.shape[0], self.tempgen.templates.shape[1])
        assert feat_t0['amp'].shape == (self.tempgen.templates.shape[0], self.tempgen.templates.shape[1])
        assert feat_t0['width'].shape == (self.tempgen.templates.shape[0], self.tempgen.templates.shape[1])
        assert feat_t0['fwhm'].shape == (self.tempgen.templates.shape[0], self.tempgen.templates.shape[1])
        assert feat_t0['ratio'].shape == (self.tempgen.templates.shape[0], self.tempgen.templates.shape[1])
        assert feat_t0['speed'].shape == (self.tempgen.templates.shape[0], self.tempgen.templates.shape[1])

    def test_cli(self):
        default_config, mearec_home = mr.get_default_config()

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["default-config"])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["gen-templates", '-n', '2', '--no-parallel', '-r', '3drot', '-nc', '2',
                                     '-ov', '20', '-off', '0', '-dt', '10', '-s', '1', '-mind', '10', '-maxd', '100',
                                     '-drst', '10', '-v'])
        print(result.output)
        assert result.exit_code == 0
        result = runner.invoke(cli, ["gen-recordings", '-t', self.test_dir + '/templates.h5', '-ne', '2', '-ni', '1',
                                     '-fe', '5', '-fi', '15', '-se', '1', '-si', '1', '-mr', '0.2',
                                     '-rp', '2', '-p', 'poisson', '-md', '1', '-mina', '10', '-maxa', '1000',
                                     '--fs', '32',  '-sr', '0', '-sj', '1', '-nl', '10', '-m', 'none',
                                     '-chn', '0', '-chf', '0', '-nseed', '10', '-hd', '30', '-cn', '-cp', '500',
                                     '-cq', '1', '-rnf', '1', '-stseed', '100', '-tseed', '10',
                                     '--no-filt', '-fc', '500', '-fo', '3', '--overlap', '-ot', '0.8', '--extract-wf',
                                     '-angt', '15', '-drvel', '10', '-tsd', '1', '-v'])
        print(result.output)
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-templates-params", '.'])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-templates-params", default_config['templates_params']])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-recordings-params", '.'])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-recordings-params", default_config['recordings_params']])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-cell-models-folder", './cell_models'])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-cell-models-folder", default_config['cell_models_folder']])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-templates-folder", '.'])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-recordings-folder", '.'])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-templates-folder", './templates', '--create'])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-recordings-folder", './recordings', '--create'])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-templates-folder", default_config['templates_folder']])
        assert result.exit_code == 0
        result = runner.invoke(cli, ["set-recordings-folder", default_config['recordings_folder']])
        assert result.exit_code == 0


# if __name__ == '__main__':
#     unittest.main()
