from __future__ import print_function, division

"""
Generates spike trains using elephant
"""


import os
from os.path import join
import time
import click
import yaml
import shutil
import numpy as np
import neo
import elephant.spike_train_generation as stg
import elephant.conversion as conv
import elephant.statistics as stat
import matplotlib.pylab as plt
import quantities as pq
from quantities import Quantity

class SpikeTrainGenerator:
    def __init__(self, params):
        '''
        Spike Train Generator: class to create poisson or gamma spike trains

        Parameters
        ----------
        n_exc: number of excitatory cells
        n_inh: number of inhibitory cells
        f_exc: mean firing rate of excitatory cells
        f_inh: mean firing rate of inhibitory cells
        st_exc: firing rate standard deviation of excitatory cells
        st_inh: firing rate standard deviation of inhibitory cells
        process: 'poisson' - 'gamma'
        gamma_shape: shape param for gamma distribution
        t_start: starting time (s)
        t_stop: stopping time (s)
        ref_period: refractory period to remove spike violation
        n_add: number of units to add at t_add time
        t_add: time to add units
        n_remove: number of units to remove at t_remove time
        t_remove: time to remove units
        '''
    
        self.params = params

        # Check quantities
        if 't_start' not in self.params.keys():
            self.params['t_start'] = 0 * pq.s
        else:
            self.params['t_start'] = self.params['t_start'] * pq.s
        if 'duration' in self.params.keys():
            self.params['t_stop'] = self.params['t_start'] + self.params['duration'] * pq.s
        if 'min_rate' not in self.params.keys():
            self.params['min_rate'] = 0.1 * pq.Hz
        else:
            self.params['min_rate'] = self.params['min_rate'] * pq.Hz
        if 'ref_per' not in self.params.keys():
            self.params['ref_per'] = 2 * pq.ms
        else:
            self.params['ref_per'] = self.params['ref_per'] * pq.ms
        if 'rates' in self.params.keys(): # all firing rates are provided
            self.params['rates'] = self.params['rates'] * pq.Hz
            self.n_neurons = len(self.params['rates'])
        else:
            rates = []
            if 'f_exc' not in self.params.keys():
                self.params['f_exc'] = 5 * pq.Hz
            else:
                self.params['f_exc'] = self.params['f_exc'] * pq.Hz
            if 'f_inh' not in self.params.keys():
                self.params['f_inh'] = 15 * pq.Hz
            else:
                self.params['f_inh'] = self.params['f_inh'] * pq.Hz
            if 'st_exc' not in self.params.keys():
                self.params['st_exc'] = 1 * pq.Hz
            else:
                self.params['st_exc'] = self.params['st_exc'] * pq.Hz
            if 'st_inh' not in self.params.keys():
                self.params['st_inh'] = 3 * pq.Hz
            else:
                self.params['st_inh'] = self.params['st_inh'] * pq.Hz
            if 'n_exc' not in self.params.keys():
                self.params['n_exc'] = 15
            if 'n_inh' not in self.params.keys():
                self.params['n_inh'] = 5

            for exc in range(self.params['n_exc']):
                rate = self.params['st_exc'] * np.random.randn() + self.params['f_exc']
                if rate < self.params['min_rate']:
                    rate = self.params['min_rate']
                rates.append(rate)
            for inh in range(self.params['n_inh']):
                rate = self.params['st_inh'] * np.random.randn() + self.params['f_inh']
                if rate < self.params['min_rate']:
                    rate = self.params['min_rate']
                rates.append(rate)
            self.params['rates'] = rates
            self.n_neurons = len(self.params['rates'])

        self.changing = False
        self.intermittent= False

        
        # self.changing = False
        # self.n_add = n_add
        # self.n_remove = n_remove
        # self.t_add = int(t_add) * pq.s
        # self.t_remove = int(t_remove) * pq.s
        # 
        # self.intermittent = False
        # self.n_int = n_int
        # self.t_int = t_int
        # self.t_burst = t_burst
        # self.t_int_sd = t_int
        # self.t_burst_sd = t_burst
        # self.f_int = f_int
        # 
        # self.idx = 0
        # 
        # 
        # if n_add != 0:
        #     if t_add == 0:
        #         raise Exception('Provide time to add units')
        #     else:
        #         self.changing = True
        # if n_remove != 0:
        #     if t_remove == 0:
        #         raise Exception('Provide time to remove units')
        #     else:
        #         self.changing = True
        # if n_int != 0:
        #     self.intermittent = True
        # 
        # 
        # if self.changing:
        #     n_tot = n_exc + n_inh
        #     perm_idxs = np.random.permutation(np.arange(n_tot))
        #     self.idxs_add = perm_idxs[:self.n_add]
        #     self.idxs_remove = perm_idxs[-self.n_remove:]
        # else:
        #     self.idxs_add = []
        #     self.idxs_remove = []
        # 
        # if self.intermittent:
        #     n_tot = n_exc + n_inh
        #     perm_idxs = np.random.permutation(np.arange(n_tot))
        #     self.idxs_int = perm_idxs[:self.n_int]
        # else:
        #     self.idxs_int = []

    def set_spiketrain(self, idx, spiketrain):
        '''
        Sets spike train idx to new spiketrain
        Parameters
        ----------
        idx: index of spike train to set
        spiketrain: new spike train

        Returns
        -------

        '''
        self.all_spiketrains[idx] = spiketrain

    def generate_spikes(self):
        '''
        Generate spike trains based on params of the SpikeTrainGenerator class.
        self.all_spiketrains contains the newly generated spike trains

        Returns
        -------

        '''

        self.all_spiketrains = []
        idx = 0
        for n in range(self.n_neurons):
            if not self.changing and not self.intermittent:
                rate = self.params['rates'][n]
                if self.params['process'] == 'poisson':
                    st = stg.homogeneous_poisson_process(rate,
                                                         self.params['t_start'], self.params['t_stop'])
                elif self.params['process'] == 'gamma':
                    st = stg.homogeneous_gamma_process(rate, self.params['rates'][n],
                                                       self.params['t_start'], self.params['t_stop'])
            else:
                raise NotImplementedError('Changing and intermittent spiketrains are not impleented yet')
            self.all_spiketrains.append(st)
            self.all_spiketrains[-1].annotate(freq=rate)
            if 'n_exc' in self.params.keys() and 'n_inh' in self.params.keys():
                if idx < self.params['n_exc']:
                    self.all_spiketrains[-1].annotate(type='exc')
                else:
                    self.all_spiketrains[-1].annotate(type='inh')
            idx += 1

        # check consistency and remove spikes below refractory period
        for idx, st in enumerate(self.all_spiketrains):
            isi = stat.isi(st)
            idx_remove = np.where(isi < self.params['ref_per'])[0]
            spikes_to_remove = len(idx_remove)
            unit = st.times.units

            while spikes_to_remove > 0:
                new_times = np.delete(st.times, idx_remove[0]) * unit
                st = neo.SpikeTrain(new_times, t_start=self.params['t_start'], t_stop=self.params['t_stop'])
                isi = stat.isi(st)
                idx_remove = np.where(isi < self.params['ref_per'])[0]
                spikes_to_remove = len(idx_remove)

            st.annotations = self.all_spiketrains[idx].annotations
            self.set_spiketrain(idx, st)


    def raster_plots(self, marker='|', markersize=5, mew=2):
        '''
        Plots raster plots of spike trains

        Parameters
        ----------
        marker: marker type (def='|')
        markersize: marker size (def=5)
        mew: marker edge width (def=2)

        Returns
        -------
        ax: matplotlib axes

        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, spiketrain in enumerate(self.all_spiketrains):
            t = spiketrain.rescale(pq.s)
            if i < self.params['n_exc']:
                ax.plot(t, i * np.ones_like(t), color='b', marker=marker, ls='', markersize=markersize, mew=mew)
            else:
                ax.plot(t, i * np.ones_like(t), color='r', marker=marker, ls='', markersize=markersize, mew=mew)
        ax.axis('tight')
        ax.set_xlim([self.params['t_start'].rescale(pq.s), self.params['t_stop'].rescale(pq.s)])
        ax.set_xlabel('Time (ms)', fontsize=16)
        ax.set_ylabel('Spike Train Index', fontsize=16)
        plt.gca().tick_params(axis='both', which='major', labelsize=14)

        return ax

    def resample_spiketrains(self, fs=None, T=None):
        '''
        Resamples spike trains. Provide either fs or T parameters
        Parameters
        ----------
        fs: new sampling frequency (quantity)
        T: new period (quantity)

        Returns
        -------
        matrix with resampled binned spike trains

        '''
        resampled_mat = []
        if not fs and not T:
            print('Provide either sampling frequency fs or time period T')
        elif fs:
            if not isinstance(fs, Quantity):
                raise ValueError("fs must be of type pq.Quantity")
            binsize = 1./fs
            binsize.rescale('ms')
            resampled_mat = []
            for sts in self.all_spiketrains:
                spikes = conv.BinnedSpikeTrain(sts, binsize=binsize).to_array()
                resampled_mat.append(np.squeeze(spikes))
        elif T:
            binsize = T
            if not isinstance(T, Quantity):
                raise ValueError("T must be of type pq.Quantity")
            binsize.rescale('ms')
            resampled_mat = []
            for sts in self.all_spiketrains:
                spikes = conv.BinnedSpikeTrain(sts, binsize=binsize).to_array()
                resampled_mat.append(np.squeeze(spikes))
        return np.array(resampled_mat)

    def add_synchrony(self, idxs, rate=0.05):
        '''
        Adds synchronous spikes between pairs of spike trains at a certain rate
        Parameters
        ----------
        idxs: list or array with the 2 indices
        rate: probability of adding a synchronous spike to spike train idxs[1] for each spike of idxs[0]

        Returns
        -------

        '''
        idx1 = idxs[0]
        idx2 = idxs[1]
        st1 = self.all_spiketrains[idx1]
        st2 = self.all_spiketrains[idx2]
        times2 = st2.times
        t_start = st2.t_start
        t_stop = st2.t_stop
        unit = times2.units
        for t1 in st1:
            rand = np.random.rand()
            if rand <= rate:
                # check time difference
                t_diff = np.abs(t1.rescale(pq.ms).magnitude-times2.rescale(pq.ms).magnitude)
                if np.all(t_diff > self.params['ref_period']):
                    times2 = np.sort(np.concatenate((np.array(times2), np.array([t1]))))
                    times2 = times2 * unit
                    st2 = neo.SpikeTrain(times2, t_start=t_start, t_stop=t_stop)
                    self.set_spiketrain(idx2, st2)


    def bursting_st(self, freq=None, min_burst=3, max_burst=10):
        pass


@click.command()
@click.option('--params', '-prm', default=None,
              help='path to params.yaml (otherwise default params are used)')
@click.option('--default', is_flag=True,
              help='shows default values for simulation')
@click.option('--fname', '-fn', default=None,
              help='template filename')
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
@click.option('--duration', '-d', default=None, type=float,
              help='duration in s (default=10)')
def run(params, **kwargs):
    """Generates spike trains for recordings"""
    # Retrieve params file
    if params is None:
        with open(join('params/spiketrain_params.yaml'), 'r') as pf:
            params_dict = yaml.load(pf)
    else:
        with open(params, 'r') as pf:
            params_dict = yaml.load(pf)

    if kwargs['default'] is True:
        print(params_dict)
        return

    spiketrain_folder = params_dict['spiketrain_folder']

    if kwargs['n_exc'] is not None:
        params_dict['n_exc'] = kwargs['n_exc']
    if kwargs['n_inh'] is not None:
        params_dict['n_inh'] = kwargs['n_inh']
    if kwargs['f_exc'] is not None:
        params_dict['f_exc'] = kwargs['f_exc']
    if kwargs['f_inh'] is not None:
        params_dict['f_inh'] = kwargs['f_inh']
    if kwargs['st_exc'] is not None:
        params_dict['st_exc'] = kwargs['st_exc']
    if kwargs['st_inh'] is not None:
        params_dict['st_inh'] = kwargs['st_inh']
    if kwargs['min_rate'] is not None:
        params_dict['min_rate'] = kwargs['min_rate']
    if kwargs['ref_per'] is not None:
        params_dict['ref_per'] = kwargs['ref_per']
    if kwargs['process'] is not None:
        params_dict['process'] = kwargs['process']
    if kwargs['min_rate'] is not None:
        params_dict['min_rate'] = kwargs['min_rate']
    if kwargs['tstart'] is not None:
        params_dict['t_start'] = kwargs['tstart']
    if kwargs['duration'] is not None:
        params_dict['duration'] = kwargs['duration']

    info = {'Params': params_dict}
    yaml.dump(info, open('tmp_info.yaml', 'w'), default_flow_style=False)

    spgen = SpikeTrainGenerator(params_dict)
    spgen.generate_spikes()

    spiketrains = spgen.all_spiketrains
    n_neurons = len(spiketrains)

    if kwargs['fname'] is None:
        fname = 'spiketrains_%d_%s' % (n_neurons, time.strftime("%d-%m-%Y:%H:%M"))
    else:
        fname = kwargs['fname']
    save_folder = join(spiketrain_folder, fname)

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    np.save(join(save_folder, 'gtst'), spiketrains)
    shutil.move('tmp_info.yaml', join(save_folder, 'info.yaml'))
    print('\nSaved spike trains in', save_folder, '\n')


if __name__ == '__main__':
    run()