import numpy as np
import neo
import elephant.spike_train_generation as stg
import elephant.statistics as stat
from copy import copy, deepcopy
from MEArec.tools import *
from pprint import pprint
import quantities as pq


class SpikeTrainGenerator:
    """
    Class for generation of spike trains called by the gen_recordings function.
    The list of parameters is in default_params/recordings_params.yaml (spiketrains field).
    """

    def __init__(self, params=None, spiketrains=None, verbose=False):
        self._verbose = verbose
        self._has_spiketrains = False
        self.params = {}
        if params is None:
            if self._verbose:
                print("Using default parameters")
        if spiketrains is None:
            self.params = deepcopy(params)
            if self._verbose:
                print('Spiketrains seed: ', self.params['seed'])
            np.random.seed(self.params['seed'])

            if 't_start' not in self.params.keys():
                params['t_start'] = 0
            self.params['t_start'] = params['t_start'] * pq.s
            if 'duration' not in self.params.keys():
                params['duration'] = 10
            self.params['t_stop'] = self.params['t_start'] + params['duration'] * pq.s
            if 'min_rate' not in self.params.keys():
                params['min_rate'] = 0.1
            self.params['min_rate'] = params['min_rate'] * pq.Hz
            if 'ref_per' not in self.params.keys():
                params['ref_per'] = 2
            self.params['ref_per'] = params['ref_per'] * pq.ms
            if 'process' not in self.params.keys():
                params['process'] = 'poisson'
            self.params['process'] = params['process']
            if 'gamma_shape' not in self.params.keys() and params['process'] == 'gamma':
                params['gamma_shape'] = 2
                self.params['gamma_shape'] = params['gamma_shape']

            if 'rates' in self.params.keys():  # all firing rates are provided
                self.params['rates'] = self.params['rates'] * pq.Hz
                self.n_neurons = len(self.params['rates'])
            else:
                rates = []
                types = []
                if 'f_exc' not in self.params.keys():
                    params['f_exc'] = 5
                self.params['f_exc'] = params['f_exc'] * pq.Hz
                if 'f_inh' not in self.params.keys():
                    params['f_inh'] = 15
                self.params['f_inh'] = params['f_inh'] * pq.Hz
                if 'st_exc' not in self.params.keys():
                    params['st_exc'] = 1
                self.params['st_exc'] = params['st_exc'] * pq.Hz
                if 'st_inh' not in self.params.keys():
                    params['st_inh'] = 3
                self.params['st_inh'] = params['st_inh'] * pq.Hz
                if 'n_exc' not in self.params.keys():
                    params['n_exc'] = 2
                self.params['n_exc'] = params['n_exc']
                if 'n_inh' not in self.params.keys():
                    params['n_inh'] = 1
                self.params['n_inh'] = params['n_inh']

                for exc in np.arange(self.params['n_exc']):
                    rate = self.params['st_exc'] * np.random.randn() + self.params['f_exc']
                    if rate < self.params['min_rate']:
                        rate = self.params['min_rate']
                    rates.append(rate)
                    types.append('e')
                for inh in np.arange(self.params['n_inh']):
                    rate = self.params['st_inh'] * np.random.randn() + self.params['f_inh']
                    if rate < self.params['min_rate']:
                        rate = self.params['min_rate']
                    rates.append(rate)
                    types.append('i')
                self.params['rates'] = rates
                self.params['types'] = types
                self.n_neurons = len(self.params['rates'])

            self.changing = False
            self.intermittent = False

            self.info = params
            self.spiketrains = False
        else:
            self.spiketrains = spiketrains
            self._has_spiketrains = True
            if params is not None:
                self.params = deepcopy(params)

    def set_spiketrain(self, idx, spiketrain):
        """
        Sets spike train idx to new spiketrain.

        Parameters
        ----------
        idx : int
            Index of spike train to set
        spiketrain : neo.SpikeTrain
            New spike train

        """
        self.spiketrains[idx] = spiketrain

    def generate_spikes(self):
        """
        Generate spike trains based on default_params of the SpikeTrainGenerator class.
        self.spiketrains contains the newly generated spike trains
        """
        if not self._has_spiketrains:
            self.spiketrains = []
            idx = 0
            for n in np.arange(self.n_neurons):
                if not self.changing and not self.intermittent:
                    rate = self.params['rates'][n]
                    if self.params['process'] == 'poisson':
                        st = stg.homogeneous_poisson_process(rate,
                                                             self.params['t_start'], self.params['t_stop'])
                    elif self.params['process'] == 'gamma':
                        st = stg.homogeneous_gamma_process(self.params['gamma_shape'], rate,
                                                           self.params['t_start'], self.params['t_stop'])
                else:
                    raise NotImplementedError('Changing and intermittent spiketrains are not impleented yet')
                self.spiketrains.append(st)
                self.spiketrains[-1].annotate(fr=rate)
                if 'n_exc' in self.params.keys() and 'n_inh' in self.params.keys():
                    if idx < self.params['n_exc']:
                        self.spiketrains[-1].annotate(type='E')
                    else:
                        self.spiketrains[-1].annotate(type='I')
                idx += 1

            # check consistency and remove spikes below refractory period
            for idx, st in enumerate(self.spiketrains):
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

                st.annotations = self.spiketrains[idx].annotations
                self.set_spiketrain(idx, st)
        else:
            print("SpikeTrainGenerator initialized with existing spike trains!")

    def add_synchrony(self, idxs, rate=0.05, time_jitt=1 * pq.ms, verbose=False):
        """
        Adds synchronous spikes between pairs of spike trains at a certain rate.

        Parameters
        ----------
        idxs : list or array
            Spike train indexes to add synchrony to
        rate : float
            Rate of added synchrony spike to spike train idxs[1] for each spike of idxs[0]
        time_jitt : quantity
            Maximum time jittering between added spikes
        verbose : bool
            If True output is verbose

        Returns
        -------
        sync_rate : float
            New synchrony rate
        fr1 : quantity
            Firing rate spike train 1
        fr2 : quantity
            Firing rate spike train 2

        """
        idx1 = idxs[0]
        idx2 = idxs[1]
        st1 = self.spiketrains[idx1]
        st2 = self.spiketrains[idx2]
        times1 = st1.times
        times2 = st2.times
        t_start = st2.t_start
        t_stop = st2.t_stop
        unit = times2.units
        added_spikes_t1 = 0
        added_spikes_t2 = 0

        all_times_shuffle = np.concatenate((times1, times2))
        all_times_shuffle = all_times_shuffle[np.random.permutation(len(all_times_shuffle))] * unit

        sync_rate = compute_sync_rate(st1, st2, time_jitt)
        if sync_rate < rate:
            for t in all_times_shuffle:
                if sync_rate <= rate:
                    # check time difference
                    if t in times1:
                        t_diff = np.abs(t.rescale(pq.ms).magnitude - times2.rescale(pq.ms).magnitude)
                        if np.all(t_diff > self.params['ref_per']):
                            t1_jitt = time_jitt.rescale(unit).magnitude * np.random.rand(1) + t.rescale(
                                unit).magnitude - \
                                      (time_jitt.rescale(unit) / 2).magnitude
                            if t1_jitt < t_stop:
                                times2 = np.sort(np.concatenate((np.array(times2), np.array(t1_jitt))))
                                times2 = times2 * unit
                                st2 = neo.SpikeTrain(times2, t_start=t_start, t_stop=t_stop)
                                added_spikes_t1 += 1
                    elif t in times2:
                        t_diff = np.abs(t.rescale(pq.ms).magnitude - times1.rescale(pq.ms).magnitude)
                        if np.all(t_diff > self.params['ref_per']):
                            t2_jitt = time_jitt.rescale(unit).magnitude * np.random.rand(1) + t.rescale(
                                unit).magnitude - \
                                      (time_jitt.rescale(unit) / 2).magnitude
                            if t2_jitt < t_stop:
                                times1 = np.sort(np.concatenate((np.array(times1), np.array(t2_jitt))))
                                times1 = times1 * unit
                                st1 = neo.SpikeTrain(times1, t_start=t_start, t_stop=t_stop)
                                added_spikes_t2 += 1
                    sync_rate = compute_sync_rate(st1, st2, time_jitt)
                else:
                    break
            if verbose:
                print("Added", added_spikes_t1, "spikes to spike train", idxs[0],
                      "and", added_spikes_t2, "spikes to spike train", idxs[1], 'Sync rate:', sync_rate)
        else:
            spiketrains = [st1, st2]
            annotate_overlapping_spikes(spiketrains)
            max_overlaps = np.floor(rate * (len(times1) + len(times2)))
            curr_overlaps = np.floor(sync_rate * (len(times1) + len(times2)))
            remove_overlaps = int(curr_overlaps - max_overlaps)
            if curr_overlaps > max_overlaps:
                st1_ovrl_idx = np.where(spiketrains[0].annotations['overlap'] == 'O')[0]
                st2_ovrl_idx = np.where(spiketrains[1].annotations['overlap'] == 'O')[0]
                perm = np.random.permutation(len(st1_ovrl_idx))[:remove_overlaps]
                st1_ovrl_idx = st1_ovrl_idx[perm]
                st2_ovrl_idx = st2_ovrl_idx[perm]
                idx_rm_1 = st1_ovrl_idx[:remove_overlaps // 2]
                idx_rm_2 = st2_ovrl_idx[remove_overlaps // 2:]
                times1 = np.delete(st1.times, idx_rm_1)
                times1 = times1 * unit
                times2 = np.delete(st2.times, idx_rm_2)
                times2 = times2 * unit
                st1 = neo.SpikeTrain(times1, t_start=t_start, t_stop=t_stop)
                st2 = neo.SpikeTrain(times2, t_start=t_start, t_stop=t_stop)
                sync_rate = compute_sync_rate(st1, st2, time_jitt)
                if verbose:
                    print("Removed", len(idx_rm_1), "spikes from spike train", idxs[0],
                          "and", len(idx_rm_2), "spikes from spike train", idxs[1], 'Sync rate:', sync_rate)

        st1.annotations = self.spiketrains[idx1].annotations
        st2.annotations = self.spiketrains[idx2].annotations
        self.set_spiketrain(idx1, st1)
        self.set_spiketrain(idx2, st2)

        fr1 = len(st1.times) / st1.t_stop
        fr2 = len(st2.times) / st2.t_stop

        return sync_rate, fr1, fr2