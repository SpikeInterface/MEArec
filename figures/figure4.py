import MEArec as mr
import matplotlib.pylab as plt
from matplotlib import gridspec
import numpy as np
import neo
import quantities as pq
from copy import deepcopy
from plotting_conventions import *
import os

save_fig = False

plt.ion()
plt.show()


plot_mod = False
plot_burst = False
plot_pca = True

# create neo spike train
spacing = 10 * pq.ms
times = np.arange(0, 30) * spacing
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

st = neo.SpikeTrain(times=times, t_stop=np.max(times))
exp = 0.1
ylim = [0.3, 1.3]

mod_no_isi, _ = mr.tools.compute_modulation(st, sdrand=0.02, n_spikes=0)
mod_isi_5, _ = mr.tools.compute_modulation(st, sdrand=0.02, n_spikes=5, max_burst_duration=300 * pq.ms, exp=exp)
mod_isi_10, _ = mr.tools.compute_modulation(st, sdrand=0.02, n_spikes=10, max_burst_duration=300 * pq.ms, exp=exp)
mod_isi_75, _ = mr.tools.compute_modulation(st, sdrand=0.02, n_spikes=100, max_burst_duration=75 * pq.ms, exp=exp)

if plot_mod:
    fig1 = plt.figure(figsize=(7, 6))
    ax1 = fig1.add_subplot(5, 1, 1)
    ax2 = fig1.add_subplot(5, 1, 2)
    ax3 = fig1.add_subplot(5, 1, 3)
    ax4 = fig1.add_subplot(5, 1, 4)
    ax5 = fig1.add_subplot(5, 1, 5)

    ax1.plot(times, np.ones(len(times)), 'k', marker='|', mew=3, markersize=5, ls='')
    ax2.plot(times, mod_no_isi, colors[0], marker='o', mew=3, markersize=2, ls='')
    ax2.axhline(1, color='gray', ls='--', alpha=0.2)
    ax3.plot(times, mod_isi_5, colors[1], marker='o', mew=3, markersize=2, ls='')
    ax3.axhline(1, color='gray', ls='--', alpha=0.2)
    ax4.plot(times, mod_isi_10, colors[2], marker='o', mew=3, markersize=2, ls='')
    ax4.axhline(1, color='gray', ls='--', alpha=0.2)
    ax5.plot(times, mod_isi_75, colors[3], marker='o', mew=3, markersize=2, ls='')
    ax5.axhline(1, color='gray', ls='--', alpha=0.2)

    ax1.axis('off')
    ax2.set_ylim(ylim)
    ax3.set_ylim(ylim)
    ax4.set_ylim(ylim)
    ax5.set_ylim(ylim)

    simplify_axes([ax2, ax3, ax4], remove_bottom=True)
    simplify_axes([ax5])
    ax5.set_xlabel('time (ms)', fontsize=20)
    ax3.set_ylabel('modulation value', fontsize=20)

    if save_fig:
        if not os.path.isdir('figure4'):
            os.mkdir('figure4')

        fig1.savefig('figure4/bursting_modulation.pdf')


# Load recordings data
tetrode_rec = 'data/recordings/recordings_6cells_tetrode_30.0_10.0uV.h5'
recgen_noburst = mr.load_recordings(tetrode_rec)

if plot_burst:
    templates = recgen_noburst.templates

    template_id = 5

    mod_array = []
    for st in recgen_noburst.spiketrains:
        mod, _ = mr.tools.compute_modulation(st, sdrand=0.02, n_spikes=10, max_burst_duration=300 * pq.ms, exp=exp)
        mod_array.append(mod)

    templates_10 = []
    for i, t in enumerate(templates):
        rand_jitt = np.random.randint(0, t.shape[0])
        temp_max = t[rand_jitt, np.unravel_index(np.argmin(t), t.shape)[1]]
        templates_10.append(np.array([temp_max * m for m in mod_array[i]]))

    templates_burst_10 = []
    for i, t in enumerate(templates):
        rand_jitt = np.random.randint(0, t.shape[0])
        temp_max = t[rand_jitt, np.unravel_index(np.argmin(t), t.shape)[1]]
        templates_burst_10.append(np.array([mr.tools.compute_stretched_template(temp_max, m, sigmoid_range=10)
                                            for m in mod_array[i]]))

    templates_burst_10f = []
    for i, t in enumerate(templates):
        rand_jitt = np.random.randint(0, t.shape[0])
        temp_max = t[rand_jitt, np.unravel_index(np.argmin(t), t.shape)[1]]
        templates_burst_10f.append(np.array([mr.tools.compute_stretched_template(temp_max, m, sigmoid_range=30)
                                             for m in mod_array[i]]))
    fig2 = plt.figure(figsize=(7, 6))
    ax3 = fig2.add_subplot(1, 1, 1)
    ax3.plot(templates_burst_10f[template_id][0], colors[2], label='Sigmoid 30')
    ax3.plot(templates_burst_10f[template_id].T, colors[2], lw=0.5)
    ax3.plot(templates_burst_10[template_id][0], colors[1], label='Sigmoid 10')
    ax3.plot(templates_burst_10[template_id].T, colors[1], lw=0.5)
    ax3.plot(templates_10[template_id][0], colors[0], label='No burst')
    ax3.plot(templates_10[template_id].T, colors[0], lw=0.5, alpha=0.2)
    ax3.set_ylabel('voltage ($\mu$V)', fontsize=20)
    ax3.legend(fontsize=18, loc='lower right')

    fig3 = plt.figure(figsize=(7, 13))
    gs = gridspec.GridSpec(7, 5)
    ax31 = fig3.add_subplot(gs[:1, :])
    ax32 = fig3.add_subplot(gs[1:3, :])
    ax33 = fig3.add_subplot(gs[3:, :])

    ts = recgen_noburst.timestamps * pq.ms
    spiketrain = recgen_noburst.spiketrains[template_id].times
    modulations = mod_array[template_id]
    cut_out_temp = recgen_noburst.info['templates']['cut_out']
    pad = recgen_noburst.info['templates']['pad_len']
    cut_outs = [int((c + p) * recgen_noburst.info['recordings']['fs'] / 1000.) for (c,p) in zip(cut_out_temp, pad)]
    spike_bin = mr.tools.resample_spiketrains(recgen_noburst.spiketrains,
                                              fs=recgen_noburst.info['recordings']['fs']*pq.Hz)
    spike_trace = mr.tools.convolve_single_template(template_id, spike_bin[template_id], templates[template_id, 0],
                                                    cut_out=cut_out_temp, modulation=True, mod_array=modulations,
                                                    bursting=True, sigmoid_range=30)

    time_interval = [5, 6] * pq.s
    idxs_st = np.where((spiketrain > time_interval[0]) & (spiketrain < time_interval[1]))
    idxs_rec = np.where((ts > time_interval[0]) & (ts < time_interval[1]))

    ax31.plot(spiketrain[idxs_st], 1.5*np.ones(len(spiketrain[idxs_st])), 'k', marker='|', mew=3, markersize=10, ls='')
    ax32.plot(spiketrain[idxs_st], modulations[idxs_st], colors[0], marker='o', mew=3, markersize=2, ls='')
    ax32.axhline(1, color='gray', ls='--', alpha=0.2)
    ax33.plot(ts[idxs_rec].rescale('s'), spike_trace[idxs_rec], lw=0.5, color='k')

    simplify_axes([ax3, ax31, ax32], remove_bottom=True)
    ax31.axis('off')
    # ax31.set_title('spike train', fontsize=20)
    ax31.set_xlim(time_interval)
    ax32.set_xlim(time_interval)
    ax32.set_yticks([0.8, 0.9, 1])
    ax32.set_ylabel('modulation value', fontsize=20)
    simplify_axes([ax33])
    ax33.set_xlim(time_interval)
    ax33.set_xlabel('time (s)', fontsize=20)
    ax33.set_ylabel('voltage ($\mu$V)', fontsize=20)

    fig2.tight_layout()
    fig3.tight_layout()

    if save_fig:
        if not os.path.isdir('figure4'):
            os.mkdir('figure4')

        fig2.savefig('figure4/bursting_templates.png', dpi=600)
        fig3.savefig('figure4/bursting_signal.pdf')


if plot_pca:
    # regenerate recording with bursting and shape-mod
    recgen_burst = deepcopy(recgen_noburst)
    recgen_noburst.extract_waveforms(cut_out=2)
    recgen_burst.params['recordings']['bursting'] = True
    recgen_burst.params['recordings']['shape_mod'] = True
    recgen_burst.generate_recordings()
    recgen_burst.extract_waveforms(cut_out=2)

    fig41 = plt.figure(figsize=(7, 6))
    fig42 = plt.figure(figsize=(7, 6))
    ax41 = fig41.add_subplot(111)
    ax42 = fig42.add_subplot(111)

    ax41, pc_scores1, pc_comp1 = mr.plot_pca_map(recgen_noburst, n_pc=1, ax=ax41, cmap='rainbow', whiten=False)
    ax42, pc_scores2, _ = mr.plot_pca_map(recgen_burst, n_pc=1, ax=ax42, cmap='rainbow', whiten=False, pc_comp=pc_comp1)

    if save_fig:
        if not os.path.isdir('figure4'):
            os.mkdir('figure4')

        fig41.savefig('figure4/nonbusting_pca.png', dpi=600)
        fig42.savefig('figure4/busting_pca.png', dpi=600)
