import MEArec as mr
import matplotlib.pylab as plt
from plotting_conventions import *
import numpy as np
import os

save_fig = False

plt.ion()
plt.show()

template_file = 'data/templates/templates_100_Neuronexus-32.h5'

tempgen = mr.load_templates(template_file, return_h5_objects=False)

recgen_random = mr.gen_recordings(tempgen=tempgen, params='figure5_params.yaml', verbose=2)
params = recgen_random.params
recgen_random.annotate_overlapping_spikes()

params['recordings']['sync_rate'] = 0
recgen_0 = mr.gen_recordings(tempgen=tempgen, params=params, verbose=2)
recgen_0.annotate_overlapping_spikes()

params['recordings']['sync_rate'] = 0.05
recgen_005 = mr.gen_recordings(tempgen=tempgen, params=params, verbose=2)
recgen_005.annotate_overlapping_spikes()

fig1 = plt.figure(figsize=(7, 9))
ax_t = fig1.add_subplot(1, 1, 1)
fig2 = plt.figure(figsize=(18, 5))
ax_r = fig2.add_subplot(1, 3, 1)
ax_0 = fig2.add_subplot(1, 3, 2)
ax_005 = fig2.add_subplot(1, 3, 3)

mr.plot_templates(recgen_random, ax=ax_t, lw=1.5, single_axes=True)
max_electrode = np.unravel_index(np.argmin(recgen_random.templates[0, 0]), recgen_random.templates[0, 0].shape)[0]
ax_t.plot(recgen_random.channel_positions[max_electrode, 1], recgen_random.channel_positions[max_electrode, 2],
          '*', color='k', markersize=5)

mr.plot_rasters(recgen_random.spiketrains, overlap=True, ax=ax_r, mew=2, markersize=15)
mr.plot_rasters(recgen_0.spiketrains, overlap=True, ax=ax_0, mew=2, markersize=15)
mr.plot_rasters(recgen_005.spiketrains, overlap=True, ax=ax_005, mew=2, markersize=15)
ax_r.set_xlabel('time (s)', fontsize=18)
ax_r.set_ylabel('')
ax_0.set_xlabel('time (s)', fontsize=18)
ax_0.set_ylabel('')
ax_005.set_xlabel('time (s)', fontsize=18)
ax_005.set_ylabel('')
ax_r.spines['left'].set_visible(False)
ax_0.spines['left'].set_visible(False)
ax_005.spines['left'].set_visible(False)
ax_r.set_ylim(-.5, 1.5)
ax_0.set_ylim(-.5, 1.5)
ax_005.set_ylim(-.5, 1.5)
ax_r.set_xlim(0, 5)
ax_0.set_xlim(0, 5)
ax_005.set_xlim(0, 5)
ax_r.set_yticks([0, 1])
ax_r.set_yticklabels(['Neuron 1', 'Neuron 2'], fontsize=18)
ax_0.set_yticks([])
ax_005.set_yticks([])

ax_r.set_title('Random synchrony rate', fontsize=20)
ax_0.set_title('Synchrony rate: 0', fontsize=20)
ax_005.set_title('Synchrony rate: 0.05', fontsize=20)
simplify_axes([ax_r, ax_0, ax_005])

fig3 = plt.figure(figsize=(7, 9))
ax_timeseries = fig3.add_subplot(111)
mr.plot_recordings(recgen_005, overlay_templates=True, start_time=0, end_time=1, lw=0.1, ax=ax_timeseries,
                   max_channels_per_template=10)
y_lim = ax_timeseries.get_ylim()
x_lim = ax_timeseries.get_xlim()

ts_lim = (x_lim[1] - x_lim[0]) // 3
ax_timeseries.plot([x_lim[0], x_lim[0] + 0.2 * ts_lim],
                   [np.min(y_lim) + 0.12 * np.abs(np.min(y_lim)),
                    np.min(y_lim) + 0.12 * np.abs(np.min(y_lim))], 'k', lw=2)
ax_timeseries.text(x_lim[0] + 0.02 * ts_lim, np.min(y_lim) + 0.05 * np.abs(np.min(y_lim)), '200 ms', fontsize=8)

y_lim = ax_t.get_ylim()
x_lim = ax_t.get_xlim()

ts_lim = (x_lim[1] - x_lim[0]) // 3
ax_t.plot([x_lim[0], x_lim[0] + 0.38 * ts_lim],
          [np.min(y_lim) + 0.12 * np.abs(np.min(y_lim)),
           np.min(y_lim) + 0.12 * np.abs(np.min(y_lim))], 'k', lw=2)
ax_t.text(x_lim[0] + 0.07 * ts_lim, np.min(y_lim) + 0.05 * np.abs(np.min(y_lim)), '5 ms', fontsize=8)

if save_fig:
    if not os.path.isdir('figure5'):
        os.mkdir('figure5')

    fig1.savefig('figure5/templates.pdf')
    fig2.savefig('figure5/rasters.pdf')
    fig3.savefig('figure5/rec.png', dpi=600)
