import MEArec as mr
import MEAutility as mu
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from plotting_conventions import *
import numpy as np
import os

save_fig = False

plt.ion()
plt.show()

template_file = 'data/templates/templates_30_tetrode.h5'
recording_file = 'data/recordings/recordings_6cells_tetrode_30.0_10.0uV.h5'

tempgen = mr.load_templates(template_file)
recgen = mr.load_recordings(recording_file)

mea = mu.return_mea(info=recgen.info['electrodes'])

fig = plt.figure(figsize=(12, 11))
gs = gridspec.GridSpec(20, 3)

ax_timeseries = fig.add_subplot(gs[:4, :])
ax_waveforms = fig.add_subplot(gs[5:8, :])
ax_pca = fig.add_subplot(gs[10:, :])

mr.plot_recordings(recgen, ax=ax_timeseries, overlay_templates=True, cmap='rainbow', start_time=5, end_time=6, lw=0.3)
mr.plot_waveforms(recgen, max_waveforms=100, electrode='max', cmap='rainbow', ax=ax_waveforms)
mr.plot_pca_map(recgen, n_pc=2, ax=ax_pca, cmap='rainbow')

y_lim = ax_timeseries.get_ylim()
x_lim = ax_timeseries.get_xlim()
y_ticks = mea.positions[:, 2]
ax_timeseries.set_yticks(y_ticks)
ax_timeseries.set_yticklabels(['Ch. 0', 'Ch. 1', 'Ch. 2', 'Ch. 3'])
ax_timeseries.axis('on')
ax_timeseries.spines['top'].set_visible(False)
ax_timeseries.spines['right'].set_visible(False)
ax_timeseries.spines['bottom'].set_visible(False)
ax_timeseries.plot([x_lim[0] + 0.02*(np.max(x_lim) - np.min(x_lim)), x_lim[0] + 0.12*(np.max(x_lim) - np.min(x_lim))],
                   [np.min(y_lim), np.min(y_lim)], 'k', lw=2)
ax_timeseries.text(x_lim[0] + 0.05*(np.max(x_lim) - np.min(x_lim)), np.min(y_lim) + 0.15*np.min(y_lim), '100 ms')

fig.subplots_adjust(bottom=0.02, top=0.98)

if save_fig:
    if not os.path.isdir('figure1'):
        os.mkdir('figure1')

    fig.savefig('figure1/figure1.png', dpi=600)
