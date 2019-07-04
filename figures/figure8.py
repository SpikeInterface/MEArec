import MEArec as mr
import matplotlib.pylab as plt
from plotting_conventions import *
import numpy as np
import yaml
import os

save_fig = False

plt.ion()
plt.show()

# several noise levels
noise_levels = [30, 20, 10, 5]

template_file = 'data/templates/templates_100_Neuronexus-32.h5'
tempgen = mr.load_templates(template_file)

with open('figure8_params.yaml', 'r') as f:
    params = yaml.load(f)

params['spiketrains']['seed'] = 0
params['templates']['seed'] = 1
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

recordings_noise = []
for n in noise_levels:
    print('noise level:', n)
    params['recordings']['noise_level'] = n
    params['recordings']['seed'] = np.random.randint(1000)
    recgen = mr.gen_recordings(tempgen=tempgen, params=params)
    recordings_noise.append(recgen)

fig1 = plt.figure(figsize=(12, 14))
ax_noise = fig1.add_subplot(111)

for i, rec in enumerate(recordings_noise):
    ax_noise = mr.plot_recordings(rec, colors=colors[i], ax=ax_noise, lw=0.2, start_time=1, end_time=2)

legend_lines = [plt.Line2D([0], [0], color=colors[0], lw=2),
                plt.Line2D([0], [0], color=colors[1], lw=2),
                plt.Line2D([0], [0], color=colors[2], lw=2),
                plt.Line2D([0], [0], color=colors[3], lw=2)]
ax_noise.legend(handles=legend_lines, labels=['30 $\mu$V', '20 $\mu$V', '10 $\mu$V', '5  $\mu$V'],
                fontsize=15, loc='upper right')

y_lim = ax_noise.get_ylim()
x_lim = ax_noise.get_xlim()

ts_lim = (x_lim[1] - x_lim[0]) // 3
ax_noise.plot([x_lim[0], x_lim[0] + 0.2 * ts_lim],
              [np.min(y_lim) + 0.12 * np.abs(np.min(y_lim)),
               np.min(y_lim) + 0.12 * np.abs(np.min(y_lim))], 'k', lw=2)
ax_noise.text(x_lim[0] + 0.01 * ts_lim, np.min(y_lim) + 0.05 * np.abs(np.min(y_lim)), '200 ms')

# several drifting velocities
drift_velocities = [60, 30, 10]

template_drift_file = 'data/templates/templates_30_Neuronexus-32_drift.h5'
tempgen_drift = mr.load_templates(template_drift_file)

params['recordings']['noise_level'] = 10
params['spiketrains']['duration'] = 60
params['spiketrains']['seed'] = 1
params['templates']['seed'] = 2
params['recordings']['drifting'] = True

recordings_drift = []
for v in drift_velocities:
    print('drift velocity:', v)
    params['recordings']['drift_velocity'] = v
    params['recordings']['seed'] = np.random.randint(1000)
    recgen = mr.gen_recordings(tempgen=tempgen_drift, params=params)
    recordings_drift.append(recgen)

fig2 = plt.figure(figsize=(12, 14))
ax_drift = fig2.add_subplot(111)

for i, rec in enumerate(recordings_drift):
    mr.plot_recordings(rec, colors=colors[i], ax=ax_drift, lw=0.1)

legend_lines = [plt.Line2D([0], [0], color=colors[0], lw=2),
                plt.Line2D([0], [0], color=colors[1], lw=2),
                plt.Line2D([0], [0], color=colors[2], lw=2)]
ax_drift.legend(handles=legend_lines, labels=['60 $\mu$m/min', '30 $\mu$m/min', '10 $\mu$m/min'],
                fontsize=15, loc='upper right')

y_lim = ax_drift.get_ylim()
x_lim = ax_drift.get_xlim()

ts_lim = (x_lim[1] - x_lim[0]) // 3
ax_drift.plot([x_lim[0], x_lim[0] + 0.16666 * ts_lim],
              [np.min(y_lim) + 0.12 * np.abs(np.min(y_lim)),
               np.min(y_lim) + 0.12 * np.abs(np.min(y_lim))], 'k', lw=2)
ax_drift.text(x_lim[0] + 0.02 * ts_lim, np.min(y_lim) + 0.05 * np.abs(np.min(y_lim)), '10 s')

if save_fig:
    if not os.path.isdir('figure8'):
        os.mkdir('figure8')
    fig1.savefig('figure8/noise.png', dpi=600)
    fig2.savefig('figure8/drift.png', dpi=600)
