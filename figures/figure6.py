import MEArec as mr
import MEAutility as mu
import matplotlib.pylab as plt
from matplotlib import gridspec
from plotting_conventions import *
import numpy as np
import yaml
import os

save_fig = False

plt.ion()
plt.show()

template_file = 'data/templates/templates_30_Neuronexus-32_drift.h5'
tempgen = mr.load_templates(template_file, return_h5_objects=False)
max_channels_per_template = 1

with open('figure6_params.yaml') as f:
    params = yaml.load(f)

fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(12, 22)

ax_t = fig.add_subplot(gs[1:-1, :7])
ax_slow = fig.add_subplot(gs[:8, 9:15])
ax_fast = fig.add_subplot(gs[:8, 16:])
ax_amp_slow = fig.add_subplot(gs[8:, 9:15])
ax_amp_fast = fig.add_subplot(gs[8:, 16:])

# select one drifting template
template_id = 341
temp = tempgen.templates[template_id]
n_steps = temp.shape[0]
cm = 'coolwarm'

# choose colormap
cmap = plt.cm.get_cmap(cm)
colors = [cmap(i / n_steps) for i in range(n_steps)]

mea = mu.return_mea(info=tempgen.info['electrodes'])

# plot drifting template
mr.plot_templates(tempgen, template_ids=template_id, drifting=True, single_axes=True, cmap=cm, ax=ax_t)
for i, t in enumerate(tempgen.locations[template_id]):
    if i == 0 or i == len(tempgen.locations[template_id]) - 1:
        ax_t.plot(t[1], t[2], '*', color=colors[i], markersize=7)
    else:
        ax_t.plot(t[1], t[2], '*', color=colors[i], markersize=1)

y_lim = ax_t.get_ylim()
x_lim = ax_t.get_xlim()

ts_lim = (x_lim[1] - x_lim[0]) // 3
ax_t.plot([x_lim[0], x_lim[0] + 0.38 * ts_lim],
          [np.min(y_lim) + 0.12 * np.abs(np.min(y_lim)),
           np.min(y_lim) + 0.12 * np.abs(np.min(y_lim))], 'k', lw=2)
ax_t.text(x_lim[0] + 0.07 * ts_lim, np.min(y_lim) + 0.05 * np.abs(np.min(y_lim)), '5 ms')

ax_t.arrow(0, 0, 0, np.max(y_lim), alpha=0.5, head_width=0.5, head_length=4, fc='k', ec='k', length_includes_head=True,
           width=0.0001)
ax_t.arrow(0, 0, np.max(x_lim), 0, alpha=0.5, head_width=2, head_length=1, fc='k', ec='k', length_includes_head=True,
           width=0.0001)
ax_t.text(-1.5, np.max(y_lim), 'y', alpha=0.5)
ax_t.text(np.max(x_lim), -7, 'x', alpha=0.5)

for ch in mea.positions:
    ax_t.plot(ch[1], ch[2], 'o', color='k', markersize=2, alpha=0.5)

# slow drifts
params['recordings']['drifting'] = True
params['recordings']['drift_mode'] = 'slow'
params['recordings']['slow_drift_velocity'] = 20
params['recordings']['modulation'] = 'none'
recgen_slow = mr.gen_recordings(tempgen=tempgen, params=params)

ax_amp_slow = mr.plot_amplitudes(recgen_slow, cmap='rainbow', ax=ax_amp_slow)
simplify_axes([ax_amp_slow])
ax_amp_slow.set_xlabel('time (s)', fontsize=15)
ax_amp_slow.set_ylabel('voltage ($\mu$V)', fontsize=15)

mr.plot_recordings(recgen_slow, lw=0.1, ax=ax_slow, overlay_templates=True,
                   max_channels_per_template=max_channels_per_template, cmap='rainbow', templates_lw=0.3)
y_lim = ax_slow.get_ylim()
x_lim = ax_slow.get_xlim()
ts_lim = (x_lim[1] - x_lim[0]) // 3
ax_slow.plot([x_lim[0], x_lim[0] + 0.166 * ts_lim],
             [np.min(y_lim) + 0.12 * np.abs(np.min(y_lim)),
              np.min(y_lim) + 0.12 * np.abs(np.min(y_lim))], 'k', lw=2)
ax_slow.text(x_lim[0] + 0.02 * ts_lim, np.min(y_lim) + 0.05 * np.abs(np.min(y_lim)), '10 s')

# fast drifts
params['recordings']['drift_mode'] = 'fast'
params['recordings']['fast_drift_period'] = 15
recgen_fast = mr.gen_recordings(tempgen=tempgen, params=params, verbose=2)

ax_amp_fast = mr.plot_amplitudes(recgen_fast, cmap='rainbow', ax=ax_amp_fast)
simplify_axes([ax_amp_fast])
ax_amp_fast.set_xlabel('time (s)', fontsize=15)
ax_amp_fast.set_ylabel('voltage ($\mu$V)', fontsize=15)

mr.plot_recordings(recgen_fast, lw=0.1, ax=ax_fast, overlay_templates=True,
                   max_channels_per_template=max_channels_per_template, cmap='rainbow', templates_lw=0.3)
y_lim = ax_fast.get_ylim()
x_lim = ax_fast.get_xlim()
ts_lim = (x_lim[1] - x_lim[0]) // 3
ax_fast.plot([x_lim[0], x_lim[0] + 0.166 * ts_lim],
             [np.min(y_lim) + 0.12 * np.abs(np.min(y_lim)),
              np.min(y_lim) + 0.12 * np.abs(np.min(y_lim))], 'k', lw=2)
ax_fast.text(x_lim[0] + 0.02 * ts_lim, np.min(y_lim) + 0.05 * np.abs(np.min(y_lim)), '10 s')

fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.95)

if save_fig:
    if not os.path.isdir('figure6'):
        os.mkdir('figure6')

    fig.savefig('figure6/drifting.png', dpi=600)
