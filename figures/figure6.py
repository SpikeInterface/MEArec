import MEArec as mr
import MEAutility as mu
import matplotlib.pylab as plt
from plotting_conventions import *
import numpy as np
import os

save_fig = False

plt.ion()
plt.show()

template_file = 'data/templates/templates_30_Neuronexus-32_drift.h5'
tempgen = mr.load_templates(template_file)

# select one drifting template
template_id = 341
temp = tempgen.templates[template_id]
n_steps = temp.shape[0]
cm = 'coolwarm'

# choose colormap
cmap = plt.cm.get_cmap(cm)
colors = [cmap(i / n_steps) for i in range(n_steps)]

mea = mu.return_mea(info=tempgen.info['electrodes'])

# plot recordings
fig1 = plt.figure(figsize=(7, 9))
ax_t = fig1.add_subplot(111)
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

recgen = mr.gen_recordings(tempgen=tempgen, params='figure6_params.yaml')

fig2 = plt.figure(figsize=(7, 4.5))
ax_wf = fig2.add_subplot(111)
mr.plot_waveforms(recgen, electrode='max', cmap='rainbow', ax=ax_wf)
ax_wf.set_ylabel('')
ax_wf.set_ylabel('voltage ($\mu$V)', fontsize=15)

# plot recordings
fig3 = plt.figure(figsize=(7, 9))
ax_r = fig3.add_subplot(111)
mr.plot_recordings(recgen, lw=0.1, ax=ax_r)
cm = plt.get_cmap('rainbow')
colors = [cm(i / len(recgen.spiketrains)) for i in np.arange(len(recgen.spiketrains))]
y_lim = ax_r.get_ylim()
x_lim = ax_r.get_xlim()
ts_lim = (x_lim[1] - x_lim[0]) // 3
ax_r.plot([x_lim[0], x_lim[0] + 0.166 * ts_lim],
          [np.min(y_lim) + 0.12 * np.abs(np.min(y_lim)),
           np.min(y_lim) + 0.12 * np.abs(np.min(y_lim))], 'k', lw=2)
ax_r.text(x_lim[0] + 0.02 * ts_lim, np.min(y_lim) + 0.05 * np.abs(np.min(y_lim)), '10 s')

amp = []
for i, st in enumerate(recgen.spiketrains):
    dir = st.annotations['final_soma_position'] - st.annotations['initial_soma_position']
    init = st.annotations['initial_soma_position']
    ax_r.quiver(init[1], init[2], dir[1], dir[2], width=0.005, color=colors[i])
    wf = st.waveforms
    mwf = np.mean(wf, axis=0)
    t = recgen.templates[i, 0, 0]
    max_elec = np.unravel_index(np.argmin(t), t.shape)[0]
    min_amp_time = np.unravel_index(np.argmin(mwf), mwf.shape)[1]
    amp.append(np.array([(w[max_elec, min_amp_time]) for w in wf]))

fig4 = plt.figure(figsize=(7, 4.5))
ax_amp = fig4.add_subplot(111)
for i, st in enumerate(recgen.spiketrains):
    max_amp = np.max(np.abs(amp[i]))
    ax_amp.plot(st, amp[i], '*', color=colors[i])

simplify_axes([ax_amp])
ax_amp.set_xlabel('time (s)', fontsize=15)
ax_amp.set_ylabel('voltage ($\mu$V)', fontsize=15)

if save_fig:
    if not os.path.isdir('figure6'):
        os.mkdir('figure6')

    fig1.savefig('figure6/templates.pdf')
    fig2.savefig('figure6/waveforms.png', dpi=600)
    fig3.savefig('figure6/rec.png', dpi=600)
    fig4.savefig('figure6/amps.pdf')
