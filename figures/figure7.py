import MEArec as mr
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from copy import deepcopy
import scipy.signal as ss
import scipy.stats as stat
from plotting_conventions import *
import numpy as np
import os

save_fig = False

plt.ion()
plt.show()

template_file = 'data/templates/templates_300_tetrode_minamp0.h5'
tempgen = mr.load_templates(template_file)

recgen_u = mr.gen_recordings(tempgen=tempgen, params='figure7_params.yaml')

recgen_dc = deepcopy(recgen_u)
recgen_dc.params['recordings']['noise_mode'] = 'distance-correlated'
recgen_dc.params['recordings']['seed'] = 0
recgen_dc.generate_recordings()

fs = recgen_dc.info['recordings']['fs']

recgen_u_col = deepcopy(recgen_u)
recgen_u_col.params['recordings']['noise_mode'] = 'uncorrelated'
recgen_u_col.params['recordings']['noise_color'] = True
recgen_u_col.params['recordings']['seed'] = 1
recgen_u_col.generate_recordings()

recgen_dc_col = deepcopy(recgen_u)
recgen_dc_col.params['recordings']['noise_mode'] = 'distance-correlated'
recgen_dc_col.params['recordings']['noise_color'] = True
recgen_dc_col.params['recordings']['seed'] = 2
recgen_dc_col.generate_recordings()

recgen_far = deepcopy(recgen_u)
recgen_far.params['recordings']['noise_mode'] = 'far-neurons'
recgen_far.params['recordings']['seed'] = 3
recgen_far.generate_recordings()

fig1 = plt.figure(figsize=(9, 9))

gs = gridspec.GridSpec(16, 19)

ax_u = fig1.add_subplot(gs[:4, :3])
ax_dc = fig1.add_subplot(gs[:4, 4:7])
ax_ucol = fig1.add_subplot(gs[:4, 8:11])
ax_dccol = fig1.add_subplot(gs[:4, 12:15])
ax_far = fig1.add_subplot(gs[:4, 16:19])

max_rec = 0
for rec in [recgen_u, recgen_dc, recgen_dc_col, recgen_u_col, recgen_far]:
    max_v = np.max(rec.recordings)
    if max_v > max_rec:
        max_rec = max_v

vscale = 1.1 * max_rec
start_time = 1
end_time = 1.2
mr.plot_recordings(recgen_u, ax=ax_u, lw=0.1, vscale=vscale, start_time=start_time, end_time=end_time)
mr.plot_recordings(recgen_dc, ax=ax_dc, lw=0.1, vscale=vscale, start_time=start_time, end_time=end_time)
mr.plot_recordings(recgen_u_col, ax=ax_ucol, lw=0.1, vscale=vscale, start_time=start_time, end_time=end_time)
mr.plot_recordings(recgen_dc_col, ax=ax_dccol, lw=0.1, vscale=vscale, start_time=start_time, end_time=end_time)
mr.plot_recordings(recgen_far, ax=ax_far, lw=0.1, vscale=vscale, start_time=start_time, end_time=end_time)


ax_u_psd = fig1.add_subplot(gs[4:8, :3])
ax_dc_psd = fig1.add_subplot(gs[4:8, 4:7])
ax_ucol_psd = fig1.add_subplot(gs[4:8, 8:11])
ax_dccol_psd = fig1.add_subplot(gs[4:8, 12:15])
ax_far_psd = fig1.add_subplot(gs[4:8, 16:19])

ax_u_cov = fig1.add_subplot(gs[8:12, :3])
ax_dc_cov = fig1.add_subplot(gs[8:12, 4:7])
ax_ucol_cov = fig1.add_subplot(gs[8:12, 8:11])
ax_dccol_cov = fig1.add_subplot(gs[8:12, 12:15])
ax_far_cov = fig1.add_subplot(gs[8:12, 16:19])

ax_u_hist = fig1.add_subplot(gs[12:, :3])
ax_dc_hist = fig1.add_subplot(gs[12:, 4:7])
ax_ucol_hist = fig1.add_subplot(gs[12:, 8:11])
ax_dccol_hist = fig1.add_subplot(gs[12:, 12:15])
ax_far_hist = fig1.add_subplot(gs[12:, 16:19])

for (rec, ax_psd, ax_cov, ax_hist) in zip([recgen_u, recgen_dc, recgen_u_col, recgen_dc_col, recgen_far],
                                          [ax_u_psd, ax_dc_psd, ax_ucol_psd, ax_dccol_psd, ax_far_psd],
                                          [ax_u_cov, ax_dc_cov, ax_ucol_cov, ax_dccol_cov, ax_far_cov],
                                          [ax_u_hist, ax_dc_hist, ax_ucol_hist, ax_dccol_hist, ax_far_hist]):
    r = rec.recordings
    f, p = ss.welch(r[0], fs=fs, scaling='spectrum', nperseg=2048)
    cov = np.cov(r)
    fidx_1000 = np.where(f > 5000)[0][0]
    fidx_100 = np.where(f < 50)[0][-1]
    ax_psd.semilogy(f[fidx_100:fidx_1000], p[fidx_100:fidx_1000])
    ax_psd.set_ylim([0.001, 6])
    ax_psd.set_yticks([])
    ax_psd.set_xlabel('Hz', fontsize=10)
    ax_cov.imshow(cov)
    ax_cov.axis('off')
    ax_hist.hist(r[0], alpha=0.5, bins=100, density=True)
    ax_hist.set_xlim(-50, 50)
    ax_hist.set_ylim([0, 0.05])
    ax_hist.text(-40, 0.045, 'Skew: ' + str(np.round(stat.skew(r[0]), 2)), fontsize=10)
    ax_hist.set_yticks([])
    ax_hist.set_xlabel('$\mu V$', fontsize=10)
    simplify_axes([ax_psd, ax_hist])

ax_u_psd.set_yticks([0.01, 0.1, 1])
ax_u_psd.set_ylabel('spectrum', fontsize=12)
ax_u_hist.set_yticks([0, 0.025, 0.05])
ax_u_hist.set_ylabel('density', fontsize=12)

ax_u.set_title('Uncorrelated\n', fontsize=15)
ax_dc.set_title('Correlated\n', fontsize=15)
ax_ucol.set_title('Colored\nuncorrelated', fontsize=15)
ax_dccol.set_title('Colored\ncorrelated', fontsize=15)
ax_far.set_title('Far neurons\n', fontsize=15)

fig1.tight_layout()

if save_fig:
    if not os.path.isdir('figure7'):
        os.mkdir('figure7')

    fig1.savefig('figure7/noise.png', dpi=600)

