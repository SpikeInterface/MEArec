import MEArec as mr
import MEAutility as mu
import matplotlib.pylab as plt
import numpy as np
import os

save_fig = False

plt.ion()
plt.show()

plot_probes = True
plot_templates = True
plot_recs = True

if plot_probes:
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    nexus = mu.return_mea('Neuronexus-32')
    npix = mu.return_mea('Neuropixels-128')
    sqmea = mu.return_mea('SqMEA-10-15')

    nexus.move([0, -200, 0])
    sqmea.move([0, 200, 0])

    miny_sqmea = np.min(sqmea.positions[:, 2])
    miny_nexus = np.min(nexus.positions[:, 2])
    miny_npix = np.min(npix.positions[:, 2])

    nexus.move([0, 0, miny_npix - miny_nexus])
    sqmea.move([0, 0, miny_npix - miny_sqmea])

    mu.plot_probe(nexus, ax=ax1, top=500, bottom=100)
    mu.plot_probe(npix, ax=ax1, bottom=150)
    mu.plot_probe(sqmea, ax=ax1, top=500, bottom=150)

    ax1.axis('off')

    ax1.set_xlim([-250, 300])
    ax1.set_ylim([-830, -300])
    fig1.tight_layout()

if plot_templates:
    fig21 = plt.figure(figsize=(7, 11))
    ax21 = fig21.add_subplot(111)
    fig22 = plt.figure(figsize=(7, 16))
    ax22 = fig22.add_subplot(111)
    fig23 = plt.figure(figsize=(8, 7))
    ax23 = fig23.add_subplot(111)
    n_t = 1

    nexus = mu.return_mea('Neuronexus-32')
    npix = mu.return_mea('Neuropixels-128')
    sqmea = mu.return_mea('SqMEA-10-15')

    nexus_templates = 'data/templates/templates_100_Neuronexus-32.h5'
    npix_templates = 'data/templates/templates_100_Neuropixels-128.h5'
    sqmea_templates = 'data/templates/templates_100_SqMEA-10-15.h5'

    tempgen_t = mr.load_templates(nexus_templates)
    tempgen_np = mr.load_templates(npix_templates)
    tempgen_sq = mr.load_templates(sqmea_templates)

    templates_t = tempgen_t.templates[1000]
    templates_np = tempgen_np.templates[1000]
    templates_sq = tempgen_sq.templates[1000]

    mu.plot_mea_recording(templates_t, nexus, colors='k', lw=2, ax=ax21)
    mu.plot_mea_recording(templates_np, npix, colors='k', lw=2, ax=ax22)
    mu.plot_mea_recording(templates_sq, sqmea, colors='k', lw=2, ax=ax23)

    fig21.tight_layout()
    fig22.tight_layout()
    fig23.tight_layout()

    if save_fig:
        if not os.path.isdir('figure3'):
            os.mkdir('figure3')

        fig21.savefig('figure3/template_nn.pdf')
        fig22.savefig('figure3/template_np.pdf')
        fig23.savefig('figure3/template_sq.pdf')

if plot_recs:
    fig31 = plt.figure(figsize=(7, 11))
    ax31 = fig31.add_subplot(111)
    fig32 = plt.figure(figsize=(7, 16))
    ax32 = fig32.add_subplot(111)
    fig33 = plt.figure(figsize=(8, 7))
    ax33 = fig33.add_subplot(111)
    seconds = 1.5
    fs = 32000

    nexus = mu.return_mea('Neuronexus-32')
    npix = mu.return_mea('Neuropixels-128')
    sqmea = mu.return_mea('SqMEA-10-15')

    nexus_rec = 'data/recordings/recordings_20cells_Neuronexus-32_30.0_10.0uV.h5'
    npix_rec = 'data/recordings/recordings_60cells_Neuropixels-128_30.0_10.0uV.h5'
    sqmea_rec = 'data/recordings/recordings_50cells_SqMEA-10-15_30.0_10.0uV.h5'

    recgen_t = mr.load_recordings(nexus_rec)
    recgen_np = mr.load_recordings(npix_rec)
    recgen_sq = mr.load_recordings(sqmea_rec)

    mr.plot_recordings(recgen_t, ax=ax31, start_time=0, end_time=seconds, lw=0.2)
    mr.plot_recordings(recgen_np, ax=ax32, start_time=0, end_time=seconds, lw=0.2, vscale=150)
    mr.plot_recordings(recgen_sq, ax=ax33, start_time=0, end_time=seconds, lw=0.2)

    fig31.tight_layout()
    fig32.tight_layout()
    fig33.tight_layout()

    if save_fig:
        if not os.path.isdir('figure3'):
            os.mkdir('figure3')

        fig21.savefig('figure3/template_nn.pdf')
        fig22.savefig('figure3/template_np.pdf')
        fig23.savefig('figure3/template_sq.pdf')
