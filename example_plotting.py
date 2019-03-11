'''Examples on how to load and plot templates, spike trains, recordings'''

import matplotlib.pylab as plt
from MEArec import tools
import MEAutility as mu

template_file = None  # insert here the templates path
recording_file = None  # insert here the recordings path


if template_file is not None:
    tempgen = tools.load_templates(template_file)
    electrode_info = tempgen.info['electrodes']
    mea = mu.return_mea(info=electrode_info)

    mu.plot_mea_recording(tempgen.templates[0], mea)


if recording_file is not None:
    recgen = tools.load_templates(template_file)
    electrode_info = recgen.info['electrodes']
    mea = mu.return_mea(info=electrode_info)

    mu.plot_mea_recording(recgen.recordings, mea)
    tools.raster_plots(recgen.spiketrains)

plt.ion()
plt.show()
