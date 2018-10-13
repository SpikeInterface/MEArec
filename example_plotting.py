from __future__ import print_function

'''Examples on how to load and plot templates, spike trains, recordings'''
import matplotlib.pylab as plt
from MEArec import tools
import MEAutility as MEA

template_folder = '/home/alessio/Documents/Codes/MEArec/data/templates/physrot/templates_30_Neuronexus-32_13-10-2018' # insert here the template path
recording_folder = '/home/alessio/Documents/Codes/MEArec/data/recordings/recordings_20cells_Neuronexus-32_10.0_10.0uV_13-10-2018:10:55' # insert here the recording path

if template_folder is not None:
    temp_dict, temp_info = tools.load_templates(template_folder)
    electrode_name = temp_info['electrodes']['electrode_name']
    mea = MEA.return_mea(electrode_name)

    MEA.plot_mea_recording(temp_dict['templates'][0], mea)


if recording_folder is not None:
    rec_dict, rec_info = tools.load_recordings(recording_folder)
    electrode_name = rec_info['recordings']['electrode_name']
    mea = MEA.return_mea(electrode_name)
    
    MEA.plot_mea_recording(rec_dict['recordings'], mea)
    tools.raster_plots(rec_dict['spiketrains'])

plt.ion()
plt.show()
