from __future__ import print_function

'''Examples on how to load and plot templates, spike trains, recordings'''
import numpy as np
import matplotlib.pylab as plt
import tools
import MEAutility as MEA

template_folder = None # insert here the template path
spiketrain_folder = None # insert here the spiketrains path
recording_folder = None # insert here the recording path

if template_folder is not None:
    templates, locs, rots, celltypes, temp_info = tools.load_templates(template_folder)
    electrode_name = temp_info['Electrodes']['electrode_name']
    mea_pos, mea_dim, mea_pitch = MEA.return_mea(electrode_name)

    tools.plot_mea_recording(templates, mea_pos, mea_pitch)

if spiketrain_folder is not None:
    spiketrains, st_info = tools.load_spiketrains(spiketrain_folder)

    tools.raster_plots(spiketrains)

if recording_folder is not None:
    recordings, times, templates, spiketrains, sources, peaks, rec_info = tools.load_recordings(recording_folder)
    electrode_name = rec_info['General']['electrode_name']
    mea_pos, mea_dim, mea_pitch = MEA.return_mea(electrode_name)

    tools.plot_mea_recording(recordings, mea_pos, mea_pitch)
