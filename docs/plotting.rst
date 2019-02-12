Loading and plotting
====================

Recordings
----------

Loading and plotting the templates and recordings can be done easily in Python using MEArec and MEAutility:

.. code-block:: python

    import MEArec as mr
    import MEAutility as mu
    import matplotlib.pylab as plt

    # load recordings
    recgen = mr.load_recordings('path-to-recording.h5')

    # create mea object
    mea = mu.return_mea(info=recgen.info['electrodes'])

    # plot recordings
    mu.plot_mea_recording(recgen.recordings, mea)
    plt.show()

.. image:: images/recordings_static.png

Drifting templates
------------------

This is an example on how to plot a drifting template.

.. code-block:: python

    import MEArec as mr
    import MEAutility as mu
    import matplotlib.pylab as plt

    # load recordings
    tempgen = mr.load_recordings('path-to-drifting-template.h5')

    # create mea object
    mea = mu.return_mea(info=tempgen.info['electrodes'])

    # select one drifting template
    temp = tempgen[100]
    n_steps = temp.shape[0]

    # choose colormap
    cmap = plt.cm.get_cmap('Reds')
    colors = [cmap(i/n_steps) for i in range(n_steps)]

    # plot recordings
    mu.plot_mea_recording(temp, mea, colors=colors)
    plt.show()

.. image:: images/templates_drift.png

Intergation with SpikeInterface
-------------------------------

MEArec is designed to help validating spike sorting algorithms. Hence, its integration with `SpikeInterface <https://github.com/SpikeInterface>`_,
a Python framework for spike sorting analysis and validation, is extremely straightforward.

Having the `spikeextractors <https://github.com/SpikeInterface/spikeextractors>`_ and
`spiketoolkit <https://github.com/SpikeInterface/spiketoolkit>`_ packages installed, one can easily load a MEArec
generated recording, run several spike sorting algorithms, and compare/validate their output:

.. code-block:: python

    import spikeextractors as se
    import spiketoolkit as st

    # load recordings and spiketrains with MEArecExtractors
    recording = se.MEArecRecordingExtractor('path-to-recording.h5')
    sorting = se.MEArecSortingExtractor('path-to-recording.h5')

    # run spike sorting
    sorting_MS4 = st.sorters.mountainsort4(recording)
    sorting_KS = st.sorters.kilosort(recording)

    # compare results
    comp_MS = st.comparison.SortingComparison(sorting, sorting_MS4)
    comp_KS = st.comparison.SortingComparison(sorting, sorting_KS)
    comp_MS_KS = st.comparison.SortingComparison(sorting_MS4, sorting_KS)

Check out more about the SpikeInterface framework with these `tutorials <https://github.com/SpikeInterface/spiketutorials>`_.

