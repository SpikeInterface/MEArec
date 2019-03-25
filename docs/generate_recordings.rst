.. _gen-recordings:

Generating Recordings
=====================

Recordings are generated combining templates and spike trains. The recordings parameters are divided in different
sections:

* :code:`spiketrains`
* :code:`templates`
* :code:`cell-types`
* :code:`recordings`

The :code:`spiketrains` part deals with the generation of spike trains, while the other 3 sections specify parameters to
assemble spike trains and templates and build the extracellular recordings.


Spike trains generation
-----------------------

The first step is the spike train generation. The user can specify the number and type of cells in 2 ways:

1. providing a list of :code:`rates` and corresponding :code:`types`: e.g. rates = [3, 3, 5], types = ['e', 'e', 'i'] will generate 3 spike trains with average firing rates 3, 3, and 5 Hz and respectively excitatory, excitatory , and inhibitory type.
2. providing :code:`n_exc`, :code:`n_inh`, :code:`f_exc`, :code:`f_inh`, :code:`st_exc`, :code:`st_min`: in this case there will be generated :code:`n_exc` excitatory spike trains with average firing rate of :code:`f_exc` and firing rate standard deviation of :code:`st_exc` (same for inhibitory spike trains)

The firinga rates generated with the second option have a minimum firing rate of :code:`min_rate` (default 0.5 Hz).

Spike trains are simulated as Poisson or Gamma processes (chosen with the parameter :code:`process`) and in the latter
case the :code:`gamma` parameter controls the curve shape.

Spikes violating the refreactory period :code:`ref_per` (default is 2 ms) are removed.

:code:`t_start` (0 s by default) is the start timestamp of the recordings in second and :code:`duration` will correspond
to the duration of the recordings.

The :code:`seed` parameter can be set to ensure reproducibility, and if :code:`null` a random seed is used.


Spike trains parameters section summary
"""""""""""""""""""""""""""""""""""""""

.. code-block:: bash

    spiketrains:
      # Default parameters for spike train generation (spiketrain_gen.py)

      # spike train generation parameters

      # rates: [3,3,5] # individual spike trains rates
      # types: ['e', 'e', 'i'] # individual spike trains class (exc-inh)
      # alternative to rates - excitatory and inhibitory settings
      n_exc: 2 # number of excitatory cells
      n_inh: 1 # number of inhibitory cells
      f_exc: 5 # average firing rate of excitatory cells in Hz
      f_inh: 15 # average firing rate of inhibitory cells in Hz
      st_exc: 1 # firing rate standard deviation of excitatory cells in Hz
      st_inh: 3 # firing rate standard deviation of inhibitory cells in Hz
      min_rate: 0.5 # minimum firing rate in Hz
      ref_per: 2 # refractory period in ms
      process: poisson # process for spike train simulation (poisson-gamma)
      gamma_shape: 2 # gamma shape (for gamma process)
      t_start: 0 # start time in s
      duration: 10 # duration in s
      seed: null # random seed for spiketrain generation


Recordings Generation
---------------------


Specyfying excitatory and inhibitory cell-types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to select the proper cell type (excitatory - inhibitory) the :code:`cell-types` section of the parameters
allows the user to specify which  strings to look for in the cell model name (from the NMC database) to assign it to
the excitatory or inhibitory set. In this example from L5 cells, all cells contining LBC (Large Basket Cells) will be
marked as inhibitory, and so on.

Cell-types parameters section summary
"""""""""""""""""""""""""""""""""""""

.. code-block:: bash

    cell_types:
      # excitatory and inhibitory cell names
      excitatory: ['STPC', 'TTPC1', 'TTPC2', 'UTPC']
      inhibitory: ['BP', 'BTC', 'ChC', 'DBC', 'LBC', 'MC', 'NBC', 'NGC', 'SBC']

Template selection and parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Templates are selected so that they match the excitatory-inhibitory spike trains (if the :code:`cell-types` section is
provided) and they follow the following rules:

* neuron locations cannot be closer than the :code:`min_dist` parameter (default 25 :math:`\mu m`)
* templates must have an amplitude of at least :code:`min_amp` (default 50 :math:`\mu V`) and at most :code:`max_amp`
(default 500 :math:`\mu V`)
* if specified, neuron locations are selected within the :code:`xlim`, :code:`ylim`, and :code:`zlim` limits

Once the templates are selected and matched to the corresponding spike train, temporal jitter is added to them to
simulate the uncertainty of the spike event within the sampling period. :code:`n_jitters` (default is 10) templates are
created by upsampling the original templates by :code:`upsample` times (default is 8) and shifting them within a
sampling period. During convolution, randomly a jittered version of the spike is selected.
Finally, the templates are linearly padded on both sides (:code:`pad_len` by default pads 3 ms before and 3 after the
duration of the template) to ensure a smooth convolution.

The :code:`overlap_threshold` allows to define spatially overlapping templates. For example, if set to 0.9 (by default)
template A and template B are marked as overlapping if on the electrode with the largest peak for template A, template
B's amplitude is greater or equal than the 90% of its peak amplitude.

The :code:`seed` parameter, randomly set if :code:`null`, ensures reproducibility.

Templates parameters section summary
"""""""""""""""""""""""""""""""""""""

.. code-block:: bash

    templates:
      # recording generation parameters
      min_dist: 25 # minimum distance between neurons
      min_amp: 50 # minimum spike amplitude in uV
      max_amp: 500 # minimum spike amplitude in uV
      xlim: null # limits for neuron depths (x-coord) in um [min, max]
      ylim: null # limits for neuron depths (y-coord) in um [min, max]
      zlim: null # limits for neuron depths (z-coord) in um [min, max]
      # (e.g 0.8 -> 80% of template B on largest electrode of template A)
      n_jitters: 10 # number of temporal jittered copies for each eap
      upsample: 8 # upsampling factor to extract jittered copies
      pad_len: [3, 3] # padding of templates in ms
      overlap_threshold: 0.8 # threshold to consider two templates spatially overlapping
      seed: null # random seed to draw eap templates


Other recordings settings
^^^^^^^^^^^^^^^^^^^^^^^^^

After the templates are selected, jittered, and padded, clean recordings are generated by convolving each template with
its corresponding spike train.
The :code:`fs` parameters permits to resample the recordings and if it is not provided recordings are created with the
same sampling frequency as the templates.

If :code:`sync_rate` is greater than 0 (and <= 1, default is 0), synchrony is added to spatially overlapping templates.
For example, if :code:`sync_rate` is 0.2, 1 out of 5 spikes on spike trains with overlapping templates will be temporally
coincident. :code:`sync_jitt` (default 1 ms) controls the jittering in time for added spikes.

The :code:`modulation` parameter is extremely important, as it controls the variablility of the amplitude modulation:
* if :code:`modulation` id :code:`none`, spikes are not modulated and each instance will have the same aplitude
* if :code:`modulation` id :code:`template`, each spike event is modulated with the same amplitude for all electrodes
* if :code:`modulation` id :code:`electrode`, each spike event is modulated with different amplitude for each electrode
* if :code:`modulation` id :code:`template-isi`, each spike event is modulated based on the inter-spike-interval, with the same amplitude for all electrodes
* if :code:`modulation` id :code:`electrode-isi`, each spike event is modulated based on the inter-spike-interval, with different amplitude for each electrode
For the :code:`template` and :code:`electrode` modulations, the amplitude is modulated as a Normal distribution with
amplitude 1 and standard deviation of :code:`sdrand` (default is 0.05).
For the :code:`template-isi` and :code:`electrode-isi` modulations, on top of the gaussian modulation the amplitude is
modulated by the previous inter-spike-intervals, to simulate the amplitude decay due to bursting. In this case, the
:code:`max_burst_duration` and :code:`n_burst_spikes` parameters control the maximum length and maximum number of spikes of a bursting event.
During a bursting event, the amplitude modulation, previous to the gaussian one, is computed as:

.. math:: mod = (\frac{avg_{ISI} / n_{consecutive}}{mem_{ISI}})^{exp}

where :math:`mod` is the resulting amplitude modulation, :math:`avg_{ISI}` is the average ISI so far during the
bursting event, :math:`n_{consecutive}` is the number of spikes occurred in the bursting period (maximum is
:code:`n_burst_spikes`) and :code:`exp` is the exponent of the decay (0.2 by default).

While isi modulation only modulates in amplitude, bursting can also modulate the spike shape. In order to model this, if
:code:`shape_mod` is True, then the templates are low-pass filtered depending on the :math:`mod` value. The :code:`bursting_fc`
parameter ([500, 12000] Hz by default) indicates how the :math:`mod` values will be mapped to the filter: For the minimum
:math:`mod` value the low pass filter will have a cutoff frequency of 500 Hz, and for the highest :math:`mod` of 12000 Hz.
The templates are filtered with the same value on all electrodes, and then, in case of an :code:`electrode`-type modulation,
the eap on each electrode to match the specific :math:`mod` for the electrode. Also for an :code:`template`-type modulation,
the eap is rescaled at the template level.

Next, noise is added to to the clean recordings. Two different noise modes can be used (using the :code:`noise_mode`
parameter):
1. :code:`uncorrelated`: additive gaussian noise (default) with a standard deviation of :code:`noise_level` (10 :math:`\mu V` by default)
2. :code:`distance-correlated`: noise is generated as a multivariate normal with covariance matrix decaying with distance between electrodes. The :code:`noise_half_distance` parameter is the distance for which correlation is 0.5.
Noise can be added in chunks (:code:`chunk_noise_duration`) as for long recordings the user can run into :code:`MemoryError`.

In order to simulate noise that resembles experimental noise, one can use the :code:`noise_color` option (default is False),
so that the noise spectrum is similar to biological noise.
If :code:`noise_color` is True, the gaussian noise is filtered with an IIR resonant filter with a peak at :code:`color_peak`
(default 500) and quality factor :code:`color_q` (default 1). Moreover, a gaussian noise floor is added to the noise.
The amplitude of the gaussian added noise is controlled by :code:`random_noise_floor` (default 1), which is the percent
of gaussian noise over the colored noise (when :code:`random_noise_floor=1` 50% of the noise is additive gaussian. The final
noise level is adjusted so that the overall standard deviation is equal to :code:`noise_level`.

Finally, and optionally, the recordings can be filtered (if :code:`filter` is :code:`True`) with a high-pass or band-pass
filter with :code:`filter_cutoff` frequency(ies) ([300, 6000] by default). If :code:`filter_cutoff` is a scalar, the signal is high-pass
filtered. The order of the Butterworth filter can be adjusted with the :code:`filter_order` frequency(ies) param.
Also filtering can be applied in chunks (:code:`chunk_filter_duration`).

For further analysis, spike events can be annotated as "O" (temporal overlapping) or "SO" (spatio-temporal overlapping)
when :code:`overlap` is set to :code:`True`. The waveforms can also be extracted and loaded to the
`Neo.Spiketrain <https://neo.readthedocs.io/en/0.4.0/core.html#example-spiketrain>`_
object if the :code:`extract_waveforms` is :code:`True`. Note that this might take some time for long recordings.

Recordings parameters section summary
"""""""""""""""""""""""""""""""""""""
.. code-block:: bash

    recordings:
        fs: null # sampling frequency in kHz (corresponds to dt=0.03125 ms)

        sync_rate: 0.1 # added synchrony rate for spatially overlapping templates

        modulation: electrode # type of spike modulation [none (no modulation) |
            # template (each spike instance is modulated with the same value on each electrode) |
            # electrode (each electrode is modulated separately) |
            # template-isi (spike amplitude is modulated depending on isi interval with the same value on each electrode)
            # electrode-isi (spike amplitude is modulated depending on isi interval and each electrode is modulated separately)]
        mrand: 1 # mean of gaussian modulation (should be 1)
        sdrand:  0.05 # standard deviation of gaussian modulation
        exp_decay: 0.2 # with isi modulation experimental decay in aplitude between consecutive spikes
        n_burst_spikes: 10 # max number of 'bursting' consecutive spikes
        max_burst_duration: 100 # duration in ms of maximum burst modulation
        shape_mod: False # if True waveforms are modulated in shape with a low pass filter depending on the isi
        bursting_fc: [1000., 12000.]  # min and max frequencies to be mapped to modulation value

        noise_level: 20 # noise standard deviation in uV
        noise_mode: uncorrelated # [uncorrelated | distance-correlated]
        noise_color: True # if True noise is colored resembling experimental noise
        noise_half_distance: 30 # (distance-correlated noise) distance between electrodes in um for which correlation is 0.5
        color_peak: 500 # (color) peak / curoff frequency of resonating filter
        color_q: 1 # (color) quality factor of resonating filter
        random_noise_floor: 1 # (color) additional noise floor
        chunk_noise_duration: 0 # chunk duration for noise addition
        seed: null # random seed for noise generation

        filter: False # if True it filters the recordings
        filter_cutoff: [300, 6000] # filter cutoff frequencies in Hz
        filter_order: 3 # filter order
        chunk_filter_duration: 0 # chunk duration for filtering

        overlap: False # if True, temporal and spatial overlap are computed for each spike (it may be time consuming)
        extract_waveforms: False # if True, waveforms are extracted from recordings


Drifting recordings
^^^^^^^^^^^^^^^^^^^

When drifting templates are generated (:ref:`drift-templates`), drifting recordings can be simulated when
:code:`drifting` is set to :code:`True`. The :code:`preferred_dir` parameter indicates the 3D vector with the
preferred direction of drift ([0,0,1], default, is upwards in the z-direction) and the :code:`angle_tol` (default is 15
degrees) corresponds to the tolerance in this direction.
The :code:`drift_velocity` controls how fast templates are 'replayed' along their trajectory (default is 5
:math:`\mu m`/min). Finally, :code:`t_start_drift` (default is 0) is the starting time from which cells start drifting.

.. code-block:: bash

        drifting: False # if True templates are drifted
        preferred_dir: [0, 0, 1]  # preferred drifting direction ([0,0,1] is positive z, direction)
        angle_tol: 15  # tolerance for direction in degrees
        drift_velocity: 30  # drift velocity in um/min
        t_start_drift: 0  # tim in s from which drifting starts

Running recording generation using CLI
--------------------------------------

Recordings can be generated using the CLI with the command: :code:`mearec gen-recordings`.
Run :code:`mearec gen-recordings --help` to display the list of available arguments, that can be used to overwrite the
default parameters or to point to another parameter .yaml file. In order to run a recording simulation, the
:code:`--templates` or :code:`-t` must be given to point to the templates to be used.

The output recordings are saved in .h5 format to the default recordings output folder.

Running recording generation using Python
-----------------------------------------

Recordings can also be generated using a Python script, or a jupyter notebook.

.. code-block:: python

    import MEArec as mr
    recgen = mr.gen_recordings(params=None, templates=None, tempgen=None)

The :code:`params` argument can be the path to a .yaml file or a dictionary containing the parameters
(if None default parameters are used). On of the :code:`templates` or :code:`tempgen` parameters must be indicated, the
former pointing to a generated templates file, the latter instead is a :code:`TemplateGenerator` object.

The :code:`gen_recordings()` function returns a gen_templates :code:`RecordingGenerator` object (:code:`recgen`).


The RecordingGenerator object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :code:`RecordingGenerator` class contains several fields:

* recordings: (n_electrodes, n_samples) recordings
* spiketrains: list of (n_spiketrains) :code:`neo.Spiketrain` objects
* templates: (n_spiketrains, n_electrodes, n_templates samples) templates
* channel_positions: (n_electrodes, 3) electrodes 3D positions
* times: (n_samples) timestamps in seconds (quantities)
* voltage_peaks: (n_spiketrains, n_electrodes) average voltage peaks on the electrodes
* spike_traces: (n_spiketrains, n_samples) clean spike trace for each spike train
* info: dictionary with parameters used


:code:`RecordingGenerator` can be saved to .h5 files as follows:

.. code-block:: python

    import MEArec as mr
    mr.save_recording_generator(recgen, filename=None)

where :code:`recgen` is a :code:`RecordingGenerator` object and :code:`filename` is the output file name.