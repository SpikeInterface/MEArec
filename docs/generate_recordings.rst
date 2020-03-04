.. _gen-recordings:

Generating Recordings
=====================

Recordings are generated combining templates and spike trains. The recordings parameters are divided in different
sections:

* :code:`spiketrains`
* :code:`templates`
* :code:`cell-types`
* :code:`recordings`
* :code:`seeds`

The :code:`spiketrains` part deals with the generation of spike trains, while the :code:`templates`, 
:code:`cell-types`, and :code:`recordings` sections specify parameters to assemble spike trains and templates and build 
the extracellular recordings. The :code:`seeds` contains all the random seeds involved in the simulations, to ensure
reproducibility.


Spike trains generation
-----------------------

The first step is the spike train generation. The user can specify the number and type of cells in 2 ways:

1. providing a list of :code:`rates` and corresponding :code:`types`: e.g. rates = [3, 3, 5], types = ['E', 'E', 'E'] 
will generate 3 spike trains with average firing rates 3, 3, and 5 Hz and respectively excitatory, excitatory , and inhibitory type.
2. providing :code:`n_exc`, :code:`n_inh`, :code:`f_exc`, :code:`f_inh`, :code:`st_exc`, :code:`st_min`: in this case 
there will be generated :code:`n_exc` excitatory spike trains with average firing rate of :code:`f_exc` and firing rate standard deviation of :code:`st_exc` (same for inhibitory spike trains)

The firinga rates generated with the second option have a minimum firing rate of :code:`min_rate` (default 0.5 Hz).

Spike trains are simulated as Poisson or Gamma processes (chosen with the parameter :code:`process`) and in the latter
case the :code:`gamma` parameter controls the curve shape.

Spikes violating the refreactory period :code:`ref_per` (default is 2 ms) are removed.

:code:`t_start` (0 s by default) is the start timestamp of the recordings in second and :code:`duration` will correspond
to the duration of the recordings.


Spike trains parameters section summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    spiketrains:
      # Default parameters for spike train generation (spiketrain_gen.py)

      # spike train generation parameters

      # rates: [3,3,5] # individual spike trains rates
      # types: ['E', 'E', 'I'] # individual spike trains class (exc-inh)
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


Recordings Generation
---------------------

Specifying excitatory and inhibitory cell-types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to select the proper cell type (excitatory - inhibitory) the :code:`cell-types` section of the parameters
allows the user to specify which  strings to look for in the cell model name (from the NMC database) to assign it to
the excitatory or inhibitory set. In this example from L5 cells, all cells contining LBC (Large Basket Cells) will be
marked as inhibitory, and so on. 
If you use custom cell models, you should overwrite this section as shown in 
`this notebook <https://github.com/alejoe91/MEArec/blob/master/notebooks/generate_recordings_with_allen_models.ipynb>`_  
using cell models from `Allen database <https://celltypes.brain-map.org/>`_.

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
Recordings can be split in times chunks using the :code:`chunk_duration` (20 s by default) parameter.
Chunks can be processed in parallel.

If :code:`sync_rate` is greater than 0 (and <= 1, default is 0), synchrony is added to spatially overlapping templates.
For example, if :code:`sync_rate` is 0.2, 1 out of 5 spikes on spike trains with overlapping templates will be temporally
coincident. :code:`sync_jitt` (default 1 ms) controls the jittering in time for added spikes.

The :code:`modulation` parameter is extremely important, as it controls the variablility of the amplitude modulation:
* if :code:`modulation` id :code:`none`, spikes are not modulated and each instance will have the same aplitude
* if :code:`modulation` id :code:`template`, each spike event is modulated with the same amplitude for all electrodes
* if :code:`modulation` id :code:`electrode`, each spike event is modulated with different amplitude for each electrode

For the :code:`template` and :code:`electrode` modulations, the amplitude is modulated as a Normal distribution with
amplitude 1 and standard deviation of :code:`sdrand` (default is 0.05).

Bursting behavior can be selected by setting :code:`bursting` to True. The number of bursting units can be selected using the
:code:`n_bursting` parameter. By default, if bursting is used, all units are bursty.
When bursting is selected, on top of the gaussian modulation the amplitude is
modulated by the previous inter-spike-intervals, to simulate the amplitude decay due to bursting. In this case, the
:code:`max_burst_duration` and :code:`n_burst_spikes` parameters control the maximum length and maximum number of spikes of a bursting event.
During a bursting event, the amplitude modulation, previous to the gaussian one, is computed as:

.. math:: mod = (\frac{avg_{ISI} / n_{consecutive}}{mem_{ISI}})^{exp}

where :math:`mod` is the resulting amplitude modulation, :math:`avg_{ISI}` is the average ISI so far during the
bursting event, :math:`n_{consecutive}` is the number of spikes occurred in the bursting period (maximum is
:code:`n_burst_spikes`) and :code:`exp` is the exponent of the decay (0.1 by default).

In addition to amplitude modulation, bursting can also modulate the spike shape. In order to model this, if
:code:`shape_mod` is True, then the templates are *stretched*  depending on the :math:`mod` value.
The stretching is obtained by projecting the template on a sigmoid-transformed scale, which effectively stretches the waveform.
The :code:`shape_stretch` parameter controls the amount of stretching (default 30). Larger :code:`shape_stretch` will result
in more shape modulation, lower values in less shape modulation.
The templates are stretched with the same value on all electrodes, and then, in case of an :code:`electrode`-type modulation,
the eap on each electrode to match the specific :math:`mod` for the electrode. Also for an :code:`template`-type modulation,
the eap is rescaled at the template level.

Next, noise is added to to the clean recordings. Three different noise modes can be used (using the :code:`noise_mode`
parameter):

1. :code:`uncorrelated`: additive gaussian noise (default) with a standard deviation of :code:`noise_level`
(10 :math:`\mu V` by default)

2. :code:`distance-correlated`: noise is generated as a multivariate normal with covariance matrix decaying with
distance between electrodes. The :code:`noise_half_distance` parameter is the distance for which correlation is 0.5.

3. :code:`far-neurons`: noise is generated by the activity of :code:`far_neurons_n` far neurons (default 300). In order to use this mode,
   it is recommended to generate templates with a small or null maximum amplitude. In fact, far neurons if their maximum amplitude
   is below :code:`far_neurons_max_amp` (default 10 :math:`\mu V`) and with an excitatory/inhibitory ratio of
   :code:`far_neurons_exc_inh_ratio` (default 0.8). Finally, a random gaussian noise floor is added, with a standard
   deviation :code:`far_neurons_noise_floor` times the one from the far neurons' activity, and the noise level is adjusted
   to match :code:`noise_level`.

When selecting :code:`uncorrelated` or :code:`distance-correlated`, one can use the :code:`noise_color` option (default is False),
so that the noise spectrum is similar to biological noise.
If :code:`noise_color` is True, the gaussian noise is filtered with an IIR resonant filter with a peak at :code:`color_peak`
(default 500) and quality factor :code:`color_q` (default 1). Moreover, a gaussian noise floor is added to the noise.
The amplitude of the gaussian added noise is controlled by :code:`random_noise_floor` (default 1), which is the percent
of gaussian noise over the colored noise (when :code:`random_noise_floor=1` 50% of the noise is additive gaussian. The final
noise level is adjusted so that the overall standard deviation is equal to :code:`noise_level`.


Finally, and optionally, the recordings can be filtered (if :code:`filter` is :code:`True`) with a high-pass or band-pass
filter with :code:`filter_cutoff` frequency(ies) ([300, 6000] by default). If :code:`filter_cutoff` is a scalar, the signal is high-pass
filtered. The order of the Butterworth filter can be adjusted with the :code:`filter_order` frequency(ies) param.

For further analysis, spike events can be annotated as "TO" (temporal overlapping) or "SO" (spatio-temporal overlapping)
when :code:`overlap` is set to :code:`True`. The waveforms can also be extracted and loaded to the
`Neo.Spiketrain <https://neo.readthedocs.io/en/0.4.0/core.html#example-spiketrain>`_
object if the :code:`extract_waveforms` is :code:`True`. Note that this might take some time for long recordings.

Recordings parameters section summary
"""""""""""""""""""""""""""""""""""""
.. code-block:: bash

    recordings:
      fs: null # sampling frequency in kHz (corresponds to dt=0.03125 ms)

      sync_rate: 0 # added synchrony rate for spatilly overlapping templates
      sync_jitt: 1 # jitter in ms for added spikes

      modulation: electrode # type of spike modulation [none (no modulation) |
        # template (each spike instance is modulated with the same value on each electrode) |
        # electrode (each electrode is modulated separately)]
      sdrand:  0.05 # standard deviation of gaussian modulation
      bursting: True # if True, spikes are modulated in amplitude depending on the isi and in shape (if shape_mod is True)
      exp_decay: 0.1 # with bursting modulation experimental decay in aplitude between consecutive spikes
      n_burst_spikes: 10 # max number of 'bursting' consecutive spikes
      max_burst_duration: 100 # duration in ms of maximum burst modulation
      shape_mod: True # if True waveforms are modulated in shape with a low pass filter depending on the isi
      shape_stretch: 30.  # min and max frequencies to be mapped to modulation value
      n_bursting: 3  # number of bursting units
      chunk_duration: 20 # chunk duration for convolution (if running into MemoryError)

      noise_level: 0 # noise standard deviation in uV
      noise_mode: uncorrelated # [uncorrelated | distance-correlated | far-neurons]
      noise_color: False # if True noise is colored resembling experimental noise
      noise_half_distance: 30 # (distance-correlated noise) distance between electrodes in um for which correlation is 0.5
      far_neurons_n: 300 # number of far noisy neurons to be simulated
      far_neurons_max_amp: 10 # maximum amplitude of far neurons
      far_neurons_noise_floor: 0.5 # percent of random noise
      far_neurons_exc_inh_ratio: 0.8 # excitatory / inhibitory noisy neurons ratio
      color_peak: 500 # (color) peak / curoff frequency of resonating filter
      color_q: 1 # (color) quality factor of resonating filter
      random_noise_floor: 1 # (color) additional noise floor

      filter: True # if True it filters the recordings
      filter_cutoff: [300, 6000] # filter cutoff frequencies in Hz
      filter_order: 3 # filter order

      overlap: False # if True, temporal and spatial overlap are computed for each spike (it may be time consuming)
      extract_waveforms: False # if True, waveforms are extracted from recordings


Drifting recordings
^^^^^^^^^^^^^^^^^^^

When drifting templates are generated (:ref:`drift-templates`), drifting recordings can be simulated when
:code:`drifting` is set to :code:`True`. The :code:`preferred_dir` parameter indicates the 3D vector with the
preferred direction of drift ([0,0,1], default, is upwards in the z-direction) and the :code:`angle_tol` (default is 15
degrees) corresponds to the tolerance in this direction.
There are three types of :code:`drift_mode`: slow, fast, and slow+fast.
The different modalities vary in terms of how the drifting template is selected for each spike during the modulated convolution.

For slow drifts, a new position is calculated moving from the initial position along the drifting direction with
a velocity of :code:`slow_drift_velocity` (default 5 :math:`\mu m`/min).
If a boundary position is reached (initial or final positions), the drift direction is reversed.

For fast drifts, the user can set the frequency at which fast drift events occur (every :code:`fast_drift_period` s, default 20 s).
When a fast drift event happens, a new template position is selected randomly among the drifting templates for each
drifting neuron, so that the amplitude of the new template on the channel in which the old template has the largest
peak is within :code:`fast_drift_min_jump` and :code:`fast_drift_min_jump` (defaults 5-20).
This is to ensure that fast drifts are not too abrupt.

Finally, when the slow+fast mode is selected, the two previously described modes are combined.

.. code-block:: bash

      drifting: False # if True templates are drifted
      drift_mode: 'slow' # drifting mode can be ['slow', 'fast', 'slow+fast']
      n_drifting: null # number of drifting units
      preferred_dir: [0, 0, 1]  # preferred drifting direction ([0,0,1] is positive z, direction)
      angle_tol: 15  # tolerance for direction in degrees
      slow_drift_velocity: 5  # drift velocity in um/min.
      fast_drift_period: 10  # period between fast drift events
      fast_drift_max_jump: 20 # maximum 'jump' in um for fast drifts
      fast_drift_min_jump: 5 # minimum 'jump' in um for fast drifts
      t_start_drift: 0  # time in s from which drifting starts

Random seeds
^^^^^^^^^^^^

The :code:`seeds` section of the recording parameters contains all the random seeds for: spike train generation
(:code:`spiketrains`), template selection (:code:`templates`), convolution operations (:code:`convolution` - including modulation,
jittering, and drifting), and noise generation (:code:`noise`). If seeds are not set, a random seed will be generated
and saved, to ensure full reproducibility of the simulations.

.. code-block:: bash

    seeds:
      spiketrains: null # random seed for spiketrain generation
      templates: null # random seed for template selection
      convolution: null # random seed for jitter selection in convolution
      noise: null # random seed for noise


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
    recgen = mr.gen_recordings(params=None, templates=None, tempgen=None, n_jobs=None, verbose=False)

The :code:`params` argument can be the path to a .yaml file or a dictionary containing the parameters
(if None default parameters are used). On of the :code:`templates` or :code:`tempgen` parameters must be indicated, the
former pointing to a generated templates file, the latter instead is a :code:`TemplateGenerator` object.
The :code:`n_jobs` argument indicates how many jobs will be used in parallel (for parallel processing, more than 1 chunks
are required).
If :code:`verbose` is True, the output shows the progress of the template simulation. :code:`verbose`=True corresponds to
:code:`verbose`=1. For a higher level of verbosity also :code:`verbose`=2 can be used.


The :code:`gen_recordings()` function returns a gen_templates :code:`RecordingGenerator` object (:code:`recgen`).

The RecordingGenerator object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :code:`RecordingGenerator` class contains several fields:

* recordings: (n_electrodes, n_samples) recordings
* spiketrains: list of (n_spiketrains) :code:`neo.Spiketrain` objects
* templates: (n_spiketrains, n_jitters, n_electrodes, n_templates samples) templates --
(n_spiketrains, n_drifting_steps, n_jitters, n_electrodes, n_templates samples) for drifting recordings
* templates_celltypes: (n_spiketrains) templates cell type
* templates_locations: (n_spiketrains, 3) templates soma locations
* templates_rotations: (n_spiketrains, 3) 3d model rotations
* channel_positions: (n_electrodes, 3) electrodes 3D positions
* timestamps: (n_samples) timestamps in seconds (quantities)
* voltage_peaks: (n_spiketrains, n_electrodes) average voltage peaks on the electrodes
* spike_traces: (n_spiketrains, n_samples) clean spike trace for each spike train
* info: dictionary with parameters used


:code:`RecordingGenerator` can be saved to .h5 files as follows:

.. code-block:: python

    import MEArec as mr
    mr.save_recording_generator(recgen, filename=None)

where :code:`recgen` is a :code:`RecordingGenerator` object and :code:`filename` is the output file name.