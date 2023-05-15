API reference
==============

Module :mod:`MEArec.generators`
-------------------------------
.. automodule:: MEArec.generators

    .. autoclass:: TemplateGenerator
        :members:
        :show-inheritance:
        :undoc-members:

    .. autoclass:: SpikeTrainGenerator
        :members:
        :show-inheritance:
        :undoc-members:

    .. autoclass:: RecordingGenerator
        :members:
        :show-inheritance:
        :undoc-members:


Module :mod:`MEArec.generation_tools`
-------------------------------------
.. automodule:: MEArec.generation_tools

    .. autofunction:: gen_templates
    .. autofunction:: gen_spiketrains
    .. autofunction:: gen_recordings


Module :mod:`MEArec.tools`
--------------------------
.. automodule:: MEArec.tools

    .. autofunction:: load_templates
    .. autofunction:: load_recordings
    .. autofunction:: save_template_generator
    .. autofunction:: save_recording_generator
    .. autofunction:: get_default_config
    .. autofunction:: get_default_cell_models_folder
    .. autofunction:: get_default_templates_params
    .. autofunction:: get_default_recordings_params
    .. autofunction:: plot_rasters
    .. autofunction:: plot_templates
    .. autofunction:: plot_recordings
    .. autofunction:: plot_waveforms
    .. autofunction:: plot_amplitudes
    .. autofunction:: plot_pca_map

Module :mod:`MEArec.drift_tools`
--------------------------------
.. automodule:: MEArec.drift_tools

    .. autofunction:: generate_drift_dict_from_params


Module :mod:`MEArec.simulate_cells`
-----------------------------------
.. automodule:: MEArec.simulate_cells

    .. autofunction:: calculate_extracellular_potential
    .. autofunction:: return_bbp_cell
    .. autofunction:: return_bbp_cell_morphology
    .. autofunction:: run_cell_model
