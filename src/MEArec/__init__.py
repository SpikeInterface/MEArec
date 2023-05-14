import importlib.metadata

__version__ = importlib.metadata.version("MEArec")

from MEArec.generation_tools import (gen_recordings, gen_spiketrains,
                                     gen_templates)
from MEArec.generators import (RecordingGenerator, SpikeTrainGenerator,
                               TemplateGenerator)
from MEArec.simulate_cells import (calc_extracellular,
                                   calculate_extracellular_potential,
                                   return_bbp_cell, return_bbp_cell_morphology,
                                   run_cell_model, simulate_templates_one_cell)
from MEArec.tools import (available_probes, convert_recording_to_new_version,
                          extract_units_drift_vector,
                          get_default_cell_models_folder, get_default_config,
                          get_default_drift_dict,
                          get_default_recordings_params,
                          get_default_templates_params, get_templates_features,
                          load_dict_from_hdf5, load_recordings,
                          load_recordings_from_file, load_templates,
                          plot_amplitudes, plot_cell_drifts, plot_pca_map,
                          plot_rasters, plot_recordings, plot_templates,
                          plot_waveforms, safe_yaml_load, save_dict_to_hdf5,
                          save_recording_generator, save_recording_to_file,
                          save_template_generator)

from MEArec.drift_tools import generate_drift_dict_from_params
