# MEArec: Fast and customizable simulation of extracellular recordings on Multi-Electrode-Arrays

MEArec is a project for using generating biophysical extracellular neural recording on Multi-Electrode Arrays (MEA). The recording generations combines a Extracellular Action Potentials (EAP) templates generation and spike trains generation. The recordings are built by convoluting and modulating EAP templates with spike trains and adding noise.

To clone this repo open your terminal and run:

`git clone https://github.com/alejoe91/MEArec.git`

## Pre-requisites and Installation

In order to install all required packages we recommend creating an conda
(https://www.anaconda.com/download/) environment using the environment files. Open your terminal and run:

For Anaconda
`conda env create -f environment.yml`

Then activate the environment:

On Linux/MacOS:
`source activate mearec`

On Windows:
`activate neurocnn`

The neural simulations rely on NEURON 7.5 (https://www.neuron.yale.edu/neuron/) (it can be downloaded from https://neuron.yale.edu/ftp/neuron/versions/) and the LFPy 2.0. NEURON should be installed manually (I you are running a Linux system add `export PYTHONPATH="/usr/local/nrn/lib/python/:$PYTHONPATH"` to your `.bashrc`. On Linux systems you also install libncurses: `sudo apt install lib32ncurses5-dev`

Moreover, these scripts make use of a package for handling MEAs (MEAutility):

```
git clone https://github.com/alejoe91/MEAutility
cd MEAutility
python setup.py install (or develop)
```

## EAP templates generation

Cell models can be downloaded from the Neocortical Micro Circuit Portal https://bbp.epfl.ch/nmc-portal/welcome
(13 models from layer 5 for testing are already included).
Newly downloaded models should be unzipped to the folder `cell_models/bbp/`

The script for generating EAP templates is `gen_templates.py`. If you run `python gen_templates.py --help` all the available arguments and options are listed. In order to check available MEA probes, just run `python gen_templates.py`, or do not provide the `--probe` option.
During the first run of the scripts, the NEURON model in the `cell_models/bbp/` will be first compiled. Simulation parameters can be changed from the `params/template_params.yaml` file, provided with an external yaml file (with the `--params` option) or overwritten with command line argument. 
EAP templates will be generated and saved in `templates\<rotation-type>\templates_<n>_<meaname>_<date>` (where n is the number of EAPs per cell model) and they can be loaded with the `tools.load_eaps(path-to-recordings)` function.


## Spike trains generation

The script for generating EAP templates is `gen_spiketrain.py`. Simulation parameters can be viewed and changed in the `params/spiketrain_params.yaml`.
Spike trains and info are saved in `spiketrains\spiketrains_<neurons>_<date>` folder (neurons is the number of neurons) and they can be loaded with the `tools.load_spiketrains(path-to-spiketrains)` function.


## Recordings generation

Once EAP templates and spike trains are generated, the `gen_recording.py` script combines to create simulated recordings. 
Run the script with `--template` or `-t` option to point to the templates path and the `--spiketrain` or `-st` for the spike trains path. In brief, the templates are selected based onthe number of available spike trains and other parameters (see `params/recording_params.yaml` for details). Then, templates are convoluted in time with the spikes to create clean recordings. During convolution, single eap can be modulated either at the template level, or at the single electrode level (eith the `--modulation` ot `-m` option - none | template | electrode). Finally, a gaussian noise is added to the clean recordings (`--noise-lev` or `-nl` allows to change the noise sd in uV) and the recordings are filtered (unless the `--no-filter` option is used).
Recordings are saved in `recordings\recording_<neurons>cells_<meaname>_<duration>s_<noise-level>uV_<date>` and they can be loaded with the `tools.load_recordings(path-to-recordings)` function.

## Loading the simulated data

The `example_plotting.py` script shows how to load eap templates, spike trains, and recordings. It also shows how to use some plotting functions in `tools.py`.

## Running via MountainLab

Two of the MEArec routines (gen_spiketrains and gen_recording) have been wrapped as [MountainLab](https://github.com/flatironinstitute/mountainlab-js) processors. After installing MountainLab, clone this repository and make a symbolic link to the MountainLab packages directory as described in the [MountainLab documentation](https://github.com/flatironinstitute/mountainlab-js), or as given here:

```
# install mountainlab
conda install -c flatiron -c conda-forge mountainlab mountainlab_pytools mlprocessors 

# clone this repository, install the requirements, and install as ML package
git clone [this-repo]
cd [this-repo-name]
pip install -r ml_requirements.txt
ln -s $PWD `ml-config package_directory`/ml_mearec
```

If everything worked, you can see the spec for the two processors by running

```
ml-spec -p mearec.gen_spiketrains
ml-spec -p mearec.gen_recording
```

which gives something like the following:

```
###############################################################################
mearec.gen_spiketrains
Generate spiketrains

INPUTS

OUTPUTS
  spiketrains_out -- MEArec spiketrains .npy output file -- neo format

PARAMETERS
  duration -- (optional) Duration of spike trains (s)
  n_exc -- (optional) Number of excitatory cells
  n_inh -- (optional) Number of inhibitory cells
  f_exc -- (optional) Mean firing rate of excitatory cells (Hz)
  f_inh -- (optional) Mean firing rate of inhibitory cells (Hz)
  min_rate -- (optional) Minimum firing rate for all cells (Hz)
  st_exc -- (optional) Firing rate standard deviation of excitatory cells (Hz)
  st_inh -- (optional) Firing rate standard deviation of inhibitory cells (Hz)
  process -- (optional) poisson or gamma
  t_start -- (optional) Starting time (s)
  ref_per -- (optional) Refractory period to remove spike violation (ms)
```

```
###############################################################################
mearec.gen_recording
Generate a MEArec recording

INPUTS
  templates -- (optional) MEArec templates .hdf5 file - generated using utils/templates_to_hdf5.py - if omitted, will download default
  spiketrains -- MEArec spiketrains .npy file

OUTPUTS
  recording_out -- MEArec recording .hdf5 file

PARAMETERS
  min_dist -- (optional) minimum distance between neurons
  min_amp -- (optional) minimum spike amplitude in uV
  noise_level -- (optional) noise standard deviation in uV
  filter -- (optional) if True it filters the recordings
  cutoff -- (optional) filter cutoff frequencies in Hz
  overlap_threshold -- (optional) threshold to consider two templates spatially overlapping (e.g 0.6 -> 60 percent of template B on largest electrode of template A)
  n_jitters -- (optional) number of temporal jittered copies for each eap
  upsample -- (optional) upsampling factor to extract jittered copies
  pad_len -- (optional) padding of templates in ms
  modulation -- (optional) # type of spike modulation none (no modulation) | template (each spike instance is modulated with the same value on each electrode) | electrode (each electrode is modulated separately)
  mrand -- (optional) mean of gaussian modulation (should be 1)
  sdrand -- (optional) standard deviation of gaussian modulation
  chunk_duration -- (optional) chunk duration in s for chunk processing (if 0 the entire recordings are generated at once)
  overlap -- (optional) if True it annotates overlapping spikes
```

An example run from the command line would be

```
ml-run-process mearec.gen_spiketrains -o spiketrains_out:spiketrains_60.npy -p duration:60
ml-run-process mearec.gen_recording -o recording_out:recording_60.h5 -i spiketrains:spiketrains_60.npy -p noise_level:3
```

This should produce a 60 second recording file named `recording_60.h5`.

Or, if you have singularity and MountainLab installed, you can run these commands in a container without installing this repository by adding the `--container=magland/MEArec:v0.1.0` argument to the `ml-run-process` commands, where the version should be updated as appropriate.

As ML processors, these can also be executed using python or jupyter notebooks using the mountainlab_pytools utilities. See the MountainLab and MountainSort documentation for more details.


