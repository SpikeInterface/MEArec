[![Build Status](https://travis-ci.org/alejoe91/MEArec.svg?branch=master)](https://travis-ci.org/alejoe91/MEArec)

# MEArec: Fast and customizable simulation of extracellular recordings on Multi-Electrode-Arrays

MEArec is a project for using generating biophysical extracellular neural recording on Multi-Electrode Arrays (MEA). The recording generations combines a Extracellular Action Potentials (EAP) templates generation and spike trains generation. The recordings are built by convoluting and modulating EAP templates with spike trains and adding noise.

To clone this repo open your terminal and run:

`git clone https://github.com/alejoe91/MEArec.git`

## Pre-requisites and Installation

The neural simulations rely on NEURON 7.5 (https://www.neuron.yale.edu/neuron/) (it can be downloaded from https://neuron.yale.edu/ftp/neuron/versions/) and the LFPy 2.0. NEURON should be installed manually (I you are running a Linux system add `export PYTHONPATH="/usr/local/nrn/lib/python/:$PYTHONPATH"` to your `.bashrc`. On Linux systems you also install libncurses: `sudo apt install lib32ncurses5-dev`. MEArec also uses LFPy (https://github.com/LFPy/LFPy), which requires mpi installation. On linux distributions, run: `sudo apt install libopenmpi-dev`.

After installing NEURON and openmpi, the MEArec package can be installed with:
```
pip install MEArec
```
or, from the cloned folder:

```
python setup.py develop
```

You could also create a conda environment (https://www.anaconda.com/download/) using the environment file. Open your terminal and run:

For Anaconda
`conda env create -f environment.yml`

Then activate the environment:

On Linux/MacOS:
`source activate mearec`

On Windows:
`activate mearec`

`mearec` is a command line interface: to show available commands you can run: `mearec --help`

```
Usage: mearec [OPTIONS] COMMAND [ARGS]...

  MEArec: Fast and customizable simulation of extracellular recordings on
  Multi-Electrode-Arrays

Options:
  --help  Show this message and exit.

Commands:
  default-config          Print default configurations
  gen-recordings          Generates recordings from TEMPLATES and...
  gen-templates           Generates EAP templates on multi-electrode arrays...
  recfromhdf5             Convert recordings from hdf5
  rectohdf5               Convert recordings to hdf5
  set-cell-models-folder  Set default cell_models folder
  set-recordings-folder   Set default recordings output folder
  set-recordings-params   Set default templates output folder
  set-templates-folder    Set default templates output folder
  set-templates-params    Set default templates output folder
  tempfromhdf5            Convert templates from hdf5
  temptohdf5              Convert templates to hdf5
```

## Configure simulations

the first time a command is run, `mearec` will generate a configuration file in `$HOME/.config/mearec/mearec.conf` and copy some default parameters in the `$HOME/.config/mearec/default_params` folder.

Before running any simulation, tt is necessary to point to the package where the cell models folder are. Cell models can be downloaded from the Neocortical Micro Circuit Portal https://bbp.epfl.ch/nmc-portal/welcome
(13 models from layer 5 for testing are already included in the repo - `cell_models/bbp/`).
The cell models folder can be set with:

`mearec set-cell-models-folder folder` (e.g. `mearec set-cell-models-folder MEArec/cell_models/bbp`)

Moreover, the user can set the default folders and params yaml files for templates, spike trains, and recording outputs with:

`mearec set-templates-folder folder`

`mearec set-spiketrains-folder folder`

`mearec set-recordings-folder folder`

`mearec set-templates-params folder`

`mearec set-spiketrains-params folder`

`mearec set-recordings-params folder`

(by default in `$HOME/.config/mearec/templates`, `$HOME/.config/mearec/spiketrains`, and `$HOME/.config/mearec/recordings`)


## EAP templates generation

The command to generate templates is:
```
mearec gen_templates
```
Run it with `--help` to show available arguments.

In order to check available MEA probes, just run `mearec gen_templates`, or do not provide the `--probe` option.
During the first run of the scripts, the NEURON model in the `cell_models/bbp/` will be first compiled. Simulation parameters can be changed from the `params/template_params.yaml` file, provided with an external yaml file (with the `--params` option) or overwritten with command line argument. 

EAP templates will be generated and saved in `templates\<rotation-type>\templates_<n>_<meaname>_<date>` (where n is the number of EAPs per cell model) and they can be loaded with the `tools.load_eaps(path-to-recordings)` function.


## Recordings generation

The command to generate recordings is:
```
mearec gen_recordings
```
Run it with `--help` to show available arguments.

Run the command with `--template` or `-t` option to point to the templates path. In brief, first spike trains are generated with the `SpikeTrainGenerator` class based on the `spiketrain` parameters in the `recording_params`. Then, the templates are selected based on the number of simulated spike trains and other parameters (`templates` parameters in the `recording_params`). Then, templates are convoluted in time with the spikes to create clean recordings. During convolution, single eap can be modulated either at the template level, or at the single electrode level (eith the `--modulation` ot `-m` option - none | template | electrode). Finally, a gaussian noise is added to the clean recordings (`--noise-lev` or `-nl` allows to change the noise sd in uV) and the recordings are filtered (unless the `--no-filter` option is used). All parameters for convolution and noise can be set in the `recordings` parameters in the `recording_params`.

Recordings are saved in `recordings\recording_<neurons>cells_<meaname>_<duration>s_<noise-level>uV_<date>` and they can be loaded with the `tools.load_recordings(path-to-recordings)` function.

## Save and load in hdf5 format

`mearec temptohdf5 | tempfromhdf5 | rectohdf5 | recfromhdf5` allow the user to convert the output folders to and from hdf5.

## Loading the simulated data

The `example_plotting.py` script shows how to load eap templates, spike trains, and recordings. It also shows how to use some plotting functions in `tools.py`.

# Running the simulations in Python (without command line interface)

It is also possible to run the simulation in the python environment.

```
import MEArec as mr

# Generate templates
temp_gen = mr.gen_templates('path-to-cell-models-folder')
rec_gen = mr.gen_recordings(tempgen = temp_gen)
```
`temp_gen` is a `TemplateGenerator` object that has `templates`, `locations`, `rotations`, `celltypes`, and `info` fields.
`rec_gen` is a `RecordingGenerator` object that has `recordings`, `positions`, `spiketrains`, `locations`, `peaks`, `sources`, and `info` fields.

The user can pass a `params` argument (either a `dict` or a path to a yaml file) to both `gen_templates` and `gen_recordings` to overwrite default simulation parameters (see `MEArec/default_params/templates_params.yaml` and `MEArec/default_params/recordings_params.yaml` for default values and explanation).
