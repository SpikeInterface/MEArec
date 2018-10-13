# MEArec: Fast and customizable simulation of extracellular recordings on Multi-Electrode-Arrays

MEArec is a project for using generating biophysical extracellular neural recording on Multi-Electrode Arrays (MEA). The recording generations combines a Extracellular Action Potentials (EAP) templates generation and spike trains generation. The recordings are built by convoluting and modulating EAP templates with spike trains and adding noise.

To clone this repo open your terminal and run:

`git clone https://github.com/alejoe91/MEArec.git`

## Pre-requisites and Installation

The neural simulations rely on NEURON 7.5 (https://www.neuron.yale.edu/neuron/) (it can be downloaded from https://neuron.yale.edu/ftp/neuron/versions/) and the LFPy 2.0. NEURON should be installed manually (I you are running a Linux system add `export PYTHONPATH="/usr/local/nrn/lib/python/:$PYTHONPATH"` to your `.bashrc`. On Linux systems you also install libncurses: `sudo apt install lib32ncurses5-dev`

After installing NEURON, the MEArec package can be installed with:
```
python setup.py develop
```
In addition, we recommend creating an conda (https://www.anaconda.com/download/) environment using the environment files. Open your terminal and run:

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
  fromhdf5                Convert templates spike trains, and recordings...
  gen-recordings          Generates recordings from TEMPLATES and...
  gen-spiketrains         Generates spike trains for recordings
  gen-templates           Generates EAP templates on multi-electrode arrays...
  set-cell-models-folder  Set default cell_models folder
  set-recordings-folder   Set default recordings output folder
  set-spiketrains-folder  Set default spiketrains output folder
  set-templates-folder    Set default templates output folder
  tohdf5                  Convert templates spike trains, and recordings to...
```

## Configure simulations

the first time a command is run, `mearec` will generate a configuration file in `$HOME/.config/mearec/mearec.conf` and copy some default parameters in the `$HOME/.config/mearec/default_params` folder.

Before running any simulation, tt is necessary to point to the package where the cell models folder are. Cell models can be downloaded from the Neocortical Micro Circuit Portal https://bbp.epfl.ch/nmc-portal/welcome
(13 models from layer 5 for testing are already included in the repo - `cell_models/bbp/`).
The cell models folder can be set with:

`mearec set-cell-models-folder folder` (e.g. `mearec set-cell-models-folder MEArec/cell_models/bbp`)

Moreover, the user can set the default folders for templates, spike trains, and recording outputs with:

`mearec set-templates-folder folder`

`mearec set-spiketrains-folder folder`

`mearec set-recordings-folder folder`

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


## Spike trains generation

The command to generate spike trains is:
```
mearec gen_spiketrains
```
Run it with `--help` to show available arguments.
Simulation parameters can be viewed and changed in the `params/spiketrain_params.yaml`.
Spike trains and info are saved in `spiketrains\spiketrains_<neurons>_<date>` folder (neurons is the number of neurons) and they can be loaded with the `tools.load_spiketrains(path-to-spiketrains)` function.


## Recordings generation

The command to generate recordings is:
```
mearec gen_recordings
```
Run it with `--help` to show available arguments.

Run the command with `--template` or `-t` option to point to the templates path and the `--spiketrain` or `-st` for the spike trains path. In brief, the templates are selected based onthe number of available spike trains and other parameters (see `params/recording_params.yaml` for details). Then, templates are convoluted in time with the spikes to create clean recordings. During convolution, single eap can be modulated either at the template level, or at the single electrode level (eith the `--modulation` ot `-m` option - none | template | electrode). Finally, a gaussian noise is added to the clean recordings (`--noise-lev` or `-nl` allows to change the noise sd in uV) and the recordings are filtered (unless the `--no-filter` option is used).
Recordings are saved in `recordings\recording_<neurons>cells_<meaname>_<duration>s_<noise-level>uV_<date>` and they can be loaded with the `tools.load_recordings(path-to-recordings)` function.

## Save and load in hdf5 format

`mearec tohdf5` and `mearec fromhdf5` allow the user to convert the output folders to and from hdf5.

## Loading the simulated data

The `example_plotting.py` script shows how to load eap templates, spike trains, and recordings. It also shows how to use some plotting functions in `tools.py`.
