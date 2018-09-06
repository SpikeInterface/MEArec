# MEArec: Fast and customizable simulation of extracellular recordings on Multi-Electrode-Arrays

MEArec is a project for using generating biophysical extracellular neural recording on Multi-Electrode Arrays (MEA). The recording generations combines a Extracellular Action Potentials (EAP) templates generation and spike trains generation. The recordings are built by convoluting and modulating EAP templates with spike trains and adding noise.

To clone this repo open your terminal and run:

`git clone https://github.com/alejoe91/MEArec.git`

## Pre-requisites

In order to install all required packages we recommend creating an conda
(https://www.anaconda.com/download/) environment using the environment files. Open your terminal and run:

For Anaconda
`conda env create -f environment.yml`

Then activate the environment:

On Linux/MacOS:
`source activate neurocnn`

On Windows:
`activate neurocnn`

The neural simulations rely on NEURON 7.5 (https://www.neuron.yale.edu/neuron/) (it can be downloaded from https://neuron.yale.edu/ftp/neuron/versions/) and the LFPy 2.0. NEURON should be installed manually.

Moreover, these scripts make use of a package for handling MEAs (MEAutility):

```
git clone https://github.com/alejoe91/MEAutility
cd MEAutility
python setup.py install
```

## EAP templates generation

Cell models can be downloaded from the Neocortical Micro Circuit Portal https://bbp.epfl.ch/nmc-portal/welcome
(13 models from layer 5 for testing are already included).
Newly downloaded models should be unzipped to the folder `cell_models/bbp/`

The script for generating EAP templates is `templates_gen.py`. If you run `python templates_gen.py --help` all the available arguments and options are listed. In order to check available MEA probes, just run `python templates_gen.py`, or do not provide the `--probe` option.
During the first run of the scripts, the NEURON model in the `cell_models/bbp/` will be first compiled and then the EAP templates will be generated and saved in `templates\bbp\<rotation-type>\templates_<n>_<meaname>_<date>` (where n is the number of EAPs per cell model)

## Spike trains generation


## Recordings generation


