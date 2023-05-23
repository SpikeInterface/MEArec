[![Build Status](https://github.com/SpikeInterface/MEArec/actions/workflows/python-package.yml/badge.svg)](https://github.com/SpikeInterface/MEArec/actions/workflows/python-package.yml/badge.svg) [![PyPI version](https://badge.fury.io/py/MEArec.svg)](https://badge.fury.io/py/MEArec)

# MEArec: Fast and customizable simulation of extracellular recordings on Multi-Electrode-Arrays

MEArec is a package for generating biophysical extracellular neural recording on Multi-Electrode Arrays (MEA). The recording generations combines a Extracellular Action Potentials (EAP) templates generation and spike trains generation. The recordings are built by convoluting and modulating EAP templates with spike trains and adding noise.

To clone this repo open your terminal and run:

`git clone https://github.com/SpikeInterface/MEArec.git`

## Installation

The MEArec package can be installed with:

```
pip install MEArec
```

To install from sources, run this from the cloned folder:

```
pip install -e .
```

## Documentation

The MEArec detailed documentation is here: https://mearec.readthedocs.io/en/latest/

### Reference

For further information please refer to the open-access Neuroinformatics article: https://doi.org/10.1007/s12021-020-09467-7

If you use the software, please cite:
```
@article{buccino2020mearec,
  title={Mearec: a fast and customizable testbench simulator for ground-truth extracellular spiking activity},
  author={Buccino, Alessio Paolo and Einevoll, Gaute Tomas},
  journal={Neuroinformatics},
  pages={1--20},
  year={2020},
  publisher={Springer}
}
```
