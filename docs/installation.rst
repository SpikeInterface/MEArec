Installation
============

MEArec is a Python package and it can be easily installed using pip:

.. code-block:: python

    pip install MEArec


If you want to install from sources and be updated with the latest development you can install with:

.. code-block:: python

    git clone https://github.com/SpikeInterface/MEArec
    cd MEArec
    pip install -e .

Requirements
------------

The following are the Python requirements, which are installed when running the pip installer.

- numpy
- click
- pyyaml
- matplotlib
- neo
- elephant
- h5py
- `MEAutility <https://github.com/alejoe91/MEAutility>`_

Additional requirements for template generatiom
-----------------------------------------------

The template generation phase requires NEURON and LFPy to be installed. These are not installed by default, but they can be easily installed with pip.

.. code-block:: bash

    pip install MEArec[templatess]


Installing NEURON
~~~~~~~~~~~~~~~~~

The template generation requires the NEURON simulator to be installed.
From MEArec version 1.7.0, NEURON version 7.8 is supported. On UNIX systems NEURON can be installed with:

.. code-block:: bash

    pip install neuron

On Windows machines, NEURON can be downloaded and installed from this `link <https://www.neuron.yale.edu/neuron/download>`_.


Installing LFPy
~~~~~~~~~~~~~~~

LFPy is used to generate extracellular templates. It is not installed by default, but it can be easily installed with:

.. code-block:: bash

    pip install LFPy>=2.2


***NOTE***: LFPy version 2.2 is required. LFPy version 2.1 is not compatible with MEArec.


Test the installation
---------------------

You can test that MEArec is correctly imported in python:

.. code-block:: python

    import MEArec as mr

And that the CLI is working. Open a terminal and run:

.. code-block:: bash

    mearec

You should get the list of available commands:

.. code-block:: bash

    Usage: mearec [OPTIONS] COMMAND [ARGS]...

      MEArec: Fast and customizable simulation of extracellular recordings on
      Multi-Electrode-Arrays

    Options:
      --help  Show this message and exit.

    Commands:
      available-probes        Print available probes.
      default-config          Print default configurations.
      gen-recordings          Generates RECORDINGS from TEMPLATES.
      gen-templates           Generates TEMPLATES with biophysical simulation.
      set-cell-models-folder  Set default cell_models folder.
      set-recordings-folder   Set default recordings output folder.
      set-recordings-params   Set default templates output folder.
      set-templates-folder    Set default templates output folder.
      set-templates-params    Set default templates output folder.