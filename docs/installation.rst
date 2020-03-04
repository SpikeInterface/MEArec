Installation
============

MEArec is a Python package and it can be easily installed using pip:

.. code-block:: python

    pip install MEArec


If you want to install from sources and be updated with the latest development you can install with:

.. code-block:: python

    git clone https://github.com/alejoe91/MEArec
    cd MEArec
    python setup.py install (or develop)

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
- MEAutility (https://github.com/alejoe91/MEAutility)

Additional requirements for template generatiom
-----------------------------------------------

Installing NEURON
~~~~~~~~~~~~~~~~~

The template generation requires NEURON. The code is tested using version 7.5 and 7.6.4,
that can be downloaded `here <https://neuron.yale.edu/ftp/neuron/versions/>`_. If you are running a Linux system
add:

.. code-block:: bash

    export PYTHONPATH="/usr/local/nrn/lib/python/:$PYTHONPATH"

to your .bashrc.

On Linux systems you also need install libncurses and libreadlines:

.. code-block:: bash

    sudo apt install lib32ncurses5-dev libreadline-dev

Installing LFPy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LFPy is used to generate extracellular templates. It is not installed by default, but it can be easily installed with:

.. code-block:: bash

    pip install LFPy


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