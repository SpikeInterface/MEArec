Command Line Interface (CLI)
============================

MEArec implements a command line interface (CLI) to make templates and recordings generation easy to use and to allow for scripting.
In order to discover the available commands, the user can use the :code:`--help` option:


.. code-block:: bash

    >> mearec --help

which outputs:

.. parsed-literal::

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
      set-recordings-params   Set default recordings parameter file.
      set-templates-folder    Set default templates output folder.
      set-templates-params    Set default templates parameter file.

Each available command can be inspected using the \texttt{-{}-help} option:


.. code-block:: bash

    >> mearec "command" --help


A list of available probes can be found by running the :code:`mearec available-probes` command.

Setting global configurations
------------------------------

At installation, MEArec creates a configuration folder (:code:`.config/mearec/`) in which global settings are stored.
The default paths to cell models folder, templates and recordings output folders and parameters can be set using the
:code:`set-` commands. By default, these files and folders are located in the configuration folder.

.. code-block:: bash

    >> mearec default-config

which outputs:

.. parsed-literal::

    {'cell_models_folder': path-to-cell_models,
     'recordings_folder': path-to-recordings-folder,
     'recordings_params': path-to-recordings-params.yaml,
     'templates_folder': path-to-templates-folder,
     'templates_params': path-to-templates-params.yaml}

At time of installation, some default files are copied in the :code:`.config/mearec` folder including:

* a small set of 13 cell models (:code:`.config/mearec/cell_models/bbp/`)

* default parameters for templates generation (:code:`.config/mearec/default_params/templates_params.yaml`)

* default parameters for recordings generation (:code:`.config/mearec/default_params/recordings_params.yaml`)

These provide the default folders to look for cell models and parameters. From the command line and Python interface
both the cell models folder and all parameters related to template and recording generation can be overridden.

The :code:`.config/mearec/recordings` and :code:`.config/mearec/templates` are the default output folders where the
templates and recordings will be saved, respectively.

The default settings can be changed with the following commands:

.. code-block:: bash

    >> mearec set-cell-models-folder
    >> mearec set-recordings-folder
    >> mearec set-recordings-params
    >> mearec set-templates-folder
    >> mearec set-templates-params
