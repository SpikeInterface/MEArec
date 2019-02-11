Setting global configurations (CLI)
===================================

The command-line interface provides a very handy way to run MEArec simulations.

MEArec commands will use some default-settings that can be changed. The default settings can be seen using this command:

.. code-block:: bash

    mearec default-config

Which will print something like:

.. code-block:: bash

    {'cell_models_folder': '/home/User/.config/mearec/cell_models/bbp',
     'recordings_folder': '/home/User/.config/mearec/recordings',
     'recordings_params': '/home/User/.config/mearec/default_params/recordings_params.yaml',
     'templates_folder': '/home/User/.config/mearec/templates',
     'templates_params': '/home/User/.config/mearec/default_params/templates_params.yaml'}

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

    mearec set-cell-models-folder
    mearec set-recordings-folder
    mearec set-recordings-params
    mearec set-templates-folder
    mearec set-templates-params
