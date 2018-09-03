# defaultconfig.py

# This is a file for default system specific configuration file, which are ignored
# if a local 'config.py' is provided (not being part of the git!!).

import os

machine_name='not specified'

# local path variables
root_folder=os.path.dirname(os.path.abspath(__file__))
data_dir=root_folder
expdata_dir=data_dir
