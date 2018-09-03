from __future__ import print_function

"""
This is a bad hack to be able to run models in series. For some reason I run into problems if I try to run different
cell models in a for loop if the kernel in not completely restarted
"""


import os, sys
import glob
from os.path import join

only_intracellular = True
model_folder = 'cell_models'
cells = 'all'


if __name__ == '__main__':

    if len(sys.argv) == 1:
        print('Use: \n\t -rot <arg> \t possible rotation arguments: Norot-physrot-3drot, \n' \
              '\t -intraonly \t only simulate intracellular,\n' \
              '\t -model <name> \t cell model name (default=bbp), \n' \
              '\t -probe <name> \t any probe name in electrodes folder, \n' \
              '\t -n <int> \t number of observations per cell type')
    else:
        if '-model' in sys.argv:
            pos = sys.argv.index('-model')
            model = sys.argv[pos+1]  # file full path
        else:
            model = 'bbp'
        if '-rot' in sys.argv:
            pos = sys.argv.index('-rot')
            rotation = sys.argv[pos+1]  # Na - NaRep - PCA - PCA3d - 3d
        else:
            rotation = 'physrot'
        if '-intraonly' in sys.argv:
            only_intracellular = True
        else:
            only_intracellular = False
        if '-probe' in sys.argv:
            pos = sys.argv.index('-probe')
            probe = sys.argv[pos+1]  # Na - NaRep - PCA - PCA3d - 3d
        else:
            probe = 'SqMEA-10-15um'
        if '-n' in sys.argv:
            pos = sys.argv.index('-n')
            nobs = int(sys.argv[pos+1])  # Na - NaRep - PCA - PCA3d - 3d
        else:
            nobs = 1000

        if model == 'bbp':
            cell_models = [f for f in os.listdir(join(model_folder, model)) if 'mods' not in f]

        else:
            raise Exception('Only Blue Brain Project cells can be used')

        if not os.path.isdir(join(model_folder, model, 'mods')):
            print('Compiling NEURON models')
            os.system('python hbp_cells.py compile')

        for numb, cell_model in enumerate(cell_models):
            print(cell_model, numb + 1, "/", len(cell_models))
            os.system("python hbp_cells.py %s %s %d %r %s %s %d"\
                      % (join(model_folder, model, cell_model), model, numb,\
                         only_intracellular, rotation, probe, nobs))


