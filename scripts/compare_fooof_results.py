from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import json
import sys
import os

from bars_bands import plot_eeg_bands
from fooof_results_one_session import get_fooof_results

##############
def main(args):
    # global flags_global
    flags_global = np.array([0,0,0])

    flag_csd = False
    ## run matplotlib in interactive mode
    plt.ion()

    print(f'arg {args[1]}') ## folder location
    print(f'arg {args[2]}') ## subject = {0:patient 1, 1:patient 2, ...}
    print(f'arg {args[3]}') ## session = {1:time zero, 2:three months, 3:six months}
    print(f'arg {args[4]}') ## ABT = {0:resting, 1:biking}
    print(f'arg {args[5]}') ## rest end = {0:off, 1:on}
    
    path=args[1]
    subject= int(args[2])
    session=int(args[3])
    abt= int(args[4])
    flag_rest_end = int(args[5])

     ## path to save figure
    path_fig = path+'a_neuroplasticity/figures/'

    # checking if the directory figures
    # exist or not.
    if not os.path.exists(path_fig):
        # if the figures directory is not present 
        # then it creates it.
        os.makedirs(path_fig)

    ## read all sessions until the session value
    fooof_results_dict={}
    info_patient_dict={}
    for id_s in np.arange(session+1):
        fooof_results_dict[id_s], info_patient_dict[id_s] = get_fooof_results(path, subject, id_s, abt, flag_rest_end)

    ## plot power differences from alpha and beta bands
    fig_filename = path_fig+f'eeg_bands_n_{subject}.png'
    plot_eeg_bands(fooof_results_dict, info_patient_dict, fig_filename)

    return 0

##########################
if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
