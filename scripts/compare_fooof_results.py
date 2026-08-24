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

    ## read all sessions until the session value
    labels_list = ['a','b','c']
    titles_list = ['rest start','biking','rest end']

    id_ref = 0
    label_ref = labels_list[id_ref]
    title_ref = titles_list[id_ref]

    id_val = 2
    label_val = labels_list[id_val]
    title_val = titles_list[id_val]

    fooof_results_dict={}
    info_patient_dict={}
    for id_s in np.arange(session+1):
        fooof_results_dict[id_s], info_patient_dict[id_s], path_base = get_fooof_results(path, subject, id_s, abt, flag_rest_end,  label_val, label_ref)

    print(f"path base: {path_base}")

    path_base = path_base + 'figures/'
    if not os.path.exists(path_base):
            # if the figures directory is not present 
            # then it creates it.
            os.makedirs(path_base)

    # for label_s in fooof_results_dict:
    #     print(f"fooof_results_dict label: {label_s}")
    #     comp_dict = fooof_results_dict[label_s]
    #     print(f"comp_dict:\n{comp_dict}")

    #     for label_diff in comp_dict:
    #         ## plot power differences from alpha and beta bands
    #         fig_filename = path_base+f'eeg_bands_n_{subject}.png'
    #         ## plot power differences between two states, for example cycling and resting start
    #         plot_eeg_bands(comp_dict[label_diff], info_patient_dict, fig_filename)
        
    ## plot power differences from alpha and beta bands
    flag_save = True
    fig_filename = path_base+f'diff_{label_val}_{label_ref}_n_{subject}.png'
    ## plot power differences between two states, for example cycling and resting start
    
    plot_eeg_bands(fooof_results_dict, info_patient_dict, fig_filename, flag_save, title_val, title_ref)

    return 0

##########################
if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
