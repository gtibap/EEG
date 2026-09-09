#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import mne
mne.set_log_level('error')

import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# from channels_tfr import selected_channels
## include modules from another directory
# sys.path.insert(0, '../../scripts')
from info_participants import subject_dict

from class_psd_all import PSD_Epochs_Class
from ica_epochs import ica_epochs_interactive, read_ica_model
from plot_psd_labels import psd_regions_visualization

######################
## global variables

sampling_rate = 1.0
y_limits = [-8,8]
ylim_global = [0, 40]
freq_range = [1,45]
thr_peaks_global = 2.0

## scale selection for visualization raw data with annotations
scale_dict = dict(mag=1e-12, grad=4e-11, eeg=100e-6, eog=150e-6, ecg=400e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)

##############################
## channels lists per regions
central_left_channels = ['E7','E13','E29','E30','E31','E35','E36','E37','E41','E42','E47','E53','E54'] ## blue
central_right_channels = ['E79','E80','E86','E87','E93','E98','E103','E104','E105','E106','E110','E111','E112'] ## orange

frontal_left_channels = ['E12','E18','E19','E20','E22','E23','E24','E26','E27','E28','E32','E33','E34'] ## green
frontal_right_channels = ['E1','E2','E3','E4','E5','E9','E10','E116','E117','E118','E122','E123','E124'] ## pink

parietal_left_channels = ['E51','E52','E58','E59','E60','E61','E66','E67','E71']
parietal_right_channels = ['E76','E77','E78','E84','E85','E91','E92','E96','E97'] ## marine blue

occipital_left_channels = ['E64','E65','E69','E70','E74']
occipital_right_channels = ['E82','E83','E89','E90','E95']

temporal_left_channels = ['E40','E46','E50']
temporal_right_channels = ['E101','E102','E109']

midline_channels = ['VREF','E6','E11','E16','E55','E62','E72','E75']

excluded_channels = ['E8','E14','E15','E17','E21','E25','E38','E39','E43','E44','E45','E48','E49','E56','E57','E63','E68','E73','E81','E88','E94','E99','E100','E107','E108','E113','E114','E115','E119','E120','E121','E125','E126','E127','E128']

event_list_ce = ['a_ce','b_ce','c_ce']
event_list_oe = ['a_oe','b_oe','c_oe']

## interactive plots
fig_a = []
ax_a = []
fig_ap = []
ax_ap = []
ax_ce_global = []
fig_ce_global = []
fig_mea = []
ax_mea = []
fig_ce=[]
fig_oe=[]
ax_ce=[]
ax_oe=[]
fig_peaks=[]
ax_peaks=[]
info_p = ''

obj_list = []
obj_global = []
region_global = ''
selectors=[]
selectors_peaks=[]
df_psd_global = pd.DataFrame()
f0_global=0
f1_global=0
flag_eyes_closed = True
flag_peaks_global = False
fig_title_mea = ''
path_fig_fooof = ''
path_csv_files = ''
path_fig = ''
path_fig_psd = ''
path_fig_boxplot = ''
path_prep = ''

##############
def create_directories(path_session):
    ## path filename boxplots
    global path_fig_boxplot, path_fig, path_fig_psd, path_fig_fooof, path_csv_files, path_prep
    path_fig = path_session+'figures/'
    path_fig_boxplot = path_session+'figures/'
    path_fig_psd = path_session+'figures/psd/'
    path_fig_fooof = path_session+'figures/fooof/'
    path_csv_files = path_session+'csv/'
    path_prep = path_session+'prep/'

    return 0


##################################################
def load_selected_epochs(raw_data, label_list_ref):
    global obj_list
    ## create same number of events and epochs from raw_data
    dt = 5 ## epoch duration in seconds
    print(f"Epochs size: {dt} seconds / each ")

    ## first, include a sequence of regular events to the raw data 
    new_events = mne.make_fixed_length_events(raw_data, start=0, stop=None, duration=dt)
    # raw_copy.add_events(new_events, replace=True)

    ## second, use events to create epochs
    epochs_ref = mne.Epochs(raw_data, new_events, tmin=0.0, tmax=dt, baseline=None, preload=True, reject=None, reject_by_annotation=True)
    # print(f"Number of epochs for ICA section:\n{len(epochs.selection)}")
    # print(f"Epochs for ICA section:\n{epochs.selection}")
    all_epochs_list = epochs_ref.selection

    ## Now, we reject bad epochs and bad channels for each selected label
    for label in label_list_ref:
        print(f"label: {label}")
        ## removing bad epochs and bad channels if they were already selected in a previous iteration
        epochs = epochs_ref.copy()

        #### read selected epochs, bad epochs, and bad channels
        try:
            # Read list of bad epochs and bad channels from file
            with open(f"{path_prep}{label}_bad_epochs.json", "r") as f:
                data = json.load(f)
        except:
            print(f"not found: {path_prep}{label}_bad_epochs.json")
            continue
        sel_epochs_list = data['sel_epochs']
        bad_epochs_list = data['bad_epochs']
        bad_channels_list = data['bad_channels']
        # print(f"sel_epochs_list:\n{sel_epochs_list}")
        print(f"bad_epochs_list: {bad_epochs_list}")
        print(f"bad_channels_list: {bad_channels_list}")

        ## keep epochs of the first_list (selection) that are not in the second list (bads)
        sel_epochs_list = np.array([x for x in sel_epochs_list if not (x in bad_epochs_list)]).astype(int)
        # print(f"sel_epochs_list:\n{sel_epochs_list}")

        ## list of bad epochs to remove
        bad_epochs_list = np.array([x for x in all_epochs_list if not (x in sel_epochs_list)]).astype(int)
        # print(f"bad_epochs_list:\n{bad_epochs_list}")

        ## drop bad epochs
        epochs.drop(bad_epochs_list.astype(int))
        ## including bad channels
        epochs.info['bads'] = bad_channels_list

        ## interactive selection of bad epochs and bad channels
        # epochs.plot(n_epochs=12, events=True, block=True, n_channels=36, scalings=scale_dict, title=f"{label} : Epochs",)
        if len(epochs.selection) > 0:
            #############
            # ICA
            print(f"ica epochs interactive...")
            root_filename = f"{path_prep}{label}"

            try:
                print(f"loading ICA model...")
                epochs = read_ica_model(epochs, label, root_filename)
            except:
                print(f"calculating ICA model...")
                epochs = ica_epochs_interactive(epochs, label, root_filename)

            ## re-referencing average
            epochs.set_eeg_reference(ref_channels="average", ch_type='eeg', projection=False,)
            ## replace bad channels by interpolation
            epochs.interpolate_bads()

            ## power spectral density (PSD) from epochs of selected channels
            freq_range = [0.5, 45]
            psd_left = epochs.compute_psd(picks=central_left_channels, exclude='bads',fmin=freq_range[0], fmax=freq_range[1])
            psd_right = epochs.compute_psd(picks=central_right_channels, exclude='bads',fmin=freq_range[0], fmax=freq_range[1])

            obj = PSD_Epochs_Class(label)
            obj.set_psd(psd_left, 'central_left')
            obj.set_psd(psd_right, 'central_right')

            obj_list.append(obj)

        else:
            print(f"Warning: {path_prep}{label}_bad_epochs.json not found")
            print(f"Warning: ICA not calculated.")
        
    return 0


###########################################
## EEG filtering and signals pre-processing
##
def main(args):
    global sampling_rate, psd_fig_name, obj_list, ylim_global, thr_peaks_global, info_p, path_fig_fooof, path_csv_files, path_fig_psd

    # to run GUI event loop
    plt.ion()

    print(f'subject id: {args[1]}') ## subject id (integer number in the dict info_participants.py)
    print(f'session: {args[2]}') ## session = {0:first session, 1:second session, and so on}

    subject= int(args[1])
    session= int(args[2])

    ## root folder for the selected subject
    path = f"../../data/a_neuroplasticity/n_{str(subject).zfill(3)}/"
    ## folder selected session
    path_session = f"{path}session_{str(session)}/"
    print(f"path: {path}")
    print(f"path session: {path_session}")

    ## create global variables of path folders/directories
    create_directories(path_session)

    ## patient info including file name of raw (data), age, sex, ais, nli, days (after trauma), 
    data_pt = subject_dict[subject]
    print(f"data pt: {data_pt}")

    raw_filename = data_pt['raw'][session]

    ##########################
    ## read raw data 
    acquisition_system = 'geodesic'
    raw_data = mne.io.read_raw_egi(path_session + raw_filename, preload=False)

    ## open annotations annotations.fif
    my_annot = mne.read_annotations(path_session + 'annotations.fif')

    ##########################
    ## raw data recording date
    print(f"\nmeasuring date: {raw_data.info['meas_date']}\n")

     ## loading raw data    
    raw_data.load_data()
    ## adding annotations to raw data
    raw_data.set_annotations(my_annot)

    ##########################
    ## exclude channels of the net boundaries that usually bring noise or artifacts
    ## geodesic system we remove channels in the boundaries
    # raw_data.info["bads"] = bad_channels_dict[acquisition_system]
    ## list of excluded channels 
    raw_data.info["bads"] = excluded_channels
    raw_data.drop_channels(raw_data.info['bads'])

    ################################
    ## Stage 1: passband and notch filters, and resampling
    low_cut =    0.5
    hi_cut  =   45.0

    print(f"Passband filter {low_cut, hi_cut} Hz...")
    raw_data.filter(l_freq=low_cut, h_freq=hi_cut, picks='eeg')

    freqs_notch = [60,]
    print(f"Notch filter {freqs_notch}...")
    
    raw_data.notch_filter(freqs=freqs_notch, picks='eeg',) ## filter_length="10s"
    # raw_data.notch_filter(freqs=freqs_notch, picks='eeg', method="spectrum_fit",)

    #########################
    ## make groups of EEG data segments with same label in order to create epochs
    label_list_ref = ['a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes','c_closed_eyes','c_opened_eyes']

    ###############################
    ## once bad epochs and bad channels were selected, next:
    ## ica decomposition for interactive artifacts removal
    print(f"ica epochs...")
    load_selected_epochs(raw_data, label_list_ref)

    info_pt = f"n_{str(subject).zfill(3)}, session: {session}"
    flag_save = False

    ################################
    ## fooof model fitting to separate aperiodic and periodic components    
    psd_regions_visualization(obj_list, info_pt, path_fig, flag_save)

    plt.show(block=True)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
