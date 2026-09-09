#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import mne
mne.set_log_level('error')
from mne.preprocessing import ICA
from mne.time_frequency import psd_array_multitaper

import os
from pathlib import Path
import time
import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from channels_tfr import selected_channels
## include modules from another directory
# sys.path.insert(0, '../../scripts')
from info_participants import subject_dict

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

    # checking if the directory figures
    # exist or not.
    if not os.path.exists(path_fig_boxplot):
        # if the figures directory is not present 
        # then create it.
        os.makedirs(path_fig_boxplot)
    
    if not os.path.exists(path_fig_fooof):
        # if the figures directory is not present 
        # then create it.
        os.makedirs(path_fig_fooof)

    if not os.path.exists(path_fig_psd):
        # if the figures directory is not present 
        # then create it.
        os.makedirs(path_fig_psd)

    if not os.path.exists(path_csv_files):
        # if the figures directory is not present 
        # then create it.
        os.makedirs(path_csv_files)
    
    Path(path_prep).mkdir(parents=True, exist_ok=True)
            
    return 0

############
def group_segments_by_label(raw_copy, label):

    annot_raw = raw_copy.annotations
    time_offset = raw_copy.first_samp / sampling_rate  ## in seconds

    arr_onset=np.array([])
    arr_durat=np.array([])
    arr_label=[]
    label_list=[]

    for ann in annot_raw:
        label_list.append(ann['description'])
        # print(f"onset, duration, description: {ann['onset'], ann['duration'], ann['description']}")
        ## extract data only of selected label
        if (label in ann['description']):
            arr_onset = np.append(arr_onset, ann['onset']-time_offset)
            arr_durat = np.append(arr_durat, ann['duration'])
            arr_label.append(ann['description'])

    ## create new_annot only if at least one annotation named "label" were found
    if len(arr_label) > 0:

        new_annot = mne.Annotations(
        onset=arr_onset,  # in seconds
        duration=arr_durat,  # in seconds, too
        description=arr_label, # label description
        )

        ## copying annotations to take as a reference
        copy_annot = new_annot.copy()

        # first annotation bad segments from time=0s until start of the first annotation
        onset=0
        duration= arr_onset[0]
        description='bad_seg'

        new_annot.append(onset, duration, description)

        # print(f"onset, duration, description:")
        for ann_a, ann_b in zip(copy_annot, copy_annot[1:]):
            # print(f"{ann['onset'], ann['duration'], ann['description']}")
            onset = ann_a['onset'] + ann_a['duration']
            duration =  ann_b['onset'] - onset
            description = 'bad_seg'
            new_annot.append(onset, duration, description)

        # last annotation: from the end of the last annotation until the end of the recording
        onset = arr_onset[-1] + arr_durat[-1]
        duration = raw_copy.duration - onset
        description = 'bad_seg'
        new_annot.append(onset, duration, description)

    else:
        print(f"{label} annotations not found")
        ## bad_seg from the beginning until the end
        new_annot = mne.Annotations(
        onset=0,  # in seconds
        duration=raw_copy.duration,  # in seconds, too
        description='bad_seg', # label description
        )

    ## set annotations of selected label + bad_seg
    raw_copy.set_annotations(new_annot)

    # ##############################
    # ## data visualization
    # scale_dict = dict(eeg=100e-6, ecg=400e-6,)
    
    # mne.viz.plot_raw(raw_copy, picks=['eeg','ecg'], start=0, duration=240, n_channels=36, scalings=scale_dict, highpass=0.5, lowpass=45.0, title=f"{label}", block=True)
    
    return raw_copy

############################
def interactive_bad_epochs_bad_channels_selection(epochs, label, path_prep):
    ## remove previously selected bad epochs and marking of bad channels
    ## id's list of generated epochs
    epochs_list = epochs.selection.astype(int)
    print(f"epochs first_list: {epochs_list}")

    ## removing bad epochs and bad channels if they were already selected in a previous iteration
    try:
        #### drop bad epochs
        # Read list of bad epochs and bad channels from file
        with open(f"{path_prep}{label}_bad_epochs.json", "r") as f:
            data = json.load(f)
        bad_epochs_list = data['bad_epochs']
        bad_channels_list = data['bad_channels']
        print(f"bad_epochs_list:\n{bad_epochs_list}")
        print(f"bad_channels_list:\n{bad_channels_list}")

        ## remove (drop) bad epochs
        drop_idx_list = []
        for id_bad in bad_epochs_list:
            ## find id of bad epoch
            drop_idx = np.where(epochs_list == id_bad)[0]
            drop_idx_list = np.concatenate((drop_idx_list, drop_idx), axis=None)
        # print(f"ids_bad_list: {drop_idx_list}")
        ## remove bad epochs by indexes
        if len(drop_idx_list) > 0:
            epochs.drop(drop_idx_list.astype(int))
        else:
            pass
        ## marking bad channels
        epochs.info['bads'] = bad_channels_list

    except:
        bad_epochs_list = []
        # print(f"{label}_bad_epochs.json: Problem trying to load bad segments and bad channels.")

    ## update first list
    first_list = epochs.selection.astype(int)
    # print(f"first_list: {first_list}")

    ## plot of PSD to help identify bad channels
    ch_exclude_list = ['VREF'] 
    epochs.plot_psd(exclude=ch_exclude_list, fmax=65)

    ## interactive selection of bad epochs and bad channels
    epochs.plot(n_epochs=36, events=True, block=True, n_channels=24, scalings=scale_dict, title=f"{label} : Epochs",)

    list_channel_bads = epochs.info['bads']
    # print(f"second list_bads: {list_channel_bads}")

    second_list = epochs.selection.astype(int)
    # print(f"second_list: {second_list}")

    ## found in the second list the epochs ids that were eliminated from the first list
    bad_epochs_ids = np.array([x for x in first_list if not (x in second_list)]).astype(int)

    # print(f"bad epochs ids: {bad_epochs_ids}")
    bad_epochs_list = np.concatenate((bad_epochs_list, bad_epochs_ids),axis=None).astype(int)

    ## save bad epochs ids and bad channels ids
    bad_epochs_dict={
            'sel_epochs': epochs_list.tolist(),
            'bad_epochs': bad_epochs_list.tolist(),
            'bad_channels' : list_channel_bads,
        }
    with open(f"{path_prep}{label}_bad_epochs.json", "w") as f:
        json.dump(bad_epochs_dict, f)

    return epochs


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

    ## create folders/directories if they do not exit yet
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

    ################################
    ## interactive selection of bad epochs and bad channels

    for label in label_list_ref:
        ## for each label segments are grouped and transformed into epochs
        print (f"searching for annotations for: {label}")
        ## copy raw data in order to separate segments by labels (annotations) and include bad segments
        raw_copy = raw_data.copy()
        ## raw_copy would includes only two labels: 'label' and 'bad_seg'
        raw_copy = group_segments_by_label(raw_copy, label)
        ########
        ## create events and epochs
        dt = 5 ## epoch duration in seconds
        print(f"Epochs size: {dt} seconds / each ")

        ## first, include a sequence of regular events to the raw data 
        new_events = mne.make_fixed_length_events(raw_copy, start=0, stop=None, duration=dt)
        # raw_copy.add_events(new_events, replace=True)

        ## second, use events to create epochs excluding bad segments
        epochs = mne.Epochs(raw_copy, new_events, tmin=0.0, tmax=dt, baseline=None, preload=True, reject=None, reject_by_annotation=True)
        print(f"Number of epochs:\n{len(epochs.selection)}")

        ######
        if len(epochs.selection) > 0:
            ## interactive selection of bad epochs and bad channels
            epochs = interactive_bad_epochs_bad_channels_selection(epochs, label, path_prep)

        else:
            print(f"{label}: Epochs not found")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
