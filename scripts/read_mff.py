#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import mne
mne.set_log_level('error')

import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import pytz
import datetime
import pathlib

######################
## Description
# Interactive annotations editing that means manual adjusting of start and ending of each label, which includes:
# * a_opened_eyes
# * a_closed_eyes
# * b_opened_eyes
# * b_closed_eyes
# Segments without label will be labeling as BAD_ to exclude them from posterior analyses
######################
######################
## labels during the EEG signals acquisition should correspond to the following
labels_ref = ['a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes',]
######################

def main(args):
    
    print(f'arg {args[1]}') ## folder location
    print(f'arg {args[2]}') ## file name .mff
    print(f'arg {args[3]}') ## file name annotations .csv
    
    path=args[1]
    fn_in=args[2]
    fn_csv=args[3]

    # function to return the file extension
    file_extension = pathlib.Path(fn_in).suffix
    print("File Extension: ", file_extension)

    ##########################
    ## read raw data
    if file_extension == '.mff':
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
    elif file_extension == '.bdf':
        raw_data = mne.io.read_raw_bdf(path + fn_in, preload=True)
    else:
        return 0
    ## read raw data
    ##########################
    # printing basic information from data
    print(f'Info:\n{raw_data.info}')
    first_sample = raw_data.first_samp / raw_data.info["sfreq"]
    last_sample = raw_data.last_samp / raw_data.info["sfreq"]
    print(f"first and last samples:\n{first_sample, last_sample} in seconds")
    ########################
    ## read raw annotations
    if fn_csv.startswith('raw'):
        ## open csv markers file (annotations that were saved with an online application that transforms xml to csv. The xml file is found inside the file (folder) with the extension .mff)
        print(f'CSV file: {fn_csv}')
        df = pd.read_csv(path + fn_csv)
        print(f'markers:\n{df}')
        # transform column data from type string to type datetime 
        df['beginTime'] = pd.to_datetime(df['beginTime'], utc=True)
        # print(f"type datetime:\n{df['beginTime']}")
        ## open csv markers file
        # subtract initial recording time from each markers time
        markersTime = df['beginTime'] - raw_data.info['meas_date']
        print(f'markers time:\n{markersTime}')
        serie_sec = markersTime.dt.total_seconds()
        print(f'markers sec:\n{serie_sec}')
        ## adding a column to the dataframe
        df['onset']=serie_sec
        print(f'annotations:\n{df}')
    
        # ## extract selected data from dataframe
        # labels = labels_ref
        
        arr_onset=np.array([])
        arr_label=np.array([])

        for d in labels_ref:
            arr_onset = np.append(arr_onset, df[df['label'] == d]['onset'].to_numpy())
            arr_label = np.append(arr_label, df[df['label'] == d]['label'].to_numpy())
            print(f'onset+label: {arr_onset}, {arr_label}')

        ## we expect every segment last 60 seconds
        arr_durat = len(arr_onset)*[60]
        ## latter on we manually adjust segments to avoid overlapping
        ######
        ## for BAD_ segments 
        arr_onset_bad = df[df['label'] == 'BAD_']['onset'].to_numpy()
        arr_label_bad = df[df['label'] == 'BAD_']['label'].to_numpy()

        ## we define every BAD segment last 2 seconds
        arr_durat_bad = len(arr_onset_bad)*[2]

        arr_onset = np.append(arr_onset, arr_onset_bad)
        arr_label = np.append(arr_label, arr_label_bad)
        arr_durat = np.append(arr_durat, arr_durat_bad)

        ## for BAD_ segments 
        ######
        print (f'arr_onset:\n{arr_onset}')
        print (f'arr_label:\n{arr_label}')
        print (f'arr_durat:\n{arr_durat}')

        my_annot = mne.Annotations(
        onset=arr_onset,  # in seconds
        duration=arr_durat,  # in seconds, too
        description=arr_label,
        )
    ###################    
    ## read edited annotations
    else:
        my_annot = mne.read_annotations(path + fn_csv)

    print(my_annot)

    ############################
    ############################
    ## adding original annotations to raw data
    raw_data.set_annotations(my_annot)
    # print(raw_data.annotations)
    ############################
    ############################
    ## signals visualization and
    ## interactive annotations editing avoiding overlaping 
    ## visualization scale
    scale_dict = dict(mag=1e-12, grad=4e-11, eeg=200e-6, eog=150e-6, ecg=5e-4, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)

    fig = raw_data.plot(start=0, duration=120, scalings=scale_dict, highpass=0.1, lowpass=100.0, block=True)
    ############################
    ############################
    ## regions that do not have labels will be labeled as BAD_
    ## BAD_ labeled segments are excluded of posterior analyses

    arr_ini = np.array([])
    arr_end = np.array([])
    arr_des = np.array([])
    label_bad = 'BAD_'

    ## first_sample in seconds
    arr_ini= np.append(arr_ini, first_sample)

    for ann in raw_data.annotations:
        
        descr = ann["description"]
        start = ann["onset"]
        end = ann["onset"] + ann["duration"]

        # print(f"annotations: {start, end, descr}")

        if descr != 'BAD_':
            ## region labeled as BAD_
            arr_end = np.append(arr_end, start)
            arr_des = np.append(arr_des, label_bad)
            ## region labeled as != BAD_
            arr_ini = np.append(arr_ini, start)
            arr_end = np.append(arr_end, end)
            arr_des = np.append(arr_des, descr)
            ## initial value of a BAD_ labeled region
            arr_ini = np.append(arr_ini, end)
        else:
            pass
    ## last region is labeled as BAD_
    arr_end = np.append(arr_end, last_sample)
    arr_des = np.append(arr_des, label_bad)

    ## annotations including BAD_ labels
    my_annot = mne.Annotations(
    onset=arr_ini,  # in seconds
    duration=arr_end - arr_ini,  # in seconds, too
    description=arr_des,    # labels
    )
    print(my_annot)
    
    ## updating labels of original data
    raw_data.set_annotations(my_annot)
    print(raw_data.annotations)

    ## data visualization
    fig = raw_data.plot(start=0, duration=120, scalings=scale_dict, highpass=1.0, lowpass=30.0, block=True)

    # save annotations
    flag = int(input("Save annotations ? (1 for yes, 0 for non)"))
    if flag == 1:
        raw_data.annotations.save(path+"annotations.csv", overwrite=True)
        raw_data.annotations.save(path+"annotations.fif", overwrite=True)
        raw_data.annotations.save(path+"annotations.txt", overwrite=True)
    else:
        pass

    plt.show()

    return 0
    

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
