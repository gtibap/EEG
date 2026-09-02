#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mne
mne.set_log_level('error')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import sys

from info_participants import subject_dict

##############
# global variables
excluded_channels = ['E8','E14','E15','E17','E21','E25','E38','E39','E43','E44','E45','E48','E49','E56','E57','E63','E68','E73','E81','E88','E94','E99','E100','E107','E108','E113','E114','E115','E119','E120','E121','E125','E126','E127','E128']


######################
## Description
# Interactive annotations editing that means manual adjusting of start and ending of each label, which includes:
# * a_opened_eyes
# * a_closed_eyes
# * b_opened_eyes
# * b_closed_eyes
######################

def main(args):

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

    ## patient info including file name of raw (data), age, sex, ais, nli, days (after trauma), 
    data_pt = subject_dict[subject]
    print(f"data pt: {data_pt}")

    raw_filename = data_pt['raw'][session]

    ##########################
    ## read raw data 
    raw_data = mne.io.read_raw_egi(path_session + raw_filename, preload=False)
    # acquisition_system = 'geodesic'

    ##########################
    ## raw data recording date
    print(f"\nmeasuring date: {raw_data.info['meas_date']}\n")

    ########################
    ## read raw annotations
    try:
        ## open annotations annotations.fif
        my_annot = mne.read_annotations(path_session + 'annotations.fif')
    except:
        ## open annotations raw_annotations.csv
        ## open csv markers file (annotations that were saved with an online application that transforms xml to csv. The xml file is found inside the file (folder) with the extension .mff)
        df = pd.read_csv(path_session + 'raw_annotations.csv')
        print(f'markers:\n{df}')
        # transform column data from type string to type datetime 
        df['beginTime'] = pd.to_datetime(df['beginTime'], utc=True)
        ## open csv markers file
        # subtract initial recording time from each markers time
        markersTime = df['beginTime'] - raw_data.info['meas_date']
        print(f'markers time:\n{markersTime}')
        serie_sec = markersTime.dt.total_seconds()
        print(f'markers sec:\n{serie_sec}')
        ## adding a column to the dataframe
        df['onset']=serie_sec

        my_annot = mne.Annotations(
        onset=df['onset'].values,  # in seconds
        description=df['label'].values,
        duration=len(df['duration'].values)*[2],  # 2 seconds for each label
        )

    ############################
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

    ########################################################################
    ## data visualization
    scale_dict = dict(eeg=100e-6, ecg=400e-6,)

    ## data visualization. Interactive annotation editing
    mne.viz.plot_raw(raw_data, picks=['eeg','ecg'], start=0, duration=240, n_channels=36, scalings=scale_dict, highpass=0.5, lowpass=45.0, title=f"EEG time series -- interactive annotation editing", block=True)

    # save annotations
    flag = int(input("Save annotations ? (1-yes, 0-non) "))
    if flag:
        print(f"saving annotations...")
        raw_data.annotations.save(path_session+"annotations.fif", overwrite=True)
    else:
        pass

    return 0

#########################
if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
