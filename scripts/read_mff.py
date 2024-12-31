#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  read_bdf.py
#  

import mne
mne.set_log_level('error')

import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import pytz
import datetime


def main(args):
    
    print(f'arg {args[1]}')
    print(f'arg {args[2]}')
    print(f'arg {args[3]}')
    
    path=args[1]
    fn_in=args[2]
    fn_csv=args[3]

    ########################
    ## open csv markers file
    print(f'CSV file: {fn_csv}')
    df = pd.read_csv(path+fn_csv)
    print(f'markers:\n{df}')
    df['beginTime'] = pd.to_datetime(df['beginTime'])
    ## open csv markers file
    ########################

    # title=args[3]
    # filename_out=args[4]

    ##########################
    ## read raw data
    # raw_data = mne.io.read_raw_bdf(args[1], preload=True)
    raw_data = mne.io.read_raw_egi(path+fn_in, preload=True)
    ## read raw data
    ##########################

    print(f'Info:\n{raw_data.info}')
    print(f"date: {raw_data.info['meas_date']}")
    time_of_first_sample = raw_data.first_samp / raw_data.info["sfreq"]
    print(f"first sample: {time_of_first_sample}")
    print(f'first and last samples: {raw_data.first_samp/ raw_data.info["sfreq"], raw_data.last_samp/ raw_data.info["sfreq"]}')

    date_time = raw_data.info['meas_date']
    now_gmt = date_time.astimezone(pytz.timezone('Europe/London'))
    print(f'now_gmt: {now_gmt}')
    print(f'time: {now_gmt.hour, now_gmt.minute, now_gmt.second, now_gmt.microsecond, now_gmt.tzname()}')

    #########
    # subtract initial recording time from each markers time
    # arr_x = [(x) for x in df['beginTime']]
    df_markersTime = df['beginTime'] - date_time
    print(f'markers time:\n{df_markersTime}')
    serie_sec = df_markersTime.dt.total_seconds()
    
    df_an = pd.DataFrame()
    df_an['onset']=serie_sec
    df_an['label']=df['label']

    print(f'annotations:\n{df_an}')

    # my_annot = mne.Annotations(
    # onset=[5.31, 64.88, 159.946, 221.796, 335.634, 395.476, 468.664, 531.538,],  # in seconds
    # duration=[59.57, 57.844, 61.85, 59.698, 59.842, 61.24, 62.874, 60.574, ],  # in seconds, too
    # description=['a_ce', 'a_oe', 'a_ce', 'a_oe', 'b_ce', 'b_oe', 'b_ce', 'b_oe', ],
    # )
    # print(my_annot)

    labels = ['a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes',]
    # print(f"closed eyes:\n{df_an[df_an['label']==labels[0]]}")
    # print(f"closed eyes:\n{df_an[df_an['label']==labels[0]]['onset'].to_numpy()}")
    # print(f"closed eyes:\n{df_an[df_an['label']==labels[0]]['label'].to_numpy()}")

    arr_onset=np.array([])
    arr_label=np.array([])

    for d in labels:
        arr_onset = np.append(arr_onset, df_an[df_an['label'] == d]['onset'].to_numpy())
        arr_label = np.append(arr_label, df_an[df_an['label'] == d]['label'].to_numpy())

    ## we expect every segment last 60 seconds
    arr_durat = len(arr_onset)*[58]

    ######################
    ## for BAD_ segments 
    arr_onset_bad = df_an[df_an['label'] == 'BAD_']['onset'].to_numpy()
    arr_label_bad = df_an[df_an['label'] == 'BAD_']['label'].to_numpy()

    ## we define every BAD segment last 2 seconds
    arr_durat_bad = len(arr_onset_bad)*[2]

    arr_onset = np.append(arr_onset, arr_onset_bad)
    arr_label = np.append(arr_label, arr_label_bad)
    arr_durat = np.append(arr_durat, arr_durat_bad)

    ## for BAD_ segments 
    ######################

    print (f'arr_onset:\n{arr_onset}')
    print (f'arr_label:\n{arr_label}')
    print (f'arr_durat:\n{arr_durat}')


    my_annot = mne.Annotations(
    onset=arr_onset,  # in seconds
    duration=arr_durat,  # in seconds, too
    description=arr_label,
    )
    print(my_annot)


    raw_data.set_annotations(my_annot)
    print(raw_data.annotations)

    # convert meas_date (a tuple of seconds, microseconds) into a float:
    meas_date = raw_data.info["meas_date"]
    orig_time = raw_data.annotations.orig_time
    print(meas_date == orig_time)


    ## choosing channels whose names start with E
    ch_names = raw_data.info['ch_names']
    ch_with_E = [x for x in ch_names if x.startswith('E')]
    # print(f'ch_with_E: {ch_with_E}')

    ## plot raw data
    # raw_data.plot()

    ## band pass filter settings
    low_cut = 0.1
    ica_low_cut = 1.0
    hi_cut  = 30    

    # band pass filter
    # raw_filt = raw_data.copy().filter(l_freq=low_cut, h_freq=hi_cut)
    raw_ica = raw_data.copy().filter(l_freq=ica_low_cut, h_freq=hi_cut)

    ## annotations
    raw_ica.set_annotations(my_annot)

    print(raw_ica.annotations)
    for ann in raw_ica.annotations:
        descr = ann["description"]
        start = ann["onset"]
        end = ann["onset"] + ann["duration"]
        print(f"original '{descr}' goes from {start} to {end}")

    # raw_filt.compute_psd(picks=ch_with_E, tmin=start_time, tmax=end_time, fmax=5).plot()


    # raw_filt.plot()
    scale_dict = dict(mag=1e-12, grad=4e-11, eeg=200e-6, eog=150e-6, ecg=5e-4, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)

    fig = raw_ica.plot(start=0, duration=120, scalings=scale_dict, block=True)
    # fig.fake_keypress("a")
    # plt.show()

    for ann in raw_ica.annotations:
        descr = ann["description"]
        start = ann["onset"]
        end = ann["onset"] + ann["duration"]
        print(f"interactive '{descr}' goes from {start} to {end}")

    ## segments of EEG signals without labels (description) are considered out of analysis; therefore, those segments will be labeled as BAD_
    ini_time = raw_ica.first_samp/ raw_data.info["sfreq"]
    end_time = raw_ica.last_samp/ raw_data.info["sfreq"]
    print(f'time start and end: {ini_time, end_time}')

    arr_ini = np.array([])
    arr_end = np.array([])
    arr_des = np.array([])
    label_bad = 'BAD_'

    arr_ini= np.append(arr_ini, ini_time)

    for ann in raw_ica.annotations:
        
        descr = ann["description"]
        start = ann["onset"]
        end = ann["onset"] + ann["duration"]

        if descr != 'BAD_':
            arr_end = np.append(arr_end, start)
            arr_des= np.append(arr_des, label_bad)

            arr_ini = np.append(arr_ini, start)
            arr_end = np.append(arr_end, end)
            arr_des = np.append(arr_des, descr)

            arr_ini = np.append(arr_ini, end)
        else:
            pass

    arr_end = np.append(arr_end, end_time)
    arr_des= np.append(arr_des, label_bad)

    df_an_all = pd.DataFrame()
    df_an_all['start']=arr_ini
    df_an_all['end']=arr_end
    df_an_all['label']=arr_des

    print(f'annotations:\n{df_an_all}')

    my_annot = mne.Annotations(
    onset=arr_ini,  # in seconds
    duration=arr_end - arr_ini,  # in seconds, too
    description=arr_des,
    )
    print(my_annot)

    # ######################
    # ## for BAD_ segments 
    # arr_onset_bad = df_an[df_an['label'] == 'BAD_']['onset'].to_numpy()
    # arr_label_bad = df_an[df_an['label'] == 'BAD_']['label'].to_numpy()

    # ## we define every BAD segment last 2 seconds
    # arr_durat_bad = len(arr_onset_bad)*[2]
    
    # my_bad_annot = mne.Annotations(
    # onset=arr_onset_bad,  # in seconds
    # duration=arr_durat_bad,  # in seconds, too
    # description=arr_label_bad,
    # )
    # print(my_bad_annot)
    # ## for BAD_ segments 
    # ######################
    # raw_ica.set_annotations(my_annot+my_bad_annot)

    raw_ica.set_annotations(my_annot)
    print(raw_ica.annotations)

    fig = raw_ica.plot(start=0, duration=120, scalings=scale_dict, block=True)
        

    plt.show()

    return 0
    

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
