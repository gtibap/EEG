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

from autoreject import AutoReject

from class_read_bdf import EEG_components

# from mne.viz import set_browser_backend
# set_browser_backend("qt")

def labels_activity(ax, pos_y):
    
    ax.axvline(x = 4, color = 'tab:gray', alpha=0.5)
    ax.axvline(x = 8, color = 'tab:gray', alpha=0.5)
    ax.axvline(x = 13,color = 'tab:gray', alpha=0.5)
    
    ax.annotate('Delta', xy=(0.5, pos_y),
                    color='blue',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                    )
    ax.annotate('Theta', xy=(4.3, pos_y),
                    color='blue',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                    )
    ax.annotate('Alpha', xy=(8.7, pos_y),
                    color='blue',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                    )
    ax.annotate('Beta', xy=(15, pos_y),
                    color='blue',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                    )
    return ax


def main(args):
    
    print(f'arg {args[1]}')
    print(f'arg {args[2]}')
    print(f'arg {args[3]}')
    # print(f'arg {args[2]}')
    # print(f'arg {args[3]}')
    # print(f'arg {args[4]}')
    
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

    my_annot = mne.Annotations(
    onset=[5.31, 64.88, 159.946, 221.796, 335.634, 395.476, 468.664, 531.538,],  # in seconds
    duration=[59.57, 57.844, 61.85, 59.698, 59.842, 61.24, 62.874, 60.574, ],  # in seconds, too
    description=['a_ce', 'a_oe', 'a_ce', 'a_oe', 'b_ce', 'b_oe', 'b_ce', 'b_oe', ],
    )
    print(my_annot)

    raw_data.set_annotations(my_annot)
    print(raw_data.annotations)

    # convert meas_date (a tuple of seconds, microseconds) into a float:
    meas_date = raw_data.info["meas_date"]
    orig_time = raw_data.annotations.orig_time
    print(meas_date == orig_time)

    # fig = raw_data.plot(start=2, duration=6)

    # arr_x = [x.total_seconds() for x in df_markersTime]
    # print(f'total seconds:\n{df_markersTime.total_seconds()}')
    # print(f'arr_x: {arr_x}')

    # print(f'Decribe:\n{raw_data.describe}')
    # print(f"{raw_data.info['dig']}")
    # print(f"{raw_data.info['ch_names']}")
    # print(f"{mne.events_from_annotations(raw_data)}")


    ## choosing channels whose names start with E
    ch_names = raw_data.info['ch_names']
    ch_with_E = [x for x in ch_names if x.startswith('E')]
    # print(f'ch_with_E: {ch_with_E}')

    ## plot raw data
    # raw_data.plot()

    ## plot power spectral density
    start_time=0
    end_time=500
    # raw_data.compute_psd(picks=ch_with_E, tmin=start_time, tmax=end_time, fmax=5).plot()

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
    plt.show()

    for ann in raw_ica.annotations:
        descr = ann["description"]
        start = ann["onset"]
        end = ann["onset"] + ann["duration"]
        print(f"interactive '{descr}' goes from {start} to {end}")

    interactive_annot = raw_ica.annotations
    ## annotations
    raw_ica.set_annotations(interactive_annot)
    fig = raw_ica.plot(start=0, duration=120, scalings=scale_dict, block=True)
    # fig.fake_keypress("a")
    plt.show()
    raw_ica.annotations.save(path+"events/saved-annotations.csv", overwrite=True)

    # ###############################
    # ## ICA components
    # # Break raw data into 1 s epochs
    # tstep = 1.0
    # events_ica = mne.make_fixed_length_events(raw_ica, duration=tstep)
    # epochs_ica = mne.Epochs(raw_ica, events_ica,
    #                         tmin=0.0, tmax=tstep,
    #                         baseline=None,
    #                         preload=True)
    
    # print(f'events_ica: {events_ica}')
    # print(f'epochs_ica: {epochs_ica}')
    # ###############################

    # #################
    # ## autoreject ###
    # ar = AutoReject(n_interpolate=[1, 2, 4],
    #                 random_state=42,
    #                 picks=mne.pick_types(epochs_ica.info, 
    #                                     eeg=True,
    #                                     eog=False
    #                                     ),
    #                 n_jobs=-1, 
    #                 verbose=False
    #                 )

    # ar.fit(epochs_ica)

    # reject_log = ar.get_reject_log(epochs_ica)

    # # fig, ax = plt.subplots(figsize=[15, 5])
    # # reject_log.plot('horizontal', ax=ax, aspect='auto')
    # ## autoreject ###
    # #################

    # ################
    # ## ICA ##
    # # ICA parameters
    # random_state = 42   # ensures ICA is reproducible each time it's run
    # ica_n_components = .99     # Specify n_components as a decimal to set % explained variance

    # # Fit ICA
    # ica = mne.preprocessing.ICA(n_components=ica_n_components,
    #                             random_state=random_state,
    #                             )
    # ica.fit(epochs_ica[~reject_log.bad_epochs], decim=3)

    # ica.plot_components()
    # ## ICA ##
    # ################
    
    # ################
    # ## identify components related to eyes movements and blinking
    # ica.exclude = []
    # num_excl = 0
    # max_ic = 2
    # z_thresh = 3.5
    # z_step = .05

    # while num_excl < max_ic:
    #     eog_indices, eog_scores = ica.find_bads_eog(epochs_ica,
    #                                                 ch_name=['E8','E9','E14','E17','E21','E22','E25',], 
    #                                                 threshold=z_thresh
    #                                                 )
    #     num_excl = len(eog_indices)
    #     z_thresh -= z_step # won't impact things if num_excl is ≥ n_max_eog
    #     print(f'num_excl, z_thresh: {num_excl, z_thresh}')

    # # assign the bad EOG components to the ICA.exclude attribute so they can be removed later
    # ica.exclude = eog_indices

    # print('Final z threshold = ' + str(round(z_thresh, 2)))
    # ## plot components' scores
    # ica.plot_scores(eog_scores)
    # ## identify components related to eyes movements and blinking
    # # #######################

    # #######################
    # ## save ica components
    # p_id = 'out'
    # ica.save(path + p_id + '-ica.fif', 
    #     overwrite=True)
    # ## save ica components
    # #######################
    

    


    ## plot filtered data
    # raw_data.plot(start=15, duration=5)
    # raw_filt.plot(start=60, duration=60)

    # sfreq = raw_data.info['sfreq']
    # meas_date = raw_data.info['meas_date']
    
    # start_eeg = meas_date.hour*3600 + meas_date.minute*60 + meas_date.second
    # print(f'start eeg: {start_eeg} s')
    

    # raw_data.plot_sensors(show_names=True)
    # raw_filt.plot()
    # mne.io.Raw.plot_sensors(raw_data, show_names=True)
    # raw_data.compute_psd().plot()
    # raw_filt.compute_psd().plot()
    plt.show()
    

    return 0
    
    
    # data_dict = raw_data.__dict__
    # print(data_dict)
    # raw_dict  = data_dict["_raw_extras"][0]
    # print(f'extras: {raw_dict}')
    
    # print(type(data_dict))
    # print(raw_dict["ch_names"])
    # print(raw_dict["ch_names"][64])
    # print(raw_dict["ch_names"][64+63])
    # print(type())
    
    
    # raw_filt.plot_psd(picks=['Fp1'], fmax=10);
    # raw_filt.plot_psd(picks=['Fp1'], fmax=10);
    
    # raw_data.plot(picks=['Fp1'])
    # raw_filt.plot(picks=['Fp1'])
    # raw_data.plot_psd(picks=['Fp1'], fmax=400);
    
    
    # data_2 = raw_data.get_data(picks=['Fp1'])
    # print(data_2.shape)
    # data_2 = raw_data.get_data(picks=['Fp1','O2'],tmin=0,tmax=60*5)
    # data_2 = raw_data.get_data(picks=['OCU3','ECG5'],tmin=0,tmax=60*5)
    # plt.plot(data_2[1])
    
    offset=-0.001

    label_signals=['P3','P4','O1','O2']
    signals = raw_filt.get_data(picks=label_signals)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,5), squeeze=False)
    ax = ax.reshape(-1)
    # ax[0].plot(signals[1]+offset, label='Fpz')
    # ax[0].plot(signals[0], label='Cz')
    ax[0].plot(signals[0]+offset*0, label=label_signals[0])
    ax[0].plot(signals[1]+offset*1, label=label_signals[1])
    ax[0].plot(signals[2]+offset*2, label=label_signals[2])
    ax[0].plot(signals[3]+offset*3, label=label_signals[3])
    
    # print()
    for ta,section,action in zip(df_events['time'].tolist(), df_events['section'].tolist(), df_events['action'].tolist()):
        ts = int((ta - start_eeg)*sfreq)
        ax[0].axvline(x = ts, color = 'b', alpha=0.5)
        print(f'{ts}, {section}, {action}')
    
    fig.canvas.draw()
    # ax[0].set_ylim([-0.0004, 0.0001])
    ax[0].set_ylim([-0.004, 0.001])
    
    ## Subject 1
    # ax[0].set_xlim([195000, 326000])
    # pos_xlabel1=222000
    # pos_xlabel2=285000

    ## Subject 2
    # ax[0].set_xlim([105000, 175000])
    # pos_xlabel1=120000
    # pos_xlabel2=155000    
    
    
    ## Subject 3
    # ax[0].set_xlim([65000, 140000])
    # pos_xlabel1=80500
    # pos_xlabel2=115500
    
    ## Subject 4
    ax[0].set_xlim([70000, 135000])
    pos_xlabel1=82000
    pos_xlabel2=115000
    
    x_labels = [item.get_text() for item in ax[0].get_xticklabels()]
    x_labels = (np.array(x_labels).astype(int)/(sfreq)).astype(int)
    print(f'x_labels {x_labels}')
    
    ax[0].set_xticklabels(x_labels)
    
    ax[0].set_yticks([offset*0,offset*1,offset*2,offset*3])
    ax[0].set_yticklabels(label_signals)
    
    # ax[0].set_ylabel('amplitude [uV]')
    ax[0].set_xlabel(f'time (s)')
    ax[0].annotate('eyes-closed', xy=(pos_xlabel1, -offset*0.7),
                    color='blue',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                    )
    ax[0].annotate('eyes-opened', xy=(pos_xlabel2, -offset*0.7),
                    color='blue',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                    )
    # plt.legend(loc='lower right')
    ax[0].set_title(f'{title}')
    
    plt.savefig(f'figures/{filename_out}.png', bbox_inches='tight')
    # plt.suptitle(f'{title}')
    # raw_data.set_montage('standard_1005')
    # raw_data.plot_sensors()
    
    # print(data_2.shape)
    # print(data_2)
    
    time_ce_a=np.array(time_ce_a)
    time_ce_b=np.array(time_ce_b)
    
    time_ce_a = ((time_ce_a - start_eeg)*sfreq).astype(int)
    time_ce_b = ((time_ce_b - start_eeg)*sfreq).astype(int)
    
    print(f'{time_ce_a}')
    print(f'{time_ce_b}')
    ## frequency components signals segments closed eyes and open eyes
    obj_signals = EEG_components(signals, sfreq)
    
    fig1, ax1 = plt.subplots(nrows=2, ncols=2, figsize=(10, 7),sharex=True, sharey=True)
    ax1 = ax1.reshape(-1)
    
     # ## vertical lines
    # ax1[0].axvline(x = 4, color = 'tab:gray', alpha=0.5)
    # ax1[0].axvline(x = 8, color = 'tab:gray', alpha=0.5)
    # ax1[0].axvline(x = 13,color = 'tab:gray', alpha=0.5)
    
    # ax1[1].axvline(x = 4, color = 'tab:gray', alpha=0.5)
    # ax1[1].axvline(x = 8, color = 'tab:gray', alpha=0.5)
    # ax1[1].axvline(x = 13,color = 'tab:gray', alpha=0.5)
    
    # ax1[2].axvline(x = 4, color = 'tab:gray', alpha=0.5)
    # ax1[2].axvline(x = 8, color = 'tab:gray', alpha=0.5)
    # ax1[2].axvline(x = 13,color = 'tab:gray', alpha=0.5)
    
    # ax1[3].axvline(x = 4, color = 'tab:gray', alpha=0.5)
    # ax1[3].axvline(x = 8, color = 'tab:gray', alpha=0.5)
    # ax1[3].axvline(x = 13,color = 'tab:gray', alpha=0.5)
    
    
    # #################
    # ## resting state
    
    # arr = np.array(time_ce_a[0])
    # arr[arr<0]=0
    # ax1 = obj_signals.freq_components(arr, ax1,'0')
    
    # arr = np.array(time_ce_a[1])
    # arr[arr<0]=0
    # ax1 = obj_signals.freq_components(arr, ax1,'1')
    
    # arr = np.array(time_ce_a[2])
    # arr[arr<0]=0
    # ax1 = obj_signals.freq_components(arr, ax1,'2')
    
    # ## resting state
    # #################
    
    #################
    ## activity-based therapy
    
    arr = np.array(time_ce_b[0])
    arr[arr<0]=0
    ax1 = obj_signals.freq_components(arr, ax1,'0')
    
    arr = np.array(time_ce_b[1])
    arr[arr<0]=0
    ax1 = obj_signals.freq_components(arr, ax1,'1')
    
    arr = np.array(time_ce_b[2])
    arr[arr<0]=0
    ax1 = obj_signals.freq_components(arr, ax1,'2')
    
    ## activity-based therapy
    #################
    
    
    # pos_y=4.0e-6
    pos_y=1.6e-5
    # label_signals=['P3','P4','O1','O2']
    ax1[0] = labels_activity(ax1[0], pos_y)
    ax1[1] = labels_activity(ax1[1], pos_y)
    ax1[2] = labels_activity(ax1[2], pos_y)
    ax1[3] = labels_activity(ax1[3], pos_y)
                    
    ax1[0].set_title(label_signals[0])
    ax1[1].set_title(label_signals[1])
    ax1[2].set_title(label_signals[2])
    ax1[3].set_title(label_signals[3])
    
    ax1[2].set_xlabel('frequency (Hz)')
    ax1[3].set_xlabel('frequency (Hz)')
    
    ax1[0].set_ylabel('PSD (V**2/Hz)')
    ax1[2].set_ylabel('PSD (V**2/Hz)')
    
    plt.suptitle(f'{title}\nFrequency components\neyes-closed')
    
    # plt.savefig(f'figures/{title}_freq.png', bbox_inches='tight')
    plt.savefig(f'figures/{filename_out}_freq.png', bbox_inches='tight')

    # plt.legend(loc='lower right')
    
    
    # ax1 = obj_signals.freq_components(arr, ax1)
    
    # obj_signals.freq_components(time_ce_a[3])
    # obj_signals.freq_components(time_ce_b[2])
    
    
    
    # print(raw_data['cal'])
    # raw_data.plot()
    # data = raw_data.get_data()
    # print(type(data))
    # print(data.shape)
    # plt.plot(data[133])
    
    # raw_data.plot_psd(fmax=100)
    
    plt.show()
    
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
