#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import mne
mne.set_log_level('error')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pathlib
import time

# def onClick(event):
#     global pause
#     print(f'pause: {pause}')
#     pause = not(pause)


ani = 0
flag=False
images=[]
spectrum = []
data_spectrum = []
fig, ax = plt.subplots(1, 1, figsize=(5,5))
draw_image = []

# fig.canvas.mpl_connect('button_press_event', onClick)

def toggle_pause(event):
        global flag
        if flag==True:
            ani.resume()
        else:
            ani.pause()
        flag = not flag

#############################
## EEG filtering and signals prepocessing

def main(args):
    global spectrum, data_spectrum, fig, ax, ani, draw_image

    ## interactive mouse pause the image visualization
    fig.canvas.mpl_connect('button_press_event', toggle_pause)

    print(f'arg {args[1]}') ## folder location
    print(f'arg {args[2]}') ## subject = {0:Mme Chen, 1:Taha, 2:Carlie, 3:Iulia, 4:A. Caron}
    print(f'arg {args[3]}') ## ABT = {0:resting, 1:biking}
    
    path=args[1]
    subject= int(args[2])
    abt= int(args[3])

    t0=0
    t1=0

    #########################
    ## data subject selection
    
    ############################
    # Mme Chen
    if subject == 0:
        path = path + 'aug04_MsChen/'
        fn_in = 'eeg_test-p3-chen_s01.bdf'
        fn_csv = 'saved-annotations.csv'

        ## read raw data
        raw_data = mne.io.read_raw_bdf(path + fn_in, preload=True)
        ## select channels
        sel_ch = np.arange(64,128)
        raw_data.pick(sel_ch)
        ## rename channels
        maps_dict = {'C1-1':'C1', 'C2-1':'C2', 'C3-1':'C3', 'C4-1':'C4', 'C5-1':'C5', 'C6-1':'C6'}
        mne.rename_channels(raw_data.info, maps_dict)
        ## electrodes montage
        raw_data.set_montage("biosemi64")
        # fig = raw_data.plot_sensors(show_names=True, sphere='eeglab')
        
        ## resting closed eyes
        t0 = 198 
        t1 = 256

    ############################
    # Mr Taha
    elif subject == 1:
        path = path + 'oct06_Taha/'
        if abt == 0: # resting
            fn_in = 'eeg_taha_test_rest.bdf'
            fn_csv = 'annotations_rest.csv'
        else:
            fn_in = 'eeg_taha_test_velo.bdf'
            fn_csv = 'annotations_velo.csv'
        ## read raw data
        raw_data = mne.io.read_raw_bdf(path + fn_in, preload=True)
        ## select channels
        sel_ch = np.arange(64,128)
        raw_data.pick(sel_ch)
        ## electrodes montage
        raw_data.set_montage("biosemi64")
        # fig = raw_data.plot_sensors(show_names=True, sphere='eeglab')

        ## resting closed eyes
        t0 = 134 
        t1 = 260
    ############################
    # Mme Carlie
    elif subject == 2:
        path = path + 'apic_data/initial_testing/p01/'
        fn_in = 'APIC_TEST_CM_20241205_023522.mff'
        fn_csv = 'saved-annotations.csv'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        # raw_data.plot_sensors(show_names=True,)

        ## resting closed eyes
        t0 = 160 
        t1 = 220
    ############################
    # Mme Iulia
    elif subject == 3:
        path = path + 'apic_data/initial_testing/p02/'
        fn_in = 'APIC_TEST_IULIA_20241217_011900.mff'
        fn_csv = 'saved-annotations.csv'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        # fig = raw_data.plot_sensors(show_names=True,)
         ## resting closed eyes
        t0 = 178 
        t1 = 236
    ############################
    # Mr Andre Caron
    elif subject == 4:
        path = path + 'neuroplasticity/n_001/'
        fn_in = 'Neuro001_session1_20250113_111350.mff'
        fn_csv = 'annotations.csv'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        ## bad channels by visual inspection
        # raw_data.info["bads"] = ['E48','E119','E126','E127']
        # fig = raw_data.plot_sensors(show_names=True,)
        ## resting closed eyes
        # t0 = 15
        # t1 = 85
        segment = "closed eyes"
        # t0 = 244
        # t1 = 304
        ## resting opened eyes
        # segment = "opened eyes"
        # t0 = 130
        # t1 = 200
        ####
        t0 = 240
        t1 = 400
        print(f'section: {t0}s - {t1}s')
    ############################
    else:
        return 0

    #############################
    ## 2D location electrodes
    # fig = raw_data.plot_sensors(show_names=True,)
    #########################    
    
    #########################
    ## reduce data size for training purposes
    # raw_data.crop(tmax=120.0)  # raw.crop() always happens in-place
    ## reduce data size for training purposes
    #########################

    ##########################
    # printing basic information from data
    print(f'Info:\n{raw_data.info}')
    # printing basic information from data
    ############################

    ############################
    ## read annotations (.csv file)
    print(f'CSV file: {fn_csv}')
    my_annot = mne.read_annotations(path + fn_csv)
    # print(f'annotations:\n{my_annot}')
    ## read annotations (.csv file)
    ############################
    ## adding annotations to raw data
    raw_data.set_annotations(my_annot)
    print(raw_data.annotations)
    ############################

    ############################
    ## signals visualization and
    ## interactive annotations editing avoiding overlaping 
    ## visualization scale

    ## scale selection
    scale_dict = dict(mag=1e-12, grad=4e-11, eeg=200e-6, eog=150e-6, ecg=300e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)

    # plot
    mne.viz.plot_raw(raw_data, start=0, duration=120, scalings=scale_dict, highpass=0.3, lowpass=60.0, block=True)
    ############################
    # Filter settings
    low_cut = 0.3
    hi_cut  = 30

    raw_filt = raw_data.copy().filter(low_cut, hi_cut)

    ############################

    print(f"bad channels: {raw_data.info['bads']}")

    # events, events_dict = mne.events_from_annotations(raw_filt)
    # print(f'events: {events}')
    # print(f'events dict: {events_dict}')

    # fig_e, ax_e = plt.subplots(figsize=[15, 5])
    # mne.viz.plot_events(events, raw_filt.info['sfreq'], axes=ax_e)
    # plt.show()

    # epochs = mne.Epochs(raw_filt, events=events, event_id=events_dict, tmin=0.0, tmax=60.0, baseline=(244,304), picks=['eeg'], preload=True,)

    # print (f'epochs: {epochs}')
    # print (f'epochs[0]: {epochs[0]}')
    # baseline=(None, 0), picks=None, preload=True, reject=None, flat=None, proj=True, decim=1, reject_tmin=None, reject_tmax=None, detrend=None, on_missing='raise', reject_by_annotation=True, metadata=None, event_repeated='error', verbose=None

    # ############################
    # Break raw data into 1 s epochs
    tstep = 1.0
    events_ica = mne.make_fixed_length_events(raw_filt, start=t0, stop=t1, duration=tstep, overlap= 0)
    # print(f'events_ica: {events_ica}')
    epochs_ica = mne.Epochs(raw_filt, events=events_ica, tmin=0.0, tmax=tstep, baseline=None, preload=True)
    # picks=['E8','E9',]
    print(f'epochs_ica: {epochs_ica}')  

    # ############################
    from autoreject import AutoReject

    ar = AutoReject(n_interpolate=[1, 2, 4],
                    random_state=42,
                    picks=mne.pick_types(epochs_ica.info, 
                                        eeg=True,
                                        eog=False
                                        ),
                    n_jobs=-1, 
                    verbose=False
                    )

    ar.fit(epochs_ica)

    reject_log = ar.get_reject_log(epochs_ica)
    fig_r, ax_r = plt.subplots(figsize=[15, 5])
    reject_log.plot('horizontal', ax=ax_r, aspect='auto')
    # ############################

    # # ICA parameters
    # random_state = 42   # ensures ICA is reproducible each time it's run
    # ica_n_components = .99     # Specify n_components as a decimal to set % explained variance

    # # Fit ICA
    # ica = mne.preprocessing.ICA(n_components=ica_n_components, random_state=random_state,)
    # ica.fit(epochs_ica, decim=3)
    # ############
    # # Plots ICA components
    # ica.plot_components(contours=0)
    # # topo_dict = {'contours':0}
    # # ica.plot_properties(epochs_ica, picks=range(0, ica.n_components_), psd_args={'fmax': hi_cut}, topomap_args=topo_dict)

    # # #############################
    # ## Finding the Right Threshold
    # ica.exclude = []
    # num_excl = 0
    # max_ic = 2
    # z_thresh = 3.5
    # z_step = .05

    # ## channels closest to the eyes
    # # ch_list = ['E25','E22','E21','E17','E14','E9','E8']
    # ch_list = ['E25','E17','E8']

    # while num_excl < max_ic:
    #     print(f'num_excl threshold: {num_excl, z_thresh}')
    #     eog_indices, eog_scores = ica.find_bads_eog(epochs_ica, ch_name=ch_list, threshold=z_thresh)
    #     num_excl = len(eog_indices)
    #     z_thresh -= z_step # won't impact things if num_excl is ≥ n_max_eog 

    # # assign the bad EOG components to the ICA.exclude attribute so they can be removed later
    # ica.exclude = eog_indices
    # print('Final z threshold = ' + str(round(z_thresh, 2)))
    # print(f'eog_indices: {eog_indices}')

    # #############################
    # z_thresh = 2.2
    # ch_list = ['E25','E17','E8']
    # eog_indices, eog_scores = ica.find_bads_eog(epochs_ica, ch_name=ch_list, threshold=z_thresh)

    # ica.exclude = eog_indices

    # ica.plot_scores(eog_scores)




    ## ocular artifacts (EOG)
    # eog_epochs = mne.preprocessing.create_eog_epochs(raw_filt, ch_name='E8', picks=['eeg'], tmin=t0, tmax=t1)
    # print(f'eog_epochs: {eog_epochs}')
    # baseline=(t0, t1)
    # eog_epochs.plot_image(combine="mean")
    # eog_epochs.average().plot_joint()


    # ## frequency spectrum
    # spectrum = raw_data.compute_psd(picks='eeg',fmin=1,fmax=120,tmin=t0, tmax=t1,) ## opened eyes
    # print(f'spectrum infor: {spectrum.info}')
    # # spectrum.plot(picks=['ECG'])
    # # sel_ch = ['E8','E9','E10']
    # sel_ch = ['E30','E55','E62','E78','E79','E119']
    # spectrum.plot(picks=sel_ch)

    # print(f'spectrum open eyes: between {t0}s and {t1}s')

    # print(f"channel names: {raw_data.info['ch_names']}")
    # # eeg_channels = [channel_name for channel_name in raw_data.info['ch_names'] if channel_name.startswith('E')]

    # eeg_channels = raw_data.info['ch_names'][0:128]
    # print(f"eeg_channels names: {eeg_channels}")
    # spectrum.plot(picks=eeg_channels, amplitude=False)


    # spectrum.plot()
    # data_spectrum = spectrum.get_data()
    # print(f'data spectrum: {data_spectrum}\nshape:{data_spectrum.shape}\nfreqs:{spectrum.freqs}')

    # ani = FuncAnimation(fig=fig, func=update, frames=len(spectrum.freqs), interval=250, repeat=False,)
    # plt.show()
    #############################

    # ecg_epochs = mne.preprocessing.create_ecg_epochs(raw_data, ch_name='E8', tmin=t0, tmax=t1)
    # ecg_epochs.plot_image(combine="mean")
    # ecg_epochs = mne.preprocessing.create_ecg_epochs(raw_data,)
    # ecg_epochs.plot_image(combine="mean")

    plt.show()

    return 0

def update(frame):
    global spectrum, data_spectrum, ax

    im, cn = mne.viz.plot_topomap(data_spectrum[:,frame], spectrum.info, contours=0, vlim=(1.0e-14, 5.0e-13), cmap='magma', axes=ax, show=False)

    print(f"updated freq: {spectrum.freqs[frame]}")
    return (0) 


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
