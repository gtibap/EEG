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
        raw_data = mne.io.read_raw_bdf(path + fn_in, preload=False)
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
        raw_data = mne.io.read_raw_bdf(path + fn_in, preload=False)
        ## select channels
        sel_ch = np.arange(64,128)
        raw_data.pick(sel_ch)
        ## electrodes montage
        raw_data.set_montage("biosemi64")
        # fig = raw_data.plot_sensors(show_names=True, sphere='eeglab')

        ## resting closed eyes
        t0 = 134 
        t1 = 198
    ############################
    # Mme Carlie
    elif subject == 2:
        path = path + 'apic_data/initial_testing/p01/'
        fn_in = 'APIC_TEST_CM_20241205_023522.mff'
        fn_csv = 'saved-annotations.csv'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=False)
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
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=False)
        # fig = raw_data.plot_sensors(show_names=True,)
    ############################
    # Mr Andre Caron
    elif subject == 4:
        path = path + 'neuroplasticity/n_001/'
        fn_in = 'Neuro001_session1_20250113_111350.mff'
        fn_csv = 'annotations.csv'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=False)
        # fig = raw_data.plot_sensors(show_names=True,)
        ## resting closed eyes
        # t0 = 15
        # t1 = 85
        ## resting opened eyes
        t0 = 130
        t1 = 200
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
    mne.viz.plot_raw(raw_data, start=0, duration=120, scalings=scale_dict, highpass=1.0, lowpass=30.0, block=True)
    ############################
    ## frequency spectrum
    spectrum = raw_data.compute_psd(picks='eeg',fmin=1,fmax=120,tmin=t0, tmax=t1,) ## opened eyes
    print(f'spectrum infor: {spectrum.info}')
    # spectrum.plot(picks=['ECG'])
    spectrum.plot(picks=['E8','E9','E10'])

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
    plt.show()
    #############################

    # ecg_epochs = mne.preprocessing.create_ecg_epochs(raw_data, ch_name='E8', tmin=t0, tmax=t1)
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
