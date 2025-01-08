#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import mne
mne.set_log_level('error')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
import time

#############################
## EEG filtering and signals prepocessing

def main(args):

    print(f'arg {args[1]}') ## folder location
    print(f'arg {args[2]}') ## subject = {0:Mme Chen, 1:Taha, 2:Carlie, 3:Iulia}
    print(f'arg {args[3]}') ## ABT = {0:resting, 1:biking}
    
    path=args[1]
    subject= int(args[2])
    abt= int(args[3])

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
    ############################
    # Mme Carlie
    elif subject == 2:
        path = path + 'apic_data/initial_testing/p01/'
        fn_in = 'APIC_TEST_CM_20241205_023522.mff'
        fn_csv = 'saved-annotations.csv'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=False)
        # fig = raw_data.plot_sensors(show_names=True,)
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
    print(f'annotations:\n{my_annot}')
    ## read annotations (.csv file)
    ############################
    ## adding annotations to raw data
    raw_data.set_annotations(my_annot)
    print(raw_data.annotations)
    ############################

    # to run GUI event loop
    plt.ion()

    ############################
    ## signals visualization and
    ## interactive annotations editing avoiding overlaping 
    ## visualization scale

    ## scale selection
    scale_dict = dict(mag=1e-12, grad=4e-11, eeg=200e-6, eog=150e-6, ecg=5e-3, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)

    # plot
    fig = mne.viz.plot_raw(raw_data, start=0, duration=120, scalings=scale_dict, highpass=1.0, lowpass=30.0, block=True)
    ############################
    ## frequency spectrum
    spectrum = raw_data.compute_psd(fmin=0,fmax=30,tmin=201, tmax=254) ## opened eyes
    # spectrum = raw_data.compute_psd(fmin=0,fmax=30,tmin=133, tmax=198) ## closed eyes
    # spectrum = raw_data.compute_psd(fmin=0,fmax=42,tmin=490, tmax=537) ## opened eyes
    # spectrum = raw_data.compute_psd(fmin=0,fmax=42,tmin=268, tmax=318) ## opened eyes
    # spectrum = raw_data.compute_psd(fmin=0,fmax=42,tmin=418, tmax=476) ## closed eyes
    # spectrum = raw_data.compute_psd(fmin=0,fmax=42,tmin=198, tmax=256) ## closed eyes
    # spectrum = raw_data.compute_psd(fmin=0,fmax=42,tmin=10, tmax=70) ## closed eyes
    print(f'spectrum: {spectrum}')

    # spectrum.plot()
    data_spectrum = spectrum.get_data()
    print(f'data spectrum: {data_spectrum}\nshape:{data_spectrum.shape}\nfreqs:{spectrum.freqs}')

    # mne.viz.plot_topomap(spectrum[:,col], spectrum.info, vlim=(0.0, 1.0), contours=0, cmap='winter', axes=ax)



    # mne.viz.plot_raw_psd(raw_data,tmin=10, tmax=70)

    # data = raw_data.get_data()

    # mne.viz.plot_topomap(, raw_data.info, vlim=(0.0, 1.0), contours=0, cmap='winter', axes=ax)

    # spectrum.plot_topomap()
    # data = raw_data.get_data()
    # print(f'shape: {data[:,5000].shape, min(data[:,5000]), max(data[:,5000])}')
    
    fig, ax = plt.subplots(1, 1, figsize=(5,5))

    # arr_rand = np.random.rand(64,100)

    col = 0
    for f in spectrum.freqs:
        # print(f'freq:{f}')
        # print(f'data: {data[:,col]}')
        print(f'freq min max: {f}, {min(data_spectrum[:,col]), max(data_spectrum[:,col])}')
        mne.viz.plot_topomap(data_spectrum[:,col], spectrum.info, contours=0, vlim=(1.0e-14, 0.5e-12), cmap='magma', axes=ax)
        # mne.viz.plot_topomap(data[:,col], raw_data.info, vlim=(-0.0001, 0.0001), contours=0, axes=ax)
        # mne.viz.plot_topomap(arr_rand[:,col], raw_data.info, vlim=(0.0, 1.0), contours=0, cmap='winter', axes=ax)
        # drawing updated values
        fig.canvas.draw()
        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        fig.canvas.flush_events()
    
        time.sleep(0.1)
        col+=1
   
    
    plt.show()

    ## save data selected channels
    # raw_data.save(path + fn_out + '_raw.fif')

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
