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
from autoreject import AutoReject
from mne.preprocessing import EOGRegression, ICA, corrmap, create_ecg_epochs, create_eog_epochs

from mne_icalabel import label_components

from bad_channels import bad_channels_dict

def onClick(event):
    global pause
    print(f'pause: {pause}')
    pause = not(pause)


ani = 0
flag=False
images=[]
spectrum = []
data_spectrum = []
draw_image = []

fig, ax = plt.subplots(1, 1, figsize=(5,5))
fig.canvas.mpl_connect('button_press_event', onClick)

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
    # fig.canvas.mpl_connect('button_press_event', toggle_pause)

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
        # t0 = 198 
        # t1 = 256

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
        # t0 = 134 
        # t1 = 260
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
        # t0 = 160 
        # t1 = 220
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
        # t0 = 178 
        # t1 = 236
    ############################
    # Mr Andre Caron
    elif subject == 4:
        path = path + 'neuroplasticity/n_001/'
        fn_in = 'Neuro001_session1_20250113_111350.mff'
        fn_csv = 'annotations.csv'
        fn_out = 'neuro_001_ann'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        ## bad channels by visual inspection
        # channels in the boundaries that could turn and have a poor contact with the patient's skin
        
        # raw_data.info["bads"] = ['E48','E119','E126','E127']
        # raw_data.interpolate_bads()

        # fig = raw_data.plot_sensors(show_names=True,)
        ## resting closed eyes
        # t0 = 15
        # t1 = 85
        # segment = "closed eyes"
        # t0 = 244
        # t1 = 304
        ## resting opened eyes
        # segment = "opened eyes"
        # t0 = 130
        # t1 = 200
        ####
        # t0 = 240
        # t1 = 400
        # print(f'{segment}: {t0}s - {t1}s')
    ############################
    else:
        return 0
    
    #############################
    ## 2D location electrodes
    # fig = raw_data.plot_sensors(show_names=True,)
    #########################    
    
    #########################    
    ## It is important to choose a reference before proceeding with the analysis.
    # raw_data.set_eeg_reference("average",ch_type='eeg',)
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
    # print(raw_data.annotations)
    ############################
    # Passband filter in place
    low_cut =   0.1
    hi_cut  = 100.0
    raw_data.filter(low_cut, hi_cut, picks='eeg')
    ############################
    ## scale selection
    scale_dict = dict(mag=1e-12, grad=4e-11, eeg=200e-6, eog=150e-6, ecg=300e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)

    ## signals visualization 
    mne.viz.plot_raw(raw_data, start=0, duration=120, scalings=scale_dict, block=False)
    # mne.viz.plot_raw(raw_data, start=0, duration=120, scalings=scale_dict, highpass=0.3, lowpass=60.0, block=False)

    ############################################
    ## cropping data according to annotations
    ## usually segments of each label have different duration

    count1 = 0
    ## we could add padding at the begining and end of each segment if required
    tpad = 0 # seconds

    ## a for resting and b for biking
    a_closed_eyes_list = []
    a_opened_eyes_list = []
    b_closed_eyes_list = []
    b_opened_eyes_list = []


    for ann in raw_data.annotations:
        # print(f'ann:\n{ann}')
        label = ann["description"]
        duration = ann["duration"]
        onset = ann["onset"]
        # print(f'annotation:{count1, onset, duration, label}')
        t1 = onset - tpad
        t2 = onset + duration + tpad
        if label == 'a_closed_eyes':
            # print('a closed eyes')
            a_closed_eyes_list.append(raw_data.copy().crop(tmin=t1, tmax=t2,))

        elif label == 'a_opened_eyes':
            # print('a opened eyes')
            a_opened_eyes_list.append(raw_data.copy().crop(tmin=t1, tmax=t2,))

        elif label == 'b_closed_eyes':
            # print('a opened eyes')
            b_closed_eyes_list.append(raw_data.copy().crop(tmin=t1, tmax=t2,))

        elif label == 'b_opened_eyes':
            # print('a opened eyes')
            b_opened_eyes_list.append(raw_data.copy().crop(tmin=t1, tmax=t2,))

        else:
            pass
        count1+=1

    print(f'size list a_closed_eyes: {len(a_closed_eyes_list)}')
    print(f'size list a_opened_eyes: {len(a_opened_eyes_list)}')
    print(f'size list b_closed_eyes: {len(b_closed_eyes_list)}')
    print(f'size list b_opened_eyes: {len(b_opened_eyes_list)}')

    ##########################
    # pre-processing selected segment: resting, closed eyes
    id = 0
    raw_cropped = a_closed_eyes_list[id]
    ## replace bad channels (selected manually) by interpolation
    raw_cropped.info["bads"] = bad_channels_dict[subject]['a_closed_eyes'][id]
    raw_cropped.interpolate_bads()
    ## re-referencing average (this technique is good for dense EEG)
    raw_closed_eyes = raw_cropped.copy().set_eeg_reference("average",ch_type='eeg',)

    # ## frequency spectrum visualization
    # mne.viz.plot_raw_psd(raw_cropped,)

    id = 0
    raw_cropped = a_opened_eyes_list[id]
    ## replace bad channels (selected manually) by interpolation
    raw_cropped.info["bads"] = bad_channels_dict[subject]['a_opened_eyes'][id]
    raw_cropped.interpolate_bads()
    # ## re-referencing average (this technique is good for dense EEG)
    raw_opened_eyes = raw_cropped.copy().set_eeg_reference("average",ch_type='eeg',)

    ## visualization selected segment
    # mne.viz.plot_raw(raw_cropped, start=0, duration=80, scalings=scale_dict, block=False)
    # mne.viz.plot_raw(raw_re_ref, start=0, duration=80, scalings=scale_dict, block=True)
    # mne.viz.plot_raw(raw_cropped, start=0, duration=80, scalings=scale_dict, highpass=0.3, lowpass=60.0, block=False)

    ##########################
    # frequency spectrums
    fig_psd, ax_psd = plt.subplots(2, 1, sharex=True, sharey=True)
    
    mne.viz.plot_raw_psd(raw_closed_eyes, ax=ax_psd[0], fmax=180)
    mne.viz.plot_raw_psd(raw_opened_eyes, ax=ax_psd[1], fmax=180)

    # print(f'raw_cropped.info: {raw_cropped.info}')
    ## visualization topographic views
    # times = np.arange(0, 60, 10)
    # raw_cropped.plot_topomap(times, ch_type='eeg', average=1.0, ncols=3, nrows="auto")

    ###########################
    ## ICA for artifact removal
    # Filter settings
    ica_low_cut  =   1.0 # For ICA, we filter out more low-frequency power
    ica_high_cut = 100.0
    raw_closed_eyes_ica = raw_closed_eyes.copy().filter(ica_low_cut, ica_high_cut)

    ##############
    ica = mne.preprocessing.ICA(
    n_components=0.99,
    max_iter="auto",
    method="infomax",
    random_state=97,
    fit_params=dict(extended=True),
    )
    ica.fit(raw_closed_eyes_ica)

    ic_labels = label_components(raw_closed_eyes_ica, ica, method="iclabel")

    # ICA0 was correctly identified as an eye blink, whereas ICA12 was
    # classified as a muscle artifact.
    print(ic_labels["labels"])
    labels = ic_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
    print(f"Excluding these ICA components: {exclude_idx}")

    ica.plot_sources(raw_closed_eyes, show_scrollbars=False, show=True)
    ica.plot_components(contours=0,colorbar=True)

    plt.show()
    return 0
    # ica.plot_properties(raw, picks=[0, 12], verbose=False)

    # blinks
    # ica.plot_overlay(raw, exclude=[0], picks="eeg")
    # ica
    ##############
    # Break raw data into 1 s epochs
    tstep = 1.0
    events_ica = mne.make_fixed_length_events(raw_closed_eyes_ica, duration=tstep)
    epochs_ica = mne.Epochs(raw_closed_eyes_ica, events_ica,
                            tmin=0.0, tmax=tstep,
                            baseline=None,
                            preload=True)
    
    ###############
    ##  autoreject
    # from autoreject import AutoReject

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

    ## plot autoreject results
    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(figsize=[15, 5])
    # reject_log.plot('horizontal', ax=ax, aspect='auto')
    # plt.show()

    #################
    ## ICA
    # ICA parameters
    random_state = 42   # ensures ICA is reproducible each time it's run
    ica_n_components = .99     # Specify n_components as a decimal to set % explained variance

    # Fit ICA
    ica = mne.preprocessing.ICA(n_components=ica_n_components,
                                random_state=random_state,
                                )
    ica.fit(epochs_ica[~reject_log.bad_epochs], decim=3)

    ## automatic ICA labeling

    ic_labels = label_components(epochs_ica[~reject_log.bad_epochs], ica, method="iclabel")

    # ICA0 was correctly identified as an eye blink, whereas ICA12 was
    # classified as a muscle artifact.
    print(f"labels components: {ic_labels['labels']}")

    ica.plot_components(contours=0,colorbar=True)

    ############################
    ## epochs of 1s from every segment that had been labeled as a_closed_eyes, a_opened_eyes, b_closed_eyes, b_opened_eyes

    ts = 10.0 # s
    delta = 1.0 # s
    arr_data = raw_cropped.get_data(picks=['eeg'], tmin=ts-delta/2, tmax=ts+delta/2)
    arr_mean = np.mean(arr_data,axis=1)
    print(f'data: {arr_mean}')
    print(f'data: {len(arr_mean)}, {arr_mean.shape}')

    pos_ch = raw_cropped.get_montage().get_positions()['ch_pos']
    print(f'pos_ch:\n{pos_ch}')

    # mne.viz.plot_evoked_topomap(mne.grand_average(diff_waves), 
                            # times=.500, average=0.200, 
                            # size=3
                        #    )
    # mne.viz.plot_topomap(arr_mean, pos_ch, ch_type='eeg',contours=0, )
    plt.show()
    return 0

    # ## frequency spectrum
    spectrum = raw_cropped.compute_psd(picks='eeg',) ## opened eyes
    data_spectrum = spectrum.get_data()
    print(f'data_spectrum size:\n{len(data_spectrum)}')

    ## first step: visualizing power spectrum density (frequency)
    mne.viz.plot_raw_psd(raw_cropped, picks=['eeg'], area_mode='std', show=True, average=False, xscale='log')
    # raw_cropped.plot_psd(area_mode='range', tmax=10.0, show=False, average=True)
    frame=0
    # im, cn = mne.viz.plot_topomap(data_spectrum[:,frame], spectrum.info, contours=0, vlim=(1.0e-14, 5.0e-13), cmap='magma')

    # ani = FuncAnimation(fig=fig, func=update, frames=len(spectrum.freqs), interval=250, repeat=False,)
    # plt.show()

    ## visualization selected segment
    # mne.viz.plot_raw(raw_cropped, start=0, duration=80, scalings=scale_dict, highpass=0.1, lowpass=45.0, block=False)

    ###########################
    ## pre-processing selected segment: resting, opened eyes
    raw_cropped = a_opened_eyes_list[0]
    ## first step: visualizing power spectrum density (frequency)
    mne.viz.plot_raw_psd(raw_cropped, picks=['eeg'], area_mode='std', show=True, average=False, xscale='log')


   

    

    # ############################
    
    '''
    # ############################
    # Filter settings
    # low_cut =  0.3
    # hi_cut  = 40.0

    filt_raw = raw_data.copy().filter(l_freq=1.0, h_freq=None)
    # raw_filt = raw_data.copy().filter(low_cut, hi_cut)
    # raw_filt.crop(tmin=t0,tmax=t1,include_tmax=True)

    # # ############################
    # ## signals visualization
    # mne.viz.plot_raw(raw_filt, scalings=scale_dict, block=False)
  
    # ##############
    #  ## selected channels for eyes-blinking
    # ch_blink = ['E25','E17','E8']

    # # events due to blinking for all the recording data excluding "BAD_" sectors (reject_by_annotation=True)
    # # eog_evoked = mne.preprocessing.create_eog_epochs(raw_filt, ch_name=ch_blink, reject_by_annotation=True).average()

    # eog_epochs = mne.preprocessing.create_eog_epochs(filt_raw, ch_name=ch_blink, reject_by_annotation=True)

    # eog_evoked = eog_epochs.average("all")

    # eog_evoked.apply_baseline(baseline=(None, -0.2))
    # ## option zero contours to avoid visualization problems
    # topo_dict = {'contours':0}
    # eog_evoked.plot_joint(topomap_args=topo_dict)
    # ##############
    # # events due to heart beats
    # ecg_epochs = mne.preprocessing.create_ecg_epochs(filt_raw, ch_name='ECG', reject_by_annotation=True)
    # ecg_evoked = ecg_epochs.average("all")
    # ecg_evoked.apply_baseline(baseline=(None, -0.2))
    # ecg_evoked.plot_joint(topomap_args=topo_dict)
    
    # ##############
    ## ICA

    ica = ICA(n_components=15, max_iter="auto", random_state=97)
    ica.fit(filt_raw, reject_by_annotation=True)
    print(f'ica:\n{ica}')

    # explained_var_ratio = ica.get_explained_variance_ratio(filt_raw)
    # for channel_type, ratio in explained_var_ratio.items():
    #     print(f"Fraction of {channel_type} variance explained by all components: {ratio}")

    # explained_var_ratio = ica.get_explained_variance_ratio(
    #     filt_raw, components=[0], ch_type="eeg"
    # )
    # # This time, print as percentage.
    # ratio_percent = round(100 * explained_var_ratio["eeg"])
    # print(
    #     f"Fraction of variance in EEG signal explained by first component: "
    #     f"{ratio_percent}%"
    # )

    # raw_data.load_data()
    # ica.plot_sources(filt_raw, show_scrollbars=False)

    # ica.plot_components(contours=0)
    # topo_dict = {'contours':0}
    # ica.plot_properties(filt_raw, picks=[0,1,2,3],topomap_args=topo_dict)

    ica.exclude = [0]  # indices chosen based on various plots above

    reconst_raw = raw_data.copy()
    ica.apply(reconst_raw)
    mne.viz.plot_raw(reconst_raw, start=0, duration=20, scalings=scale_dict, highpass=0.3, lowpass=40.0, block=False)

    # blinks
    # ica.plot_overlay(filt_raw, exclude=[0], picks="eeg")


#     # perform regression on the evoked blink response
#     model_evoked = EOGRegression(picks="eeg", picks_artifact=ch_blink,).fit(eog_evoked)
#     fig = model_evoked.plot(vlim=(None, 0.4),contours=0,)
#     fig.set_size_inches(3, 2)

#     # for good measure, also show the effect on the blink evoked
#     eog_evoked_clean = model_evoked.apply(eog_evoked)
#     eog_evoked_clean.apply_baseline()
#     eog_evoked_clean.plot("all")
#     fig.set_size_inches(6, 6)


#     order = np.concatenate(
#     [# plotting order: EOG first, then EEG
#         mne.pick_types(raw_filt.info, meg=False, eeg=True),
#     ])

#     raw_kwargs = dict(
#     events=eog_epochs.events,
#     order=order,
#     start=0,
#     duration=120,
#     n_channels=20,
#     scalings=dict(eeg=200e-6, eog=250e-6),
# )
#     # regress (using coefficients computed previously) and plot
#     raw_clean = model_evoked.apply(raw_filt)
#     raw_clean.plot(**raw_kwargs)

#     ## scale selection
#     scale_dict = dict(mag=1e-12, grad=4e-11, eeg=200e-6, eog=250e-6, ecg=300e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)
#     # # plot
#     mne.viz.plot_raw(raw_filt, start=0, duration=120, scalings=scale_dict, block=True,)

#     ##############




    # print(f"bad channels: {raw_data.info['bads']}")

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


    # # ############################
    # # Break raw data into epochs
    # tstep = 5.0
    # events_ica = mne.make_fixed_length_events(raw_filt, start=t0, stop=t1, duration=tstep, overlap= 0)
    # # print(f'events_ica: {events_ica}')
    # epochs_ica = mne.Epochs(raw_filt, events=events_ica, tmin=0.0, tmax=tstep, baseline=None, preload=True)
    # # picks=['E8','E9',]
    # print(f'epochs_ica: {epochs_ica}')  

    # # ############################
    # ## identify bad recordings
    # ar = AutoReject(n_interpolate=[1, 2, 4],
    #                 random_state=42,
    #                 picks=mne.pick_types(epochs_ica.info, eeg=True, eog=False),
    #                 n_jobs=-1,
    #                 verbose=False)

    # ar.fit(epochs_ica)

    # reject_log = ar.get_reject_log(epochs_ica)
    # fig_r, ax_r = plt.subplots(figsize=[15, 5])
    # reject_log.plot('horizontal', ax=ax_r, aspect='auto')
    # # ############################


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

    ani = FuncAnimation(fig=fig, func=update, frames=len(spectrum.freqs), interval=250, repeat=False,)
    plt.show()
    #############################

    # ecg_epochs = mne.preprocessing.create_ecg_epochs(raw_data, ch_name='E8', tmin=t0, tmax=t1)
    # ecg_epochs.plot_image(combine="mean")
    # ecg_epochs = mne.preprocessing.create_ecg_epochs(raw_data,)
    # ecg_epochs.plot_image(combine="mean")

    '''

    plt.show()
    
    return 0

def update(frame):
    global spectrum, data_spectrum, ax, fig

    im, cn = mne.viz.plot_topomap(data_spectrum[:,frame], spectrum.info, contours=0, vlim=(1.0e-14, 5.0e-13), cmap='magma', axes=ax, show=False)

    # manually fiddle the position of colorbar
    ax_x_start = 0.95
    ax_x_width = 0.04
    ax_y_start = 0.1
    ax_y_height = 0.9
    cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
    clb = fig.colorbar(im, cax=cbar_ax)
    clb.ax.set_title("topographic view",fontsize=16) # title on top of colorbar

    print(f"updated freq: {spectrum.freqs[frame]}")
    return (0) 


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
