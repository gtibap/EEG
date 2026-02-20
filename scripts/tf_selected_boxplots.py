#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import mne
from mne.preprocessing import ICA
from mne.time_frequency import psd_array_multitaper

import os
from pathlib import Path
import time
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from channels_tfr import selected_channels
## include modules from another directory
sys.path.insert(0, '../../scripts')
from bad_channels import bad_channels_dict
from list_participants import participants_list

from class_tf import TF_components


sampling_rate = 1.0
y_limits = [-8,8]

## scale selection for visualization raw data with annotations
scale_dict = dict(mag=1e-12, grad=4e-11, eeg=100e-6, eog=150e-6, ecg=400e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)


#############################
#############################
def crop_fun(raw_data, t1, t2):
    raw = raw_data.copy().crop(tmin=t1, tmax=t2,)
    # raw.set_meas_date(None)
    # mne.io.anonymize_info(raw.info)
    # print(f'first sample after crop: {raw.first_samp}')
    ann = raw.annotations
    # print(f'crop annotations:{len(ann)}\n{ann}')
    raw.annotations.delete(np.arange(len(ann)))
    return raw
#############################
#############################

#############################
#############################
def get_eeg_segments(raw_data,):    
    ## prefix:
    ## a:resting; b:biking
    baseline_list = []
    a_closed_eyes_list = []
    a_opened_eyes_list = []
    b_closed_eyes_list = []
    b_opened_eyes_list = []
    c_closed_eyes_list = []
    c_opened_eyes_list = []

    for ann in raw_data.annotations:
        # print(f'ann:\n{ann}')
        label = ann["description"]
        duration = ann["duration"]
        onset = ann["onset"]
        # print(f'annotation:{count1, onset, duration, label}')
        t1 = onset
        t2 = onset + duration
        if label == 'baseline':
            baseline_list.append(crop_fun(raw_data, t1, t2))

        elif label == 'a_closed_eyes':
            a_closed_eyes_list.append(crop_fun(raw_data, t1, t2))

        elif label == 'a_opened_eyes':
            a_opened_eyes_list.append(crop_fun(raw_data, t1, t2))

        elif label == 'b_closed_eyes':
            b_closed_eyes_list.append(crop_fun(raw_data, t1, t2))

        elif label == 'b_opened_eyes':
            b_opened_eyes_list.append(crop_fun(raw_data, t1, t2))

        elif label == 'c_closed_eyes':
            c_closed_eyes_list.append(crop_fun(raw_data, t1, t2))

        elif label == 'c_opened_eyes':
            c_opened_eyes_list.append(crop_fun(raw_data, t1, t2))

        else:
            pass

    print(f'size list baseline: {len(baseline_list)}')
    print(f'size list a_closed_eyes: {len(a_closed_eyes_list)}')
    print(f'size list a_opened_eyes: {len(a_opened_eyes_list)}')
    print(f'size list b_closed_eyes: {len(b_closed_eyes_list)}')
    print(f'size list b_opened_eyes: {len(b_opened_eyes_list)}')
    print(f'size list c_closed_eyes: {len(c_closed_eyes_list)}')
    print(f'size list c_opened_eyes: {len(c_opened_eyes_list)}')

    ## eeg data to a dictionary
    eeg_data_dict={}
    eeg_data_dict['baseline'] = baseline_list
    eeg_data_dict['a_closed_eyes'] = a_closed_eyes_list
    eeg_data_dict['a_opened_eyes'] = a_opened_eyes_list
    eeg_data_dict['b_closed_eyes'] = b_closed_eyes_list
    eeg_data_dict['b_opened_eyes'] = b_opened_eyes_list
    eeg_data_dict['c_closed_eyes'] = c_closed_eyes_list
    eeg_data_dict['c_opened_eyes'] = c_opened_eyes_list

    return eeg_data_dict
#############################
#############################

#############################
def baseline_normalization(obj_list, selected_segs_dict, ch_list, label_ref):
    ## dictionary with labels and ids of selected segements
    id_ref = selected_segs_dict[label_ref]
    ## get the first segment with label equal to label_ref
    id=0
    while (obj_list[id].get_label_simple() != label_ref) or (obj_list[id].get_id() != id_ref):
        # print(f"{id} label: {obj_list[id].get_label()}")
        id+=1
    ##
    ## selected segment to calculate reference baseline 
    seg_ref = obj_list[id]

    print(f"Selected segment to calculate reference baseline: {seg_ref.get_label()}_{seg_ref.get_id()}")

    ## get reference for baseline normalization
    # print(f"calculating average values for time-frequency normalization...")
    tf_ref, freq_tf = seg_ref.get_tf_baseline()
    
    # print(f"baseline normalization, shape tf_ref: {tf_ref.shape}\nfreq: {freq_tf}")
    # fig, ax = plt.subplots(1,3, sharex=True, sharey=True)
    # for ax_id, arr in enumerate(tf_ref):
    #     ax[ax_id].plot(freq_tf, arr)

    ## normalization of each of the  selected segments
    for obj in obj_list:
        ## find the selected segment for each label
        if obj.get_selected_flag():
            print(f"{obj.get_label()}-{obj.get_id()}:")
            # print(f"Tf normalization... ")
            obj.tf_normalization(tf_ref)

    # ## normalization of each of the  selected segments
    # for sel_label in selected_segs_dict:
    #     ## label segment and segment id
    #     sel_id = selected_segs_dict[sel_label]
    #     print(f"selected label and id: {sel_label, sel_id}")
    #     ## for each segment calculate ICA and identify components of noise/artifacts
    #     ## calculate time-freq transformation for each segment
    #     for obj in obj_list:
    #         ## label simple includes: a_oe, a_ce, b_oe, b_ce, c_oe, c_ce
    #         label_seg = obj.get_label_simple()
    #         id_seg    = obj.get_id()
    #         print(f"obj label and id: {label_seg, id_seg}")
    #         ## find the selected segment for each label
    #         if (label_seg == sel_label) and (id_seg == sel_id):
    #             print(f"{obj.get_label()}-{obj.get_id()}:")
    #             # print(f"Tf normalization... ")
    #             obj.tf_normalization(tf_ref)

    # for obj in obj_list:
    #     # print(f"Tf normalization... ")
    #     obj.tf_normalization(tf_ref)

    # ## get reference for baseline normalization
    # print(f"calculating average values for time-frequency normalization...")
    # tf_ref, freq_tf = seg_ref.get_tf_baseline()

    # print(f"baseline normalization, shape tf_ref: {tf_ref.shape}\nfreq: {freq_tf}")
    # fig2, ax2 = plt.subplots(1,3, sharex=True, sharey=True)
    # for ax2_id, arr in enumerate(tf_ref):
    #     ax2[ax2_id].plot(freq_tf, arr)

    return 0

####################################################################################
def eeg_segmentation(eeg_data_dict, eeg_filt_dict, label_seg_list, path, session):

    obj_list = []
    ## instantiate objects of the class TF_components
    for label_seg in label_seg_list:
        id_seg = 0
        ## usually three repetitions per segment of closed and open eyes during resting and cycling. Baseline is an exception
        for raw_seg, filt_seg in zip(eeg_data_dict[label_seg], eeg_filt_dict[label_seg]):

            ## instanciate object per each segment (baseline, open eyes, closed eyes)
            obj = TF_components(path, session, raw_seg, filt_seg, label_seg, id_seg)
            obj_list.append(obj)

            id_seg+=1

    return obj_list

#################################################
def annotation_bad_channels_and_segments(obj_list, flag_update):
    ## for each segment observe and identify bad channels and bad segments
    for obj in obj_list:
        # interactive selection of bad segments and bad channels
        print(f"{obj.get_label()}-{obj.get_id()}: interactive selection of bad segments and bad channels...")
        obj.selection_bads(flag_update)
        print(f"bad channels: {obj.get_bad_channels()}\n")

    return 0

#################################################
def set_selected_segments(obj_list, selected_segs_dict):
    ## for each segment of each state
    for sel_label in selected_segs_dict:
        ## label segment and segment id
        sel_id = selected_segs_dict[sel_label]
        print(f"sel_label and id: {sel_label, sel_id}")
        ## for each segment calculate ICA and identify components of noise/artifacts
        for obj in obj_list:
            ## label simple includes: a_oe, a_ce, b_oe, b_ce, c_oe, c_ce
            label_seg = obj.get_label_simple()
            id_seg    = obj.get_id()
            print(f"obj label and id: {label_seg, id_seg}")
            ## find the selected segment for each label
            if (label_seg == sel_label) and (id_seg == sel_id):
                obj.set_selected_flag()

    return 0

###################################################
def ica_artifacts_reduction(obj_list, selected_segs_dict, flag_update):
    ## ica only for selected segments     
    for obj in obj_list:
        ## find the selected segment for each label
        if obj.get_selected_flag():
            # interactive selection of bad segments and bad channels
            print(f"{obj.get_label(), obj.get_id()}: interactive selection of ICA components to exclude...")
            # re-referencing appli. average before ICA
            obj.re_referencing()
            ##
            print("ICA components...")
            # obj.ica_components(flag_update)
            obj.ica_components_interactive(flag_update)
            # print(f"excluded ica components: {obj.get_ica_exclude()}\n")

    ## iterate list segment labels that could includes: a_ce, a_oe, b_ce, b_oe, c_ce, c_oe
    # for sel_label in selected_segs_dict:
    #     ## label segment and segment id
    #     sel_id = selected_segs_dict[sel_label]
    #     print(f"sel_label and id: {sel_label, sel_id}")
    #     ## for each segment calculate ICA and identify components of noise/artifacts
    #     for obj in obj_list:
    #         ## label simple includes: a_oe, a_ce, b_oe, b_ce, c_oe, c_ce
    #         label_seg = obj.get_label_simple()
    #         id_seg    = obj.get_id()
    #         print(f"obj label and id: {label_seg, id_seg}")
    #         ## find the selected segment for each label
    #         if (label_seg == sel_label) and (id_seg == sel_id):
    #             # interactive selection of bad segments and bad channels
    #             print(f"{obj.get_label(), obj.get_id()}: interactive selection of ICA components to exclude...")
    #             # re-referencing appli. average before ICA
    #             obj.re_referencing()
    #             ##
    #             print("ICA components...")
    #             # obj.ica_components(flag_update)
    #             obj.ica_components_interactive(flag_update)
    #             # print(f"excluded ica components: {obj.get_ica_exclude()}\n")
    return 0

#############################################################    
def calculate_tf(obj_list, selected_segs_dict, ch_list,):
    ## tf calculation only for the selected segments of each state that includes: a_ce, a_oe, b_ce, b_oe, c_ce, c_oe
    for obj in obj_list:
        ## find the selected segment for each label
        if obj.get_selected_flag():
            print(f"{obj.get_label()}-{obj.get_id()}:")
            print("Bad channels interpolation...")
            obj.bads_interpolation()
            
            print(f"Current source density (Laplacian surface)...")
            obj.apply_csd()
            # print(f"plot csd data...")
            # seg_ref.plot_time_series('csd', 'After Laplacian surface filter (current source density)')
            
            # print(f"PSD selected channels: {ch_list}")
            # obj.psd_selected_chx(ch_list)

            print(f"Time-frequency transformation...")
            obj.tf_calculation(ch_list)

    # ## iterate list segment labels that could includes: a_ce, a_oe, b_ce, b_oe, c_ce, c_oe
    # for sel_label in selected_segs_dict:
    #     ## label segment and segment id
    #     sel_id = selected_segs_dict[sel_label]
    #     print(f"selected label and id: {sel_label, sel_id}")
    #     ## for each segment calculate ICA and identify components of noise/artifacts
    #     ## calculate time-freq transformation for each segment
    #     for obj in obj_list:
    #         ## label simple includes: a_oe, a_ce, b_oe, b_ce, c_oe, c_ce
    #         label_seg = obj.get_label_simple()
    #         id_seg    = obj.get_id()
    #         print(f"obj label and id: {label_seg, id_seg}")
    #         ## find the selected segment for each label
    #         if (label_seg == sel_label) and (id_seg == sel_id):
    #             print(f"{obj.get_label()}-{obj.get_id()}:")
    #             print("Bad channels interpolation...")
    #             obj.bads_interpolation()
                
    #             print(f"Current source density (Laplacian surface)...")
    #             obj.apply_csd()
    #             # print(f"plot csd data...")
    #             # seg_ref.plot_time_series('csd', 'After Laplacian surface filter (current source density)')
                
    #             # print(f"PSD selected channels: {ch_list}")
    #             # obj.psd_selected_chx(ch_list)

    #             print(f"Time-frequency transformation...")
    #             obj.tf_calculation(ch_list)

    return 0

##########################################################
def tf_freq_bands(obj_list, eeg_system, ch_name_list):
    ##
    # print(f"Power per frequency bands... ")
    ## alpha, theta, beta bands activity selected channels
    # df_ch_list = pd.DataFrame()
    # ## selected channels
    # ch_name_list = ['Cz','C3','C4']
    for ch_name in ch_name_list:
        ## for each selected channel
        for obj in obj_list:
            ## for every segment (obj) of each selected channel (ch_name)
            ## df_ch_bands : theta, beta, alpha activity of selected channels
            if obj.get_selected_flag():
                obj.channel_bands_power(ch_name, eeg_system)
    ##
    ## add bad annotations in df_ch_bands
    for obj in obj_list:
        ## for every segment (obj) of each selected channel (ch_name)
        ## df_ch_bands : theta, beta, alpha activity of selected channels 
        ## add a mask to mark band segments
        if obj.get_selected_flag():
            obj.set_annotations_freq_bands()
        # df = obj.get_df_ch_bands()
        # print(f"dataframe {obj.get_label()}--{obj.get_id()}, df shape {df.shape}:\n{df}")
        # print(f"df columns:\n{list(df.columns.values)}")
    ##

    return 0

##########################################################
def plot_tfr(obj_list, eeg_system, ch_name_list):

    # print(f"Power per frequency bands... ")
    ## alpha, theta, beta bands activity selected channels
    # df_ch_list = pd.DataFrame()
    # ## selected channels
    # ch_name_list = ['Cz','C3','C4']
    for ch_name in ch_name_list:
        ## for each selected channel
        for obj in obj_list:
            ## for every segment (obj) of each selected channel (ch_name)
            ## save figure of time-frequency analysis for each selected channel
            obj.tfr_norm_plot(ch_name, eeg_system,)

    return 0

#############################################################
# def display_psd_segments(obj_list, label_seg_list):
#     ## display simultaneusly psd of the same label_seg
#     fig_psd, ax_psd = plt.subplots(nrows=2, ncols=2, figsize=(9,4), sharey=True, sharex=True)
#     ax_psd = ax_psd.flatten()
#     ## 
#     for label_seg in label_seg_list:

#         for obj in obj_list:
#             if obj.get_label() == label_seg:
        

#     return 0
#################################################
def display_segments(obj_list, label_seg_list):
    ## for each segment observe and identify bad channels and bad segments
    for label_seg in label_seg_list:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9,4), sharey=True, sharex=True)
        ax = ax.flatten()
        id_ax = 0
        for obj in obj_list:
            # interactive selection of bad segments and bad channels
            print(f"{obj.get_label()}-{obj.get_id()}: interactive selection of bad segments and bad channels...")
            if obj.get_label() == label_seg:
                ## load bad channels and bad annotations
                ax[id_ax], id_seg = obj.data_visualization(ax[id_ax])
                ax[id_ax].set_title(f"id = {id_seg}")
                id_ax+=1
        fig.suptitle(f"{label_seg}")
        plt.show(block=True)
        # flag_bad_ch = input('Include more bad channels? (1 (True), 0 (False)): ')

    return 0


###########################################
## EEG filtering and signals pre-processing
##
def main(args):
    global sampling_rate

    ## interactive mouse pause the image visualization
    # fig.canvas.mpl_connect('button_press_event', toggle_pause)

    print(f'arg {args[1]}') ## folder location
    print(f'arg {args[2]}') ## subject = {0:patient 1, 1:patient 2, ...}
    print(f'arg {args[3]}') ## session = {1:time zero, 2:three months, 3:six months}
    print(f'arg {args[4]}') ## ABT = {0:resting, 1:biking}
    
    path=args[1]
    subject= int(args[2])
    session=int(args[3])
    abt= int(args[4])

    fn_in=''
    
    t0=0
    t1=0

    #########################
    ## selected data
    print(f'path:{path}\nsubject:{subject}\nsession:{session}\nabt:{abt}\n')

    #########################
    ## new path, eeg filename (fn_in), annotations filename (fn_csv), eeg raw data (raw_data)
    path, fn_in, fn_csv, raw_data, fig_title, rows_plot, acquisition_system, info_p, Dx, selected_segs_dict = participants_list(path, subject, session, abt)
    if fn_csv == '':
        print(f'It could not find the selected subject. Please check the path, and the selected subject number in the list of participants.')
        return 0
    else:
        pass

    ## create folder (if it does not exist) to save preprocesing parameters
    # Path(path+'session_'+str(session)+"/prep").mkdir(parents=True, exist_ok=True)

    ## path filename for baseline normalization
    filename_tr_ref = path+'session_'+str(session)+f'/prep/'+'tf_mean_baseline.npy'

    ## path filename boxplots
    path_fig_boxplot = path+'session_'+str(session)+f'/figures/'
    # checking if the directory figures
    # exist or not.
    if not os.path.exists(path_fig_boxplot):
        # if the figures directory is not present 
        # then create it.
        os.makedirs(path_fig_boxplot)

    ############################
    ## read annotations (.csv file)
    my_annot = mne.read_annotations(path + fn_csv[0])
    print(f'annotations:\n{my_annot}')
    ## adding annotations to raw data
    raw_data.set_annotations(my_annot)
    ##

    ## exclude channels of the net boundaries that usually bring noise or artifacts
    ## geodesic system we remove channels in the boundaries
    raw_data.info["bads"] = bad_channels_dict[acquisition_system]
    raw_data.drop_channels(raw_data.info['bads'])
    
    ##########################
    # printing basic information from data
    print(f'raw data filename: {fn_in}')
    print(f'annotations filename: {fn_csv}')
    print(f'raw data info:\n{raw_data.info}')
    # printing basic information from data
    ############################
    ## sampling rate
    sampling_rate = raw_data.info['sfreq']
    ############################
    ## run matplotlib in interactive mode
    plt.ion()
    ############################

    ################################
    ## Stage 1: high pass filter (in place)
    #################################
    low_cut =    0.5
    hi_cut  =   45.0
    raw_data.filter(l_freq=low_cut, h_freq=hi_cut, picks='eeg')
    ## filter 1 Hz high pass for ICA
    raw_filt = raw_data.copy().filter(l_freq=1.0, h_freq=hi_cut, picks=['eeg'])

    ## time-series data visualization
    # mne.viz.plot_raw(raw_data, picks=['eeg','ecg'], start=0, duration=240, scalings=scale_dict, highpass=None, lowpass=None, title=f'EEG after 0.5-45 Hz pass band filtering ({acquisition_system})', block=False)
    ## time-series data visualization
    # mne.viz.plot_raw(raw_filt, picks=['eeg','ecg'], start=0, duration=240, scalings=scale_dict, highpass=None, lowpass=None, title=f'EEG after 1.0-45 Hz pass band filtering ({acquisition_system})', block=True)

    ###########################################
    ## cropping data according to every section (annotations), namely 'baseline','a_closed_eyes','a_opened_eyes','b_closed_eyes', and 'b_opened_eyes'
    print(f"segments raw data")
    eeg_data_dict = get_eeg_segments(raw_data,)
    print(f"segments raw-filtered data")
    eeg_filt_dict = get_eeg_segments(raw_filt,)

    ###############################################
    # each segment has its own properties. We create an object de la class TF_components per segment that includes the raw data and filtered data
    # label_seg_list = ['baseline','a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes']
    label_seg_list = ['a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes','c_closed_eyes','c_opened_eyes']

    #####################################################
    ## data segmentation, each segment would be an object 
    obj_list = eeg_segmentation(eeg_data_dict, eeg_filt_dict, label_seg_list, path, session)

    #############################################
    ## observe EEG signals to identify and select bad segments and bad channels from each segment
    flag_update = int(input(f"Update bad-channels or bad-segments (0-False, 1-True)?: "))
    annotation_bad_channels_and_segments(obj_list, flag_update)

    ##############################################
    ## display PSD and time series signals of each state or label (a_closed_eyes, a_opened_eyes, ...)
    flag_selection = int(input(f"Update segments' selection (0-False, 1-True)?: "))
    if flag_selection:
        ## visual selection of selected segments for each state
        ## the selected segments are manually set in a list_participant.py, in the "selected_ids_dict"
        display_segments(obj_list, label_seg_list)
        return 0
    
    ## set a flag for each obj to identify selected segments from each state (a_ce, a_oe, b_ce, b_oe, c_ce, c_oe)
    set_selected_segments(obj_list, selected_segs_dict)

    #############################################
    ## apply ICA to try to remove components of noise and artifacts
    flag_update = int(input(f"Update ICA components or ICA components selection (0-False, 1-True)?: "))
    # flag_update = True
    ## apply ICA to the selected segments, one for each state: a_oe, a_ce, b_oe, b_ce, c_oe, c_ce
    ica_artifacts_reduction(obj_list, selected_segs_dict, flag_update)

    ##################
    ## selected channels; the two lists of channels are equivalent, one in the 128 electrodes and the other in the 64 electrodes
    ch_name_128 = ['VREF','E36','E104']
    ch_name_10_10 = ['Cz','C3','C4']
    
    #################################################
    ## calculate time-frequency transformations for each segment
    ## bad-channels interpolation and current-source-density filter are applied before tf-analysis
    calculate_tf(obj_list, selected_segs_dict, ch_name_128,)

    #################################################
    print(f"Baseline normalization...")
    ## reference label for baseline normalization.
    # We chose the first segment of open eyes during resting
    label_ref = 'a_oe'
    baseline_normalization(obj_list, selected_segs_dict, ch_name_128, label_ref)

    # ## save calculated the time-frequency reference for a posterior normalization of each segment
    # try:
    #     print(f"saving baseline reference... ",end='')
    #     np.save(filename_tr_ref, tf_ref)
    #     print(f"done.")
    # except:
    #     print(f"Error: something went wrong saving time-frequency analysis.")

    # print(f"tf_ref shape {tf_ref.shape}")
    # print(f"tf_ref:\n{tf_ref}")

    # ###########################
    ## values of frequency bands (median values) over time
    print(f"Power per frequency bands... ")
    tf_freq_bands(obj_list, acquisition_system, ch_name_10_10)

    # #############################################
    # # optional
    # print(f"Saving figures time-frequency analysis...")
    # plot_tfr(obj_list, acquisition_system, ch_name_10_10)
    # # optional
    # #############################################

    #############################
    ## boxplots: beta band
    ## sel_ch: 'Cz', 'C3', or 'C4'
    # sel_ch = ch_name_10_10[2]
    ## sel_band: 'beta', 'beta_l', 'beta_h'
    sel_band = 'beta_l'
    label_band = 'EEG low-beta band [12-20 Hz]'
    
    ## boxplots
    fig_box, ax_box = plt.subplots(nrows=2, ncols=3, figsize=(9,6), sharey=True,)
    ax_box = ax_box.flatten()

    ## plot's columns [1,0,2] --> channels [Cz,C3,C4] 
    ax_ch_list = [[1,4],[0,3],[2,5]]
    labels = [f'before\ncycling',f'during\ncycling',f'after\ncycling']

    #############
    #############
    #########
    ##############
    ##########

    for sel_ch, ax_ch in zip(ch_name_10_10, ax_ch_list):
        # for sel_band in ['beta_l',]:
        # create_fig_boxplot(obj_list, label_seg_list, sel_ch, sel_band)
        # create_fig_boxplot_average(obj_list, label_seg_list, sel_ch, sel_band)
        c_list, o_list = create_fig_boxplot_single(obj_list, label_seg_list, sel_ch, sel_band)

            # obj.plot_curves_beta_bands(ch_name_list)
        # obj.boxplots_beta_bands(ch_name_list)
        ax_box[ax_ch[0]].boxplot(c_list, sym='', tick_labels=labels,)
        ax_box[ax_ch[1]].boxplot(o_list, sym='', tick_labels=labels,)

        ax_box[ax_ch[0]].set_title(f"channel: {sel_ch}")
        # ax_box[ax_ch[1]].set_title(f"{sel_ch} -- open eyes")

        ax_box[ax_ch[0]].annotate(f'closed eyes', xy=(.975, .975), xycoords='axes fraction', horizontalalignment='right', verticalalignment='top', fontsize=10)
        ax_box[ax_ch[1]].annotate('open eyes', xy=(.975, .975), xycoords='axes fraction', horizontalalignment='right', verticalalignment='top', fontsize=10)
        
        # ax_box[0].set_ylabel(f"closed eyes")
        # ax_box[3].set_ylabel(f"open eyes")
        # ax_box[0].legend([f"closed eyes"],loc="upper right",)
        # ax_box[1].legend([f"closed eyes"],loc="upper right",)
        # ax_box[2].legend([f"closed eyes"],loc="upper right",)
        # ax_box[3].legend([f"open eyes"],  loc="upper right",)
        # ax_box[4].legend([f"open eyes"],  loc="upper right",)
        # ax_box[5].legend([f"open eyes"],  loc="upper right",)

        # ax_box[0].set_title(f"resting (beginning)")
        # ax_box[1].set_title(f"cycling (middle)")
        # ax_box[2].set_title(f"resting (end)")
        # ax_box[3].set_title(f"resting (beginning)")
        # ax_box[4].set_title(f"cycling (middle)")
        # ax_box[5].set_title(f"resting (end)")

    ax_box[0].set_ylim([-17,17])

    fig_box.supylabel(f"dB change from baseline")
    # fig_box.supxlabel(f"cycling")
    fig_box.suptitle(f"{info_p}\n{label_band}")
    ##save fig
    fig_box.savefig(f"{path_fig_boxplot}{sel_band}_boxplot.png")

    #############################
    # ## estimate psd from tf normalized data (self.raw_seg)
    # psds, freqs = psd_array_multitaper(x, sfreq)
    
    # create_fig_psds(obj_list, label_seg_list, sel_ch,)
    
    plt.show(block=True)

    return 0


def create_fig_boxplot_single(obj_list, label_seg_list, sel_ch, sel_band):
    ## create fig
    # fig_box, ax_box = plt.subplots(nrows=2, ncols=1, figsize=(9,6), sharey=True,)
    # ax_box = ax_box.flatten()
    
    # label_seg_list = ['a_ce','a_oe','b_ce','b_oe','c_ce','c_oe']
    ## index's list to visualizer results during each label_seg in a subplots figure
    # labels = [f'before\ncycling',f'during\ncycling',f'after\ncycling']
    ax_list = [0,1,0,1,0,1]
    c_list = []
    o_list = []
    for label_seg, ax_id in zip(label_seg_list, ax_list):
        # print(f"label_seg: {label_seg}")
        data, accu = get_data_boxplot(obj_list, label_seg, sel_ch, sel_band)
        # print(f"data {label_seg} len(accu):\n{len(accu)}")
        if ax_id == 0:
            c_list.append(accu)
        else:
            o_list.append(accu)
        
    # # obj.plot_curves_beta_bands(ch_name_list)
    # # obj.boxplots_beta_bands(ch_name_list)
    # ax_box[0].boxplot(c_list, sym='', tick_labels=labels,)
    # ax_box[0].set_title(f"{'closed eyes'}")
    # ax_box[1].boxplot(o_list, sym='', tick_labels=labels,)
    # ax_box[1].set_title(f"{'open eyes'}")
    
    # # ax_box[0].set_ylabel(f"closed eyes")
    # # ax_box[3].set_ylabel(f"open eyes")
    # # ax_box[0].legend([f"closed eyes"],loc="upper right",)
    # # ax_box[1].legend([f"closed eyes"],loc="upper right",)
    # # ax_box[2].legend([f"closed eyes"],loc="upper right",)
    # # ax_box[3].legend([f"open eyes"],  loc="upper right",)
    # # ax_box[4].legend([f"open eyes"],  loc="upper right",)
    # # ax_box[5].legend([f"open eyes"],  loc="upper right",)

    # # ax_box[0].set_title(f"resting (beginning)")
    # # ax_box[1].set_title(f"cycling (middle)")
    # # ax_box[2].set_title(f"resting (end)")
    # # ax_box[3].set_title(f"resting (beginning)")
    # # ax_box[4].set_title(f"cycling (middle)")
    # # ax_box[5].set_title(f"resting (end)")

    # ax_box[0].set_ylim([-17,17])

    # fig_box.supylabel(f"dB change from baseline")
    # # fig_box.supxlabel(f"cycling")
    # fig_box.suptitle(f"{sel_band} band -- {sel_ch} accumulated iterations")

    return c_list, o_list


def create_fig_boxplot_average(obj_list, label_seg_list, sel_ch, sel_band):
    ## create fig
    fig_box, ax_box = plt.subplots(nrows=2, ncols=3, figsize=(9,6), sharey=True,)
    ax_box = ax_box.flatten()
    
    # label_seg_list = ['a_ce','a_oe','b_ce','b_oe','c_ce','c_oe']
    ## index's list to visualizer results during each label_seg in a subplots figure
    ax_list = [0,3,1,4,2,5]
    for label_seg, ax_id in zip(label_seg_list, ax_list):
        # print(f"label_seg: {label_seg}")
        data, accu = get_data_boxplot(obj_list, label_seg, sel_ch, sel_band)
        
        print(f"data boxplot length accu:\n{len(accu)}")
        
        # obj.plot_curves_beta_bands(ch_name_list)
        # obj.boxplots_beta_bands(ch_name_list)
        ax_box[ax_id].boxplot(accu, sym='')
        ax_box[ax_id].set_title(f"{label_seg}")
    
    # ax_box[0].set_ylabel(f"closed eyes")
    # ax_box[3].set_ylabel(f"open eyes")
    # ax_box[0].legend([f"closed eyes"],loc="upper right",)
    # ax_box[1].legend([f"closed eyes"],loc="upper right",)
    # ax_box[2].legend([f"closed eyes"],loc="upper right",)
    # ax_box[3].legend([f"open eyes"],  loc="upper right",)
    # ax_box[4].legend([f"open eyes"],  loc="upper right",)
    # ax_box[5].legend([f"open eyes"],  loc="upper right",)

    # ax_box[0].set_title(f"resting (beginning)")
    # ax_box[1].set_title(f"cycling (middle)")
    # ax_box[2].set_title(f"resting (end)")
    # ax_box[3].set_title(f"resting (beginning)")
    # ax_box[4].set_title(f"cycling (middle)")
    # ax_box[5].set_title(f"resting (end)")

    ax_box[0].set_ylim([-17,17])

    fig_box.supylabel(f"dB change from baseline")
    fig_box.supxlabel(f"iteration number")
    fig_box.suptitle(f"{sel_band} band -- {sel_ch} accumulated iterations")

    return 0


def create_fig_boxplot(obj_list, label_seg_list, sel_ch, sel_band):
    ## create fig
    fig_box, ax_box = plt.subplots(nrows=2, ncols=3, figsize=(9,6), sharey=True,)
    ax_box = ax_box.flatten()
    
    # label_seg_list = ['a_ce','a_oe','b_ce','b_oe','c_ce','c_oe']
    ## index's list to visualizer results during each label_seg in a subplots figure
    ax_list = [0,3,1,4,2,5]
    for label_seg, ax_id in zip(label_seg_list, ax_list):
        # print(f"label_seg: {label_seg}")
        data, accu = get_data_boxplot(obj_list, label_seg, sel_ch, sel_band)
        print(f"data boxplot length:\n{len(accu)}")

        # print(f"data_dict:\n{data}")
        # obj.plot_curves_beta_bands(ch_name_list)
        # obj.boxplots_beta_bands(ch_name_list)
        ax_box[ax_id].boxplot(data, sym='')
        ax_box[ax_id].set_title(f"{label_seg}")
    
    # ax_box[0].set_ylabel(f"closed eyes")
    # ax_box[3].set_ylabel(f"open eyes")
    # ax_box[0].legend([f"closed eyes"],loc="upper right",)
    # ax_box[1].legend([f"closed eyes"],loc="upper right",)
    # ax_box[2].legend([f"closed eyes"],loc="upper right",)
    # ax_box[3].legend([f"open eyes"],  loc="upper right",)
    # ax_box[4].legend([f"open eyes"],  loc="upper right",)
    # ax_box[5].legend([f"open eyes"],  loc="upper right",)

    # ax_box[0].set_title(f"resting (beginning)")
    # ax_box[1].set_title(f"cycling (middle)")
    # ax_box[2].set_title(f"resting (end)")
    # ax_box[3].set_title(f"resting (beginning)")
    # ax_box[4].set_title(f"cycling (middle)")
    # ax_box[5].set_title(f"resting (end)")

    ax_box[0].set_ylim([-17,17])

    fig_box.supylabel(f"dB change from baseline")
    fig_box.supxlabel(f"iteration number")
    fig_box.suptitle(f"{sel_band} band -- {sel_ch}")

    return 0

def get_data_boxplot(obj_list, label_seg, sel_ch, sel_band):
    data = []
    accu = []
    for obj in obj_list:
        ## includes data from all iterations of selected label_seg
        if obj.get_label() == label_seg:
            # print(f"{label_seg}: {obj.get_id()}")
            df = obj.get_df_ch_bands()
            ## exclude data of annot. bad (e.i. mask=1)
            df = df.loc[df['mask']==0]
            ## get beta-low component of selected channel for boxplot
            data.append(df[f'{sel_ch}_{sel_band}'].to_list())
            accu.extend(df[f'{sel_ch}_{sel_band}'].to_list())
        else:
            pass
            # print(f"pass")

    return data, accu


def create_fig_psds(obj_list, label_seg_list, sel_ch,):
    ## create fig
    fig_box, ax_box = plt.subplots(nrows=2, ncols=3, figsize=(9,6), sharey=True, sharex=True)
    ax_box = ax_box.flatten()
    
    # label_seg_list = ['a_ce','a_oe','b_ce','b_oe','c_ce','c_oe']
    ## index's list to visualizer results during each label_seg in a subplots figure
    ax_list = [0,3,1,4,2,5]
    for label_seg, ax_id in zip(label_seg_list, ax_list):
        # print(f"label_seg: {label_seg}")
        data = get_data_psds(obj_list, label_seg, sel_ch,)
        # print(f"data_dict:\n{data}")
        # obj.plot_curves_beta_bands(ch_name_list)
        # obj.boxplots_beta_bands(ch_name_list)
        for xd in data:
            ax_box[ax_id].plot(xd[0], xd[1])
        
        ax_box[ax_id].set_title(f"{label_seg}")
    
    # ax_box[0].set_ylabel(f"closed eyes")
    # ax_box[3].set_ylabel(f"open eyes")
    # ax_box[0].legend([f"closed eyes"],loc="upper right",)
    # ax_box[1].legend([f"closed eyes"],loc="upper right",)
    # ax_box[2].legend([f"closed eyes"],loc="upper right",)
    # ax_box[3].legend([f"open eyes"],  loc="upper right",)
    # ax_box[4].legend([f"open eyes"],  loc="upper right",)
    # ax_box[5].legend([f"open eyes"],  loc="upper right",)

    # ax_box[0].set_title(f"resting (beginning)")
    # ax_box[1].set_title(f"cycling (middle)")
    # ax_box[2].set_title(f"resting (end)")
    # ax_box[3].set_title(f"resting (beginning)")
    # ax_box[4].set_title(f"cycling (middle)")
    # ax_box[5].set_title(f"resting (end)")

    # ax_box[0].set_ylim([-17,17])

    # fig_box.supylabel(f"dB change from baseline")
    # fig_box.supxlabel(f"iteration number")
    # fig_box.suptitle(f"{sel_band} band -- {sel_ch}")

    return 0

def get_data_psds(obj_list, label_seg, sel_ch,):
    data = []
    for obj in obj_list:
        ## includes data from all iterations of selected label_seg
        if obj.get_label() == label_seg:
            # print(f"{label_seg}: {obj.get_id()}")
            raw = obj.get_raw_seg()
            ## exclude data of annot. bad (e.i. mask=1)
            # df = df.loc[df['mask']==0]
            ## get data selected channel
            ch_data = raw.get_data(picks=['E104'],reject_by_annotation='omit')
            ## get beta-low component of selected channel for boxplot
            ## estimate psd from tf normalized data (self.raw_seg)
            psds, freqs = psd_array_multitaper(ch_data[0], obj.get_sfreq(), fmin=4, fmax=75)
            data.append([freqs, 10*np.log10(psds)])
        else:
            pass
            # print(f"pass")
    return data

    # ###############################################
    # ##
    # ## tf data normalization
    # print(f"normalization {label_ref} segment...")
    # seg_ref.tf_normalization(tf_ref)
    # ## tf plot
    # print(f"plot normalized time-frequency analysis...")
    # # seg_ref.tf_plot(flag_norm=False)
    # ch_name_list = ['VREF','E36','E104']
    # for ch_name in ch_name_list:
    #     seg_ref.tf_plot(ch_name, flag_norm=True)
    #     ## band separation
    #     print(f"plot frequency bands...")
    #     seg_ref.channel_bands_power(ch_name)

    
    

    plt.show(block=True)

    return 0

    

    for label_seg in label_seg_list:
        ## create folders to save preprocesing parameters
        Path(path+'session_'+str(session)+f"/prep/{label_seg}/").mkdir(parents=True, exist_ok=True)
        Path(path+'session_'+str(session)+f"/prep/{label_seg}/figures/").mkdir(parents=True, exist_ok=True)

        filename_fig_tf0 = path+'session_'+str(session)+f'/prep/{label_seg}/figures/'+f'tf_1.png'
        filename_fig_tf1 = path+'session_'+str(session)+f'/prep/{label_seg}/figures/'+f'tf_2.png'
        filename_fig_bands = path+'session_'+str(session)+f'/prep/{label_seg}/figures/'+f'tf_3.png'
        filename_fig_box = path+'session_'+str(session)+f'/prep/'+f'boxplots_comparison.png'

        select_bads(eeg_data_dict)

        id_seg = 0
        for raw_seg, filt_seg in zip(eeg_data_dict[label_seg], eeg_filt_dict[label_seg]):
        
            filename_bad_ch = path+'session_'+str(session)+f'/prep/{label_seg}/'+f'bad_ch_{id_seg}.csv'
            filename_annot  = path+'session_'+str(session)+f'/prep/{label_seg}/'+f'annot_{id_seg}.fif'
            filename_ica    = path+'session_'+str(session)+f'/prep/{label_seg}/'+f'ica_{id_seg}.fif.gz'
        

        ##
        ## to load pre-calculated (selected) values
        flag_prep = input(f'Load pre-selected bad-channels and bad-segments (1-True, 0-False) ?: ')
        flag_prep = 1 if (flag_prep == '') else int(flag_prep)

        if flag_prep==1:
            ## load bad channel list
            load_df = pd.read_csv(filename_bad_ch)
            bad_ch_list = load_df['bad_ch'].to_list()
            raw_seg.info['bads'] = bad_ch_list
            ##
            ## read annotation bad_seg
            my_annot = mne.read_annotations(filename_annot)
            # print(f'annotations:\n{my_annot}')
            ## adding annotations to raw data
            raw_seg.set_annotations(my_annot)

            mne.viz.plot_raw(raw_seg, picks=['eeg','ecg'], start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title='Time-series signals EEG (baseline)\nPlease select bad segments and bad channels interactively', block=True)

        else:
            pass
        #########################################################
        ## iterative bad segments and bad channels identification
        ##
        flag_bad_ch = True
        while flag_bad_ch:
            ##
            ## raw data visualization (baseline)
            ## visual observation of time-series and psd from raw data helps to identify bad channels
            ## using butterfly view (choosing 'b' in the time-series plot) helps to identify bad segments
            ##
            ## interactively, include annotations of bad segments (with the label 'bad_seg'), namely, sections of the data where the majority of channels are affected by noise or artefacts
            ##
            ## interactively, select bad channels: flat lines or noisy channels
            ##
            mne.viz.plot_raw(raw_seg, picks=['eeg','ecg'], start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title=f"Time-series signals EEG {label_seg} -- Please select bad segments and bad channels interactively", block=True)
            ##
            ## power spectral density (psd)
            ##
            ## interactively, identify channels that are out of the tendency (ouliers) as bad channels
            ## 
            fig_psd, ax_psd = plt.subplots(nrows=1, ncols=1, figsize=(9,4), sharey=True, sharex=True)
            mne.viz.plot_raw_psd(raw_seg, picks=['eeg'], exclude=['VREF'], ax=ax_psd, fmin=0.9, fmax=101, xscale='log',)
            ##
            ax_psd.set_title('EEG power spectral density (baseline)')
            ##
            bad_ch_list = raw_seg.info['bads']
            ##
            flag_bad_ch = input('Include more bad channels? (1 (True), 0 (False)): ')
            flag_bad_ch = 0 if (flag_bad_ch == '') else int(flag_bad_ch)
            # print(f"flag bad_ch: {flag_bad_ch}")
        ##
        ##
        # print(f"excluded bad channels: {bad_ch_list}")
        ##
        flag_rewrite = input(f"(Re-)Write bad channels and bad-segments (1-True, 0-False)?: ")
        flag_rewrite = 0 if flag_rewrite == '' else int(flag_rewrite)
        ##
        if flag_rewrite==1:
            ## save bad channels list to csv
            data_dict = {}
            data_dict['bad_ch'] = bad_ch_list
            df = pd.DataFrame(data_dict)
            # print(f"dataframe:\n{df}")
            df.to_csv(filename_bad_ch)
            ##
            ## save annotations to .fif
            ## annotation time is referenced to the time of first_samp, and that is different for each section
            time_offset = raw_seg.first_samp / sampling_rate  ## in seconds
            ## get and rewrite annotations minus time-offset
            interactive_annot = raw_seg.annotations
            annot_offset = ann_remove_offset(interactive_annot, time_offset)
            ## saving annotations
            annot_offset.save(filename_annot, overwrite=True)
        else:
            pass
            # print(f"Bad channels and annotations were not saved.")
            # return 0
        ##
        ## print(f"bad_ch_list: {bad_ch_list}")
        ##
        ## bad segments and bad channels identification
        ###########################################################
        ## re-refencing
        ##
        ## after bad channels selection, we apply average re-referencing (which exclude bad channels)
        ##
        ## the average referencing improve signals quality that we can see in the time-series and psd
        ##
        raw_seg.set_eeg_reference(ref_channels="average", ch_type='eeg')
        ##
        ## re-refencing
        ###########################################################
        ## raw data visualization (baseline)
        # mne.viz.plot_raw(raw_seg2, start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title='initial baseline average ref.', block=False)

        # mne.viz.plot_raw_psd(raw_seg2, exclude=['VREF'], ax=ax_psd[1], fmin=0.9, fmax=101, xscale='log',)
        # ax_psd[1].set_title('PSD Baseline, EEG, Average Ref.')
        ###########################################################
        ## ICA to remove blinks and other artifacts
        ##
        flag_ica = input('Load pre-calculated ICA components (1-True, 0-False) ?: ')
        flag_ica = 1 if (flag_ica == '') else int(flag_ica)
        ##
        if flag_ica==1 :
            # filename_ica = path+'session_'+str(session)+'/ica/'+'ica-baseline.fif.gz'
            ## print(f'filename: {filename_ica}')
            try:
                ica = mne.preprocessing.read_ica(filename_ica, verbose=None)
            except:
                print(f'Pre-calculated ICA was not found.')
                flag_ica = False
        else:
            ## calcalate ICA components
            ica = ICA(n_components= 0.99, method='picard', max_iter="auto", random_state=97)
            ##
            ## ica works better with a signal with offset 0; a high pass filter with a 1 Hz cutoff frequency could improve that condition
            filt_raw = raw_seg.copy().filter(l_freq=1.0, h_freq=None)
            ##
            ## ICA fitting model to the filtered raw data
            ica.fit(filt_raw, reject_by_annotation=True)
            ##
            print(f"saving ica model in {filename_ica}")
            ica.save(filename_ica, overwrite=True)
            ##
            ## interactive selection of ICA components to exclude
            ## ica components visualization
        ica.plot_components(inst=raw_seg, contours=0,)
        ica.plot_sources(raw_seg, start=0, stop=240, show_scrollbars=False, block=True)
        ##
        ## manual selection de components to exclude (not necessary if they were choosen interactively)
        # ica.exclude = [0,4] # indices that were chosen# based on previous ICA calculations
        # print(f'ICA blink component: {ica.exclude}')
        raw_seg_ica = raw_seg.copy()
        ica.apply(raw_seg_ica)
        ##
        ## interpolate bad channels
        raw_seg_ica.interpolate_bads()
        ##
        ## Surface Laplacian (current source density)
        raw_seg_csd = mne.preprocessing.compute_current_source_density(raw_seg_ica)
        ##
        ## data visualization
        mne.viz.plot_raw(raw_seg, start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, filtorder=4, title=f'{label_seg} before ICA', block=False)
        mne.viz.plot_raw(raw_seg_ica, start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, filtorder=4, title=f'{label_seg} after ICA', block=False)
        mne.viz.plot_raw(raw_seg_csd, start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, filtorder=4, title=f'{label_seg} after Surface Laplacian', block=True)
        ##
        ## ICA to remove blinks and other artifacts
        ###########################################################
        ## Time-frequency (tf) decomposition from each EEG channel
        ##
        ## average of tf-components per frequency per channel
        ## we save that information as a reference for data normalization
        ##
        ## time frequency data visualization
        ## logarithmic
        start=  0.60 # 10^start,  
        stop =  1.50 # 10^stop
        num  = 50 # samplesq
        freqs = np.logspace(start, stop, num=num,)
        # print(f'log freqs: {freqs}')

        # tfr_bl = raw_seg_ica.compute_tfr('morlet',freqs, picks=['eeg'])
        # data_bl, times_bl, freqs_bl = tfr_bl.get_data(picks=['eeg'],return_times=True, return_freqs=True)
        
        tfr_bl = raw_seg_csd.compute_tfr('morlet',freqs, reject_by_annotation=False,)
        data_bl, times_bl, freqs_bl = tfr_bl.get_data(return_times=True, return_freqs=True)

        # print(tfr_bl)
        print(f"{label_seg} type (tfr_power): {type(tfr_bl)}")
        print(f"data shape:\n{data_bl.shape}")
        print(f"data:\n{data_bl}")
        ##
        # chx = 0
        # data_chx = data_bl[chx]
        # print(f"data chx shape: {data_chx.shape}")
        # print(f"data chx:\n{data_chx}")
        # mean_chx = np.mean(data_chx, axis=1)
        # print(f"mean chx shape: {mean_chx.shape}")
        # print(f"mean chx:\n{mean_chx}")
        #  print(f"times:\n{times}")
        # print(f"freqs: {len(freqs_bl)}\n{freqs_bl}")

        ## data visualization
        # tfr_bl.plot(picks=['VREF'], title='auto', yscale='auto', show=False)
        # tfr_bl.plot(picks=['VREF'], title='auto', yscale='linear', show=False)
        # tfr_bl.plot(picks=['VREF'], title='auto', yscale='log', show=False)
        
        # ## mean along time samples
        # # for each channel, an average for each frequency
        # mean_bl = np.mean(data_bl, axis=2)
        # # print(f"mean data shape: {mean_bl.shape}")
        # # print(f"mean arr:\n{mean_bl}")
        # #
        # # save mean_bl for data normalization
        # np.save(filename_tr_ref, mean_bl)

        # print(f"loading mean_bl...")
        # mean_bl = np.load(filename_tr_ref)
        # print(f"mean arr:\n{mean_bl}")
        # print(f"done")

        ###############################
        ## time-frequency power normalization for each channel
        dim_ch, dim_fr, dim_t = data_bl.shape
        print(f"bl dim_ch, dim_fr, dim_t: {dim_ch, dim_fr, dim_t}")
        
        ## initialization new array
        data_bl_norm = np.zeros((dim_ch, dim_fr, dim_t))

        id_ch=0
        print(f"normalization ch:\n")
        for mean_ch, arr_num in zip(mean_bl, data_bl):
            # mean for each frequency per channel
            ## mean_ch is an array with a number of elements equal to the number of evaluated frequencies
            ## each element of the array represents the mean value of time samples per each frequency
            arr_den = np.repeat(mean_ch, dim_t ,axis=0).reshape(len(mean_ch),-1)
            arr_dB = 10*np.log10(arr_num / arr_den)
            # print(f"mean_ch ch arr_res:{mean_ch.shape} {id_ch}, {arr_dB.shape}")
            # data_bl[id_ch] = arr_dB
            data_bl_norm[id_ch] = arr_dB
            print(f"{id_ch}", end=", ")
            id_ch+=1
        print(f"")

        ## baseline scaling
        ## dB = 10*log10( matrix_time_freq / mean_for_each_freq_baseline )
        # data_2 = data*20
        tfr_bl_norm = tfr_bl.copy()
        tfr_bl_norm._data = data_bl_norm

        ## measure of central tendecy -- median of power bands per time-sample per EEG-channel
        df_all = average_power_bands(data_bl_norm ,times_bl, freqs_bl)

        print(f"df all:\n{df_all}")

        ## selected channel
        ch_label = 'VREF'

        # data_norm, times_vref, freqs_vref = tfr_bl_norm.get_data(picks=[ch_label],return_times=True, return_freqs=True)
        # print(f"vref data shape: {data_norm.shape}")
        ##
        ## data visualization
        fig_tf0, ax_tf0 = plt.subplots(nrows=1, ncols=1, figsize=(16,4), sharey=True, sharex=True)
        fig_tf1, ax_tf1 = plt.subplots(nrows=1, ncols=1, figsize=(16,4), sharey=True, sharex=True)

        vlim = (-12,12)
        mask_tf = np.zeros((dim_fr, dim_t)).astype(bool)
        # tfr_bl.plot(picks=['VREF'], title='auto', yscale='auto', show=False)
        
        tfr_bl_norm.plot(picks=[ch_label], title=f'Power ({ch_label}) normalized\ndB change from baseline', yscale='auto', vlim = vlim, axes=ax_tf0, show=False)
        # tfr_bl_norm.plot(picks=[ch_label], title=f'Power ({ch_label}) normalized\ndB change from baseline', yscale='auto', vlim = vlim, show=False)

        data_vref, times_vref, freqs_vref = tfr_bl_norm.get_data(picks=[ch_label],return_times=True, return_freqs=True)
        print(f"{ch_label} data shape: {data_vref.shape}")
        ##
        ## normalized tf matrix to dataframe [VREF]
        df_tf = pd.DataFrame(data_vref[0])
        df_tf['freq'] = freqs_vref
        # print(f"df_tf:\n{df_tf}")
        ##
        ## accumulative power per band per every time sample
        ##
        ## theta (4-8 Hz), alpha (8-12 Hz), beta (12-30 Hz)
        # selecting rows based on condition
        df_theta = df_tf.loc[(df_tf['freq'] >= 4)  & (df_tf['freq'] < 8)]
        df_alpha = df_tf.loc[(df_tf['freq'] >= 8)  & (df_tf['freq'] < 12)]
        df_beta  = df_tf.loc[(df_tf['freq'] >= 12) & (df_tf['freq'] < 30)]

        # print(f"df_theta:\n{df_theta}")
        # print(f"df_theta shape:\n{df_theta.shape}")
        ##
        ## exclude column freq
        df_theta = df_theta.loc[:,df_theta.columns != 'freq']
        df_alpha = df_alpha.loc[:,df_alpha.columns != 'freq']
        df_beta  =  df_beta.loc[:,df_beta.columns  != 'freq']

        ## calculate mean power in the theta band for each time sample
        # theta_mean = df_theta.mean(axis=0).to_numpy()
        theta_median = df_theta.median(axis=0).to_numpy()
        alpha_median = df_alpha.median(axis=0).to_numpy()
        beta_median  = df_beta.median(axis=0).to_numpy()


        tfr_bl_norm.plot(picks=[ch_label], title=f'Power ({ch_label}) normalized\ndB change from baseline', yscale='auto', vlim = vlim, mask=mask_tf, mask_alpha=0.7, mask_cmap='coolwarm', axes=ax_tf1, show=False)

        ax_tf1.axhline(y=4.0, alpha=1.0, drawstyle='steps',linestyle='--', color='tab:green')
        ax_tf1.axhline(y=8.0, alpha=1.0, drawstyle='steps',linestyle='--', color='tab:green')
        ax_tf1.axhline(y=12.0, alpha=1.0, drawstyle='steps',linestyle='--', color='tab:green')
        ax_tf1.axhline(y=30.0, alpha=1.0, drawstyle='steps',linestyle='--', color='tab:green')

        # ax_tf[1].plot(times_vref, freqs_vref_max)
        # ax_tf[1].axhline(y=8.0, alpha=0.3, drawstyle='steps',linestyle='--')

        fig_bands, ax_bands = plt.subplots(nrows=3, ncols=1, figsize=(9,6), sharey=True, sharex=True)
        ax_bands[0].plot(times_vref, beta_median,  label='beta [12-30 Hz]')
        ax_bands[1].plot(times_vref, alpha_median, label='alpha [8-12 Hz]')
        ax_bands[2].plot(times_vref, theta_median, label='theta [4-8 Hz]')

        ax_bands[0].set_ylim([-15,15])
        # ax_bands[0].legend(loc='upper right')
        # ax_bands[1].legend(loc='upper right')
        # ax_bands[2].legend(loc='upper right')

        ax_bands[-1].set_xlabel('Time [s]')
        ax_bands[0].set_title(f'Power({ch_label}) -- dB change from baseline [median]')

        data_dict = {
            f'{label_seg}_beta' : beta_median,
            f'{label_seg}_alpha' : alpha_median,
            f'{label_seg}_theta' : theta_median,
        }
        df_bands = pd.DataFrame(data_dict)
        df_bands_all = pd.concat([df_bands_all, df_bands],axis=1)

        ## save figures
        fig_tf0.savefig(filename_fig_tf0, dpi=fig_tf0.dpi, bbox_inches='tight', pad_inches=0.5)
        fig_tf1.savefig(filename_fig_tf1, dpi=fig_tf0.dpi, bbox_inches='tight', pad_inches=0.5)
        fig_bands.savefig(filename_fig_bands, dpi=fig_tf0.dpi, bbox_inches='tight', pad_inches=0.5)


        # ##################
        # ## topoplot for a selected moment in time
        # ##
        # # t_sel = 8.4 # 6.25 4.25 seconds 
        # t_sel = float(input(f"select time: "))


        # df_sel = df_all.loc[(df_all['time'] > (t_sel-0.002)) & (df_all['time'] < (t_sel+0.002))]
        # print(f'df_sel:\n{df_sel}')
        
        # arr_theta = get_data_band(df_sel, 'theta')
        # arr_alpha = get_data_band(df_sel, 'alpha')
        # arr_beta  = get_data_band(df_sel, 'beta')

        # fig_tp, ax_tp = plt.subplots(nrows=3, ncols=1, figsize=(2.5, 9),)
        
        # im, cn = mne.viz.plot_topomap(arr_beta, tfr_bl_norm.info, vlim=vlim, contours=0, cmap='coolwarm',axes=ax_tp[0], ) 
        # im, cn = mne.viz.plot_topomap(arr_alpha, tfr_bl_norm.info, vlim=vlim, contours=0, cmap='coolwarm',axes=ax_tp[1],) 
        # im, cn = mne.viz.plot_topomap(arr_theta, tfr_bl_norm.info, vlim=vlim, contours=0, cmap='coolwarm',axes=ax_tp[2],) 
        # # im, cn = mne.viz.plot_topomap(arr_data[0], tfr_bl_norm.info, vlim=vlim, contours=0, cmap='coolwarm',axes=ax_tp[1]) # raw_seg  cmap='magma' 'coolwarm', 'RdBu_r'
        
        # ax_tp[0].set_title('beta [12-30 Hz]')
        # ax_tp[1].set_title('alpha [8-12 Hz]')
        # ax_tp[2].set_title('theta [4-8 Hz]' )

        # # Make colorbar
        # cbar_ax = fig_tp.add_axes([0.03, 0.075, 0.94, 0.02])
        # fig_tp.colorbar(im, cax=cbar_ax, label='dB change from baseline', location='bottom')

        # fig_tp.suptitle(f"Topographic maps\ntime = {t_sel} s")
        # ##
        # ## topoplot for a selected moment in time
        # ##################

    print(f"df bands all:\n{df_bands_all}")
    ##
    ## boxplots
    ##
    fig_box, ax_box = plt.subplots(nrows=1, ncols=3, figsize=(9,6), sharey=True,)
    df_bands_all.boxplot(['a_closed_eyes_theta','b_closed_eyes_theta'], showfliers=False, ax=ax_box[0])
    df_bands_all.boxplot(['a_closed_eyes_alpha','b_closed_eyes_alpha'], showfliers=False, ax=ax_box[1])
    df_bands_all.boxplot(['a_closed_eyes_beta' ,'b_closed_eyes_beta'], showfliers=False, ax=ax_box[2])

    ax_box[0].set_xticks([1, 2], ['a', 'b',])
    ax_box[1].set_xticks([1, 2], ['a', 'b',])
    ax_box[2].set_xticks([1, 2], ['a', 'b',])

    ax_box[0].set_title(f"theta band")
    ax_box[1].set_title(f"alpha band")
    ax_box[2].set_title(f"beta band")

    fig_box.savefig(filename_fig_box, dpi=fig_tf0.dpi, bbox_inches='tight', pad_inches=0.5)

    # d1 = [
    #     df_band
    #     , alpha_median, beta_median]
    # ax_box[0].boxplot(d, showfliers=False)
    # ax_box[0].set_title(f"{label_seg} boxplot mean power in frequency bands")
    # ax_box[0].set_xticks([1, 2, 3], ['theta', 'alpha', 'beta'])

    

    plt.show(block=True)
    return 0

    
    
    
    ###########################
    ## interpolation of bad channels per section
    eeg_data_dict = channels_interpolation(eeg_data_dict, subject, session)
    
    ###########################
    ## set annotations of bad segments per section (interactive annotations previously made [inspection.py])
    # ann_filename = path + fn_csv[1] + '.pkl'
    path_ann = path+'session_'+str(session)+"/new_annotations/"
    eeg_data_dict = set_bad_segments(eeg_data_dict, path_ann, fn_csv[1])

    #########################
    ## ICA for blink removal using precalculate ICA (ica_blinks.py)
    eeg_data_dict = ica_appl_func(path, eeg_data_dict, subject, session, scale_dict)

    ##########################
    ## current source density
    ## Surface Laplacian
    csd_eeg_data_dict = csd_fun(eeg_data_dict)

    ##########################################################
    # ##########################################################
    # ## when rest and bike are in two different files, we save baseline for rest, and we open baseline for bike
    # ## save baseline
    # flag_bl = input('save baseline ? (1 (True), 0 (False)): ')
    # if int(flag_bl)==1:
    #     eeg_data_dict['baseline'][0].save(path + 'baseline.fif.gz')
    # else:
    #     pass
    
    # ## load baseline
    # flag_bl = input('load baseline ? (1 (True), 0 (False)): ')
    # if int(flag_bl)==1:
    #     eeg_data_dict['baseline'] = [mne.io.read_raw_fif(path + 'baseline.fif.gz',)]
    # else:
    #     pass
    # 
    # print(f'baseline:\n{eeg_data_dict["baseline"]}')
    # ###########################################################
    ##########################################################

    ## At this point, blink artifacts have been removed and the Surface Lapacian has been applied to the eeg data. Additionally, evident artifacts were annotated interactively and labeled as "bad" to exclude them from posterior calculations

    ################################################################################
    ## compare topographical maps before and after the filter: surface Laplacian
    ####################
    ## compare topographical maps before and after the filter: surface Laplacian

    ## VREF is excluded because the signal's  values are zero (problem with normalization)
    # bl_spectrum = baseline_spectrum(eeg_data_dict, flag='')
    # raw_data_mod = raw_data.copy().pick('all', exclude=['VREF'])
    # topomaps_normalization(raw_data_mod, eeg_data_dict, bl_spectrum, session, path,flag='')

    # csd_bl_spectrum = baseline_spectrum(csd_eeg_data_dict, flag='csd')
    # topomaps_normalization(raw_data, csd_eeg_data_dict, csd_bl_spectrum, session, path, flag='csd')
    ####################
    ## compare topographical maps before and after the filter: surface Laplacian
    ################################################################################

    ##############################################
    ## time-frequency representation
    ## wavelets
    print(f'time-frequency representation')
    ## linear
    start=  4.0 # Hz
    stop = 45.0 # Hz
    num  = 50 # samples
    # freqs = np.arange(4.0, 45.0, 1)
    # freqs = np.linspace(start, stop, num=num,)

    ## logarithmic
    start=  0.60 # 10^start,  
    stop =  1.65 # 10^stop
    num  = 50 # samples
    freqs = np.logspace(start, stop, num=num,)
    print(f'log freqs: {freqs}')

    baseline_list = csd_eeg_data_dict['baseline']

    # mne.viz.plot_raw(baseline_list[0], start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title='baseline csd data', block=True)


    tfr_bl = baseline_list[0].compute_tfr('morlet',freqs,)
    print(f"baseline type(tfr_power): {type(tfr_bl)}")
    print(tfr_bl)

    data_bl, times_bl, freqs_bl = tfr_bl.get_data(picks=['all'],return_times=True, return_freqs=True)
    # picks=['AFz','Cz','POz']
    print(f"data:\n{data_bl.shape}")
    print(f"data:\n{data_bl}")
    #
    #  print(f"times:\n{times}")
    print(f"freqs:\n{freqs_bl}")

    ## data visualization
    # tfr_bl.plot(picks=['VREF'], title='auto', yscale='linear', show=False)
    
    ## mean along time samples
    ## for each channel, an average for each frequency
    mean_bl = np.mean(data_bl, axis=2)
    print(f"mean data: {mean_bl.shape}")


    #####################
    label = 'b_closed_eyes'
    idx=0
    ## time frequency transformation for a segment of a_closed_eyes
    a_closed_list = csd_eeg_data_dict[label]

    mne.viz.plot_raw(a_closed_list[idx], start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title=f"{label} {idx}", block=False)


    tfr_ac = a_closed_list[idx].compute_tfr('morlet',freqs, reject_by_annotation=False)

    print(f"ac type(tfr_power): {type(tfr_ac)}")
    print(tfr_ac)

    data_ac, times_ac, freqs_ac = tfr_ac.get_data(picks=['all'],return_times=True, return_freqs=True)
    # picks=['AFz','Cz','POz']
    print(f"data ac:\n{data_ac.shape}")
    print(f"data ac:\n{data_ac}")
    # print(f"times:\n{times}")
    # print(f"freqs ac:\n{freqs_ac}")

    channels = selected_channels[subject]['session_'+str(session)][label][idx]
    print(f'channels: {channels}')

    # for ch in channels:
    #     tfr_ac.plot(picks=[ch], title='auto', yscale='log',)

    dim_ch_ac, dim_fr_ac, dim_t_ac = data_ac.shape
    print(f"ac dim_ch, dim_fr, dim_t: {dim_ch_ac, dim_fr_ac, dim_t_ac}")

    id_ch=0
    for mean_ch, arr_num in zip(mean_bl, data_ac):
        # mean for each frequency per channel
        ## mean_ch is an array with a number of elements equal to the number of evaluated frequencies
        ## each element of the array represents the mean value of time samples per each frequency
        arr_den = np.repeat(mean_ch, dim_t_ac ,axis=0).reshape((len(mean_ch),-1))
        arr_dB = 10*np.log10(arr_num / arr_den)
        # print(f"mean_ch ch arr_res:{mean_ch.shape} {id_ch}, {arr_dB.shape}")
        data_ac[id_ch] = arr_dB
        id_ch+=1

    ## baseline scaling
    ## dB = 10*log10( matrix_time_freq / mean_for_each_freq_baseline )
    # data_2 = data*20
    tfr_ac._data = data_ac

    ## create folder if it does not exit already
    Path(f"{path}session_{session}/figures/tfr").mkdir(parents=True, exist_ok=True)
    
    fig_tfr, axs_tfr = plt.subplots(3, 1, figsize=(12.0, 10.0))
    axs_tfr = axs_tfr.flat

    vlim = (-10,10)
    for ch, ax in zip(channels, axs_tfr):
        tfr_ac.plot(picks=[ch], title=None, yscale='log', vlim=vlim, axes=ax)
        ax.set_title(f"")
        ax.set_xlabel(f"")

    axs_tfr[-1].set_xlabel(f"Time (s)")        
    filename_fig = f"{path}session_{session}/figures/tfr/{label}.png"
    fig_tfr.savefig(filename_fig, transparent=True)
    
    # ## wavelets
    # ## time-frequency representation
    # ##############################################

    plt.show(block=True)
    return 0

    ##########################


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
