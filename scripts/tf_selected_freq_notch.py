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
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.widgets as mwidgets
from matplotlib.backend_bases import MouseButton

# Import the FOOOF object
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic
from fooof.plts.spectra import plot_spectra
from fooof.plts.annotate import plot_annotated_peak_search

from channels_tfr import selected_channels
## include modules from another directory
sys.path.insert(0, '../../scripts')
from bad_channels import bad_channels_dict
from list_participants import participants_list

from class_tf_notch import TF_components


sampling_rate = 1.0
y_limits = [-8,8]
ylim_global = [0, 40]
freq_range = [1,45]

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

## interactive plots
ax_ce_global = []
fig_ce_global = []
fig_mea = []
ax_mea = []
fig_ce=[]
fig_oe=[]
ax_ce=[]
ax_oe=[]

obj_list = []
selectors=[]
df_psd_global = pd.DataFrame()
f0_global=0
f1_global=0
flag_eyes_closed = True

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

#######################################
#############################
def baseline_normalization_two_ref(obj_list, selected_segs_dict,):
    ## dictionary with labels and ids of selected segements
    label_ref_oe = 'a_oe'
    label_ref_ce = 'a_ce'
    id_ref_oe = selected_segs_dict[label_ref_oe]
    id_ref_ce = selected_segs_dict[label_ref_ce]

    ## get ref normalization for closed-eyes and open-eyes
    for obj in obj_list:
        if (obj.get_label_simple() == label_ref_oe) and (obj.get_id() == id_ref_oe):
            ## ref for open-eyes
            seg_ref_oe = obj
        elif (obj.get_label_simple() == label_ref_ce) and (obj.get_id() == id_ref_ce):
            ## ref for closed-eyes
            seg_ref_ce = obj
        else:
            pass

    print(f"Selected segment to calculate reference baseline open eyes: {seg_ref_oe.get_label()}_{seg_ref_oe.get_id()}")
    print(f"Selected segment to calculate reference baseline closed eyes: {seg_ref_ce.get_label()}_{seg_ref_ce.get_id()}")

    ## get reference for baseline normalization
    # print(f"calculating average values for time-frequency normalization...")
    tf_ref_oe, freq_tf_oe = seg_ref_oe.get_tf_baseline()
    tf_ref_ce, freq_tf_ce = seg_ref_ce.get_tf_baseline()
    
    ## normalization of each of the  selected segments
    for obj in obj_list:
        ## find the selected segment for each label
        if obj.get_selected_flag():
            print(f"{obj.get_label()}-{obj.get_id()}:")
            # print(f"Tf normalization... ")
            if obj.get_label_simple() in ['a_oe','b_oe','c_oe']:
                ## normalization open-eyes segments
                obj.tf_normalization(tf_ref_oe)
            elif obj.get_label_simple() in ['a_ce','b_ce','c_ce']:
                ## normalization closed-eyes segments
                obj.tf_normalization(tf_ref_ce)
            else:
                pass

    return 0


##################################################################
def plot_curves_cycling(obj_list, filename, fig_title):

    print(f"plot normalized curves...")
    fig, ax = plt.subplots(3,3,sharex=True,sharey=True, figsize=(10,7))

    ## zero line of reference
    for id in np.arange(3):
        ax[0][id].axhline(y=0.0, color='black', linestyle='-', linewidth = 0.5, alpha=0.5)
        ax[1][id].axhline(y=0.0, color='black', linestyle='-', linewidth = 0.5, alpha=0.5)
        ax[2][id].axhline(y=0.0, color='black', linestyle='-', linewidth = 0.5, alpha=0.5)

        ax[0][id].grid(color = 'gray', linestyle = '--', linewidth = 0.5, alpha=0.5)
        ax[1][id].grid(color = 'gray', linestyle = '--', linewidth = 0.5, alpha=0.5)
        ax[2][id].grid(color = 'gray', linestyle = '--', linewidth = 0.5, alpha=0.5)

    for obj in obj_list:
        ## find the selected segment for each label
        if obj.get_selected_flag():
            print(f"{obj.get_label()}-{obj.get_id()}:")
            arr_mean, arr_freqs = obj.get_tf_mean()
            # print(f"mean values:\n{arr_mean}\nfreqs: {arr_freqs}")
            label = obj.get_label_simple()
            if label.startswith('a'):
                ## resting 1
                row = 0
            elif label.startswith('b'):
                ## cycling
                row = 1
            else:
                ## resting 2
                row = 2

            ## Cz, C3, C4 
            # ax[row][0].plot(arr_freqs, arr_mean[1])
            # ax[row][1].plot(arr_freqs, arr_mean[0])
            # ax[row][2].plot(arr_freqs, arr_mean[2])
            ax[row][0].semilogx(arr_freqs, arr_mean[1])
            ax[row][1].semilogx(arr_freqs, arr_mean[0])
            ax[row][2].semilogx(arr_freqs, arr_mean[2])
    
    ax[0][0].set_title(f"C3")
    ax[0][1].set_title(f"Cz")
    ax[0][2].set_title(f"C4")

    # ax[0][-1].legend(['rbc_ce', 'rbc_oe'], bbox_to_anchor=(1, 1))
    # ax[1][-1].legend(['c_ce', 'c_oe'], bbox_to_anchor=(1, 1))
    # ax[2][-1].legend(['rac_ce', 'rac_oe'], bbox_to_anchor=(1, 1))
    for id in np.arange(3):
        ax[0][id].annotate('resting 1 (closed-, open-eyes)', xy=(0.05, 0.05), xytext=(0.05, 0.05), xycoords='axes fraction')
        ax[1][id].annotate('cycling (closed-, open-eyes)',   xy=(0.05, 0.05), xytext=(0.05, 0.05), xycoords='axes fraction')
        ax[2][id].annotate('resting 2 (closed-, open-eyes)', xy=(0.05, 0.05), xytext=(0.05, 0.05), xycoords='axes fraction')

    ax[0][0].set_ylim(-150,350)
    # ax[0][0].set_xlim( 1.0, 45)

    fig.supxlabel('frequency [Hz]')
    fig.supylabel('Percentage (%)')
    fig.suptitle(fig_title)
    fig.savefig(filename)

    return 0


##################################################################
def plot_curves_tf_diff(obj_list, filename, fig_title):

    print(f"plot normalized curves...")
    fig, ax = plt.subplots(3,3,sharex=True,sharey=True, figsize=(10,7))

    ## zero line of reference
    for id in np.arange(3):
        ax[0][id].axhline(y=0.0, color='black', linestyle='-', linewidth = 0.5, alpha=0.5)
        ax[1][id].axhline(y=0.0, color='black', linestyle='-', linewidth = 0.5, alpha=0.5)
        ax[2][id].axhline(y=0.0, color='black', linestyle='-', linewidth = 0.5, alpha=0.5)

        ax[0][id].grid(color = 'gray', linestyle = '--', linewidth = 0.5, alpha=0.5)
        ax[1][id].grid(color = 'gray', linestyle = '--', linewidth = 0.5, alpha=0.5)
        ax[2][id].grid(color = 'gray', linestyle = '--', linewidth = 0.5, alpha=0.5)

    for obj in obj_list:
        ## find the selected segment for each label
        if obj.get_selected_flag():
            print(f"{obj.get_label()}-{obj.get_id()}:")
            arr_mean, arr_freqs = obj.get_tf_mean()
            # print(f"mean values:\n{arr_mean}\nfreqs: {arr_freqs}")
            label = obj.get_label_simple()
            if label.startswith('b_ce'):
                ## cycling closed-eyes
                row = 0
                arr_cycl_ce = arr_mean
                color = 'tab:blue'
                label = 'closed-eyes'
                plot_ax_tf(ax[row], arr_freqs, arr_cycl_ce, color, label)
            elif label.startswith('b_oe'):
                ## cycling open-eyes
                row = 0
                arr_cycl_oe = arr_mean
                color = 'tab:orange'
                label = 'open-eyes'
                plot_ax_tf(ax[row], arr_freqs, arr_cycl_oe, color, label)
            elif label.startswith('c_ce'):
                ## resting 2 closed-eyes
                row = 1
                arr_rest2_ce = arr_mean
                color = 'tab:blue'
                label = 'closed-eyes'
                plot_ax_tf(ax[row], arr_freqs, arr_rest2_ce, color, label)
            elif label.startswith('c_oe'):
                ## resting 2 open-eyes
                row = 1
                arr_rest2_oe = arr_mean
                color = 'tab:orange'
                label = 'open-eyes'
                plot_ax_tf(ax[row], arr_freqs, arr_rest2_oe, color, label)
            else:
                pass

            ## Cz, C3, C4 
            # ax[row][0].plot(arr_freqs, arr_mean[1])
            # ax[row][1].plot(arr_freqs, arr_mean[0])
            # ax[row][2].plot(arr_freqs, arr_mean[2])
            # ax[row][0].semilogx(arr_freqs, arr_mean[1], color=color) ## Cz
            # ax[row][1].semilogx(arr_freqs, arr_mean[0], color=color) ## C3
            # ax[row][2].semilogx(arr_freqs, arr_mean[2], color=color) ## C4
        
    ## subtract cycling from resting 2
    row = 2
    arr_diff_ce = arr_rest2_ce - arr_cycl_ce
    color = 'tab:blue'
    label = 'closed-eyes'
    plot_ax_tf(ax[row], arr_freqs, arr_diff_ce, color, label)    
    # ax[row][0].semilogx(arr_freqs, arr_diff_ce[1], color=color) ## Cz
    # ax[row][1].semilogx(arr_freqs, arr_diff_ce[0], color=color) ## C3
    # ax[row][2].semilogx(arr_freqs, arr_diff_ce[2], color=color) ## C4

    row = 2
    arr_diff_oe = arr_rest2_oe - arr_cycl_oe
    color = 'tab:orange'
    label = 'open-eyes'
    plot_ax_tf(ax[row], arr_freqs, arr_diff_oe, color, label)    
    # ax[row][0].semilogx(arr_freqs, arr_diff_oe[1], color=color) ## Cz
    # ax[row][1].semilogx(arr_freqs, arr_diff_oe[0], color=color) ## C3
    # ax[row][2].semilogx(arr_freqs, arr_diff_oe[2], color=color) ## C4

    # row = 2
    # arr_diff_mix = arr_rest2_ce - arr_cycl_oe
    # color = 'tab:green'
    # label = 'closed-open'    
    # plot_ax_tf(ax[row], arr_freqs, arr_diff_mix, color, label)    
    # ax[row][0].semilogx(arr_freqs, arr_diff_mix[1], color=color) ## Cz
    # ax[row][1].semilogx(arr_freqs, arr_diff_mix[0], color=color) ## C3
    # ax[row][2].semilogx(arr_freqs, arr_diff_mix[2], color=color) ## C4

    ax[0][0].set_title(f"C3")
    ax[0][1].set_title(f"Cz")
    ax[0][2].set_title(f"C4")

    # ax[0][-1].legend(['rbc_ce', 'rbc_oe'], bbox_to_anchor=(1, 1))
    # ax[1][-1].legend(['c_ce', 'c_oe'], bbox_to_anchor=(1, 1))
    # ax[2][-1].legend(['rac_ce', 'rac_oe'], bbox_to_anchor=(1, 1))
    for id in np.arange(3):
        ax[0][id].annotate('cycling ',   xy=(0.05, 0.05), xytext=(0.05, 0.05), xycoords='axes fraction')
        ax[1][id].annotate('resting 2', xy=(0.05, 0.05), xytext=(0.05, 0.05), xycoords='axes fraction')
        ax[2][id].annotate('difference', xy=(0.05, 0.05), xytext=(0.05, 0.05), xycoords='axes fraction')
        ax[0][id].legend()
        ax[1][id].legend()
        ax[2][id].legend()

        

    ax[0][0].set_ylim(-300,300)
    # ax[0][0].set_xlim( 1.0, 45)
    # ax[0][0].set_ylabel('symlogy')

    fig.supxlabel('frequency [Hz]')
    fig.supylabel('Percentage (%)')
    fig.suptitle(fig_title)
    fig.savefig(filename)

    return 0

##############################################
def plot_ax_tf(ax, freqs, arr, color, label):

    ax[0].semilogx(freqs, arr[1], color=color, label=label) ## Cz
    ax[1].semilogx(freqs, arr[0], color=color, label=label) ## C3
    ax[2].semilogx(freqs, arr[2], color=color, label=label) ## C4

    return 0



##################################################################
def plot_curves_tf_diff_2(obj_list, filename, fig_title):

    print(f"plot normalized curves...")
    fig, ax = plt.subplots(2,3,sharex=True,sharey=True, figsize=(8, 5), constrained_layout=True)

    ## zero line of reference
    for id in np.arange(3):
        ax[0][id].axhline(y=0.0, color='black', linestyle='-', linewidth = 0.5, alpha=0.5)
        ax[1][id].axhline(y=0.0, color='black', linestyle='-', linewidth = 0.5, alpha=0.5)

        ax[0][id].grid(color = 'gray', linestyle = '--', linewidth = 0.5, alpha=0.5)
        ax[1][id].grid(color = 'gray', linestyle = '--', linewidth = 0.5, alpha=0.5)

    for obj in obj_list:
        ## find the selected segment for each label
        if obj.get_selected_flag():
            print(f"{obj.get_label()}-{obj.get_id()}:")
            arr_mean, arr_freqs = obj.get_tf_mean()
            # print(f"mean values:\n{arr_mean}\nfreqs: {arr_freqs}")
            label = obj.get_label_simple()
            if label.startswith('b_ce'):
                ## cycling closed-eyes
                row = 0
                arr_cycl_ce = arr_mean
                color = 'tab:orange'
                label = 'cycling'
                plot_ax_tf(ax[row], arr_freqs, arr_cycl_ce, color, label)
            elif label.startswith('b_oe'):
                ## cycling open-eyes
                row = 1
                arr_cycl_oe = arr_mean
                color = 'tab:orange'
                label = 'cycling'
                plot_ax_tf(ax[row], arr_freqs, arr_cycl_oe, color, label)
            elif label.startswith('c_ce'):
                ## resting 2 closed-eyes
                row = 0
                arr_rest2_ce = arr_mean
                color = 'tab:blue'
                label = f'post-\ncycling'
                plot_ax_tf(ax[row], arr_freqs, arr_rest2_ce, color, label)
            elif label.startswith('c_oe'):
                ## resting 2 open-eyes
                row = 1
                arr_rest2_oe = arr_mean
                color = 'tab:blue'
                label = f'post-\ncycling'
                plot_ax_tf(ax[row], arr_freqs, arr_rest2_oe, color, label)
            else:
                pass

    ax[0][0].set_title(f"C3")
    ax[0][1].set_title(f"Cz")
    ax[0][2].set_title(f"C4")

    for id in np.arange(3):
        ax[0][id].annotate('closed-eyes',   xy=(0.05, 0.05), xytext=(0.05, 0.05), xycoords='axes fraction')
        ax[1][id].annotate('open-eyes', xy=(0.05, 0.05), xytext=(0.05, 0.05), xycoords='axes fraction')

    lgd_1 = ax[0][-1].legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
    lgd_2 = ax[1][-1].legend(loc='upper left', bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

    # ax[0][0].set_xlim( 1.0, 45)

    # ax[0][0].set_ylim(-300,300)
    ax[0][0].set_ylim(-1e3,1e3)
    ax[0][0].set_yscale('symlog')

    ax[0][0].set_ylabel(f"change from baseline (%)")
    ax[1][0].set_ylabel(f"change from baseline (%)")

    

    ax[1][0].set_xlabel(f"frequency (Hz)")
    ax[1][1].set_xlabel(f"frequency (Hz)")
    ax[1][2].set_xlabel(f"frequency (Hz)")

    fig.suptitle(fig_title)
    fig.savefig(filename, bbox_extra_artists=(lgd_1,lgd_2),)

    return 0


####################################################################################
def eeg_segmentation(eeg_data_dict, label_seg_list, path, session, pt_info):

    obj_list = []
    ## instantiate objects of the class TF_components
    for label_seg in label_seg_list:
        id_seg = 0
        ## usually three repetitions per segment of closed and open eyes during resting and cycling. Baseline is an exception
        for raw_seg in eeg_data_dict[label_seg]:

            ## instanciate object per each segment (baseline, open eyes, closed eyes)
            obj = TF_components(path, session, raw_seg, label_seg, id_seg, pt_info)
            obj_list.append(obj)

            id_seg+=1

    return obj_list

#################################################
def annotation_bad_channels_and_segments(obj_list, flag_update):
    ## for each segment observe and identify bad channels and bad segments
    for obj in obj_list:
        ## find the selected segment for each label
        if obj.get_selected_flag():
            # interactive selection of bad segments and bad channels
            print(f"{obj.get_label()}-{obj.get_id()}: interactive selection of bad segments and bad channels...")
            obj.selection_bads(flag_update)
            print(f"bad channels: {obj.get_bad_channels()}\n")

    return 0

#################################################
def annotation_bad_channels(obj_list, flag_update):
    ## for each segment observe and identify bad channels and bad segments
    for obj in obj_list:
        ## find the selected segment for each label
        if obj.get_selected_flag():
            # interactive selection of bad segments and bad channels
            print(f"{obj.get_label()}-{obj.get_id()}: interactive selection of bad segments and bad channels...")
            # obj.selection_bads(flag_update)
            obj.selection_bad_channels(flag_update)
            print(f"bad channels: {obj.get_bad_channels()}\n")

    return 0

#################################################
def set_selected_segments(obj_list, selected_segs_dict, dt, flag_update):
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
            # print(f"obj label and id: {label_seg, id_seg}")
            ## find the selected segment for each label
            if (label_seg == sel_label) and (id_seg == sel_id):
                ## selected segment
                obj.set_selected_flag()
                ## include bad segments
                obj.bad_segments_update(flag_update)
                ## making equal-spaced events for the selected segment, which is required to make epochs
                obj.create_events(dt)
                ## visualize raw data with events
                # obj.plot_raw_data()
                ## create epochs based on events
                obj.create_epochs(dt)
                ## epochs cleaning
                ## Ransac in some cases produce unestable results for some channels
                ## Hence, we exclude this step of artifacts_removal
                # obj.artifacts_removal()
                ## include bad channels
                obj.bad_channels_update(flag_update)

    return 0

###################################################
def ica_artifacts_reduction(obj_list, flag_update, event_list):
    ## ica only for selected segments
    for obj in obj_list:
        ## find the selected segment for each label
        if obj.get_selected_flag():
            ## selected events (closed eyes, open eyes)
            if obj.get_label_simple() in event_list:
                # interactive selection of bad segments and bad channels
                print(f"{obj.get_label(), obj.get_id()}: interactive selection of ICA components to exclude...")
                ## applying re-referencing before ICA could propagate artifacts to all electrodes
                ## hence, first we will remove artefacts uisng ICA and after we apply re-referencing
                ##
                print("ICA components...")
                # obj.ica_components(flag_update)
                # obj.ica_components_interactive(flag_update)
                # obj.ica_epochs(flag_update)
                obj.ica_epochs_interactive(flag_update)

    return 0

######################################################
def redefine_reference(obj_list, event_list):
    ## average re-referencing, bad-channels interpolation, and spatial filtering
    for obj in obj_list:
        ## find the selected segment for each label
        if obj.get_selected_flag():
            ## selected events (closed eyes, open eyes)
            if obj.get_label_simple() in event_list:
                # interactive selection of bad segments and bad channels
                print(f"{obj.get_label(), obj.get_id()}")
                #average re-referencing, bad-channels interpolation, and spatial filtering...
                
                # re-referencing appli. average after ICA
                # print(f"re-referencing...")
                # obj.re_referencing()

                print("Bad channels interpolation...")
                obj.bads_interpolation()
                
                print(f"Current source density (Laplacian surface)...")
                obj.apply_csd()

                # print(f"PSD from EEG epochs after ICA...")
                # obj.display_psd_eeg()
                # plt.show(block=True)

    return 0

##############################
def average_psd_regions(obj_list, event_list):
    ## 
    freq_range = [1,45]
    
    fig, ax = plt.subplots(2,2, sharex=True, sharey=True, figsize=(12,6))
    ax = ax.flatten()
    id=0
    acc_obj = 0
    for obj in obj_list:
        ## find the selected segment for each label
        if obj.get_selected_flag():
            # interactive selection of bad segments and bad channels
            # print(f"{obj.get_label(), obj.get_id()}")
            if acc_obj == 2:
                ## save fig and create a new one
                fig, ax = plt.subplots(2,2, sharex=True, sharey=True, figsize=(12,6))
                ax = ax.flatten()
                id=0
                acc_obj = 0
            else:
                pass

            if obj.get_label_simple() in event_list:
                print(f"{obj.get_label(), obj.get_id()}")
                ## PSD average per regions and
                region ='central_left'
                print(f"average; {obj.get_label()}; {region}")
                obj.calculate_average_psd_model(central_left_channels, freq_range, region, ax[id])
                id+=1
                
                region ='central_right'
                print(f"average; {obj.get_label()}; {region}")
                obj.calculate_average_psd_model(central_right_channels, freq_range, region, ax[id])
                id+=1
                acc_obj+=1


            # print (f"psd and freqs:\n{psd_central_left}\n{freqs}")
            # psd_central_right, freqs = obj.get_average_psd(central_right_channels)

    return 0

##############################################
##############################
def plot_psd_quantiles(obj_list, event_list, info_p, ylim, path, flag_save):
    global ax_ce_global, fig_ce_global, flag_eyes_closed, fig_ce, fig_oe, ax_ce, ax_oe
    ##
    print(f"event list: {event_list}")
    ##
    freq_range = [1,45]
    ## ids to define a subplot order for ax_ce and ax_oe
    ax_ce_dict = {'a_ce':0, 'b_ce':2, 'c_ce':4}
    ax_oe_dict = {'a_oe':0, 'b_oe':2, 'c_oe':4}

    ## how many rows in the figures depends of how many segments were recorded
    ## for closed and open eyes
    sum_ce=0
    sum_oe=0
    ## count number of selected segments [a_ce, a_oe, ...]
    for obj in obj_list:
        ## find the selected segment for each label
        if obj.get_selected_flag() and ('ce' in obj.get_label_simple()):
            sum_ce+=1
        elif obj.get_selected_flag() and ('oe' in obj.get_label_simple()):
            sum_oe+=1
        else:
            pass
    
    fig_ce, ax_ce = plt.subplots(sum_ce, 2, sharex=True, sharey=True, figsize=(12,6))
    fig_oe, ax_oe = plt.subplots(sum_oe, 2, sharex=True, sharey=True, figsize=(12,6))
    ax_ce = ax_ce.flatten()
    ax_oe = ax_oe.flatten()

    for obj in obj_list:
        ## find the selected segment for each label
        if obj.get_selected_flag():
            ## get the label of the selected object [a_ce, a_oe, ...]
            label_eyes = obj.get_label_simple()
            ## separate closed eyes and open eyes
            if 'ce' in label_eyes:
                ## closed eyes [a_ce, b_ce, c_ce]
                id_ax = ax_ce_dict[label_eyes]
                region ='central_left'
                obj.calculate_average_psd_model(central_left_channels, freq_range, region, ax_ce[id_ax])
                region ='central_right'
                obj.calculate_average_psd_model(central_right_channels, freq_range, region, ax_ce[id_ax+1])
            else:
                ## open eyes [a_oe, b_oe, c_oe]
                id_ax = ax_oe_dict[label_eyes]
                region ='central_left'
                obj.calculate_average_psd_model(central_left_channels, freq_range, region, ax_oe[id_ax])
                region ='central_right'
                obj.calculate_average_psd_model(central_right_channels, freq_range, region, ax_oe[id_ax+1])

    
    ## ax limits, closed eyes, open eyes
    ax_ce[0].set_ylim(ylim[0], ylim[1])
    ax_oe[0].set_ylim(ylim[0], ylim[1])

    ax_ce[0].set_xlim(freq_range[0]-1, freq_range[1]+1)
    ax_oe[0].set_xlim(freq_range[0]-1, freq_range[1]+1)

    # ## ax titles closed-eyes
    # ax_ce[0].set_title(f'left central region\nresting (before cycling)')
    # ax_ce[1].set_title(f'right central region\nresting (before cycling)')
    # ## ax titles open eyes
    # ax_oe[0].set_title(f'left central region\nresting (before cycling)')
    # ax_oe[1].set_title(f'right central region\nresting (before cycling)')

    ## labels x axes
    ax_ce[-2].set_xlabel(f'frequency [Hz]')
    ax_ce[-1].set_xlabel(f'frequency [Hz]')
    ax_oe[-2].set_xlabel(f'frequency [Hz]')
    ax_oe[-1].set_xlabel(f'frequency [Hz]')


    fig_ce.suptitle(f'{info_p}\nEYES CLOSED')
    fig_oe.suptitle(f'{info_p}\nEYES OPEN')

    # Creating legend with color box
    gray_patch = mpatches.Patch(color='tab:gray', alpha=0.5, label=f'Q3-Q1\ninterquantil\nrange')

    fig_ce.legend(handles=[gray_patch], loc="upper right") ## loc="outside right upper"
    fig_oe.legend(handles=[gray_patch], loc="upper right")

    if flag_save:
        fig_ce.savefig(path+'psd_ce.png',bbox_inches='tight')
        fig_oe.savefig(path+'psd_oe.png',bbox_inches='tight')
    else:
        pass

    ###############
    ## mouse, and keyboard interactions with figures and plots
    # ax_ce_global = ax_ce
    # fig_ce_global = fig_ce
    # flag_eyes_closed = True
    ## run actions described on on_click once the mouse's left-click is pressed over the figure EYES CLOSED
    # fig_ce_global.canvas.mpl_connect('button_press_event', on_click)
    fig_ce.canvas.mpl_connect('button_press_event', on_click)
    fig_oe.canvas.mpl_connect('button_press_event', on_click)


    return 0

##############################################
def onselect(vmin, vmax):
    global f0_global, f1_global
    print(vmin, vmax)

    f0_global = vmin
    f1_global = vmax

    return 0

######################################
def on_click(event):
    global ax_index

    # ax_copy = np.copy(ax_ce_global)

    # print(f"onclick event.inaxes: {event.inaxes}")
    ## is the mouse left-button pressed ?
    if event.button is MouseButton.LEFT:
        # print(f"button left")
        ##
        print(f"event.canvas.figure: {event.canvas.figure}")
        print(f"fig_ce: {fig_ce}")
        print(f"fig_oe: {fig_oe}")

        if event.canvas.figure == fig_ce:
            flag_eyes_closed = True
            ax_copy = np.copy(ax_ce)
            fig_title = f"EYES CLOSED"
            print(f"flag_eyes_closed: {flag_eyes_closed}")
        elif event.canvas.figure == fig_oe:
            flag_eyes_closed = False
            ax_copy = np.copy(ax_oe)
            fig_title = f"EYES OPEN"
            print(f"flag_eyes_closed: {flag_eyes_closed}")
        else:
            print(f"flag_eyes_closed: not found")
            return 0
    
        # print(f"event.inaxes: {event.inaxes}")
        ## is the mouse over any subplot?
        if event.inaxes in ax_copy:
            ## which subplot?
            ax_index = np.argwhere(event.inaxes == ax_copy)[0][0]
            ## subplot index
            print(f"selected ax: {ax_index}")
            ## open a new window with the signals of the selected subplot
            signal_measurements(ax_index, flag_eyes_closed, fig_title)
        else:
            pass
            # print(f"event.inaxes out of ax")
    else:
        pass
        # print(f"other button")

    return 0

####################################
def signal_measurements(ax_index, flag_eyes_closed, fig_title):
    ## open a new figure and plot graphical info of the selected subplot 
    global fig_mea, ax_mea, emg_list, selectors, df_psd_global

    ## close previous figure
    if type(fig_mea) != type([]):
        plt.close(fig_mea)
    else:
        pass

    ## subplots graphics order
    ax_ce_dict_global = {0:'a_ce', 1:'a_ce', 2:'b_ce', 3:'b_ce', 4:'c_ce', 5:'c_ce'}
    ax_oe_dict_global = {0:'a_oe', 1:'a_oe', 2:'b_oe', 3:'b_oe', 4:'c_oe', 5:'c_oe'}

    ## selected subplot state
    if flag_eyes_closed:
        sel_label = ax_ce_dict_global[ax_index]
    else:
        sel_label = ax_oe_dict_global[ax_index]
    ## selected region
    ## odd or even (left or right head side)
    if ax_index % 2 == 0:
        # even
        region ='central_left'
        sel_channels = central_left_channels
    else:
        # odd
        region ='central_right'
        sel_channels = central_right_channels

    ## close previous figure (if it is open)
    if type(fig_mea) != type([]):
        plt.close(fig_mea)
    else:
        pass
    
    ## creates a figure to plot the selected stimulation responses 
    n_rows = 1
    n_cols = 1
    fig, ax = plt.subplots(n_rows, n_cols, sharex=True, figsize=(10*n_cols, 5*n_rows))
     ## ax limits, closed eyes, open eyes
    ax.set_ylim(ylim_global[0], ylim_global[1])
    ax.set_xlim(freq_range[0]-1, freq_range[1]+1)
    fig.suptitle(f"{fig_title}")
    
    for obj in obj_list:
        ## find the selected segment for each label
        if obj.get_selected_flag():
            ## get the label of the selected object [a_ce, a_oe, ...]
            label_eyes = obj.get_label_simple()
            ## separate closed eyes and open eyes
            if sel_label in label_eyes:
                ## closed eyes [a_ce, b_ce, c_ce]
                ## plot a graphical representation of the PSD of the selected subplot
                obj.plot_psd_quantiles(sel_channels, freq_range, region, ax)
                # obj.fit_fooof(freq_range)
                df_psd_global = obj.get_psd_quantiles()
                break

    span = mwidgets.SpanSelector(ax, onselect, 'horizontal', interactive=True, useblit=True, props=dict(facecolor='blue', alpha=0.2))
    selectors.append(span)

    fig_mea = fig
    ax_mea = ax
    fig_mea.canvas.mpl_connect('key_press_event', on_press)
        
    plt.show()

    return 0

#########################
def on_press(event):
    global ax_seg, fig_seg, ax_mea
    # print('press', event.key)
    sys.stdout.flush()

    range_freqs = [f0_global, f1_global]
    fig = fig_mea

    # Set whether to plot in log-log space
    plt_log = False

    print(f"pressed: {event.key}")
    print(f'freq range: {range_freqs}')
    ## measuring amplitude peak to peak
    if event.key == 'a':
        print(f"FOOOF: aperiodic and periodic components' estimation")
        # df_psd_global
        # fm = FOOOF(aperiodic_mode='fixed', peak_width_limits=[0.5, 12], max_n_peaks=5, min_peak_height=1.0)
        fm = FOOOF(aperiodic_mode='fixed', peak_width_limits=[1.0, 25.0], max_n_peaks=3, min_peak_height=3.0)
        fm.add_data(df_psd_global['freqs'].to_numpy(), df_psd_global['psd_q2'].to_numpy(), range_freqs)
        # Fit the power spectrum model
        fm.fit(df_psd_global['freqs'].to_numpy(), 10**(df_psd_global['psd_q2'].to_numpy()), range_freqs)
        # Do an initial aperiodic fit - a robust fit, that excludes outliers
        # This recreates an initial fit that isn't ultimately stored in the FOOOF object
        init_ap_fit = gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))

        # Plot the initial aperiodic fit
        _, ax = plt.subplots(figsize=(12, 10))
        # plot_spectra(fm.freqs, fm.power_spectrum, plt_log,
        #             label='Original Power Spectrum', color='black', ax=ax)
        # plot_spectra(fm.freqs, init_ap_fit, plt_log, label='Initial Aperiodic Fit',
        #             color='blue', alpha=0.5, linestyle='dashed', ax=ax)

        # # Recompute the flattened spectrum using the initial aperiodic fit
        init_flat_spec = fm.power_spectrum - init_ap_fit

        # # Plot the flattened the power spectrum
        plot_spectra(fm.freqs, init_flat_spec, plt_log, label='Flattened Spectrum', color='black', ax=ax)
        # # Plot the iterative approach to finding peaks from the flattened spectrum
        # plot_annotated_peak_search(fm)

        # # Plot the peak fit: created by re-fitting all of the candidate peaks together
        # plot_spectra(fm.freqs, fm._peak_fit, plt_log, color='green', label='Final Periodic Fit')
        # # Plot the peak removed power spectrum, created by removing peak fit from original spectrum
        # plot_spectra(fm.freqs, fm._spectrum_peak_rm, plt_log, label='Peak Removed Spectrum', color='black')

        # Plot the final aperiodic fit, calculated on the peak removed power spectrum
        # _, ax = plt.subplots(figsize=(12, 10))
        # plot_spectra(fm.freqs, fm._spectrum_peak_rm, plt_log, label='Peak Removed Spectrum', color='black', ax=ax)
        # plot_spectra(fm.freqs, fm._ap_fit, plt_log, label='Final Aperiodic Fit', color='blue', alpha=0.5, linestyle='dashed', ax=ax)
        
        # Plot full model, created by combining the peak and aperiodic fits
        # plot_spectra(fm.freqs, fm.fooofed_spectrum_, plt_log, label='Full Model', color='red')
        plot_spectra(fm.freqs, fm._peak_fit, plt_log, label='Full Model', color='red', ax=ax)

        # Print out the model results
        fm.print_results()

        # Plot the full model fit of the power spectrum
        #  The final fit (red), and aperiodic fit (blue), are the same as we plotted above
        # fm.plot(plt_log)

    else:
        pass

    plt.show()

    return 0
        

######################################################
def plot_psd_responses(obj_list, event_list, path_fig):
    ## comparison aperiodic models among resting-cycling-resting, open-eyes, closed-eyes
    fig_a, ax_a = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(9,6))
    fig_b, ax_b = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(9,6))
    fig_c, ax_c = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(9,6))
    fig_d, ax_d = plt.subplots(nrows=3, ncols=2, sharex=True, sharey=True, figsize=(9,6))
    
    ax_d = ax_d.flatten()
    
    id_d = 0
    for obj in obj_list:
        ## find the selected segment for each label
        if obj.get_selected_flag():
            if obj.get_label_simple() in event_list:

                print(f"{obj.get_label(), obj.get_id()}")
                region ='central_left'
                obj.get_plot_psd_model(ax_a[:,0], ax_b[:,0], ax_c[:,0], ax_d[id_d], region,)
                id_d += 1
                # ax_ap[0].legend()
                # ax_ap[0].set_title(f"{region}")

                region ='central_right'
                obj.get_plot_psd_model(ax_a[:,1], ax_b[:,1], ax_c[:,1], ax_d[id_d], region,)
                id_d += 1
                # ax_ap[1].legend()
                # ax_ap[1].set_title(f"{region}")

    ax_a[0][0].set_title(f"central left channels")
    ax_a[0][1].set_title(f"central right channels")

    ax_b[0][0].set_title(f"central left channels")
    ax_b[0][1].set_title(f"central right channels")

    ax_c[0][0].set_title(f"central left channels")
    ax_c[0][1].set_title(f"central right channels")

    ax_d[0].set_title(f"central left channels")
    ax_d[1].set_title(f"central right channels")

    ## save figures
    # fig.suptitle(f"{info_p}")
    fig_a.savefig(path_fig+'fooof_a.png', bbox_inches ="tight")
    fig_b.savefig(path_fig+'fooof_b.png', bbox_inches ="tight")
    fig_c.savefig(path_fig+'fooof_c.png', bbox_inches ="tight")
    fig_d.savefig(path_fig+'fooof_d.png', bbox_inches ="tight")

    return 0

##############################################
def plot_psd_responses_all(obj_list, event_list_ce, event_list_oe, path_fig, info_p):
    ## comparison aperiodic models among resting-cycling-resting, open-eyes, closed-eyes
    fig_a, ax_a = plt.subplots(nrows=3, ncols=4, sharex=False, sharey=False, figsize=(12,6), layout='constrained')
    # fig_b, ax_b = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(10,7))
    # fig_c, ax_c = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(10,7))
    # fig_d, ax_d = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(9,6))
    # fig_e, ax_e = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(10,7))

    ## colored regions frequency bands [theta, alpha, beta]
    ## theta (4-8 Hz), alpha (8-12 Hz), beta (12-30 Hz)
    # ax_a[0].axvspan(mu-2*sigma, mu-sigma, color='0.95')
    ##theta (4-8 Hz)
    set_bands_ax(ax_a)
    
    flags = [0,0,0]

    for obj in obj_list:
        ## find the selected segment for each label
        if obj.get_selected_flag():

            if obj.get_label_simple() in event_list_ce:
                ## closed eyes
                print(f"{obj.get_label(), obj.get_id()}")
                region ='central_left'
                obj.get_plot_psd_model(ax_a[:,0], region,)
                flags = label_flags(obj, flags)

                region ='central_right'
                obj.get_plot_psd_model(ax_a[:,2], region,)
                flags = label_flags(obj, flags)
                

            elif obj.get_label_simple() in event_list_oe:
                ## open eyes
                print(f"{obj.get_label(), obj.get_id()}")
                region ='central_left'
                obj.get_plot_psd_model(ax_a[:,1], region,)
                flags = label_flags(obj, flags)
                
                region ='central_right'
                obj.get_plot_psd_model(ax_a[:,3], region,)
                flags = label_flags(obj, flags)

    set_xy_lim(ax_a)
    # ax_b[0][0].set_ylim(-0.2, 1.0)
    # ax_c[0][0].set_ylim(-6.25, -2.00)

    set_labels_ax_a(ax_a)
    ## legend
    fig_a = set_legend(fig_a, flags)

    # set_labels_ax(ax_b)
    # set_labels_ax(ax_c)

    set_title_ax(ax_a[0])
    # set_title_ax(ax_b[0])
    # set_title_ax(ax_c[0])
    
    # set_subtitle_fig(fig_a)
    # set_subtitle_fig(fig_b)
    # set_subtitle_fig(fig_c)

    set_annotation_ax (ax_a[0,:], '(PSD)')
    set_annotation_ax (ax_a[1,:], '(APERIODIC COMP.)')
    set_annotation_ax (ax_a[2,:], '(PSD) - (APERIODIC COMP.)')

    # fig_a.text(0.32, 0.925, "central left channels", ha='center', fontsize=12,)
    # fig_a.text(0.72, 0.925, "central right channels", ha='center', fontsize=12,)

    set_grid_ax(ax_a)
    # set_grid_ax(ax_b)
    # set_grid_ax(ax_c)

    ## save figures
    # text_subtitle = "\ncentral left channels \t \t \t central right channels\n".replace("\t", "    ")
    fig_a.suptitle(f"{info_p}\n",)
    # fig_b.suptitle(f"{info_p}")
    # fig_c.suptitle(f"{info_p}")


    fig_a.savefig(path_fig+'fooof_a.png', bbox_inches ="tight")
    # fig_b.savefig(path_fig+'fooof_b.png', bbox_inches ="tight")
    # fig_c.savefig(path_fig+'fooof_c.png', bbox_inches ="tight")

    return 0


##############################################
def plot_psd_responses_only4(obj_list, event_list_ce, event_list_oe, path_fig, info_p):
    ## comparison aperiodic models among resting-cycling-resting, open-eyes, closed-eyes
    fig_a, ax_a = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(12,6), layout='constrained')
    # fig_b, ax_b = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(10,7))
    # fig_c, ax_c = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(10,7))
    # fig_d, ax_d = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(9,6))
    # fig_e, ax_e = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(10,7))

    ## colored regions frequency bands [theta, alpha, beta]
    ## theta (4-8 Hz), alpha (8-12 Hz), beta (12-30 Hz)
    # ax_a[0].axvspan(mu-2*sigma, mu-sigma, color='0.95')
    ##theta (4-8 Hz)
    ax_a = ax_a.flatten()
    set_bands_ax4plot(ax_a)

    flags = [0,0,0]

    for obj in obj_list:
        ## find the selected segment for each label
        ## At the beginning, one of each condition was selected, i.e. a_ce, a_oe, b_ce, b_oe, c_ce, c_oe
        if obj.get_selected_flag():

            ## a_ce, b_ce, c_ce
            if obj.get_label_simple() in event_list_ce:
                ## closed eyes
                print(f"{obj.get_label(), obj.get_id()}")
                region ='central_left'
                # obj.get_plot_psd_model(ax_a[:,0], region,)
                obj.get_plot_psd(ax_a[0], region,)
                flags = label_flags(obj, flags)

                region ='central_right'
                # obj.get_plot_psd_model(ax_a[:,2], region,)
                obj.get_plot_psd(ax_a[1], region,)
                flags = label_flags(obj, flags)
                
            ## a_oe, b_oe, c_oe
            elif obj.get_label_simple() in event_list_oe:
                ## open eyes
                print(f"{obj.get_label(), obj.get_id()}")
                region ='central_left'
                # obj.get_plot_psd_model(ax_a[:,1], region,)
                obj.get_plot_psd(ax_a[2], region,)
                flags = label_flags(obj, flags)
                
                region ='central_right'
                # obj.get_plot_psd_model(ax_a[:,3], region,)
                obj.get_plot_psd(ax_a[3], region,)
                flags = label_flags(obj, flags)


    ## x and y limits
    ax_a[0].set_xlim(-1, 47.0)
    # ax_a[0].set_ylim(-6.25, -1.75)


    set_labels_ax_4only(ax_a)
    ## legend
    fig_a = set_legend(fig_a, flags)

    # set_labels_ax(ax_b)
    # set_labels_ax(ax_c)

    # set_title_ax(ax_a[0])
    set_title_ax4only(ax_a)
    # set_title_ax(ax_b[0])
    # set_title_ax(ax_c[0])

    # set_subtitle_fig(fig_a)
    # set_subtitle_fig(fig_b)
    # set_subtitle_fig(fig_c)

    # set_annotation_ax (ax_a[0,:], '(PSD)')
    # set_annotation_ax (ax_a[1,:], '(APERIODIC COMP.)')
    # set_annotation_ax (ax_a[2,:], '(PSD) - (APERIODIC COMP.)')

    # fig_a.text(0.32, 0.925, "central left channels", ha='center', fontsize=12,)
    # fig_a.text(0.72, 0.925, "central right channels", ha='center', fontsize=12,)

    set_grid_ax4only(ax_a)
    
    # set_grid_ax(ax_b)
    # set_grid_ax(ax_c)

    ## save figures
    # text_subtitle = "\ncentral left channels \t \t \t central right channels\n".replace("\t", "    ")
    fig_a.suptitle(f"{info_p}\n",)
    # fig_b.suptitle(f"{info_p}")
    # fig_c.suptitle(f"{info_p}")


    fig_a.savefig(path_fig+'fooof_psd_a.png', bbox_inches ="tight")
    # fig_b.savefig(path_fig+'fooof_b.png', bbox_inches ="tight")
    # fig_c.savefig(path_fig+'fooof_c.png', bbox_inches ="tight")

    return 0

##################
def plot_psd_responses_median(obj_list, event_list_ce, event_list_oe, path_fig, info_p):
    ## comparison aperiodic models among resting-cycling-resting, open-eyes, closed-eyes
    fig_a, ax_a = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(12,6), layout='constrained')
    # fig_b, ax_b = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(10,7))
    # fig_c, ax_c = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(10,7))
    # fig_d, ax_d = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(9,6))
    # fig_e, ax_e = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(10,7))

    ## colored regions frequency bands [theta, alpha, beta]
    ## theta (4-8 Hz), alpha (8-12 Hz), beta (12-30 Hz)
    # ax_a[0].axvspan(mu-2*sigma, mu-sigma, color='0.95')
    ##theta (4-8 Hz)
    ax_a = ax_a.flatten()
    
    ymin = 5
    ymax = 35
    set_bands_ax4plot(ax_a, ymax)

    flags = [0,0,0]

    for obj in obj_list:
        ## find the selected segment for each label
        ## At the beginning, one of each condition was selected, i.e. a_ce, a_oe, b_ce, b_oe, c_ce, c_oe
        if obj.get_selected_flag():

            ## a_ce, b_ce, c_ce
            if obj.get_label_simple() in event_list_ce:
                ## closed eyes
                print(f"{obj.get_label(), obj.get_id()}")
                region ='central_left'
                # obj.get_plot_psd_model(ax_a[:,0], region,)
                obj.get_plot_psd(ax_a[0], region,)
                flags = label_flags(obj, flags)

                region ='central_right'
                # obj.get_plot_psd_model(ax_a[:,2], region,)
                obj.get_plot_psd(ax_a[1], region,)
                flags = label_flags(obj, flags)
                
            ## a_oe, b_oe, c_oe
            elif obj.get_label_simple() in event_list_oe:
                ## open eyes
                print(f"{obj.get_label(), obj.get_id()}")
                region ='central_left'
                # obj.get_plot_psd_model(ax_a[:,1], region,)
                obj.get_plot_psd(ax_a[2], region,)
                flags = label_flags(obj, flags)
                
                region ='central_right'
                # obj.get_plot_psd_model(ax_a[:,3], region,)
                obj.get_plot_psd(ax_a[3], region,)
                flags = label_flags(obj, flags)


    ## x and y limits
    ax_a[0].set_xlim(-1, 47.0)
    # ax_a[0].set_ylim(-6.25, -1.75)
    # ax_a[0].set_ylim(ymin,ymax)


    set_labels_ax_4only(ax_a)
    ## legend
    fig_a = set_legend(fig_a, flags)

    # set_labels_ax(ax_b)
    # set_labels_ax(ax_c)

    # set_title_ax(ax_a[0])
    set_title_ax4only(ax_a)
    # set_title_ax(ax_b[0])
    # set_title_ax(ax_c[0])

    # set_subtitle_fig(fig_a)
    # set_subtitle_fig(fig_b)
    # set_subtitle_fig(fig_c)

    # set_annotation_ax (ax_a[0,:], '(PSD)')
    # set_annotation_ax (ax_a[1,:], '(APERIODIC COMP.)')
    # set_annotation_ax (ax_a[2,:], '(PSD) - (APERIODIC COMP.)')

    # fig_a.text(0.32, 0.925, "central left channels", ha='center', fontsize=12,)
    # fig_a.text(0.72, 0.925, "central right channels", ha='center', fontsize=12,)

    set_grid_ax4only(ax_a)
    
    # set_grid_ax(ax_b)
    # set_grid_ax(ax_c)

    ## save figures
    # text_subtitle = "\ncentral left channels \t \t \t central right channels\n".replace("\t", "    ")
    fig_a.suptitle(f"{info_p}\n",)
    # fig_b.suptitle(f"{info_p}")
    # fig_c.suptitle(f"{info_p}")


    fig_a.savefig(path_fig+'fooof_psd_a.png', bbox_inches ="tight")
    # fig_b.savefig(path_fig+'fooof_b.png', bbox_inches ="tight")
    # fig_c.savefig(path_fig+'fooof_c.png', bbox_inches ="tight")

    return 0

###################
def set_legend(fig, flags):
    # blue_line = mlines.Line2D([], [], color='tab:blue', label="\nresting\nbefore\nbiking\n")
    # orange_line = mlines.Line2D([], [], color='tab:orange', label="biking")
    # green_line = mlines.Line2D([], [], color='tab:green', label="\nresting\nafter\nbiking\n")
    blue_line = mlines.Line2D([], [], color='tab:blue', label="rest start")
    orange_line = mlines.Line2D([], [], color='tab:orange', label="biking")
    green_line = mlines.Line2D([], [], color='tab:green', label="rest end")

    print(f"sum flags = {sum(flags)}")
    if sum(flags) == 3:
        handles_list=[blue_line, orange_line, green_line]
    elif sum(flags) == 2:
        handles_list=[blue_line, orange_line,]
    else:
        handles_list=[blue_line,]

    fig.legend(handles=handles_list, loc="outside right upper")

    return fig

###################
def label_flags(obj, flags):
    label = obj.get_label_simple()
    if label in ['a_ce','a_oe']:
        flags[0] = 1
    elif label in ['b_ce','b_oe']:
        flags[1] = 1
    elif label in ['c_ce','c_oe']:
        flags[2] = 1
    return flags


###################
def set_xy_lim(ax):
    for id in np.arange(4):
        ax[0][id].set_ylim(-6.25, -1.75)
        ax[1][id].set_ylim(-6.25, -1.75)
        ax[2][id].set_ylim(-0.75, 1.75)

    for id in np.arange(4):
        ax[0][id].set_xlim(-1, 47.0)
        ax[1][id].set_xlim(-1, 47.0)
        ax[2][id].set_xlim(-1, 47.0)   
            
    return ax

###################

def set_bands_ax(ax):

    ## create particular grid
    for id in np.arange(4):
        ## include annotations
        ## theta (4-8 Hz), alpha (8-12 Hz), beta (12-30 Hz)
        ## dash lines
        ax[0][id].axvline(x=4, linestyle='--', linewidth=0.75, color='tab:gray')
        ax[1][id].axvline(x=4, linestyle='--', linewidth=0.75, color='tab:gray')
        ax[2][id].axvline(x=4, linestyle='--', linewidth=0.75, color='tab:gray')

        ax[0][id].axvline(x=8, linestyle='--', linewidth=0.75, color='tab:gray')
        ax[1][id].axvline(x=8, linestyle='--', linewidth=0.75, color='tab:gray')
        ax[2][id].axvline(x=8, linestyle='--', linewidth=0.75, color='tab:gray')

        ax[0][id].axvline(x=12, linestyle='--', linewidth=0.75, color='tab:gray')
        ax[1][id].axvline(x=12, linestyle='--', linewidth=0.75, color='tab:gray')
        ax[2][id].axvline(x=12, linestyle='--', linewidth=0.75, color='tab:gray')

        ax[0][id].axvline(x=30, linestyle='--', linewidth=0.75, color='tab:gray')
        ax[1][id].axvline(x=30, linestyle='--', linewidth=0.75, color='tab:gray')
        ax[2][id].axvline(x=30, linestyle='--', linewidth=0.75, color='tab:gray')
        ## include annotations
        ## theta (4-8 Hz)
        ax[0][id].annotate('$\\theta$', xy=(5,-2.3), xytext=(5,-2.3), fontsize=10)
        ax[1][id].annotate('$\\theta$', xy=(5,-2.3), xytext=(5,-2.3), fontsize=10)
        ax[2][id].annotate('$\\theta$', xy=(5, 1.4), xytext=(5, 1.4), fontsize=10)
        ## alpha (8-12 Hz)
        ax[0][id].annotate('$\\alpha$', xy=(9,-2.3), xytext=(9,-2.3), fontsize=10)
        ax[1][id].annotate('$\\alpha$', xy=(9,-2.3), xytext=(9,-2.3), fontsize=10)
        ax[2][id].annotate('$\\alpha$', xy=(9, 1.4), xytext=(9, 1.4), fontsize=10)
        ## beta (12-30 Hz)
        ax[0][id].annotate('$\\beta$', xy=(20,-2.3), xytext=(20,-2.3), fontsize=10)
        ax[1][id].annotate('$\\beta$', xy=(20,-2.3), xytext=(20,-2.3), fontsize=10)
        ax[2][id].annotate('$\\beta$', xy=(20, 1.4), xytext=(20, 1.4), fontsize=10)

        xticks_list = [0,4,8,12,30]
        xticks_labels = ['0','4','8','12','30']
        ax[0][id].set_xticks(xticks_list, labels= xticks_labels, fontsize=10)
        ax[1][id].set_xticks(xticks_list, labels= xticks_labels, fontsize=10)
        ax[2][id].set_xticks(xticks_list, labels= xticks_labels, fontsize=10)

    # for ax in ax_list.flatten():
        # ax.grid(linestyle='--', alpha=0.5, lw=1.0)
    return 0

##########################
def set_bands_ax4plot(ax_list, ymax):

    ## create particular grid
    for ax in ax_list:
        ## include annotations
        ## theta (4-8 Hz), alpha (8-12 Hz), beta (12-30 Hz)
        ## dash lines
        ax.axvline(x=4,  linestyle='--', linewidth=0.75, color='tab:gray')
        ax.axvline(x=8,  linestyle='--', linewidth=0.75, color='tab:gray')
        ax.axvline(x=12, linestyle='--', linewidth=0.75, color='tab:gray')
        ax.axvline(x=30, linestyle='--', linewidth=0.75, color='tab:gray')
       
        ## include annotations
        ## theta (4-8 Hz)
        ax.annotate('$\\theta$', xy=(5, 0.8*ymax), xytext=(5, 0.8*ymax), fontsize=10)
        ax.annotate('$\\alpha$', xy=(9, 0.8*ymax), xytext=(9, 0.8*ymax), fontsize=10)
        ax.annotate('$\\beta$', xy=(20, 0.8*ymax), xytext=(20,0.8*ymax), fontsize=10)

        xticks_list = [0,4,8,12,30]
        xticks_labels = ['0','4','8','12','30']
        ax.set_xticks(xticks_list, labels= xticks_labels, fontsize=10)

    return 0

####################
def set_grid_ax(ax):
    ## hide grid
    for id in np.arange(4):
        ax[0][id].grid(visible=False)
        ax[1][id].grid(visible=False)
        ax[2][id].grid(visible=False)

def set_grid_ax4only(ax_list):
    ## hide grid
    for ax in ax_list:
        ax.grid(lw=0.5, ls='--', alpha=0.5)

    return 0

def set_annotation_ax (ax_list, text):
    for ax in ax_list:
        ax.text(0.01, 0.01, text, transform=ax.transAxes, size=11, bbox=dict(facecolor='white', edgecolor='white', alpha=0.5,))
    return 0

def set_subtitle_fig(fig):
    fig.text(0.32, 0.925, "central left channels", ha='center', fontsize=12,)
    fig.text(0.72, 0.925, "central right channels", ha='center', fontsize=12,)
    return 0
    
def set_labels_ax(ax,):
    ax[0][0].set_ylabel(f"log10(power)", fontsize=12)
    ax[1][0].set_ylabel(f"log10(power)", fontsize=12)
    ax[0][1].set_ylabel(f"")
    ax[1][1].set_ylabel(f"")
    ax[0][2].set_ylabel(f"")
    ax[1][2].set_ylabel(f"")
    ax[0][3].set_ylabel(f"")
    ax[1][3].set_ylabel(f"")


    ax[1][0].set_xlabel(f"frequency (Hz)", fontsize=12)
    ax[1][1].set_xlabel(f"frequency (Hz)", fontsize=12)
    ax[1][2].set_xlabel(f"frequency (Hz)", fontsize=12)
    ax[1][3].set_xlabel(f"frequency (Hz)", fontsize=12)
    
    ax[0][0].set_xlabel(f"",)
    ax[0][1].set_xlabel(f"",)
    ax[0][2].set_xlabel(f"",)
    ax[0][3].set_xlabel(f"",)

    ax[0][0].tick_params(axis='x',reset=True, labelsize=12)
    ax[0][1].tick_params(axis='x',reset=True, labelsize=12)
    ax[0][2].tick_params(axis='x',reset=True, labelsize=12)
    ax[0][3].tick_params(axis='x',reset=True, labelsize=12)

    return 0

def set_labels_ax_a(ax,):

    ## remove legends
    for id in np.arange(3):
        ax[id][0].get_legend().set_visible(False)
        ax[id][1].get_legend().set_visible(False)
        ax[id][2].get_legend().set_visible(False)
        ax[id][3].get_legend().set_visible(False)
        
    ## y axis labels

    ax[0][0].set_ylabel(f"log10(power)", fontsize=11)
    ax[1][0].set_ylabel(f"log10(power)", fontsize=11)
    ax[2][0].set_ylabel(f"log10(power)", fontsize=11)
    # ax[3][0].set_ylabel(f"log10(power)", fontsize=11)

    for id in np.arange(1,4):
        ax[0][id].set_ylabel(f"")
        ax[1][id].set_ylabel(f"")
        ax[2][id].set_ylabel(f"")
        # ax[3][id].set_ylabel(f"")

    ## x axis labels

    for id in np.arange(2):
        ax[id][0].set_xlabel(f"")
        ax[id][1].set_xlabel(f"")
        ax[id][2].set_xlabel(f"")
        ax[id][3].set_xlabel(f"")

    ax[2][0].set_xlabel(f"frequency (Hz)", fontsize=11)
    ax[2][1].set_xlabel(f"frequency (Hz)", fontsize=11)
    ax[2][2].set_xlabel(f"frequency (Hz)", fontsize=11)
    ax[2][3].set_xlabel(f"frequency (Hz)", fontsize=11)

    # ax[0][0].tick_params(axis='x',reset=True, labelsize=12)
    # ax[0][1].tick_params(axis='x',reset=True, labelsize=12)
    # ax[0][2].tick_params(axis='x',reset=True, labelsize=12)
    # ax[0][3].tick_params(axis='x',reset=True, labelsize=12)

    return 0


################
def set_labels_ax_4only(ax_list,):

    ## remove legends
    for ax in ax_list:
        try:
            ax.get_legend().set_visible(False)
        except:
            print("legend not found")
        
    ## y axis labels
    ax_list[0].set_ylabel(f"Power\n[$dB(mV/m^2)^2/Hz$]", fontsize=11)
    ax_list[1].set_ylabel(f"")
    ax_list[2].set_ylabel(f"Power\n[$dB(mV/m^2)^2/Hz$]", fontsize=11)
    ax_list[3].set_ylabel(f"")
    

    ## x axis labels
    ax_list[0].set_xlabel(f"")
    ax_list[1].set_xlabel(f"")
    ax_list[2].set_xlabel(f"frequency (Hz)", fontsize=11)
    ax_list[3].set_xlabel(f"frequency (Hz)", fontsize=11)

    return 0

###########
def set_title_ax(ax,):
    ##
    ax[0].set_title(f"central left channels\nclosed eyes", loc='left')
    ax[1].set_title(f"central left channels\nopen eyes", loc='left')
    ax[2].set_title(f"central right channels\nclosed eyes", loc='left')
    ax[3].set_title(f"central right channels\nopen eyes", loc='left')

    ##
    # ax[0].set_title(f"central left channels")
    # ax[1].set_title(f"central right channels")
    # ax[2].set_title(f"central left channels")
    # ax[3].set_title(f"central right channels")
    return 0

###########
def set_title_ax4only(ax,):
    ##
    ax[0].set_title(f"average central left region\nclosed eyes", loc='center')
    ax[1].set_title(f"average central right region\nclosed eyes",loc='center')
    ax[2].set_title(f"open eyes",   loc='center')
    ax[3].set_title(f"open eyes",  loc='center')

    ##
    # ax[0].set_title(f"central left channels")
    # ax[1].set_title(f"central right channels")
    # ax[2].set_title(f"central left channels")
    # ax[3].set_title(f"central right channels")
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
            obj.apply_csd(ch_list)
            # print(f"plot csd data...")
            # seg_ref.plot_time_series('csd', 'After Laplacian surface filter (current source density)')
            
            ## power spetral density
            # print(f"PSD selected channels: {ch_list}")
            # obj.psd_selected_chx(ch_list)
            # print(f"data: {data_psd}")
            # print(f"freqs: {freqs_psd}")
            ## dataframe 

            # print(f"Plot PSD selected channels: {ch_list}")
            # obj.plot_psd_selected_chx(ch_list)

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

########################################################
def plot_psd_per_channels(obj_list, ch_list):
    ## plot curves
    fig, ax = plt.subplots(2,3, sharex=True, sharey=True)
    ax = ax.flatten()
    # ax.set_title(f"PSD (EEG)")
    # ax[0].set_xlim(4,101)
    # ax[0].set_ylim(-60,10)

    ## from each segment
    for obj in obj_list:
        ## find the selected segment for each label
        if obj.get_selected_flag():
            print(f"{obj.get_label()}-{obj.get_id()}:")
            ## get dataframe of PSD selected channels
            df = obj.get_df_psd()
            freqs = df['freq'].to_list()
            # print(f"{obj.get_label_simple()}")
            label = obj.get_label_simple()

            ## plot closed-eyes first row, and open-eyes in the second row
            if label.endswith('ce'):
                ## plot first row C3, Cz, C4
                ax[0].semilogx(freqs, df[ch_list[1]].to_list(), label=label[0])
                ax[1].semilogx(freqs, df[ch_list[0]].to_list(), label=label[0])
                ax[2].semilogx(freqs, df[ch_list[2]].to_list(), label=label[0])
            else:
                ## plot second row C3, Cz, C4
                ax[3].semilogx(freqs, df[ch_list[1]].to_list(), label=label[0])
                ax[4].semilogx(freqs, df[ch_list[0]].to_list(), label=label[0])
                ax[5].semilogx(freqs, df[ch_list[2]].to_list(), label=label[0])

    for id in np.arange(6):
        ax[id].legend()            
            
            # ## power spetral density
            # print(f"PSD selected channels: {ch_list}")
            # obj.psd_selected_chx(ch_list)
    return 0


########################################################
def plot_psd_per_channels_mne(obj_list, ch_list, filename, info_p):
    ## plot curves
    fig, ax = plt.subplots(2,3, figsize=(10,5), sharex=True, sharey=True)
    ax = ax.flatten()

    flag_a = False
    flag_b = False
    flag_c_oe = False
    flag_c_ce = False

    ## from each segment
    for obj in obj_list:
        ## find the selected segment for each label
        if obj.get_selected_flag():
            print(f"{obj.get_label()}-{obj.get_id()}:")
            ## get dataframe of PSD selected channels
            # df = obj.get_df_psd()
            # freqs = df['freq'].to_list()
            # print(f"{obj.get_label_simple()}")
            label = obj.get_label_simple()

            if label.startswith('a'):
                ## resting pre-cycling
                color = 'tab:blue'
                flag_a = True
            elif label.startswith('b'):
                ## cycling
                color = 'tab:orange'
                flag_b = True
            elif label.startswith('c'):
                ## resting post-cycling
                color = 'tab:green'
                if label.endswith('oe'):
                    flag_c_oe = True
                elif label.endswith('ce'):
                    flag_c_ce = True
                else:
                    pass
            else:
                color = 'black'

            ## plot closed-eyes first row, and open-eyes in the second row
            if label.endswith('ce'):
                ## closed-eyes first row plot
                ## plot first row C3, Cz, C4
                obj.plot_psd_selected_chx_mne(ch_list[1], ax[0], color)
                obj.plot_psd_selected_chx_mne(ch_list[0], ax[1], color)
                obj.plot_psd_selected_chx_mne(ch_list[2], ax[2], color)
            else:
                ## open-eyes second row plot
                ## plot second row C3, Cz, C4
                obj.plot_psd_selected_chx_mne(ch_list[1], ax[3], color)
                obj.plot_psd_selected_chx_mne(ch_list[0], ax[4], color)
                obj.plot_psd_selected_chx_mne(ch_list[2], ax[5], color)

    # ax.set_title(f"PSD (EEG)")
    ax[0].set_xlim(4,101)
    ax[0].set_ylim(-40,10)

    ax[0].set_title(f"C3 - closed-eyes")
    ax[1].set_title(f"Cz - closed-eyes")
    ax[2].set_title(f"C4 - closed-eyes")
    ax[3].set_title(f"C3 - open-eyes")
    ax[4].set_title(f"Cz - open-eyes")
    ax[5].set_title(f"C4 - open-eyes")

    ax[3].set_xlabel(f"frequency [Hz]")
    ax[4].set_xlabel(f"frequency [Hz]")
    ax[5].set_xlabel(f"frequency [Hz]")

    ax[1].set_ylabel(f"")
    ax[2].set_ylabel(f"")
    ax[4].set_ylabel(f"")
    ax[5].set_ylabel(f"")

    ## legend for each plot
    label_rest_1 = mlines.Line2D([], [], color='tab:blue', label='pre_cycling')
    label_cycling = mlines.Line2D([], [], color='tab:orange', label='cycling')
    label_rest_2 = mlines.Line2D([], [], color='tab:green', label='post_cycling')
    for id in np.arange(6):
        if flag_a:
            ax[id].legend(handles=[label_rest_1])
        if flag_a and flag_b:
            ax[id].legend(handles=[label_rest_1, label_cycling])
        if flag_a and flag_b and flag_c_ce and id < 3:
            ax[id].legend(handles=[label_rest_1, label_cycling,label_rest_2])
        if flag_a and flag_b and flag_c_oe and id >= 3:
            ax[id].legend(handles=[label_rest_1, label_cycling,label_rest_2])

    ## save fig
    fig.suptitle(f"{info_p}")
    fig.savefig(filename, bbox_inches ="tight")            

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
                ## only for selected segments of each state (a_ce, a_oe, b_ce, ...)
                obj.channel_bands_power(ch_name, eeg_system)
                # print (f"{obj.get_label()}_{obj.get_id()} beta band frequency components...")
                # obj.channel_beta_band_power(ch_name, eeg_system)
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
def tf_freq_bands_boxplots(obj_list,):
    ##
    # print(f"Power per frequency bands... ")
    ## alpha, theta, beta bands activity selected channels
    # df_ch_list = pd.DataFrame()
    # ## selected channels
    # ch_name_list = ['Cz','C3','C4']
    # for ch_name in ch_name_list:
        ## for each selected channel
    for obj in obj_list:
        ## for every segment (obj) of each selected channel (ch_name)
        ## df_ch_bands : theta, beta, alpha activity of selected channels
        if obj.get_selected_flag():
            ## only for selected segments of each state (a_ce, a_oe, b_ce, ...)
            obj.channels_bandpower()
            # print (f"{obj.get_label()}_{obj.get_id()} beta band frequency components...")
            # obj.channel_beta_band_power(ch_name, eeg_system)
    ##
    # ## add bad annotations in df_ch_bands
    # for obj in obj_list:
    #     ## for every segment (obj) of each selected channel (ch_name)
    #     ## df_ch_bands : theta, beta, alpha activity of selected channels 
    #     ## add a mask to mark band segments
    #     if obj.get_selected_flag():
    #         obj.set_annotations_freq_bands()
    #     # df = obj.get_df_ch_bands()
    #     # print(f"dataframe {obj.get_label()}--{obj.get_id()}, df shape {df.shape}:\n{df}")
    #     # print(f"df columns:\n{list(df.columns.values)}")
    # ##

    return 0


###############################
def tf_beta_plot(obj_list, ch_name_list):
    ## plot curves of beta band components
    for obj in obj_list[8:10]:
        ## for every segment (obj) generate a fig for each selected channel (C3, Cz, C4)
        if obj.get_selected_flag():
            ## save figure of time-frequency analysis for each selected channel
            obj.curves_beta_plot(ch_name_list,)
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
            if obj.get_selected_flag():
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
def display_segments(obj_list, label_seg_list, ch_excl_list):
    ## for each segment observe and identify bad channels and bad segments
    for label_seg in label_seg_list:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9,4), sharey=True, sharex=True)
        ax = ax.flatten()
        id_ax = 0
        for obj in obj_list:
            ### (interactive selection of bad segments and bad channels)
            ## visualization for the selection of the less noisy and more representative segment for each state, i.e. a_ce, a_oe, b_ce, b_oe
            # print(f"{obj.get_label()}-{obj.get_id()}: interactive selection of bad segments and bad channels...")
            print(f"{obj.get_label()}-{obj.get_id()}: segments' visualization...")
            if obj.get_label() == label_seg:
                ## load bad channels and bad annotations
                ax[id_ax], id_seg = obj.data_visualization(ax[id_ax], ch_excl_list)
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
    global sampling_rate, psd_fig_name, excluded_channels, obj_list, ylim_global

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
    path, fn_in, fn_csv, raw_data, fig_title, flag_notch, acquisition_system, info_p, Dx, selected_segs_dict, ch_excl_list, ylims = participants_list(path, subject, session, abt)
    if fn_csv == '':
        print(f'It could not find the selected subject. Please check the path, and the selected subject number in the list of participants.')
        return 0
    else:
        pass

    ylim_global = ylims
    ## create folder (if it does not exist) to save preprocesing parameters
    # Path(path+'session_'+str(session)+"/prep").mkdir(parents=True, exist_ok=True)

    ## path filename for baseline normalization
    filename_tr_ref = path+'session_'+str(session)+f'/prep/'+'tf_mean_baseline.npy'

    ## path filename boxplots
    path_fig_boxplot = path+'session_'+str(session)+f'/figures/'
    path_fig_psd = path+'session_'+str(session)+f'/figures/psd/'
    path_fig_fooof = path+'session_'+str(session)+f'/figures/fooof/'
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

    ################################################
    ## read annotations (.csv file)
    ## annotations of events during recording session that includes: resting, cycling, closed-eyes, open eyes
    my_annot = mne.read_annotations(path + fn_csv[0])
    print(f'annotations:\n{my_annot}')
    ## adding annotations to raw data
    raw_data.set_annotations(my_annot)
    ##

    ## exclude channels of the net boundaries that usually bring noise or artifacts
    ## geodesic system we remove channels in the boundaries
    # raw_data.info["bads"] = bad_channels_dict[acquisition_system]
    ## list of excluded channels 
    print(f"excluded channels:\n{excluded_channels}")
    raw_data.info["bads"] = excluded_channels
    raw_data.drop_channels(raw_data.info['bads'])

    ## additonal channels to exclude and interpolate because of problems in their signals
    raw_data.info["bads"] = ch_excl_list
    
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

    ########################################################################
    ## Preprocessing Starts
    ########################################################################

    ################################
    ## Stage 1: passband and notch filters, and resampling
    low_cut =    1.0
    hi_cut  =   45.0

    print(f"Passband filter {low_cut, hi_cut} Hz...")
    raw_data.filter(l_freq=low_cut, h_freq=hi_cut, picks='eeg')

    if flag_notch:
        print(f"Notch filter...")
        freqs_notch = [60,]
        raw_data.notch_filter(freqs=freqs_notch, picks='eeg', method="spectrum_fit",) ## filter_length="10s"
    # raw_data.notch_filter(freqs=freqs_notch, picks='eeg',) ## filter_length="10s"

    freq_resampling = 250.0 ## usually half of the original sampling frequency (500 Hz), i.e. raw_data.info['sfreq'] / 2.0
    print(f"Resampling (freq: {freq_resampling} Hz)...")
    raw_data.resample(sfreq=freq_resampling, method="polyphase",)

    ###########################################
    ## cropping data according to every section (annotations), namely 'baseline','a_closed_eyes','a_opened_eyes','b_closed_eyes', and 'b_opened_eyes'
    print(f"segments raw data")
    eeg_data_dict = get_eeg_segments(raw_data,)
    # print(f"segments raw-filtered data")
    # eeg_filt_dict = get_eeg_segments(raw_filt,)

    ###############################################
    # each segment has its own properties. We create an object de la class TF_components per segment that includes the raw data and filtered data
    # label_seg_list = ['baseline','a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes']
    label_seg_list = ['a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes','c_closed_eyes','c_opened_eyes']

    #####################################################
    ## data segmentation, each segment would be an object 
    obj_list = eeg_segmentation(eeg_data_dict, label_seg_list, path, session, info_p)

    ##############################################
    ## display PSD and time series signals of each state or label (a_closed_eyes, a_opened_eyes, ...)
    flag_selection = int(input(f"Update segments' selection (0-False, 1-True)?: "))
    if flag_selection:
        ## visual selection of selected segments for each state
        ## the selected segments are manually set in a list_participant.py, in the "selected_ids_dict"
        ch_excl_list.append('VREF')
        display_segments(obj_list, label_seg_list, ch_excl_list)
        return 0
    
    #################
    flag_update_bads = int(input(f"Update bad segments and/or bad channels (0-False, 1-True)?: "))
    ## set a flag for each obj to identify selected EEG recording section from each state (a_ce, a_oe, b_ce, b_oe, c_ce, c_oe)
    ## in the selected sections include bad segments manually by visual inspection
    ## create events equally spaced by dt seconds
    ## from the events create epochs without overlap and with a duration of dt seconds
    ## from the epochs, apply an automatic noise reduction method (Ransac, from the PREP pipeline)
    dt = 5 ## seconds
    # nobj = 3 ## limit of number of sections for testing purposes
    # sel_objs = obj_list[:nobj]
    sel_objs = obj_list
    set_selected_segments(sel_objs, selected_segs_dict, dt, flag_update_bads)

    # #############################################
    # ## observe EEG signals to identify and select bad channels from each remained section
    # flag_update = int(input(f"Update bad-channels (0-False, 1-True)?: "))
    # # annotation_bad_channels_and_segments(obj_list[:nobj], flag_update)
    # annotation_bad_channels(obj_list[:nobj], flag_update)

    ###############################################
    event_list_ce = ['a_ce','b_ce','c_ce']
    event_list_oe = ['a_oe','b_oe','c_oe']
    event_list = event_list_ce + event_list_oe
    # #############################################
    ## apply ICA to try to remove components of noise and artifacts
    flag_update = int(input(f"Update ICA components or ICA components selection (0-False, 1-True)?: "))
    # flag_update = True
    ## apply ICA to the selected segments, one for each state: a_oe, a_ce, b_oe, b_ce, c_oe, c_ce
    ica_artifacts_reduction(sel_objs, flag_update, event_list)

    ###############################################
    ## re-refrencing
    redefine_reference(sel_objs, event_list)

    ## bad channels interpolation, and spatial filtering (Laplacian surface)
    #######################################################################
    ## Preprocessing ends
    #######################################################################
    ## psd visualization
    print(f"plot quantiles")
    save_psd_plots = True
    plot_psd_quantiles(obj_list, event_list, info_p, ylims, path_fig_psd, save_psd_plots)

    #####################
    plt.show(block=True)
    return 0

    ## PSD average per regions
    ## central left and right
    average_psd_regions(sel_objs, event_list)
    plot_psd_responses_median(sel_objs, event_list_ce, event_list_oe, path_fig_fooof, info_p)

    # plot_psd_responses_all(sel_objs, event_list_ce, event_list_oe, path_fig_fooof, info_p)
    
    #####################
    plt.show(block=True)
    return 0

   
    ##################
    ## selected channels; the two lists of channels are equivalent, one in the 128 electrodes and the other in the 64 electrodes
    ch_name_128 = ['VREF','E36','E104']
    ch_name_10_10 = ['Cz','C3','C4']
    
    #################################################
    ## calculate time-frequency transformations for each segment
    ## bad-channels interpolation and current-source-density filter are applied before tf-analysis
    calculate_tf(obj_list, selected_segs_dict, ch_name_128,)

    

   


    # ##################################################
    # ## optional
    # print(f"Plot PSD per selected channels...")
    # psd_plot_filename = path_fig_boxplot + 'psd_selected_chx.png'
    # # plot_psd_per_channels(obj_list, ch_name_128)
    # plot_psd_per_channels_mne(obj_list, ch_name_128, psd_plot_filename, info_p)

    #################################################
    print(f"Baseline normalization...")
    ## reference label for baseline normalization.
    # We chose the first segment of open eyes during resting
    # label_ref = 'a_oe'
    # We chose the first segment of closed eyes during resting
    # label_ref = 'a_ce'
    # baseline_normalization(obj_list, selected_segs_dict, ch_name_128, label_ref)
    ## normalization using two references:
    ## ref for open-eyes: a_oe
    ## ref for closed-eyes: a_ce
    baseline_normalization_two_ref(obj_list, selected_segs_dict,)

    # #############################################
    # # optional
    # print(f"Saving figures time-frequency analysis...")
    # plot_tfr(obj_list, acquisition_system, ch_name_10_10)
    # # optional
    # #############################################

    # frequency responses of the cycling effect, being resting the reference
    plot_filename = path_fig_boxplot + 'normalized_curves.png'
    # fig_title = f"{info_p}\nPercentage change from baseline"
    fig_title = info_p
    # plot_curves_cycling(obj_list, plot_filename, fig_title)
    # plot_curves_tf_diff(obj_list, plot_filename, fig_title)
    plot_curves_tf_diff_2(obj_list, plot_filename, fig_title)

    plt.show(block=True)
    return 0



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
    ## optional
    ## values of frequency bands (median values) over time
    print(f"Power per frequency bands... ")
    # tf_freq_bands(obj_list, acquisition_system, ch_name_10_10)
    tf_freq_bands_boxplots()

    ######################################
    ## plot curves of Beta-band activity. The data is obtained by the previous function: tf_freq_bands()
    # print(f"plot beta band activity...")
    # tf_beta_plot(obj_list, ch_name_10_10)

    

    #############################
    ## boxplots: beta band
    ## sel_ch: 'Cz', 'C3', or 'C4'
    # sel_ch = ch_name_10_10[2]
    ## sel_band: 'beta', 'beta_l', 'beta_h'
    ##
    # sel_band = 'beta_l'
    # label_band = 'EEG low-beta band [12-20 Hz]'
    ##
    sel_band = 'beta'
    label_band = 'EEG beta band [12-30 Hz]'
    
    ## boxplots
    fig_box, ax_box = plt.subplots(nrows=2, ncols=3, figsize=(9,6), sharey=True,)
    ax_box = ax_box.flatten()

    ## plot's columns [1,0,2] --> channels [Cz,C3,C4] 
    ax_ch_list = [[1,4],[0,3],[2,5]]
    labels = [f'before\ncycling',f'during\ncycling',f'after\ncycling']

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
        if obj.get_label() == label_seg and obj.get_selected_flag():
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
