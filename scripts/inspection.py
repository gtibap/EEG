#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import mne
mne.set_log_level('error')

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

import json
import pickle

import pathlib
import time
from autoreject import AutoReject
from mne.preprocessing import EOGRegression, ICA, corrmap, create_ecg_epochs, create_eog_epochs

from mne_icalabel import label_components

from bad_channels import bad_channels_dict
from list_participants import participants_list
from selected_sequences import selected_sequences_dict
from blinks_components import blinks_components_dict


ani = 0
flag=False
images=[]
spectrum = []
data_spectrum = []
draw_image = []

frame_slider = []
data_eeg =[]
raw_closed_eyes = []
ax_topoplot = []
axfreq = []
fig_topoplot = []
cbar_ax = []
sampling_rate = 1.0
raw_data=[]
sigmoid_signal = []



#######################
def ann_remove_offset(interactive_annot, time_offset):
    arr_onset=np.array([])
    arr_durat=np.array([])
    arr_label=[]

    for ann in interactive_annot:
        arr_onset = np.append( arr_onset, ann['onset']-time_offset )
        arr_durat = np.append(arr_durat, ann['duration'])
        arr_label.append(ann['description'])

    my_annot = mne.Annotations(
    onset=arr_onset,  # in seconds
    duration=arr_durat,  # in seconds, too
    description=arr_label,
    )
    return my_annot


#######################
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
## EEG filtering and signals pre-processing

def main(args):
    global spectrum, data_spectrum, fig, ax, ani, draw_image, frame_slider, data_eeg, raw_closed_eyes, ax_topoplot, axfreq, fig_topoplot, cbar_ax, sampling_rate, raw_data

    print(f'arg {args[1]}') ## folder location
    print(f'arg {args[2]}') ## subject = {0:Mme Chen, 1:Taha, 2:Carlie, 3:Iulia, 4:A. Caron}
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
    path, fn_in, fn_csv, raw_data, fig_title, rows_plot = participants_list(path, subject, session, abt)
    if fn_csv == '':
        print(f'It could not find the selected subject. Please check the path, and the selected subject number in the list of participants.')
        return 0
    else:
        pass
    
    ##########################
    # printing basic information from data
    print(f'raw data filename: {fn_in}')
    print(f'annotations filename: {fn_csv[0]}')
    print(f'raw data info:\n{raw_data.info}')
    # printing basic information from data
    ############################
    ## sampling rate
    sampling_rate = raw_data.info['sfreq']
    ############################
    ## run matplotlib in interactive mode
    plt.ion()
    
    ################################
    ## Stage 1: high pass filter (in place)
    #################################
    low_cut =    0.5
    hi_cut  =   None
    raw_data.filter(l_freq=low_cut, h_freq=hi_cut, picks='eeg')

    ############################
    ## read annotations (.csv file)
    my_annot = mne.read_annotations(path + fn_csv[0])
    print(f'inital annotations:\n{my_annot}')
    ## adding annotations to raw data
    raw_data.set_annotations(my_annot)
    ############################
    ## scale selection for visualization raw data with annotations
    scale_dict = dict(mag=1e-12, grad=4e-11, eeg=100e-6, eog=150e-6, ecg=300e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)
    ## signals visualization (channels' voltage vs time)
    mne.viz.plot_raw(raw_data, start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title=fig_title, block=False)
    ###########################################
    
    ## cropping data according to annotations
    ## prefix:
    ## a:resting; b:biking
    baseline_list = []
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

        else:
            pass

    print(f'size list baseline: {len(baseline_list)}')
    print(f'size list a_closed_eyes: {len(a_closed_eyes_list)}')
    print(f'size list a_opened_eyes: {len(a_opened_eyes_list)}')
    print(f'size list b_closed_eyes: {len(b_closed_eyes_list)}')
    print(f'size list b_opened_eyes: {len(b_opened_eyes_list)}')

    ## eeg data to a dictionary
    eeg_data_dict={}
    eeg_data_dict['baseline'] = baseline_list
    eeg_data_dict['a_closed_eyes'] = a_closed_eyes_list
    eeg_data_dict['a_opened_eyes'] = a_opened_eyes_list
    eeg_data_dict['b_closed_eyes'] = b_closed_eyes_list
    eeg_data_dict['b_opened_eyes'] = b_opened_eyes_list


    labels_list = ['baseline','a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes',]
    label_title = ['baseline','resting closed eyes','resting opened eyes','ABT closed eyes','ABT opened eyes']

    ########################
    ## redefine annotations per segment
    new_eeg_data_dict = {}
    annotations_dict={}

    update_annotations = input('Update annotations (include new bad segments) ? (1-True, 0-False): ')
    if int(update_annotations)==1:
        
        ## annotate bad segments to exclude them of posterior calculations
        for ax_number, section in enumerate(labels_list):
            print(f'{section, ax_number} generating interactive annotations')
            ## signals visualization
            # if eeg_data_dict[section] != None:
            ann_list = []
            eeg_list= []
            
            for eeg_segment in eeg_data_dict[section]:
                ## channels' voltage vs time
                ## signals visualization to annotate bad segments (interactive annotation)
                mne.viz.plot_raw(eeg_segment, start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title=label_title[ax_number], block=True)

                ## annotation time is referenced to the time of first_samp, and that is different for each section
                time_offset = eeg_segment.first_samp / sampling_rate  ## in seconds
                ## get and rewrite annotations minus time-offset
                interactive_annot = eeg_segment.annotations
                ann_list.append( ann_remove_offset(interactive_annot, time_offset) )

                eeg_list.append(eeg_segment)

            annotations_dict[section] = ann_list
            eeg_data_dict[section] = eeg_list

        save_ann = input('save annotations? (1-True, 0-False): ')
        if int(save_ann)==1:
        # writing dictionary to a binary file
            with open(path + fn_csv[1] + '.pkl', 'wb') as file:
                pickle.dump(annotations_dict, file)
        else:
            pass
    
    else:
        try:
            filename = path + fn_csv[1] + '.pkl'
            print(f'filename annotations: {filename}')
            with open(filename, 'rb') as file:
                annotations_dict = pickle.load(file)
            print(f'annotations:\n{annotations_dict}')
        except FileNotFoundError:
            print(f'problem open new annotations file')
            return 0
        
        ## annotate bad segments to exclude them of posterior calculations
        for ax_number, section in enumerate(labels_list):
            eeg_list= []
            for eeg_segment, ann in zip(eeg_data_dict[section], annotations_dict[section]):
                ## channels' voltage vs time
                eeg_segment.set_annotations(ann)
                eeg_list.append(eeg_segment)

            eeg_data_dict[section] = eeg_list


    # power spectrum density visualization
    fig_title = "power spectrum density"

    for ax_number, section in enumerate(labels_list):
        print(f'section: {section}')
        ## signals visualization
        fig_psd = plt.figure(fig_title, figsize=(8, 5))
        number_ax = len(eeg_data_dict[section])
        print(f'number ax: {number_ax}')
        ax_psd = [[]]*number_ax
        idx=0
        for eeg_segment in eeg_data_dict[section]:
            print(f'eeg_segment')
            ## channels' spectrum of frequencies
            ax_psd[idx] = fig_psd.add_subplot(number_ax,1,idx+1)
            mne.viz.plot_raw_psd(eeg_segment, exclude=[], ax=ax_psd[idx], fmax=100, reject_by_annotation=True,)
            ax_psd[idx].set_title(label_title[ax_number] +'_'+str(idx))
            ax_psd[idx].set_ylim([-20, 50])
            ## channels' voltage vs time
            mne.viz.plot_raw(eeg_segment, start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title=label_title[ax_number]+'_'+str(idx), block=False)

            idx+=1

        plt.show(block=True)

    plt.show(block=True)
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
