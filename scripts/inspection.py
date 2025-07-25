#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import mne
mne.set_log_level('error')

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from matplotlib.widgets import Button, Slider
from pathlib import Path

import json
import pickle

import pathlib
import time
# from autoreject import AutoReject
# from mne.preprocessing import EOGRegression, ICA, corrmap, create_ecg_epochs, create_eog_epochs
# from mne_icalabel import label_components

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
ch_names_list = []



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
    # ch_names = ch_names_list
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

##################################
def get_eeg_segments(raw_data,):    
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

    return eeg_data_dict


#######################################
def get_eeg_segments_two(raw_r, raw_v):  
    ## prefix:
    ## a:resting; b:biking
    baseline_list = []
    a_closed_eyes_list = []
    a_opened_eyes_list = []
    b_closed_eyes_list = []
    b_opened_eyes_list = []

    for ann in raw_r.annotations:
        # print(f'ann:\n{ann}')
        label = ann["description"]
        duration = ann["duration"]
        onset = ann["onset"]
        # print(f'annotation:{count1, onset, duration, label}')
        t1 = onset
        t2 = onset + duration
        if label == 'baseline':
            baseline_list.append(crop_fun(raw_r, t1, t2))

        elif label == 'a_closed_eyes':
            a_closed_eyes_list.append(crop_fun(raw_r, t1, t2))

        elif label == 'a_opened_eyes':
            a_opened_eyes_list.append(crop_fun(raw_r, t1, t2))
        else:
            pass


    for ann in raw_v.annotations:
        # print(f'ann:\n{ann}')
        label = ann["description"]
        duration = ann["duration"]
        onset = ann["onset"]
        # print(f'annotation:{count1, onset, duration, label}')
        t1 = onset
        t2 = onset + duration

        if label == 'b_closed_eyes':
            b_closed_eyes_list.append(crop_fun(raw_v, t1, t2))

        elif label == 'b_opened_eyes':
            b_opened_eyes_list.append(crop_fun(raw_v, t1, t2))

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

    return eeg_data_dict


#############################
## EEG filtering and signals pre-processing

def main(args):
    global spectrum, data_spectrum, fig, ax, ani, draw_image, frame_slider, data_eeg, raw_closed_eyes, ax_topoplot, axfreq, fig_topoplot, cbar_ax, sampling_rate, raw_data, ch_names_list

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
    if abt == 0:
        path, fn_in, fn_csv, raw_data, fig_title, rows_plot, acquisition_system = participants_list(path, subject, session, 0)
        ## exclude channels of the net boundaries that usually bring noise or artifacts
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
        ## channels names
        ch_names_list = raw_data.info['ch_names']
        ############################
        ## run matplotlib in interactive mode
        plt.ion()
        ############################
        
        ################################
        ## Stage 1: high pass filter (in place)
        #################################
        low_cut =    0.5
        hi_cut  =   None
        raw_data.filter(l_freq=low_cut, h_freq=hi_cut, picks='eeg')

        ############################
        ## read annotations (.csv file)
        my_annot = mne.read_annotations(path + fn_csv[0])
        print(f'annotations:\n{my_annot}')
        ## adding annotations to raw data
        raw_data.set_annotations(my_annot)
        
        ############################
        ## scale selection for visualization raw data with annotations
        scale_dict = dict(mag=1e-12, grad=4e-11, eeg=100e-6, eog=150e-6, ecg=300e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)
        ## signals visualization (channels' voltage vs time)
        # filt_raw_data = raw_data.copy().filter(l_freq=1.0, h_freq=45.0)
        # highpass=1.0, lowpass=45.0,
        mne.viz.plot_raw(raw_data, highpass=1.0, lowpass=45.0, filtorder=4, start=0, duration=240, scalings=scale_dict, title=fig_title, block=False)
        ###########################################
        
        ###########################################
        ## cropping data according to every section (annotations)
        eeg_data_dict = get_eeg_segments(raw_data,)

    else:
        path_r, fn_in_r, fn_csv_r, raw_data_r, fig_title_r, rows_plot_r, acquisition_system = participants_list(path, subject, session, 0)
        path, fn_in_v, fn_csv_v, raw_data_v, fig_title_v, rows_plot_v, acquisition_system = participants_list(path, subject, session, 1)

        ## exclude channels of the net boundaries that usually bring noise or artifacts
        raw_data_r.info["bads"] = bad_channels_dict[acquisition_system]
        raw_data_v.info["bads"] = bad_channels_dict[acquisition_system]

        raw_data_r.drop_channels(raw_data_r.info['bads'])
        raw_data_v.drop_channels(raw_data_v.info['bads'])

        ##########################
        # printing basic information from data
        # print(f'raw data filename: {fn_in}')
        # print(f'annotations filename: {fn_csv}')
        # print(f'raw data info:\n{raw_data.info}')
        # printing basic information from data
        ############################
        ## sampling rate
        sampling_rate = raw_data_r.info['sfreq']
        print(f"sampling rate: {sampling_rate}")
        ############################
        ## channels names
        ch_names_list = raw_data_r.info['ch_names']
        print(f"channels names: {ch_names_list}")
        ############################
        ## run matplotlib in interactive mode
        plt.ion()
        ############################
        
        ################################
        ## Stage 1: high pass filter (in place)
        #################################
        low_cut =   0.5
        hi_cut  =   None
        raw_data_r.filter(l_freq=low_cut, h_freq=hi_cut, picks='eeg')
        raw_data_v.filter(l_freq=low_cut, h_freq=hi_cut, picks='eeg')

        ############################
        ## read annotations (.csv file)
        my_annot_r = mne.read_annotations(path + fn_csv_r[0])
        my_annot_v = mne.read_annotations(path + fn_csv_v[0])
        # print(f'annotations:\n{my_annot}')
        ## adding annotations to raw data
        raw_data_r.set_annotations(my_annot_r)
        raw_data_v.set_annotations(my_annot_v)
        
        ############################
        ## scale selection for visualization raw data with annotations
        scale_dict = dict(mag=1e-12, grad=4e-11, eeg=100e-6, eog=150e-6, ecg=300e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)
        ## signals visualization (channels' voltage vs time)
        # filt_raw_data = raw_data.copy().filter(l_freq=1.0, h_freq=45.0)
        # highpass=1.0, lowpass=45.0,
        fig_title = f"EEG subject {subject} resting"
        mne.viz.plot_raw(raw_data_r, highpass=1.0, lowpass=45.0, filtorder=4, start=0, duration=240, scalings=scale_dict, title=fig_title, block=False)
        fig_title = f"EEG subject {subject} passive biking"
        mne.viz.plot_raw(raw_data_v, highpass=1.0, lowpass=45.0, filtorder=4, start=0, duration=240, scalings=scale_dict, title=fig_title, block=False)
        ###########################################

        ###########################################
        ## cropping data according to every section (annotations)
        eeg_data_dict = get_eeg_segments_two(raw_data_r, raw_data_v)



    # #############################
    # #########################
    # ## new path, eeg filename (fn_in), annotations filename (fn_csv), eeg raw data (raw_data)
    # path, fn_in, fn_csv, raw_data, fig_title, rows_plot, acquisition_system = participants_list(path, subject, session, abt)
    # if fn_csv == '':
    #     print(f'It could not find the selected subject. Please check the path, and the selected subject number in the list of participants.')
    #     return 0
    # else:
    #     pass
    
    # ## exclude channels of the net boundaries that usually bring noise or artifacts
    # raw_data.info["bads"] = bad_channels_dict[acquisition_system]
    # raw_data.drop_channels(raw_data.info['bads'])

    # ##########################
    # # printing basic information from data
    # print(f'raw data filename: {fn_in}')
    # print(f'annotations filename: {fn_csv[0]}')
    # print(f'raw data info:\n{raw_data.info}')
    # # printing basic information from data
    # ############################
    # ## sampling rate
    # sampling_rate = raw_data.info['sfreq']
    # ############################
    # ## sampling rate
    # ch_names_list = raw_data.info['ch_names']
    # ############################

    # ## run matplotlib in interactive mode
    # plt.ion()
    
    # ################################
    # ## Stage 1: high pass filter (in place)
    # #################################
    # low_cut =    0.5
    # hi_cut  =   None
    # raw_data.filter(l_freq=low_cut, h_freq=hi_cut, picks='eeg')

    # ############################
    # ## read annotations (.csv file)
    # my_annot = mne.read_annotations(path + fn_csv[0])
    # print(f'inital annotations:\n{my_annot}')
    # ## adding annotations to raw data
    # raw_data.set_annotations(my_annot)
    
    # ############################
    # ## scale selection for visualization raw data with annotations
    # scale_dict = dict(mag=1e-12, grad=4e-11, eeg=100e-6, eog=150e-6, ecg=300e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)
    # ## signals visualization (channels' voltage vs time)
    # # filt_raw_data = raw_data.copy().filter(l_freq=1.0, h_freq=45.0)
    # # highpass=1.0, lowpass=45.0,
    # mne.viz.plot_raw(raw_data, highpass=1.0, lowpass=45.0, filtorder=4, start=0, duration=240, scalings=scale_dict, title=fig_title, block=False)
    # ###########################################
    
    # eeg_data_dict = get_eeg_segments(raw_data)



    labels_list = ['baseline','a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes',]
    label_title = ['baseline','resting closed eyes','resting opened eyes','ABT closed eyes','ABT opened eyes']

    ########################
    ## redefine annotations per segment
    new_eeg_data_dict = {}
    annotations_dict={}

    ## create folder new annotations if it does not exit already
    Path(path+'session_'+str(session)+"/new_annotations").mkdir(parents=True, exist_ok=True)
    path_ann = path+'session_'+str(session)+"/new_annotations/"
    print(f"path annotations: {path_ann}")

    update_annotations = input('Update annotations (include new bad segments) ? (1-True, 0-False): ')
    if int(update_annotations)==1:
        
        ## annotate bad segments to exclude them of posterior calculations
        for ax_number, section in enumerate(labels_list):
            print(f'{section, ax_number} generating interactive annotations')
            ## signals visualization
            # if eeg_data_dict[section] != None:
            ann_list = []
            eeg_list= []
            
            for id_section, eeg_segment in enumerate(eeg_data_dict[section]):
                print(f"{section}: {id_section}")
                ## channels' voltage vs time
                ## signals visualization to annotate bad segments (interactive annotation)
                # filt_eeg_segment = eeg_segment.copy().filter(l_freq=1.0, h_freq=45.0)
                
                
                mne.viz.plot_raw(eeg_segment, start=0, duration=240, highpass=1.0, lowpass=45.0, filtorder=4, scalings=scale_dict, title=label_title[ax_number], block=True)

                ## annotation time is referenced to the time of first_samp, and that is different for each section
                time_offset = eeg_segment.first_samp / sampling_rate  ## in seconds
                ## get and rewrite annotations minus time-offset
                interactive_annot = eeg_segment.annotations
                ann_list.append( ann_remove_offset(interactive_annot, time_offset) )

                eeg_list.append(eeg_segment)

            annotations_dict[section] = ann_list
            eeg_data_dict[section] = eeg_list

        save_ann = input('save annotations? (1-True, 0-False): ')
        
        ## saving annotations
        if int(save_ann)==1:
            for ax_number, section in enumerate(labels_list):
                ann_list = annotations_dict[section]
                for id, ann in enumerate(ann_list):
                    filename = section+str(id).zfill(2)+'.fif'
                    ann.save(path_ann+filename,overwrite=True)
                    print(f"annotations {filename} saved")

        # writing dictionary to a binary file
            # with open(path + fn_csv[1] + '.pkl', 'wb') as file:
                # pickle.dump(annotations_dict, file)
        else:
            pass
    
    else:
        ## read annotations
        all_filenames = os.listdir(path_ann)
        print(f'annotations filenames: {all_filenames}')

        for ax_number, section in enumerate(labels_list):
            
            start_name = section
            files_list = [i for i in all_filenames if i.startswith(start_name)]
            
            ann_list = []
            for filename in sorted(files_list):
                print(f'reading: {filename}')
                ann_list.append( mne.read_annotations(path_ann+filename,))
            
            annotations_dict[section] = ann_list

        # try:
        #     filename = path + fn_csv[1] + '.pkl'
        #     print(f'filename annotations: {filename}')
        #     with open(filename, 'rb') as file:
        #         annotations_dict = pickle.load(file)
        #     print(f'annotations:\n{annotations_dict}')
        # except FileNotFoundError:
        #     print(f'problem open new annotations file')
        #     return 0
        
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
    ## create folder psd if it does not exit already
    path_psd = path+'session_'+str(session)+'/figures/psd/'
    Path(path_psd).mkdir(parents=True, exist_ok=True)

    for ax_number, section in enumerate(labels_list):
        print(f'section: {section}')
        ## signals visualization
        fig_psd = plt.figure(fig_title, figsize=(9, 7))
        number_ax = len(eeg_data_dict[section])
        print(f'number ax: {number_ax}')
        ax_psd = [[]]*number_ax
        idx=0
        for eeg_segment in eeg_data_dict[section]:
            print(f'eeg_segment')
            ## channels' spectrum of frequencies
            ax_psd[idx] = fig_psd.add_subplot(number_ax,1,idx+1)
            mne.viz.plot_raw_psd(eeg_segment, exclude=[], ax=ax_psd[idx], fmax=100, reject_by_annotation=True, xscale='log',)
            # ax_psd[idx].set_title(label_title[ax_number] +'_'+str(idx))
            ax_psd[idx].set_title('')
            ax_psd[idx].set_ylim([-20, 50])
            ## channels' voltage vs time
            # filt_raw = eeg_segment.copy().filter(l_freq=1.0, h_freq=45.0)
            # highpass=1.0, lowpass=45.0,
            mne.viz.plot_raw(eeg_segment, start=0, duration=240, highpass=1.0, lowpass=45.0, filtorder=4, scalings=scale_dict, title=label_title[ax_number]+'_'+str(idx), block=False)

            idx+=1

        plt.show(block=True)
        fig_psd.suptitle(label_title[ax_number])
        fig_psd.savefig(path_psd+section, transparent=False,)

    plt.show(block=True)
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
