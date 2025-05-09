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
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from pathlib import Path

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
arr_psd = []
channel_id=0

# y_limits = (None, None)
# y_limits = (-0.4e-3, 0.4e-3)
y_limits = [-8,8]

## scale selection for visualization raw data with annotations
scale_dict = dict(mag=1e-12, grad=4e-11, eeg=200e-6, eog=150e-6, ecg=300e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)


# fig, ax = plt.subplots(1, 1, figsize=(5,5))
# fig.canvas.mpl_connect('button_press_event', onClick)

def toggle_pause(event):
        global flag
        if flag==True:
            ani.resume()
        else:
            ani.pause()
        flag = not flag


#############################
# The function to be called anytime a slider's value changes
def update(val):
    global fig_topoplot

    frame = math.floor(frame_slider.val*sampling_rate)
    # print(f'!!! update frame: {frame_slider.val}, {sampling_rate}, {frame} !!!')
    # print(f'!!! data_eeg: {data_eeg.shape} !!!')
    # y_limit = 0.4e-3
    im, cn = mne.viz.plot_topomap(data_eeg[:,frame], raw_data.info, vlim=y_limits, contours=0, axes=ax_topoplot, cmap='magma')
    # colorbar
    fig_topoplot.colorbar(im, cax=cbar_ax)
    fig_topoplot.canvas.draw_idle()
    return 0

#############################
## Bad channels identification
def set_bad_channels(data_dict, subject, section, sequence):
    ## EEG signals of selected section (a_opened_eyes, a_closed_eyes, b_opened_eyes, b_closed_eyes)
    raw_cropped = data_dict[section][sequence]
    ## include bad channels previously identified
    raw_cropped.info["bads"] = bad_channels_dict[subject][section][sequence]
    ## interpolate only selected bad channels; exclude bad channels that are in the extremes or those who do not have good number of surounding good channels
    ch_excl_interp = bad_channels_dict[subject][section+'_excl'][sequence]
    raw_cropped.interpolate_bads(exclude=ch_excl_interp)
    ## re-referencing average (this technique is good for dense EEG)
    # data_dict[section][sequence] = raw_cropped.set_eeg_reference("average",ch_type='eeg',)
    data_dict[section][sequence] = raw_cropped
    return data_dict

#######################
def remove_eeg_bad(eeg_segment, ann_list):
    t1_list = []
    t2_list = []
    for ann in ann_list:
        print(f'{ann}')
        ## iterate annotations labeled : 'bad' to cut them using time sections
        t1_list.append(ann["onset"])
        t2_list.append(ann["onset"] + ann["duration"])

    ## crop bad segments
    if len(t1_list) > 0:
        cropped_data = crop_bad_segments(eeg_segment, t1_list, t2_list)
    else:
        cropped_data = eeg_segment.copy()
    
    return cropped_data

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
def crop_bad_segments(raw_data, t1_list, t2_list):
    global sigmoid_signal
    
    t0 = 0  ## initial time
    t_end = (raw_data.last_samp - raw_data.first_samp) / sampling_rate

    raw_list = []
    for t1, t2 in zip(t1_list, t2_list):
        ## cut valid segment
        raw_crop = crop_fun(raw_data, t0, t1)
        ## sigmoid signal to smooth segment's borders
        sigmoid_signal = set_sigmoid_fun(raw_crop)
        ## apply sigmoid signal to the valid segment (signals multiplication)
        raw_crop.apply_function(mult_functions, picks=['all'])
        ## concatenate resultant segment
        raw_list.append(raw_crop)
        ## redefine t0 to select the next valid segment
        t0 = t2
    ## concatenate last valid segment
    raw_crop = crop_fun(raw_data, t0, t_end)
    sigmoid_signal = set_sigmoid_fun(raw_crop)
    raw_crop.apply_function(mult_functions, picks=['all'])
    raw_list.append(raw_crop)
    ## concatenate raws
    print(len(raw_list))
    if len(raw_list) > 0:
        ## concatenate several segments same label  to have continuous data
        new_raw_data = mne.concatenate_raws(raw_list, )
        ## delete all annotations, including bad annotations generated in the concatenation operation
        ann = new_raw_data.annotations
        ## actual operation that deletes all annotations (by indexes)
        new_raw_data.annotations.delete(np.arange(len(ann)))
    else:
        new_raw_data=None

    return new_raw_data

#######################
def crop_fun(raw_data, t1, t2):
    raw = raw_data.copy().crop(tmin=t1, tmax=t2,)
    # raw.set_meas_date(None)
    # mne.io.anonymize_info(raw.info)
    # print(f'first sample after crop: {raw.first_samp}')
    ann = raw.annotations
    print(f'crop annotations:{len(ann)}\n{ann}')
    raw.annotations.delete(np.arange(len(ann)))
    return raw

def mult_functions(input):

    # print(f'before input: {input}\ntype: {type(input)} length: {len(input)}')
    input = sigmoid_signal * input
    # print(f'after  input: {input}\ntype: {type(input)} length: {len(input)}')

    return (input)


def set_sigmoid_fun(raw):
    
    data = raw.get_data(picks=['all'])
    # print(f'data size: {data.shape}')

    n_signals = data.shape[0]
    n_samples = data.shape[1]

    freq = sampling_rate # sampling rate (samples per second)

    tmin= 0 # seconds
    tmax= n_samples / freq # seconds

    # print(f'freq nsignals nsamples tmin tmax: {freq, n_signals, n_samples, tmin, tmax}')

    t = np.linspace(tmin, tmax,num=n_samples, endpoint=True )

    t0=0.5  ## sigmoid signal translation (time in seconds)
    s0=20   ## sigmoid scale factor... how fast the sigmoid signal make its transition from zero to one (at the begining) or from one to zero (at the end)
    f_ini = 1 / (1 + np.exp(-s0*(t-(tmin+t0))))
    f_end = 1 / (1 + np.exp( s0*(t-(tmax-t0))))

    f_ref = f_ini*f_end
    # print(f'f_ref: {f_ref}\n shape: {f_ref.shape}')

    return f_ref

def concat_fun(eeg_data_dict):
    global sigmoid_signal

    labels_list = ['a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes',]
    new_eeg_data_dict = {}

    for id_label, label in enumerate(labels_list):
        
        print(f'label: {label}')

        raw_list = []
        for id, raw in enumerate(eeg_data_dict[label]):
            ## To create a smooth transition among concatenated signals, we multiply the raw data with a function that makes the EEG signals' boundaries close to zero (we create a function based on sigmoide functions).
            sigmoid_signal = set_sigmoid_fun(raw)
            ## sigmoid signal and raw data multiplication
            raw.apply_function(mult_functions, picks=['all'])
            ## append results
            raw_list.append(raw)

        if len(raw_list) > 0:
            ## concatenate several segments same label  to have continuous data
            new_eeg_data_dict[label] = mne.concatenate_raws(raw_list, )
            ## delete all annotations, including bad annotations generated in the concatenation operation
            ann = new_eeg_data_dict[label].annotations
            ## actual operation that deletes all annotations (by indexes)
            new_eeg_data_dict[label].annotations.delete(np.arange(len(ann)))
        else:
            new_eeg_data_dict[label]=None
    # 
    return new_eeg_data_dict


def interpolation_and_concatenation(eeg_data_dict, subject, session):
    global sigmoid_signal

    labels_list = ['a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes',]
    labels_opened_eyes = ['a_opened_eyes','b_opened_eyes',]

    # fig_sigmoid = [[]]*len(labels_list)

    for id_label, label in enumerate(labels_list):
        
        print(f'label: {label}')
        ## for visualization
        # fig_sigmoid[id_label] = plt.figure(label, figsize=(12, 5))
        # ax_topoplot = fig_sigmoid.subplots(1, 2, sharex=True, sharey=True)

        raw_list = []
        for id, raw in enumerate(eeg_data_dict[label]):
            # print(f'raw data:\n{len(raw)}')
            raw.info["bads"] = bad_channels_dict[subject]['session_'+str(session)][label][id]
            raw.interpolate_bads()
            # print(f'subject, session, label, id: {subject, session, label, id}')
            # print(f'bad channels:{raw.info["bads"]}')

            # if label in (labels_opened_eyes):
            #     ica_simple_func(raw, subject, session)


            ## To create a smooth transition among concatenated signals, we multiply the raw data with a function that makes the EEG signals' boundaries close to zero (we create a function based on sigmoide functions).
            # print(f'calling sigmoid function {id}')
            sigmoid_signal = set_sigmoid_fun(raw)

            ## sigmoid signal and raw data multiplication
            raw.apply_function(mult_functions, picks=['all'])

            # print(f'raw annotations: {raw.annotations}')
            raw_list.append(raw)

        if len(raw_list) > 0:
            ## concatenate several segments same label           
            eeg_data_dict[label] = mne.concatenate_raws(raw_list, )
            ## delete all annotations form each concatenated raw data
            ann = eeg_data_dict[label].annotations
            # print(f'annotations concatenated: {ann}\count: {len(ann)}')
            eeg_data_dict[label].annotations.delete(np.arange(len(ann)))
            # print(f'eeg_data_dict[{label}]: {eeg_data_dict[label]}')
        else:
            eeg_data_dict[label]=None
    # 
    return eeg_data_dict


def channels_interpolation(eeg_data_dict, subject, session):

    labels_list = ['baseline','a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes',]

    for id_label, label in enumerate(labels_list):
        print(f'label: {label}')
        new_raw_list = []
        for id, raw in enumerate(eeg_data_dict[label]):
            # print(f'raw data:\n{len(raw)}')
            raw.info["bads"] = bad_channels_dict[subject]['session_'+str(session)][label][id]
            raw.interpolate_bads()
            new_raw_list.append(raw)
        eeg_data_dict[label] = new_raw_list

    return eeg_data_dict

##
def csd_fun(eeg_data_dict):
    print(f'current source density calculation...')
    labels_list = ['baseline','a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes',]

    new_eeg_data_dict = {}
    for id_label, label in enumerate(labels_list):
        new_raw_list = []
        for id, raw in enumerate(eeg_data_dict[label]):
            print(f'label: {label, id}')
            new_raw_list.append(mne.preprocessing.compute_current_source_density(raw))
        new_eeg_data_dict[label] = new_raw_list

    return new_eeg_data_dict

def visualization_raw(eeg_data_dict):
    labels_list = ['a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes',]

    for id_label, label in enumerate(labels_list):
        for id, raw in enumerate(eeg_data_dict[label]):
            mne.viz.plot_raw(raw, start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title=label+'_'+str(id), block=True)

    return 0


##########################
def ica_appl_func(path, eeg_data_dict, subject, session, scale_dict):
    print(f'Inside ica function.\nSubject, session: {subject, session}')

    labels_list = ['baseline','a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes',]

    for id_label, label in enumerate(labels_list):
        new_raw_list = []
        for id, raw in enumerate(eeg_data_dict[label]):
            print(f'label: {label, id}')
            ## read precalculated ICA per each section
            # filename_ica = path+'ica/'+label+'_'+str(id)+'-ica.fif.gz'
            filename_ica = path+'session_'+str(session)+'/ica/'+label+'_'+str(id)+'-ica.fif.gz'
            ## print(f'filename: {filename_ica}')
            ica = mne.preprocessing.read_ica(filename_ica, verbose=None)
            ## exclude component associated to blinks            
            ica.exclude = blinks_components_dict[subject]['session_'+str(session)][label][id]  # indices that were chosen based on previous ICA calculations (ica_blinks.py)
            # print(f'ICA blink component: {ica.exclude}')
            reconst_raw = raw.copy()

            ica.apply(reconst_raw)
            new_raw_list.append(reconst_raw)
            ################
            ## visualization for comparison
            ## before ICA
            # mne.viz.plot_raw(raw, start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title='Before ICA', block=False)
            ## after ICA
            # mne.viz.plot_raw(reconst_raw, start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title='After ICA', block=True)
            ################
        eeg_data_dict[label] = new_raw_list

    return eeg_data_dict

   

#############################
## topographic views
def plot_topographic_view(raw_data):
    global frame_slider, data_eeg, axfreq, cbar_ax, fig_topoplot
    ## spatial visualization (topographical maps)

    # Passband filter in place
    low_cut =    0.3
    hi_cut  =   45.0
    data = raw_data.copy().filter(l_freq=low_cut, h_freq=hi_cut, picks='eeg')

    data_eeg = data.get_data(picks=['eeg'])
    df_eeg = data.to_data_frame(picks=['eeg'], index='time')
    # print(f'shape data:\n{data_eeg.shape}\n{data_eeg}')
    # print(f'dataframe data:\n{df_eeg}')

    init_frame = 0
    im, cn = mne.viz.plot_topomap(data_eeg[:,init_frame], raw_data.info, vlim=y_limits, contours=0, axes=ax_topoplot, cmap='magma')

    # Make a horizontal slider to control the frequency.
    axfreq = fig_topoplot.add_axes([0.25, 0.1, 0.65, 0.03])

    # Make colorbar
    cbar_ax = fig_topoplot.add_axes([0.05, 0.25, 0.03, 0.65])
    fig_topoplot.colorbar(im, cax=cbar_ax)
    # clb.ax.set_title("topographic view",fontsize=16) # title on top of colorbar
    # fig_topoplot.add_axes([0.25, 0.1, 0.65, 0.03])
    # valmin=0, valmax=len(df_eeg)/sampling_rate,
    frame_slider = Slider( ax=axfreq, label='Time [s]', valmin=0, valmax=len(df_eeg)/sampling_rate, valinit=init_frame/sampling_rate, )

    # register the update function with each slider
    frame_slider.on_changed(update)
    return 0


#############################
# The function to be called anytime a slider's value changes
def update_psd(val):
    global fig_topoplot

    frame = math.floor(frame_slider.val)
    print(f'!!! update frame: {frame_slider.val}, {sampling_rate}, {frame} !!!')
    # print(f'!!! data_eeg: {data_eeg.shape} !!!')
    # y_limit = 0.4e-3
    # vlim=y_limits,
    # y_limits = [-10,10]
    print(f'arr_psd: {arr_psd.shape}\n{arr_psd[:,frame]}')
    im, cn = mne.viz.plot_topomap(arr_psd[:,frame], raw_data.info, vlim=y_limits, contours=0, axes=ax_topoplot, cmap='magma')
    # colorbar
    fig_topoplot.colorbar(im, cax=cbar_ax, label='dB change from baseline')
    fig_topoplot.canvas.draw_idle()
    return 0

#############################
## topographic views
def plot_topomap_psd(raw_data, arr_psd, arr_freqs):
    global frame_slider, data_eeg, axfreq, cbar_ax, fig_topoplot
    ## spatial visualization (topographical maps)

    ## average every frequency band
    init_frame = 10
    freq = arr_freqs[init_frame]
    print(f'inside plot topomap, freq: {freq}')
    # y_limits = [-10,10]
    im, cn = mne.viz.plot_topomap(arr_psd[:,init_frame], raw_data.info, vlim=y_limits, contours=0, axes=ax_topoplot, cmap='magma')

    # Make a horizontal slider to control the frequency.
    axfreq = fig_topoplot.add_axes([0.25, 0.1, 0.65, 0.03])

    # Make colorbar
    cbar_ax = fig_topoplot.add_axes([0.05, 0.25, 0.03, 0.65])
    fig_topoplot.colorbar(im, cax=cbar_ax, label='dB change from baseline')
    frame_slider = Slider( ax=axfreq, label='Time [s]', valmin=0, valmax=len(arr_freqs), valinit=init_frame, )

    # register the update function with each slider
    frame_slider.on_changed(update_psd)
    return 0

#############################
## average frequency components for every frequency band
## topographic views 
# delta: 0 - 4 Hz
# theta: 4 - 8 Hz
# alpha: 8 - 12 Hz
# beta: 12 - 30 Hz
# gamma: 30 - 45 Hz


def plot_topomap_bands(raw_data, arr_psd, arr_freqs, label, filename):

    fig_bands, axs_bands = plt.subplots(1, 4, figsize=(12.5, 4.0))
    print(f'arr_psd and arr_freqs shape: {arr_psd.shape , arr_freqs.shape}')

    ## theta band
    fmin_list=[4,8,12,30]
    fmax_list=[8,12,30,45]
    title_list=['theta [4-8 Hz]', 'alpha [8-12 Hz]', 'beta [12-30 Hz]', 'gamma [30-45 Hz]',]
    for fmin, fmax, ax, title in zip(fmin_list, fmax_list, axs_bands.flat, title_list):
    
        idx_range_freq = np.argwhere((arr_freqs >= fmin)&(arr_freqs <= fmax))
        id_min = np.min(idx_range_freq)
        id_max = np.max(idx_range_freq)

        print(f'idx min and max and values: {id_min, id_max, arr_freqs[id_min], arr_freqs[id_max],}')

        arr_mean = np.mean(arr_psd[:,id_min:id_max], axis=1)
        # print(f'arr mean shape: {arr_mean.shape}')

        # init_frame = 10
        # freq = arr_freqs[init_frame]
        # print(f'inside plot topomap, freq: {freq}')
        # # y_limits = [-10,10]
        im, cn = mne.viz.plot_topomap(arr_mean, raw_data.info, vlim=y_limits, contours=0, axes=ax, ) # cmap='magma'
        ax.title.set_text(title)
    
    # Make colorbar
    cbar_ax = fig_bands.add_axes([0.02, 0.25, 0.03, 0.60])
    fig_bands.colorbar(im, cax=cbar_ax, label='dB change from baseline')

    fig_bands.suptitle(label, size='large', weight='bold')

    fig_bands.savefig(filename, transparent=True)
    

    # clb.ax.set_title("topographic view",fontsize=16) # title on top of colorbar
    # fig_topoplot.add_axes([0.25, 0.1, 0.65, 0.03])
    # valmin=0, valmax=len(df_eeg)/sampling_rate,

    # Make a horizontal slider to control the frequency.
    # axfreq = fig_topoplot.add_axes([0.25, 0.1, 0.65, 0.03])

    # frame_slider = Slider( ax=axfreq, label='Time [s]', valmin=0, valmax=len(arr_freqs), valinit=init_frame, )

    # register the update function with each slider
    # frame_slider.on_changed(update_psd)

    return 0

###########################################
def annotations_bad_segments(eeg_data_dict, subject, session,scale_dict):

    labels_list = ['a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes',]
    label_title = ['resting closed eyes','resting opened eyes','ABT closed eyes','ABT opened eyes']

    ########################
    annotations_dict={}

    update_annotations = input('Generate annotations ? (1-True, 0-False): ')
    if int(update_annotations)==1:
        
        ## interactive bad annotations
        for ax_number, section in enumerate(labels_list):
            # print(f'{section, ax_number} generating interactive annotations')
            ## signals visualization
            ann_list=[]
            # if eeg_data_dict[section] != None:
            for idx, eeg_segment in enumerate(eeg_data_dict[section]):
                ## channels' voltage vs time
                print(f'annotations: {section, idx}')
    
                ## signals visualization to annotate bad segments (interactive annotation)
                mne.viz.plot_raw(eeg_segment, start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title=(label_title[ax_number]+' cycle '+str(idx)), block=True)
    
                ## annotation time is referenced to the time of first_samp, and that is different for each section
                time_offset = eeg_segment.first_samp / sampling_rate  ## in seconds
                ## get and rewrite annotations minus time-offset
                interactive_annot = eeg_segment.annotations
                if len(interactive_annot) >0:
                    print(f'remove offset...')
                    ann_list.append(ann_remove_offset(interactive_annot, time_offset))
                    print(f'remove offset done.')
                else:
                    print(f'no annotations')
                    ann_list.append(interactive_annot)

            annotations_dict[section] = ann_list

        save_ann = input('save annotations? (1-True, 0-False): ')
        if int(save_ann)==1:
        # writing dictionary to a binary file
            with open('../data/results_ICA/ann_pat_'+str(subject)+'_session_'+str(session)+'.pkl', 'wb') as file:
                pickle.dump(annotations_dict, file)
        else:
            pass
    ## open annotations file from disk
    else:
        try:
            with open('../data/results_ICA/ann_pat_'+str(subject)+'_session_'+str(session)+'.pkl', 'rb') as file:
                annotations_dict = pickle.load(file)
            print(f'annotations:\n{annotations_dict}')
        except FileNotFoundError:
            annotations_dict = {}
        
        ## annotations visualization
        for ax_number, section in enumerate(labels_list):
            ## set previous annotations
            print(f'section and number: {section}')
            # print(f'first sample: {eeg_data_dict[section].first_samp}')
            eeg_list=[]
            for idx, eeg_segment in enumerate(eeg_data_dict[section]):

                eeg_segment.set_annotations(annotations_dict[section][idx])
                eeg_list.append(eeg_segment)

                # mne.viz.plot_raw(eeg_segment, start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title=(label_title[ax_number]+' cycle '+str(idx)), block=True)

            eeg_data_dict[section] = eeg_list

    plt.show(block=True)
    return eeg_data_dict

#############################
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

#############################
def set_bad_segments(eeg_data_dict, path_ann, fn):
    ## set annotations
    annotations_dict = {}
    labels_list = ['baseline','a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes',]

    all_filenames = os.listdir(path_ann)
    print(f'annotations filenames: {all_filenames}')

    for ax_number, section in enumerate(labels_list):
        
        start_name = section+fn
        files_list = [i for i in all_filenames if i.startswith(start_name)]
        
        ann_list = []
        for filename in sorted(files_list):
            print(f'reading: {filename}')
            ann_list.append( mne.read_annotations(path_ann+filename,))
        
        annotations_dict[section] = ann_list
    # try:
    #     with open(ann_filename, 'rb') as file:
    #         annotations_dict = pickle.load(file)
    #     print(f'annotations:\n{annotations_dict}')
    # except FileNotFoundError:
    #     annotations_dict = {}
    
    ## annotate bad segments to exclude them of posterior calculations
    for ax_number, section in enumerate(labels_list):
        eeg_list= []
        for eeg_segment, ann in zip(eeg_data_dict[section], annotations_dict[section]):
            ## channels' voltage vs time
            eeg_segment.set_annotations(ann)
            eeg_list.append(eeg_segment)

        eeg_data_dict[section] = eeg_list

    return eeg_data_dict

##########################
def set_psd_fun(input):
    global channel_id

    input = arr_psd[channel_id,:]
    channel_id+=1

    return (input)

#############################
## EEG filtering and signals pre-processing

def main(args):
    global spectrum, data_spectrum, fig, ax, ani, draw_image, frame_slider, data_eeg, raw_closed_eyes, ax_topoplot, axfreq, fig_topoplot, cbar_ax, sampling_rate, raw_data, arr_psd

    ## interactive mouse pause the image visualization
    # fig.canvas.mpl_connect('button_press_event', toggle_pause)

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
    hi_cut  =   None
    raw_data.filter(l_freq=low_cut, h_freq=hi_cut, picks='eeg')

    ############################
    ## read annotations (.csv file)
    my_annot = mne.read_annotations(path + fn_csv[0])
    print(f'annotations:\n{my_annot}')
    ## adding annotations to raw data
    raw_data.set_annotations(my_annot)
    
    ###########################################
    ## cropping data according to every section (annotations)
    eeg_data_dict = get_eeg_segments(raw_data,)

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
    eeg_data_dict = csd_fun(eeg_data_dict)

    ##########################################################
    ## when rest and bike are in two different files, we save baseline for rest, and we open baseline for bike
    ## save baseline
    flag_bl = input('save baseline ? (1 (True), 0 (False)): ')
    if int(flag_bl)==1:
        eeg_data_dict['baseline'][0].save(path + 'baseline.fif.gz')
    else:
        pass
    
    ## load baseline
    flag_bl = input('load baseline ? (1 (True), 0 (False)): ')
    if int(flag_bl)==1:
        eeg_data_dict['baseline'] = [mne.io.read_raw_fif(path + 'baseline.fif.gz',)]
    else:
        pass
    ###########################################################

    print(f'baseline:\n{eeg_data_dict["baseline"]}')

    ## visualization raw after processing
    # visualization_raw(eeg_data_dict)

    ## At this point, blink artifacts have been removed and the Surface Lapacian has been applied to the eeg data. Additionally, evident artifacts were annotated interactively and labeled as "bad" to exclude them from posterior calculations

    ####################
    ## comparison between closed and opened eyes
    ## baseline
    list_bl = []
    for raw_baseline in eeg_data_dict['baseline']: 
        ## median values of psd closed and opened eyes

        ## power spectral density (psd) from first resting closed eyes
        psd_bl = raw_baseline.compute_psd(fmin=0,fmax=45,reject_by_annotation=True)
        data_bl, freq_bl = psd_bl.get_data(return_freqs=True)
        print(f'arr_bl : {data_bl.shape}')

        list_bl.append(data_bl)

    arr_bl=np.array(list_bl)
    median_bl = np.median(arr_bl, axis=0)
    print(f'median_bl : {median_bl.shape}')

    ## resting
    # list_c = []
    # list_o = []
    ## create folder if it does not exit already
    Path(f"{path}session_{session}/figures/topomaps").mkdir(parents=True, exist_ok=True)
    

    labels_list=['a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes']
    titles_list=['rest closed eyes','rest opened eyes','bike closed eyes','bike opened eyes']
    # for raw_c, raw_o in zip(eeg_data_dict['a_closed_eyes'], eeg_data_dict['a_opened_eyes']): 
    for label, title in zip(labels_list, titles_list):
        print(f'label: {label}\ntitle: {title}')
        list_psd = []

        if len(eeg_data_dict[label]) > 0:

            for raw in eeg_data_dict[label]: 
                ## median values of psd closed and opened eyes

                ## power spectral density (psd) from first resting closed eyes
                psd_raw = raw.compute_psd(fmin=0,fmax=45,reject_by_annotation=True)
                # psd_raw_o = raw_o.compute_psd(fmin=0,fmax=45,reject_by_annotation=True)

                data_psd, freq_psd = psd_raw.get_data(return_freqs=True)
                # data_o, freq_o = psd_raw_o.get_data(return_freqs=True)

                print(f'data_psd : {data_psd.shape}')
                # print(f'arr_o : {data_o.shape}')

                list_psd.append(data_psd)
                # list_o.append(data_o)

            arr_psd=np.array(list_psd)
            print(f'arr_psd shape: {arr_psd.shape}')
            # arr_o=np.array(list_o)

            median_psd = np.median(arr_psd, axis=0)
            # median_o = np.median(arr_o, axis=0)

            print(f'median_psd : {median_psd.shape}')
            # print(f'median_o : {median_o.shape}')

            ##########
            ## normalization for each channel using baseline
            norm_psd = median_psd / median_bl
            # norm_o = median_o / median_bl

            # arr_psd = norm_psd

            ## copy spectrum as template
            # psd_ref = psd_raw.copy()
            ## multiplied by 1e-6 as a scale factor for visualization because the visualizaiton function multiply data by 1e6
            # psd_ref._data = norm_psd*1e-6
            # psd_ref.plot()


            norm_psd = 10*np.log10(norm_psd)

            # # power spectrum density visualization
            # fig_title = "power spectrum density"
            # fig_psd = plt.figure(fig_title, figsize=(12, 5))
            # ax_psd = fig_psd.add_subplot(1,1,1)
            # for v in norm_psd:
            #     # ax_psd.plot(freqs,10*np.log(v))
            #     ax_psd.plot(freq_psd,v)

            filename_fig = f"{path}session_{session}/figures/topomaps/{label}_topo.png"
            plot_topomap_bands(raw_data, norm_psd, freq_psd, title, filename_fig)

        else:
            pass


    plt.show(block=True)
    return 0

    ##########################


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
