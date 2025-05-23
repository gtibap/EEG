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

y_limits = (None, None)
# y_limits = (-0.4e-3, 0.4e-3)
    


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
    y_limit = 0.4e-3
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
def remove_eeg_bad(eeg_data_dict, annotations_dict, section):
    t1_list = []
    t2_list = []
    for ann in annotations_dict[section]:
        print(f'{ann}')
        ## iterate annotations labeled : 'bad' to cut them using time sections
        t1_list.append(ann["onset"])
        t2_list.append(ann["onset"] + ann["duration"])

    ## crop bad segments
    if len(t1_list) > 0:
        cropped_data = crop_bad_segments(eeg_data_dict[section], t1_list, t2_list)
    else:
        cropped_data = eeg_data_dict[section].copy()
    
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

    labels_list = ['a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes',]

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
    labels_list = ['a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes',]

    for id_label, label in enumerate(labels_list):
        print(f'label: {label}')
        eeg_data_dict[label] = mne.preprocessing.compute_current_source_density(eeg_data_dict[label])

    return eeg_data_dict

##########################
def ica_simple_func(raw, subject, session):
    print(f'Inside ica function.\nSubject, session: {subject, session}')
    ## ICA components
    ica = ICA(n_components= 0.99, method='fastica', max_iter="auto", random_state=97)
    
    ## high pass filter at 1 Hz to raw data before ICA calculation
    filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)
    ## ICA fitting model to the filtered raw data
    ica.fit(filt_raw)

    ica.plot_sources(raw, show_scrollbars=False, block=False)
    ica.plot_components(inst=raw, contours=0)

    # ica.exclude = [2]  # indices chosen based on various plots above
    # reconst_raw = raw.copy()
    # ica.apply(reconst_raw)

    # blinks
    # ica.plot_overlay(raw, exclude=[2], picks="eeg")

    # plt.show(block=True)
    # return 0
    
    # ica.exclude = [2]  # indices chosen based on various plots above

    # ica.apply() changes the Raw object in-place, so let's make a copy first:
    # reconst_raw = raw.copy()
    # ica.apply(reconst_raw)
    return 0

##########################
def ica_func(eeg_data_dict, subject, session):
    print(f'Inside ica function.\nSubject, session: {subject, session}')
    ## ICA components
    ica = ICA(n_components= 0.99, method='fastica', max_iter="auto", random_state=97)

    labels_list = ['a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes',]
    new_eeg_data_dict = {}

    for id_label, label in enumerate(labels_list):
        new_raw_list = []
        for id, raw in enumerate(eeg_data_dict[label]):
            print(f'label: {label, id}')
            # print(f'raw data:\n{len(raw)}')
            filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)
            ## ICA fitting model to the filtered raw data
            ica.fit(filt_raw)
            #################
            ## ica components visualization
            # ica.plot_sources(raw, show_scrollbars=False, block=False)
            # ica.plot_components(inst=raw, contours=0,)
            # plt.show(block=True)
            #################
            ica.exclude = blinks_components_dict[subject]['session_'+str(session)][label][id]  # indices chosen based on various plots above
            print(f'ICA blink component: {ica.exclude}')
            reconst_raw = raw.copy()
            ica.apply(reconst_raw)

            new_raw_list.append(reconst_raw)
        new_eeg_data_dict[label] = new_raw_list

    return new_eeg_data_dict

   

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
## EEG filtering and signals pre-processing

def main(args):
    global spectrum, data_spectrum, fig, ax, ani, draw_image, frame_slider, data_eeg, raw_closed_eyes, ax_topoplot, axfreq, fig_topoplot, cbar_ax, sampling_rate, raw_data

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

    update_ICA = input("Update ICA calculations ? (1-True, 0-False) ")

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
    my_annot = mne.read_annotations(path + fn_csv)
    print(f'annotations:\n{my_annot}')
    ## adding annotations to raw data
    raw_data.set_annotations(my_annot)
    ############################
    ## scale selection for visualization raw data with annotations
    scale_dict = dict(mag=1e-12, grad=4e-11, eeg=200e-6, eog=150e-6, ecg=300e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)
    ## signals visualization (channels' voltage vs time)
    # mne.viz.plot_raw(raw_data, start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title=fig_title, block=False)
    ###########################################
    ## topographical view channels' voltage
    # adjust the main plot to make room for the sliders
    # fig_topoplot, ax_topoplot = plt.subplots(1, 1, sharex=True, sharey=True)
    # fig_topoplot.subplots_adjust(bottom=0.25)
    # fig_topoplot.suptitle(fig_title)
    # ## topographical map; we apply band pass filter (0.3 - 45 Hz) only for visualization 
    # plot_topographic_view(raw_data)
    ############################################
    if int(update_ICA)==1:
        ## cropping data according to annotations
        ## prefix:
        ## a:resting; b:biking
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
            if label == 'a_closed_eyes':
                a_closed_eyes_list.append(crop_fun(raw_data, t1, t2))

            elif label == 'a_opened_eyes':
                a_opened_eyes_list.append(crop_fun(raw_data, t1, t2))

            elif label == 'b_closed_eyes':
                b_closed_eyes_list.append(crop_fun(raw_data, t1, t2))

            elif label == 'b_opened_eyes':
                b_opened_eyes_list.append(crop_fun(raw_data, t1, t2))

            else:
                pass

        print(f'size list a_closed_eyes: {len(a_closed_eyes_list)}')
        print(f'size list a_opened_eyes: {len(a_opened_eyes_list)}')
        print(f'size list b_closed_eyes: {len(b_closed_eyes_list)}')
        print(f'size list b_opened_eyes: {len(b_opened_eyes_list)}')

        ## eeg data to a dictionary
        eeg_data_dict={}
        eeg_data_dict['a_closed_eyes'] = a_closed_eyes_list
        eeg_data_dict['a_opened_eyes'] = a_opened_eyes_list
        eeg_data_dict['b_closed_eyes'] = b_closed_eyes_list
        eeg_data_dict['b_opened_eyes'] = b_opened_eyes_list

        #########################
        ## interpolation of bad channels and contatenation of segments with same label
        # eeg_data_dict = interpolation_and_concatenation(eeg_data_dict, subject, session)
        eeg_data_dict = channels_interpolation(eeg_data_dict, subject, session)
        
        #########################
        ## ICA for blink removal
        eeg_data_dict = ica_func(eeg_data_dict, subject, session)

        ##########################
        ## segments concatenation
        eeg_data_dict = concat_fun(eeg_data_dict)

        ##########################
        ## current source density
        ## Surface Laplacian 
        eeg_data_dict = csd_fun(eeg_data_dict)

        ##########################
        ## save results
        # writing dictionary to a binary file
        with open('../data/results_ICA/pat_'+str(subject)+'_session_'+str(session)+'.pkl', 'wb') as file:
            pickle.dump(eeg_data_dict, file)
    else:
        # Reading dictionary from the binary file
        with open('../data/results_ICA/pat_'+str(subject)+'_session_'+str(session)+'.pkl', 'rb') as file:
            eeg_data_dict = pickle.load(file)


    ##########################

    labels_list = ['a_closed_eyes','a_opened_eyes','b_closed_eyes','b_opened_eyes',]
    label_title = ['resting closed eyes','resting opened eyes','ABT closed eyes','ABT opened eyes']

    ########################
    new_eeg_data_dict = {}
    annotations_dict={}

    update_annotations = input('Generate annotations ? (1-True, 0-False): ')
    if int(update_annotations)==1:
        
        ## remove data segments in the selected data using bad annotations
        for ax_number, section in enumerate(labels_list):
            print(f'{section, ax_number} generating interactive annotations')
            ## signals visualization
            if eeg_data_dict[section] != None:
                ## channels' voltage vs time
                ## signals visualization to annotate bad segments (interactive annotation)
                mne.viz.plot_raw(eeg_data_dict[section], start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title=label_title[ax_number], block=True)

                ## annotation time is referenced to the time of first_samp, and that is different for each section
                time_offset = eeg_data_dict[section].first_samp / sampling_rate  ## in seconds
                ## get and rewrite annotations minus time-offset
                interactive_annot = eeg_data_dict[section].annotations
                annotations_dict[section] = ann_remove_offset(interactive_annot, time_offset)

                ## crop segments labeled bad
                new_eeg_data_dict[section] = remove_eeg_bad(eeg_data_dict, annotations_dict, section)

                mne.viz.plot_raw(eeg_data_dict[section], start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title=label_title[ax_number], block=False)

                mne.viz.plot_raw(new_eeg_data_dict[section], start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title=label_title[ax_number], block=True)

            else:
                pass

        save_ann = input('save annotations? (1-True, 0-False): ')
        if int(save_ann)==1:
        # writing dictionary to a binary file
            with open('../data/results_ICA/ann_pat_'+str(subject)+'_session_'+str(session)+'.pkl', 'wb') as file:
                pickle.dump(annotations_dict, file)
        else:
            pass
    
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

            if len(annotations_dict[section])>0:
                # for ann in annotations_dict[section]:
                    # print(f'annotations before: {ann}')
                eeg_data_dict[section].set_annotations(annotations_dict[section])
                # for ann in eeg_data_dict[section].annotations:
                    # print(f'annotations after: {ann}')
            else:
                # print(f'annotations not found')
                pass

            ## crop segments labeled bad
            new_eeg_data_dict[section] = remove_eeg_bad(eeg_data_dict, annotations_dict, section)   

            # print(f'{section, ax_number} annotations:')
            ## signals visualization
            # if eeg_data_dict[section] != None:
                # mne.viz.plot_raw(eeg_data_dict[section], start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title=label_title[ax_number], block=False)

                # mne.viz.plot_raw(new_eeg_data_dict[section], start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title=label_title[ax_number], block=True)

    # power spectrum density visualization
    fig_title = "power spectrum density"
    fig_psd = plt.figure(fig_title, figsize=(12, 5))
    ax_psd = [[]]*4

    for ax_number, section in enumerate(labels_list):
        ## signals visualization
        if eeg_data_dict[section] != None:
            ## channels' voltage vs time
            # mne.viz.plot_raw(new_eeg_data_dict[section], start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title=label_title[ax_number], block=False)
            ## channels' spectrum of frequencies
            ax_psd[ax_number] = fig_psd.add_subplot(2,2,ax_number+1)
            mne.viz.plot_raw_psd(new_eeg_data_dict[section], exclude=[], ax=ax_psd[ax_number], fmax=100)
            ax_psd[ax_number].set_title(label_title[ax_number])
            ax_psd[ax_number].set_ylim([-20, 50])

            # print(f'compute psd topomap for each frequency band...')
            # spectrum = new_eeg_data_dict[section].compute_psd().plot_topomap(contours=0,)
            # print(f'done. Type: {type(spectrum)}')

        else:
            pass
    
        

    # fig_psd.tight_layout()


    plt.show(block=True)
    return 0



    '''
    ax_number = 0
    ##########################
    if len(a_closed_eyes_list) > 0 and len(a_opened_eyes_list) > 0:
        # pre-processing selected segment: resting, closed- and opened-eyes

        section = 'a_closed_eyes'
        # sequence = selected_sequences_dict[subject][section]
        # eeg_data_dict = set_bad_channels(eeg_data_dict, subject, section, sequence)

        ## signals visualization
        ## band pass filter (0.3 - 45 Hz) only for visualization
        mne.viz.plot_raw(eeg_data_dict[section], start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title=section, block=False)

        ## spectrum closed eyes resting
        ax_psd[ax_number] = fig_psd.add_subplot(2,2,ax_number+1)

        mne.viz.plot_raw_psd(eeg_data_dict[section], exclude=['VREF'], ax=ax_psd[ax_number], fmax=180)
        ax_psd[ax_number].set_title('rest closed eyes')
        ax_number+=1

        section = 'a_opened_eyes'
        # sequence = selected_sequences_dict[subject][section]
        # eeg_data_dict = set_bad_channels(eeg_data_dict, subject, section, sequence)        

        ## signals visualization
        ## band pass filter (0.3 - 45 Hz) only for visualization
        mne.viz.plot_raw(eeg_data_dict[section], start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title=section, block=False)

        ax_psd[ax_number] = fig_psd.add_subplot(2,2,ax_number+1)
        ## spectrum opened eyes resting
        mne.viz.plot_raw_psd(eeg_data_dict[section], exclude=['VREF'], ax=ax_psd[ax_number], fmax=180)
        ax_psd[ax_number].set_title('rest opened eyes')
        ax_number+=1

    else:
        pass

    if len(b_closed_eyes_list) > 0 and len(b_opened_eyes_list) > 0:
        # pre-processing selected segment: biking, closed- and opened-eyes

        section = 'b_closed_eyes'
        # sequence = selected_sequences_dict[subject][section]
        # eeg_data_dict = set_bad_channels(eeg_data_dict, subject, section, sequence)

        ## signals visualization
        ## band pass filter (0.3 - 45 Hz) only for visualization
        mne.viz.plot_raw(eeg_data_dict[section], start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title=section, block=False)

        ax_psd[ax_number] = fig_psd.add_subplot(2,2,ax_number+1)
        ## spectrum closed eyes ABT
        mne.viz.plot_raw_psd(eeg_data_dict[section], exclude=['VREF'], ax=ax_psd[ax_number], fmax=180)
        ax_psd[ax_number].set_title('ABT closed eyes')
        ax_number+=1
        

        section = 'b_opened_eyes'
        # sequence = selected_sequences_dict[subject][section]
        # eeg_data_dict = set_bad_channels(eeg_data_dict, subject, section, sequence)

        ## signals visualization
        ## band pass filter (0.3 - 45 Hz) only for visualization
        mne.viz.plot_raw(eeg_data_dict[section], start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title=section, block=False)

        ax_psd[ax_number] = fig_psd.add_subplot(2,2,ax_number+1)
        ## spectrum opened eyes ABT 
        mne.viz.plot_raw_psd(eeg_data_dict[section], exclude=['VREF'], ax=ax_psd[ax_number], fmax=180)
        ax_psd[ax_number].set_title('ABT opened eyes')
        

    else:
        pass
    
    ax_psd[0].set_ylim([-30, 45])    

    plt.show(block=True)
    return 0
    '''
    ## visualization topographic views
    # useful for bad-electrodes identification
    # times = np.arange(0, 60, 10)
    # raw_cropped.plot_topomap(times, ch_type='eeg', average=1.0, ncols=3, nrows="auto")
    data_eeg = raw_closed_eyes.get_data(picks=['eeg'])
    df_eeg = raw_closed_eyes.to_data_frame(picks=['eeg'], index='time')
    print(f'shape: raw_closed_eyes:\n{data_eeg.shape}\n{data_eeg}')
    print(f'dataframe: raw_closed_eyes:\n{df_eeg}')
    # mne.viz.plot_topomap()

    # adjust the main plot to make room for the sliders
    fig_topoplot, ax_topoplot = plt.subplots(1, 1, sharex=True, sharey=True)
    fig_topoplot.subplots_adjust(bottom=0.25)

    init_frame = 0
    sampling_rate = raw_closed_eyes.info['sfreq']

    # vlim=(1.0e-14, 5.0e-13)
    ## temporal visualization
    mne.viz.plot_raw(raw_closed_eyes, start=0, duration=80, highpass=0.3, lowpass=30.0, butterfly=False, scalings=scale_dict, block=False)
    
    ## spatial visualization (topographical maps)
    im, cn = mne.viz.plot_topomap(data_eeg[:,init_frame], raw_closed_eyes.info, contours=0, axes=ax_topoplot, cmap='magma')

    # Make a horizontal slider to control the frequency.
    axfreq = fig_topoplot.add_axes([0.25, 0.1, 0.65, 0.03])

    
    cbar_ax = fig_topoplot.add_axes([0.05, 0.25, 0.03, 0.65])
    clb = fig_topoplot.colorbar(im, cax=cbar_ax)
    # clb.ax.set_title("topographic view",fontsize=16) # title on top of colorbar
    # fig_topoplot.add_axes([0.25, 0.1, 0.65, 0.03])
    frame_slider = Slider( ax=axfreq, label='Time [s]', valmin=0, valmax=len(df_eeg)/sampling_rate, valinit=init_frame/sampling_rate, )

    # register the update function with each slider
    frame_slider.on_changed(update)

    ##########################
    plt.show(block=True)
    # plt.show()
    return 0
    ##########################

    ## ICA for artifact removal
    # Filter settings
    ica_low_cut  =  1.0 # For ICA, we filter out more low-frequency power
    ica_high_cut = None
    raw_closed_eyes_ica = raw_closed_eyes.copy().filter(l_freq=ica_low_cut, h_freq=ica_high_cut)

    ##############
    ica = mne.preprocessing.ICA(
    n_components=0.99,
    max_iter="auto",
    random_state=97,
    method="infomax",
    fit_params=dict(extended=True),
    )
    ica.fit(raw_closed_eyes_ica)

    ic_labels = label_components(raw_closed_eyes_ica, ica, method="iclabel")

    # ICA0 was correctly identified as an eye blink, whereas ICA12 was
    # classified as a muscle artifact.
    print(f'ic_labels:\n{ic_labels}')
    print(ic_labels["labels"])

    # for idx, label in enumerate(ic_labels["labels"]):
    #     print(f'{idx}: {label}')
    
    # exclude_idx = [idx for idx, label in enumerate(ic_labels["labels"]) if label not in ["brain", "other"]]
    # print(f"Excluding these ICA components: {exclude_idx}")

    # signals' components
    ica.plot_sources(raw_closed_eyes, show_scrollbars=False, show=True)
    # topographic maps
    ica.plot_components(inst=raw_closed_eyes, contours=0,colorbar=True)

    ## ica components related to noise and artifacts in the EEG signals
    # ica.exclude = [2, 3, 4, 6, 7, 14, 38, 47,]  # indices chosen based on various plots above

    # ica.plot_overlay(inst=raw_closed_eyes, picks="eeg")

    # ## apply ica to the raw data
    # # ica.apply() changes the Raw object in-place, so let's make a copy first:
    # reconst_raw = raw_closed_eyes.copy()
    # ica.apply(reconst_raw)

    # raw_closed_eyes.plot(show_scrollbars=False)
    # reconst_raw.plot(show_scrollbars=False)
    # del reconst_raw

    plt.show(block=True)
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

# def update(frame):
#     global spectrum, data_spectrum, ax, fig

#     im, cn = mne.viz.plot_topomap(data_spectrum[:,frame], spectrum.info, contours=0, vlim=(1.0e-14, 5.0e-13), cmap='magma', axes=ax, show=False)

#     # manually fiddle the position of colorbar
#     ax_x_start = 0.95
#     ax_x_width = 0.04
#     ax_y_start = 0.1
#     ax_y_height = 0.9
#     cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
#     clb = fig.colorbar(im, cax=cbar_ax)
#     clb.ax.set_title("topographic view",fontsize=16) # title on top of colorbar

#     print(f"updated freq: {spectrum.freqs[frame]}")
#     return (0) 


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
