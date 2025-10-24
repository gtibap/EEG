#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import mne
from mne.preprocessing import EOGRegression, ICA, corrmap, create_ecg_epochs, create_eog_epochs

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
from blinks_components import blinks_components_dict


sampling_rate = 1.0
y_limits = [-8,8]

## scale selection for visualization raw data with annotations
scale_dict = dict(mag=1e-12, grad=4e-11, eeg=100e-6, eog=150e-6, ecg=400e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)

#############################
#############################
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
#############################
#############################

#############################
#############################
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
#############################
#############################

##########################
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
##########################
##########################

#############################   
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

        # print(f'idx min and max and values: {id_min, id_max, arr_freqs[id_min], arr_freqs[id_max],}')

        arr_mean = np.mean(arr_psd[:,id_min:id_max], axis=1)
        # print(f'arr mean shape: {arr_mean.shape}')

        im, cn = mne.viz.plot_topomap(arr_mean, raw_data.info, vlim=y_limits, contours=0, axes=ax, )
        ax.title.set_text(title)
    
    # Make colorbar
    cbar_ax = fig_bands.add_axes([0.02, 0.25, 0.03, 0.60])
    fig_bands.colorbar(im, cax=cbar_ax, label='dB change from baseline')

    fig_bands.suptitle(label, size='large', weight='bold')

    fig_bands.savefig(filename, transparent=True)

    return 0
#############################   
#############################

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
#############################

#############################
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
    
    ## annotate bad segments to exclude them of posterior calculations
    for ax_number, section in enumerate(labels_list):
        eeg_list= []
        for eeg_segment, ann in zip(eeg_data_dict[section], annotations_dict[section]):
            ## channels' voltage vs time
            eeg_segment.set_annotations(ann)
            eeg_list.append(eeg_segment)

        eeg_data_dict[section] = eeg_list

    return eeg_data_dict
#############################
#############################

#############################
#############################
def baseline_spectrum(eeg_data_dict, flag):
    ## baseline spectrum of frequencies for normalization
    list_bl = []
    for raw_baseline in eeg_data_dict['baseline']: 
        ## median values of psd closed and opened eyes

        ## power spectral density (psd) from first resting closed eyes
        if flag=='csd':
            psd_bl = raw_baseline.compute_psd(fmin=0,fmax=45,reject_by_annotation=True)
        else:
            psd_bl = raw_baseline.compute_psd(fmin=0,fmax=45,reject_by_annotation=True, exclude=['VREF'])

        data_bl, freq_bl = psd_bl.get_data(return_freqs=True)
        print(f'arr_bl : {data_bl.shape}')

        list_bl.append(data_bl)

    arr_bl=np.array(list_bl)
    median_bl = np.median(arr_bl, axis=0)

    return median_bl
#############################
#############################

#############################
#############################
def topomaps_normalization(raw_data, eeg_data_dict, median_bl, session, path,flag):
    ## create folder if it does not exit already to save topomap images
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
                if flag=='csd':
                    psd_raw = raw.compute_psd(fmin=0,fmax=45,reject_by_annotation=True)
                else:
                    psd_raw = raw.compute_psd(fmin=0,fmax=45,reject_by_annotation=True, exclude=['VREF'])

                data_psd, freq_psd = psd_raw.get_data(return_freqs=True)

                print(f'data_psd : {data_psd.shape}')

                list_psd.append(data_psd)

            arr_psd=np.array(list_psd)
            print(f'arr_psd shape: {arr_psd.shape}')

            median_psd = np.median(arr_psd, axis=0)

            print(f'median_psd : {median_psd.shape}')

            ##########
            ## normalization for each channel using baseline
            print(f'median_psd / median_bl: {median_psd.shape, median_bl.shape}')
            norm_psd = median_psd / median_bl

            norm_psd = 10*np.log10(norm_psd)

            filename_fig = f"{path}session_{session}/figures/topomaps/{label}_{flag}.png"
            plot_topomap_bands(raw_data, norm_psd, freq_psd, title, filename_fig)

        else:
            pass

    return 0
#############################
#############################

#############################
## EEG filtering and signals pre-processing

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
    path, fn_in, fn_csv, raw_data, fig_title, rows_plot, acquisition_system, info_p, Dx = participants_list(path, subject, session, abt)
    if fn_csv == '':
        print(f'It could not find the selected subject. Please check the path, and the selected subject number in the list of participants.')
        return 0
    else:
        pass

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

    ## create folder to save preprocesing parameters
    Path(path+'session_'+str(session)+"/prep").mkdir(parents=True, exist_ok=True)
    filename_bad_ch = path+'session_'+str(session)+'/prep/'+'bad_ch_baseline.csv'
    
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
    ##
    ## time-series data visualization
    mne.viz.plot_raw(raw_data, picks=['eeg','ecg'], start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title=f'EEG time-series ({acquisition_system})', block=True)

    ###########################################
    ## cropping data according to every section (annotations), namely 'baseline','a_closed_eyes','a_opened_eyes','b_closed_eyes', and 'b_opened_eyes'
    eeg_data_dict = get_eeg_segments(raw_data,)

    ###########################################
    ## baseline selection
    ##
    ## if baseline is not present, we choose the first a_opened_eyes (resting) as baseline
    ##
    if len(eeg_data_dict['baseline']) > 0:
        ## baseline was collected during open eyes and fixation at the begining of the session
        raw_seg = eeg_data_dict['baseline'][0]
    elif len(eeg_data_dict['a_opened_eyes']) > 0:
        ## baseline was not collected at the begining -->> we take as baseline the first period of open-eyes during resting 
        raw_seg = eeg_data_dict['a_opened_eyes'][0]
    else:
        print('Baseline was not found.')
        return 0
    ##
    ## baseline selection
    

    flag_prep = int(input(f'load preprocessed values (1-True, 0-False) ?: '))
    if flag_prep:
        ## load bad channel list
        load_df = pd.read_csv(filename_bad_ch)
        bad_ch_list = load_df['bad_ch'].to_list()
        raw_seg.info['bads'] = bad_ch_list

        mne.viz.plot_raw(raw_seg, picks=['eeg','ecg'], start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title='Time-series signals EEG (baseline)\nPlease select bad segments and bad channels interactively', block=True)

    else:
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
            mne.viz.plot_raw(raw_seg, picks=['eeg','ecg'], start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title='Time-series signals EEG (baseline)\nPlease select bad segments and bad channels interactively', block=True)
            ##
            ## power spectral density (psd)
            ##
            ## interactively, identify channels that are out of the tendency (ouliers) as bad channels
            ## 
            fig_psd, ax_psd = plt.subplots(nrows=1, ncols=1, figsize=(9,4), sharey=True, sharex=True)
            mne.viz.plot_raw_psd(raw_seg, picks=['eeg'], exclude=['VREF'], ax=ax_psd, fmin=0.9, fmax=101, xscale='log',)

            ax_psd.set_title('EEG power spectral density (baseline)')

            flag_bad_ch = int(input('Include more bad channels? (1 (True), 0 (False)): '))
            print(f"flag bad_ch: {flag_bad_ch}")
        ##
        bad_ch_list = raw_seg.info['bads']
        print(f"excluded bad channels: {bad_ch_list}")

        data_dict = {}
        data_dict['bad_ch'] = bad_ch_list
        df = pd.DataFrame(data_dict)
        print(f"dataframe:\n{df}")

        df.to_csv(filename_bad_ch)

    # load_df = pd.read_csv(filename_bad_ch)
    # print(f"bad_ch_list_load: {load_df['bad_ch'].to_list()}")
    print(f"bad_ch_list: {bad_ch_list}")

    return 0

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
    flag_ica = int(input('Load the pre-calculated ICA (1-True, 0-False) ?: '))
    ##
    if flag_ica:
        filename_ica = path+'session_'+str(session)+'/ica/'+'ica-baseline.fif.gz'
        ## print(f'filename: {filename_ica}')
        try:
            ica = mne.preprocessing.read_ica(filename_ica, verbose=None)
        except:
            print(f'Pre-calculated ICA was not found.')
            flag_ica = False
    else:
        pass

    if not(flag_ica):    
        ##
        ica = ICA(n_components= 0.99, method='picard', max_iter="auto", random_state=97)
        ##
        ## ica works better with a signal with offset 0; a high pass filter with a 1 Hz cutoff frequency could improve that condition
        filt_raw = raw_seg.copy().filter(l_freq=1.0, h_freq=None)
        ##
        ## ICA fitting model to the filtered raw data
        ica.fit(filt_raw, reject_by_annotation=True)
        ##
        print(f"saving ica model in {path+'session_'+str(session)+'/ica/'+'ica-baseline.fif.gz'}")
        ica.save(path+'session_'+str(session)+'/ica/'+'ica-baseline.fif.gz', overwrite=True)
        ##
    else:
        pass
        ## interactive selection of ICA components to exclude
        ## ica components visualization
    ica.plot_components(inst=raw_seg, contours=0,)
    ica.plot_sources(raw_seg, start=0, stop=240, show_scrollbars=False, block=True)
        ##
        # ica_excl_list = ica.exclude
        # print(f"ica exclude components: {ica_excl_list}")
        ##

    ## manual selection de components to exclude (not necessary if they were choosen interactively)
    # ica.exclude = [0,4] # indices that were chosen# based on previous ICA calculations
    # print(f'ICA blink component: {ica.exclude}')
    raw_seg_ica = raw_seg.copy()
    ica.apply(raw_seg_ica)
    ##
    ## data visualization
    mne.viz.plot_raw(raw_seg, start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, filtorder=4, title=f'baseline before ICA', block=False)
    mne.viz.plot_raw(raw_seg_ica, start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, filtorder=4, title=f'baseline after ICA', block=True)

    return 0
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
    stop =  1.65 # 10^stop
    num  = 50 # samples
    freqs = np.logspace(start, stop, num=num,)
    # print(f'log freqs: {freqs}')

    tfr_bl = raw_seg_ica.compute_tfr('morlet',freqs, picks=['eeg'])
    print(f"baseline type (tfr_power): {type(tfr_bl)}")
    # print(tfr_bl)

    data_bl, times_bl, freqs_bl = tfr_bl.get_data(picks=['eeg'],return_times=True, return_freqs=True)
    # picks=['AFz','Cz','POz']
    print(f"data:\n{data_bl.shape}")
    # print(f"data:\n{data_bl}")
    #
    #  print(f"times:\n{times}")
    print(f"freqs: {len(freqs_bl)}\n{freqs_bl}")

    ## data visualization
    # tfr_bl.plot(picks=['VREF'], title='auto', yscale='linear', show=False)
    # tfr_bl.plot(picks=['VREF'], title='auto', yscale='log', show=False)
    
    ## mean along time samples
    ## for each channel, an average for each frequency
    mean_bl = np.mean(data_bl, axis=2)
    print(f"mean data: {mean_bl.shape}")

    ###############################
    ## data normalization
    dim_ch, dim_fr, dim_t = data_bl.shape
    print(f"bl dim_ch, dim_fr, dim_t: {dim_ch, dim_fr, dim_t}")

    id_ch=0
    for mean_ch, arr_num in zip(mean_bl, data_bl):
        # mean for each frequency per channel
        ## mean_ch is an array with a number of elements equal to the number of evaluated frequencies
        ## each element of the array represents the mean value of time samples per each frequency
        arr_den = np.repeat(mean_ch, dim_t ,axis=0).reshape((len(mean_ch),-1))
        arr_dB = 10*np.log10(arr_num / arr_den)
        # print(f"mean_ch ch arr_res:{mean_ch.shape} {id_ch}, {arr_dB.shape}")
        data_bl[id_ch] = arr_dB
        id_ch+=1

    ## baseline scaling
    ## dB = 10*log10( matrix_time_freq / mean_for_each_freq_baseline )
    # data_2 = data*20
    tfr_bl_norm = tfr_bl.copy()
    tfr_bl_norm._data = data_bl

    ## data visualization
    vlim = (-10,10)    
    tfr_bl.plot(picks=['VREF'], title='auto', yscale='auto', show=False)
    
    tfr_bl_norm.copy().plot(picks=['VREF'], title='Power (VREF) normalized', yscale='auto', vlim=vlim, show=False)


    tfr_bl.plot_topo(picks='eeg', title='topo map', yscale='auto', show=True,)
    tfr_bl_norm.copy().plot_topo(picks='eeg', title='topo map', yscale='auto', show=True, vmin=vlim[0],vmax=vlim[1])

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
