#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import mne
from mne.preprocessing import EOGRegression, ICA, corrmap, create_ecg_epochs, create_eog_epochs

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import make_splrep


class TF_components:

    def __init__(self, path, session, raw_seg, filt_seg, label_seg, id_seg):
        ## EEG data per segment
        self.raw_seg = raw_seg
        self.filt_seg = filt_seg
        self.id_seg = id_seg
        self.label_seg = label_seg

        if label_seg == 'a_closed_eyes':
            self.label = 'a_ce'
            self.title_fig = 'resting (before cycling) closed-eyes'
        elif label_seg == 'a_opened_eyes':
            self.label = 'a_oe'
            self.title_fig = 'resting (before cycling) open-eyes'
        elif label_seg == 'b_closed_eyes':
            self.label = 'b_ce'
            self.title_fig = 'ABT (cycling) closed-eyes'
        elif label_seg == 'b_opened_eyes':
            self.label = 'b_oe'
            self.title_fig = 'ABT (cycling) open-eyes'
        elif label_seg == 'c_closed_eyes':
            self.label = 'c_ce'
            self.title_fig = 'resting (after cycling) closed-eyes'
        elif label_seg == 'c_opened_eyes':
            self.label = 'c_oe'
            self.title_fig = 'resting (after cycling) open-eyes'
        else:
            self.label = ''

        # for ica calculations
        self.raw_seg_ica = []
        # for current source density calculations
        self.raw_seg_csd = []
        # for time-frequency analysis
        self.activity_alpha_band = []
        self.activity_beta_band = []
        self.activity_theta_band = []
        self.baseline_tf = []
        self.data_tf = []
        self.df_bands = []
        self.freqs_tf = []
        self.mask_data_tf = []
        self.tfr_seg = []
        self.tfr_seg_norm = []
        self.times_tf = []

        # pandas data frames
        self.df_ch_bands = pd.DataFrame()

        ## create folder to save preprocesing parameters
        Path(path+'session_'+str(session)+"/prep/"+self.label_seg).mkdir(parents=True, exist_ok=True)
        self.root_name = path+'session_'+str(session)+'/prep/'+self.label_seg+'/'
        self.filename_annot  = f"{self.root_name}{self.label}_{id_seg}_bad_annot.fif"
        self.filename_bad_ch = f"{self.root_name}{self.label}_{id_seg}_bad_channels.csv"
        self.filename_ex_ica = f"{self.root_name}{self.label}_{id_seg}_ica_excluded.csv"
        self.filename_ica    = f"{self.root_name}{self.label}_{id_seg}_ica_model.fif.gz"

        ## create folder to save figures
        Path(f"{path}session_{session}/prep/{self.label_seg}/figures/").mkdir(parents=True, exist_ok=True)
        root_figs = f"{path}session_{session}/prep/{self.label_seg}/figures/"
        self.psd_filename    = f"{root_figs}{self.label}_{id_seg}_psd.png"
        self.psd_ica_filename= f"{root_figs}{self.label}_{id_seg}_psd_ica.png"
        self.psd_chx_filename= f"{root_figs}{self.label}_{id_seg}_psd_chx.png"
        self.raw_filename    = f"{root_figs}{self.label}_{id_seg}_raw.png"
        self.ica_s_filename  = f"{root_figs}{self.label}_{id_seg}_ica_sources.png"
        self.ica_c_filename  = f"{root_figs}{self.label}_{id_seg}_ica_components"
        self.csd_filename    = f"{root_figs}{self.label}_{id_seg}_csd.png"
        self.tf_ch_filename  = f"{root_figs}{self.label}_{id_seg}_tf"
        self.tf_cu_filename  = f"{root_figs}{self.label}_{id_seg}_tf_curves"
        self.tf_bp_filename  = f"{root_figs}{self.label}_{id_seg}_tf_boxplot"
        
        ## ica parameters to calculate ICA components
        self.ica = ICA(n_components= 0.99, method='picard', max_iter="auto", random_state=97)

        ## bad channels and bad annotations lists
        self.bad_annot_list = []
        self.bad_ch_list = []
        self.ica_exclude = []
        self.time_bad_segs = []

        self.sampling_rate = raw_seg.info['sfreq']
        ## scale selection for visualization raw data with annotations
        self.scale_dict = dict(mag=1e-12, grad=4e-11, eeg=100e-6, eog=150e-6, ecg=400e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)
        ## flag to identify selected segment
        self.flag_selection = False

    ###################################
    def load_channels_annot_bads(self):
        ## read previous bad channels and segments
        try:
            print(f"Loading bad channels... ",end='')
            ## load bad channel list
            load_df = pd.read_csv(self.filename_bad_ch)
            self.bad_ch_list = load_df['bad_ch'].to_list()
            self.raw_seg.info['bads'] = self.bad_ch_list
            print(f"done.")
            ##
        except:
            print(f"Bad channels file was not found.")
        ## read bad annotations list
        try:
            print(f"Loading bad segments... ",end='')
            ## read annotation bad_seg
            my_annot = mne.read_annotations(self.filename_annot)
            # print(f'annotations:\n{my_annot}')
            ## adding annotations to raw data
            self.raw_seg.set_annotations(my_annot)
            print(f"done.")
        except:
            print(f"Bad segments file was not found.")

        return 0
    
    def data_visualization(self, ax_plot):
        # display time-series signals
        fig_raw = mne.viz.plot_raw(self.raw_seg, picks=['eeg','ecg'], start=0, duration=240, n_channels=36, scalings=self.scale_dict, highpass=1.0, lowpass=45.0, title=f"{self.label_seg}_{self.id_seg} (EEG) -- Please select bad segments and bad channels interactively", block=False)
        ## display PSD from time-series signals
        mne.viz.plot_raw_psd(self.raw_seg, picks=['eeg'], exclude=['VREF'], ax=ax_plot, fmin=0.9, fmax=101, xscale='log',)
        return ax_plot, self.id_seg

    ##################################
    def selection_bads(self, flag_update):
        ## load bad channels and bad segments
        ## to the raw data (self.raw_seg)
        self.load_channels_annot_bads()
        #########################################################
        ## iterative bad segments and bad channels identification
        ##
        flag_bad_ch = True
        while (flag_bad_ch) and (flag_update) :
            ##
            ## raw data visualization (baseline)
            ## visual observation of time-series and psd from raw data helps to identify bad channels
            ## using butterfly view (choosing 'b' in the time-series plot) helps to identify bad segments
            ##
            ## interactively, include annotations of bad segments (with the label 'bad_seg'), namely, sections of the data where the majority of channels are affected by noise or artefacts
            ##
            ## interactively, select bad channels: flat lines or noisy channels
            ##
            fig_raw = mne.viz.plot_raw(self.raw_seg, picks=['eeg','ecg'], start=0, duration=240, n_channels=36, scalings=self.scale_dict, highpass=1.0, lowpass=45.0, title=f"{self.label_seg}_{self.id_seg} (EEG) -- Please select bad segments and bad channels interactively", block=False)
            ##
            ## power spectral density (psd)
            ##
            ## interactively, identify channels that are out of the tendency (ouliers) as bad channels
            ## 
            ## we exclude VREF because it is the reference (0 volts all the time)
            fig_psd, ax_psd = plt.subplots(nrows=1, ncols=1, figsize=(9,4), sharey=True, sharex=True)
            mne.viz.plot_raw_psd(self.raw_seg, picks=['eeg'], exclude=['VREF'], ax=ax_psd, fmin=0.9, fmax=101, xscale='log',)
            ##
            ax_psd.set_title(f"PSD (EEG) -- {self.label_seg}_{self.id_seg}")
            # Save figures
            fig_psd.savefig(self.psd_filename)
            fig_raw.grab().save(self.raw_filename)
            ##
            flag_bad_ch = input('Include more bad channels? (1 (True), 0 (False)): ')
            flag_bad_ch = 0 if (flag_bad_ch == '') else int(flag_bad_ch)
            # print(f"flag bad_ch: {flag_bad_ch}")

        ## bad channels and bad annotations to filtered version of raw data        
        self.update_channels_annot_bads()
        ## save on disk bad channels and bad annotations
        self.save_channels_annot_bads()
        
        return 0

    #######################
    def update_channels_annot_bads(self):
        ## bad channels
        self.bad_ch_list = self.raw_seg.info['bads']
        print(f"bad channels: {self.bad_ch_list}")
        ## bad annotations
        interactive_annot = self.raw_seg.annotations
        time_offset = self.raw_seg.first_samp / self.sampling_rate  ## in seconds
        self.bad_annot_list = self.ann_remove_offset(interactive_annot, time_offset)

        ## setting same bad channels and bad segments for the filtered data
        self.filt_seg.set_annotations(self.bad_annot_list)
        self.filt_seg.info['bads'] = self.bad_ch_list

        return 0
    
    ###########################################
    def save_channels_annot_bads(self):
        ## save selected bad channels and segments
        try:
            ## save bad channels list to csv through a pandas dataframe
            data_dict = {}
            data_dict['bad_ch'] = self.bad_ch_list
            df = pd.DataFrame(data_dict)
            # print(f"dataframe:\n{df}")
            df.to_csv(self.filename_bad_ch)
        except:
            print(f"Error: something went wrong saving bad channels.")
        ## save selected bad annotations
        try:
            ## save annotations to .fif
            self.bad_annot_list.save(self.filename_annot, overwrite=True)
        except:
            print(f"Error: something went wrong saving bad annotations.")
        
        return 0
    
    #######################
    def ann_remove_offset(self, interactive_annot, time_offset):
        arr_onset=np.array([])
        arr_durat=np.array([])
        arr_label=[]

        for ann in interactive_annot:
            arr_onset = np.append(arr_onset, ann['onset']-time_offset)
            arr_durat = np.append(arr_durat, ann['duration'])
            arr_label.append(ann['description'])

        my_annot = mne.Annotations(
        onset=arr_onset,  # in seconds
        duration=arr_durat,  # in seconds, too
        description=arr_label,
        )
        return my_annot
    
    ##################################
    def re_referencing(self):
        ## this function excludes bad channels
        self.raw_seg.set_eeg_reference(ref_channels="average", ch_type='eeg')
        self.filt_seg.set_eeg_reference(ref_channels="average", ch_type='eeg')
        return 0
    
    ##################################
    def read_ica_model(self):
        try:
            print(f"loading pre-calculated ICA model... ", end='')
            self.ica = mne.preprocessing.read_ica(self.filename_ica, verbose=None)
            print(f"done.")
            read_ica_flag = True
        except:
            print(f'Pre-calculated ICA was not found.')
            read_ica_flag = False

        return read_ica_flag
    
    ##################################
    def read_ica_excluded_comp(self):
        try:
            print(f"loading pre-selected excluded ICA components... ", end='')
            df_ex_ica = pd.read_csv(self.filename_ex_ica)
            self.ica.exclude = df_ex_ica['ex_ica'].to_list()
            print(f"done.")
            read_ica_flag = True
        except:
            print(f'Pre-selected excluded ICA components were not found.')
            read_ica_flag = False

        return read_ica_flag
    
    #####################################
    def save_fig_ica_comp(self, fig_ica_comp):
        try:
            ## save ICA plot components
            if isinstance(fig_ica_comp, list):
                id = 0
                for fig in fig_ica_comp:
                    fig.savefig(f"{self.ica_c_filename}_{id}.png")
                    id+=1
            else:
                fig.savefig(f"{self.ica_c_filename}.png")
        except:
            print(f"Error: something went wrong saving ICA components.")
        return 0
    ######################################
    def ica_components(self, flag_update):
        
        if flag_update:
            ## update list of excluded ICA components or (re-)calculate ICA components
            flag_ica = input('Re-calculate ICA components (1-true, 0-False) ?: ')
            flag_ica = 0 if (flag_ica == '') else int(flag_ica)
            ## re-calculate ICA components    
            if flag_ica==1 :
                ## ica works better with clean (denoised) EEG signals with 0 offset (a high pass filter with a 1 Hz cutoff frequency could improve that condition, that is why we use the filtered version of the data [self.filt_seg])
                ## ICA fitting model to the filtered raw data
                self.ica.fit(self.filt_seg, reject_by_annotation=True)
                self.ica.exclude = []
            else:
                ## read pre-caluculated ICA model
                self.read_ica_model()
            ##
            ## eeg signals visualization (raw data with bad annot and bad channels)
            fig_raw = mne.viz.plot_raw(self.raw_seg, picks=['eeg','ecg'], start=0, duration=240, scalings=self.scale_dict, highpass=1.0, lowpass=45.0, title=f"{self.label_seg}_{self.id_seg} (EEG)", block=False)
            ## plot_components shows 2D-topomaps of the ICA components
            fig_ica_comp = self.ica.plot_components(inst=self.raw_seg.copy(), contours=0, show=True, title=f"{self.label_seg}-{self.id_seg} -- ICA components")
            ## save figure ica plot components
            self.save_fig_ica_comp(fig_ica_comp)
            ## interactive selection of ICA components to exclude
            self.ica.plot_sources(inst=self.raw_seg, start=0, stop=240, show_scrollbars=False, show=True, title=f"{self.label_seg}-{self.id_seg} -- ICA components", block=True)

            print(f"ica excluded components: {self.ica.exclude}")
            # blinks
            # self.ica.plot_overlay(self.raw_seg,)
        else:
            pass
    
        ## optional ############
        ## display EEG before ICA
        # mne.viz.plot_raw(self.raw_seg.copy(), picks=['eeg','ecg'], start=0, duration=240, scalings=self.scale_dict, highpass=1.0, lowpass=45.0, title=f"{self.label_seg}_{self.id_seg} (EEG)\n Before ICA", block=False)
        ## optional ############

        ## selected ica components to exclude
        self.ica_exclude = self.ica.exclude
        print(f"ica excluded components: {self.ica_exclude}")
        ## apply ica
        copy_raw_seg = self.raw_seg.copy()
        ## ica in place
        self.ica.apply(self.raw_seg)

        # optional ############
        # display EEG after ICA
        # print(f"EEG signals display after ICA...")
        fig_psd_ica, ax_psd_ica = plt.subplots(nrows=2, ncols=1, figsize=(9,4), sharey=True, sharex=True)

        mne.viz.plot_raw_psd(copy_raw_seg, picks=['eeg'], fmin=0.9, fmax=101, xscale='log', ax=ax_psd_ica[0], show=False,)
        mne.viz.plot_raw_psd(self.raw_seg, picks=['eeg'], fmin=0.9, fmax=101, xscale='log', ax=ax_psd_ica[1], show=False,)

        ax_psd_ica[0].set_title(f"PSD(EEG) before ICA")
        ax_psd_ica[1].set_title(f"PSD(EEG) after ICA")

        ## display eeg signals before ICA
        mne.viz.plot_raw(copy_raw_seg, picks=['eeg','ecg'], start=0, duration=240, scalings=self.scale_dict, highpass=1.0, lowpass=45.0, title=f"{self.label_seg}_{self.id_seg} (EEG)\n Before ICA", block=False)
        ## display eeg signals after ICA
        mne.viz.plot_raw(self.raw_seg, picks=['eeg','ecg'], start=0, duration=240, scalings=self.scale_dict, highpass=1.0, lowpass=45.0, title=f"{self.label_seg}_{self.id_seg} (EEG)\n After ICA", block=True)
        # optional ############

        ## save ica fitted model and excluded components
        # print(f"saving ica model in {filename_ica}")
        self.save_ica_model()
        ## save plot ica sources
        self.save_plot_ica_sources()
    
        return 0
    ## ica components()
    ######################################
    
    #################################
    def save_ica_model(self):
        try:
            ## save ICA model
            self.ica.save(self.filename_ica, overwrite=True)
        except:
            print(f"Error: something went wrong writing ICA model.")
        return 0
    
     #################################
    def save_ica_excluded_comp(self):
        try:
            ## save ICA excluded components
            df_ex_ica = pd.DataFrame()
            df_ex_ica['ex_ica'] = self.ica.exclude
            df_ex_ica.to_csv(self.filename_ex_ica)
        except:
            print(f"Error: something went wrong writing ICA excluded components.")
        return 0
    
    #################################
    def save_plot_ica_sources(self):
        ## save figure ICA sources
        fig_ica_sources = self.ica.plot_sources(self.raw_seg, start=0, stop=240, show_scrollbars=False, show=False, block=False)
        try:
            ## save ICA plot sources
            fig_ica_sources.grab().save(self.ica_s_filename)
        except:
            print(f"Error: something went wrong saving ICA sources.")

        return 0

    ##############################
    def display_psd_rawdata(self):
        fig_psd, ax_psd = plt.subplots(nrows=1, ncols=1, figsize=(9,4), sharey=True, sharex=True)
        mne.viz.plot_raw_psd(self.raw_seg, picks=['eeg'], fmin=0.9, fmax=101, xscale='log', ax=ax_psd, show=False,)
        ax_psd.set_title(f"PSD(EEG) before ICA")
        return 0
    ##################################################
    def ica_components_interactive(self, flag_update_ica):

        recal_ica_flag = int(input(f"Do you want to recalculate an ICA model (yes 1, no 0) ?: "))
        # if read_ica_flag:
        #     # print(f"An ICA model was found.")
        # else:
        #     keep_ica_flag = False
        #     ## read pre-caluculated ICA model
        #     read_ica_flag = self.read_ica_model()

        
        ## update list of excluded ICA components or (re-)calculate ICA components
        # flag_ica = input('Re-calculate ICA components (1-true, 0-False) ?: ')
        # flag_ica = 0 if (flag_ica == '') else int(flag_ica)
        if flag_update_ica:
            flag_ica = 1
            ## re-calculate ICA components    
            while flag_ica==1 :
                ## display psd eeg raw data
                self.display_psd_rawdata()
                ## copy of raw data
                copy_raw_seg = self.raw_seg.copy()
                
                if recal_ica_flag==False:
                    print(f"reading the previous ICA model...")
                    self.read_ica_model()
                    self.read_ica_excluded_comp()
                else:
                    ## ica works better with clean (denoised) EEG signals with 0 offset (a high pass filter with a 1 Hz cutoff frequency could improve that condition, that is why we use the filtered version of the data [self.filt_seg])
                    ## ICA fitting model to the filtered raw data
                    print(f"creating an ICA model...")
                    self.ica.fit(self.filt_seg, reject_by_annotation=True)
                    self.ica.exclude = []
                    ## save ICA model and excluded components
                    self.save_ica_model()
                ##    
                ## eeg signals visualization (raw data with bad annot and bad channels)
                fig_raw = mne.viz.plot_raw(self.raw_seg.copy(), picks=['eeg','ecg'], start=0, duration=240, scalings=self.scale_dict, highpass=1.0, lowpass=45.0, title=f"{self.label_seg}_{self.id_seg} (EEG)", block=False)
                ## plot_components shows 2D-topomaps of the ICA components
                fig_ica_comp = self.ica.plot_components(inst=self.raw_seg, contours=0, show=True, title=f"{self.label_seg}-{self.id_seg} -- ICA components")
                ## save figure ica plot components
                self.save_fig_ica_comp(fig_ica_comp)
                ###############
                ## interactive selection of ICA components to exclude
                self.ica.plot_sources(inst=self.raw_seg, start=0, stop=240, show_scrollbars=False, show=True, title=f"{self.label_seg}-{self.id_seg} -- ICA components", block=True)
                print(f"ica excluded components: {self.ica.exclude}")
                ## save ICA excluded components that were selected interactively
                self.save_ica_excluded_comp()
                ################
                ## visual comparison before and after ICA
                self.ica.apply(copy_raw_seg)
                self.display_comparison_ica(self.raw_seg.copy(), copy_raw_seg)
                ## update ica calculation flag
                option_ica = int(input(f"0: Save the current model\n1: Re-calculate ICA components\n2: Redefine list of exclusion ICA components\n ?: "))
                # option_ica = 0 if (flag_ica == '') else int(flag_ica)
                if option_ica==1:
                    ## update self.filt_reg bad channels and bad segments
                    ## bad channels and bad annotations to filtered version of raw data        
                    fig_raw = mne.viz.plot_raw(self.raw_seg, picks=['eeg','ecg'], start=0, duration=240, scalings=self.scale_dict, highpass=1.0, lowpass=45.0, title=f"{self.label_seg}_{self.id_seg} (EEG)\nPlease select bad channels and/or bad segments", block=True)
                    ## bad channels and bad annotations are copy to the self.filt_seg
                    self.update_channels_annot_bads()
                    ## save on disk bad channels and bad annotations
                    self.save_channels_annot_bads()
                    ## recalculate ICA components
                    # read_ica_flag=False
                    recal_ica_flag=True
                    ## keep in the loop
                    flag_ica = 1
                elif option_ica==2:
                    ## same ICA model but choosing other components to exclude
                    ## does not recalculate ICA components
                    # read_ica_flag=True
                    recal_ica_flag=False
                    ## keep in the loop
                    flag_ica = 1
                else:
                    ## apply the ICA model to the raw data on place
                    # self.ica.apply(self.raw_seg)
                    # ## save on disk bad channels and bad annotations
                    # self.save_channels_annot_bads()
                    ## break the loop
                    flag_ica = 0

                print(f"continuous loop: {flag_ica}")
        else:
            print(f"reading the previous ICA model...")
            self.read_ica_model()
            self.read_ica_excluded_comp()

        ## optional ############
        ## display EEG before ICA
        # mne.viz.plot_raw(self.raw_seg.copy(), picks=['eeg','ecg'], start=0, duration=240, scalings=self.scale_dict, highpass=1.0, lowpass=45.0, title=f"{self.label_seg}_{self.id_seg} (EEG)\n Before ICA", block=False)
        ## optional ############

        ## selected ica components to exclude
        self.ica_exclude = self.ica.exclude
        print(f"ica excluded components: {self.ica_exclude}")
        ## apply ica
        # copy_raw_seg = self.raw_seg.copy()
        ## ica in place
        self.ica.apply(self.raw_seg)

        # # optional ############
        # # display EEG after ICA
        # # print(f"EEG signals display after ICA...")
        # fig_psd_ica, ax_psd_ica = plt.subplots(nrows=2, ncols=1, figsize=(9,4), sharey=True, sharex=True)

        # mne.viz.plot_raw_psd(copy_raw_seg, picks=['eeg'], fmin=0.9, fmax=101, xscale='log', ax=ax_psd_ica[0], show=False,)
        # mne.viz.plot_raw_psd(self.raw_seg, picks=['eeg'], fmin=0.9, fmax=101, xscale='log', ax=ax_psd_ica[1], show=False,)

        # ax_psd_ica[0].set_title(f"PSD(EEG) before ICA")
        # ax_psd_ica[1].set_title(f"PSD(EEG) after ICA")

        # ## display eeg signals before ICA
        # mne.viz.plot_raw(copy_raw_seg, picks=['eeg','ecg'], start=0, duration=240, scalings=self.scale_dict, highpass=1.0, lowpass=45.0, title=f"{self.label_seg}_{self.id_seg} (EEG)\n Before ICA", block=False)
        # ## display eeg signals after ICA
        # mne.viz.plot_raw(self.raw_seg, picks=['eeg','ecg'], start=0, duration=240, scalings=self.scale_dict, highpass=1.0, lowpass=45.0, title=f"{self.label_seg}_{self.id_seg} (EEG)\n After ICA", block=True)
        # # optional ############

        ## save ica fitted model and excluded components
        # print(f"saving ica model in {filename_ica}")
        # self.save_ica_model()
        ## save plot ica sources
        self.save_plot_ica_sources()
        ## save figure psd before and after ICA
        # fig_psd_ica.savefig(self.psd_ica_filename)
    
        return 0
    ## ica components_interactive()

    ######################################
    ## display psd rawdata before and after ICA
    def display_comparison_ica(self, raw_before_ica, raw_after_ica):
        # optional ############
        # display EEG after ICA
        # print(f"EEG signals display after ICA...")
        fig_psd_ica, ax_psd_ica = plt.subplots(nrows=2, ncols=1, figsize=(9,4), sharey=True, sharex=True)

        mne.viz.plot_raw_psd(raw_before_ica, picks=['eeg'], fmin=0.9, fmax=101, xscale='log', ax=ax_psd_ica[0], show=False,)
        mne.viz.plot_raw_psd(raw_after_ica, picks=['eeg'], fmin=0.9, fmax=101, xscale='log', ax=ax_psd_ica[1], show=False,)

        ax_psd_ica[0].set_title(f"PSD(EEG) before ICA")
        ax_psd_ica[1].set_title(f"PSD(EEG) after ICA")

        ## display eeg signals before ICA
        mne.viz.plot_raw(raw_before_ica, picks=['eeg','ecg'], start=0, duration=240, scalings=self.scale_dict, highpass=1.0, lowpass=45.0, title=f"{self.label_seg}_{self.id_seg} (EEG)\n Before ICA", block=False)
        ## display eeg signals after ICA
        mne.viz.plot_raw(raw_after_ica, picks=['eeg','ecg'], start=0, duration=240, scalings=self.scale_dict, highpass=1.0, lowpass=45.0, title=f"{self.label_seg}_{self.id_seg} (EEG)\n After ICA", block=True)
        # optional ############
        fig_psd_ica.savefig(self.psd_ica_filename)

        return 0
    
    ##################################
    def bads_interpolation(self):
        # print("Bad channels interpolation")
        # self.raw_seg_ica.interpolate_bads()
        self.raw_seg.interpolate_bads()
        return 0
    
    ##################################
    def apply_csd(self):
        # self.raw_seg_csd = mne.preprocessing.compute_current_source_density(self.raw_seg_ica)
        ## in place
        self.raw_seg = mne.preprocessing.compute_current_source_density(self.raw_seg)
        
        ## save figure csd
        fig_csd = mne.viz.plot_raw(self.raw_seg, start=0, duration=240, scalings=self.scale_dict, highpass=None, lowpass=None, filtorder=4, title=f'EEG after Surface Laplacian', show=False, block=False)
        try:
            fig_csd.grab().save(self.csd_filename)
        except:
            print(f"Error: something went wrong saving csd.")

        return 0
    
    def psd_selected_chx(self, channels_list):

        fig_psd, ax_psd = plt.subplots(nrows=1, ncols=1, figsize=(9,4), sharey=True, sharex=True)
        fig_psd = mne.viz.plot_raw_psd(self.raw_seg, picks=channels_list, fmin=0.9, fmax=101, xscale='log', ax=ax_psd, show=False,)
        try:
            fig_psd.grab().save(self.psd_chx_filename)
        except:
            print(f"Error: something went wrong saving csd.")

            ##
            # ax_psd.set_title(f"PSD (EEG) -- {self.label_seg}_{self.id_seg}")
            # Save figures
            # fig_psd.savefig(self.psd_filename)

        return 0
    
    ##################################
    def tf_calculation(self, ch_list):
        ## Time-frequency (tf) decomposition from each EEG channel
        ## logarithmic scale frequencies
        start= 0.60 # 10^start,  
        stop = 1.50 # 10^stop
        num  = 75 # samplesq
        freqs = np.logspace(start, stop, num=num,)
        # print(f'log freqs: {freqs}')

        # tfr_bl = raw_seg_ica.compute_tfr('morlet',freqs, picks=['eeg'])
        # data_bl, times_bl, freqs_bl = tfr_bl.get_data(picks=['eeg'],return_times=True, return_freqs=True)
        # print(f"Time-frequency analysis (Morlet wavelet)")
        # self.tfr_seg = self.raw_seg_csd.compute_tfr('morlet', freqs, reject_by_annotation=False)
        ## in place
        self.tfr_seg = self.raw_seg.compute_tfr('morlet', freqs, picks=ch_list, reject_by_annotation=False)
        ## time-frequency data
        # self.data_tf, self.times_tf, self.freqs_tf = self.tfr_seg.get_data(return_times=True, return_freqs=True)
        # print(f"data tf analysis shape: {self.data_tf.shape}")
        # print(f"times tf analysis shape: {self.times_tf.shape}")
        # print(f"freqs tf analysis shape: {self.freqs_tf.shape}")
        # print(f"times tf analysis: {self.times_tf}")
        # print(f"freqs tf analysis: {self.freqs_tf}")
        ##
        ## adding time and annotations to results of frequency bands
        self.set_annotations_freq_bands()
        
        return 0

    ###############################
    def get_tf_baseline(self):

        data_tf, times_tf, freqs_tf = self.tfr_seg.get_data(return_times=True, return_freqs=True)
        ## create a mask to avoid data of bad segments
        mask_data_tf = np.zeros(data_tf.shape)
        # self.mask_ann = np.zeros(len(self.times_tf))
        ## annotations bad segments
        for ann in self.bad_annot_list:
            if ann['description'].startswith('bad'):
                onset = ann['onset']
                duration = ann['duration']
                print(f"bad segment:\nonset: {onset}\nduration: {duration}")
                ## identify samples inside the bad segment
                idx_times = np.nonzero((times_tf>=onset) & (times_tf<(onset+duration)))
                ## initial (t0) and final (t1) samples of the bad segment
                t0 = idx_times[0][0]
                t1 = idx_times[0][-1]
                # print(f"samples bad segment (t0, t1): ({t0,t1})")
                ## a mask for all channels, all frequencies, and same range in time (between t0 and t1)
                mask_data_tf[:,:,t0:t1]=1
                # self.mask_ann[t0:t1]=1
        ## including a mask to avoid bad segments in the mean calculation
        data_tf_masked = np.ma.array(data_tf, mask=mask_data_tf)
        ## mean values per frequency per channel (ref for baseline normalization)
        self.baseline_tf = data_tf_masked.mean(axis=2)
        # self.baseline_tf = self.baseline_tf.data

        # print(f"ref baseline data shape: {self.baseline_tf.shape}")
        # print(f"ref baseline data:\n{self.baseline_tf}")
        return self.baseline_tf.data, freqs_tf
    
    ###############################
    def tf_normalization(self, tf_ref):

        data_tf, times_tf, freqs_tf = self.tfr_seg.get_data(return_times=True, return_freqs=True)
        ## time-frequency power normalization for each channel
        dim_ch, dim_fr, dim_t = data_tf.shape
        # print(f"bl dim_ch, dim_fr, dim_t: {dim_ch, dim_fr, dim_t}")
        
        ## initialization new array for baseline normalization
        data_tf_norm = np.zeros(data_tf.shape)

        id_ch=0
        # print(f"normalization ch:")
        for mean_ch, arr_num in zip(tf_ref, data_tf):
            # mean for each frequency per channel
            ## mean_ch is an array with a number of elements equal to the number of evaluated frequencies
            ## each element of the array represents the mean value of time samples per each frequency
            arr_den = np.repeat(mean_ch, dim_t, axis=0).reshape(len(mean_ch),-1)
            arr_dB = 10*np.log10(arr_num / arr_den)
            # print(f"mean_ch ch arr_res:{mean_ch.shape} {id_ch}, {arr_dB.shape}")
            # data_bl[id_ch] = arr_dB
            data_tf_norm[id_ch] = arr_dB
            # print(f"{id_ch}", end=", ")
            id_ch+=1
        # print(f"")
        # copy of the tf transformation
        # self.tfr_seg_norm = self.tfr_seg.copy()
        self.tfr_seg._data = data_tf_norm

        return 0
    
    ###############################
    def channel_bands_power(self, ch_label, eeg_system):
        
        ## 128-channel geodesic to 10-10 equivalent names
        ch_name = self.get_ch_equivalent(ch_label, eeg_system)

        ## separate components frequency bands theta, alpha, and beta
        data_ch, times_ch, freqs_ch = self.tfr_seg.get_data(picks=[ch_name],return_times=True, return_freqs=True)

        print(f"Channel {ch_label}/{ch_name} data shape: {data_ch.shape}")
        # print(f"times shape: {times_ch.shape}")
        # print(f"freqs shape: {freqs_ch.shape}")

        # plot matrix selected channel (as an image)
        # fig_tf, ax_tf = plt.subplots(nrows=3, ncols=1, figsize=(16,4), sharey=True, sharex=True)
        # ax_tf[0].imshow(data_ch[0], aspect='auto', cmap='coolwarm', vmin=-12, vmax=12)

        ## normalized tf matrix to dataframe [VREF]
        df_tf = pd.DataFrame(data_ch[0])
        # print(f"df_tf shape: {df_tf.shape}")
        # rows-->freqs (from the lowest to highest freqs), columns-->times
        df_tf['freq'] = freqs_ch
        # print(f"df_tf:\n{df_tf}")
        ##
        ## accumulative power per band per every time sample
        ##
        ## theta (4-8 Hz), alpha (8-12 Hz), beta (12-30 Hz)
        # selecting group of rows based on frequency band values
        df_theta  = df_tf.loc[(df_tf['freq'] >= 4)  & (df_tf['freq'] < 8)]
        df_alpha  = df_tf.loc[(df_tf['freq'] >= 8)  & (df_tf['freq'] < 12)]
        df_beta   = df_tf.loc[(df_tf['freq'] >= 12) & (df_tf['freq'] < 30)]
        df_beta_l = df_tf.loc[(df_tf['freq'] >= 12) & (df_tf['freq'] < 21)]
        df_beta_h = df_tf.loc[(df_tf['freq'] >= 21) & (df_tf['freq'] < 30)]

        # print(f"df_theta  shape: {df_theta.shape}")
        # print(f"df_alpha  shape: {df_alpha.shape}")
        # print(f"df_beta shape:   {df_beta.shape}")

        # print(f"df_theta:\n{df_theta}")
        # print(f"df_theta shape:\n{df_theta.shape}")
        ##
        ## exclude column freq
        df_theta  = df_theta.loc[:,df_theta.columns != 'freq']
        df_alpha  = df_alpha.loc[:,df_alpha.columns != 'freq']
        df_beta   = df_beta.loc[:,df_beta.columns  != 'freq']
        df_beta_l = df_beta_l.loc[:,df_beta_l.columns  != 'freq']
        df_beta_h = df_beta_h.loc[:,df_beta_h.columns  != 'freq']

        ## calculate median value for each time sample for each freq. band
        activity_theta_band  = df_theta.median(axis=0).to_numpy()
        activity_alpha_band  = df_alpha.median(axis=0).to_numpy()
        activity_beta_band   = df_beta.median(axis=0).to_numpy()
        activity_beta_l_band = df_beta_l.median(axis=0).to_numpy()
        activity_beta_h_band = df_beta_h.median(axis=0).to_numpy()

        # print(f"theta median shape: {self.activity_theta_band.shape}")
        # print(f"alpha median shape: {self.activity_alpha_band.shape}")
        # print(f"beta median shape:  {self.activity_beta_band.shape}")

        # ## plot curves frequency bands
        # self.plot_curves_bands(ch_label)
        # self.plot_boxplots_bands(ch_label)

        data_dict = {
            f'{ch_label}_theta': activity_theta_band,
            f'{ch_label}_alpha': activity_alpha_band,
            f'{ch_label}_beta' : activity_beta_band,
            f'{ch_label}_beta_l' : activity_beta_l_band,
            f'{ch_label}_beta_h' : activity_beta_h_band,
        }
        df_bands = pd.DataFrame(data_dict)

        ## concat freq bands activity of selected channels
        self.df_ch_bands = pd.concat([self.df_ch_bands, df_bands], axis=1,)

        return 0
    
    ###############################
    ## plot beta-low and beta-high of selected channels
    def plot_curves_beta_bands(self, ch_list):
        ## from the dataframe self.df_ch_bands make plots beta-low for selected channels
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9,6), sharey=True, sharex=True)

        # time
        time = self.df_ch_bands[f"time"].to_numpy()

        for ch_name in ch_list:
            beta_l = self.df_ch_bands[f"{ch_name}_beta_l"].to_numpy()
            beta_h = self.df_ch_bands[f"{ch_name}_beta_h"].to_numpy()

            ax[0].plot(time, beta_l, label=f'{ch_name}')
            ax[1].plot(time, beta_h, label=f'{ch_name}')
        
        # ax_bands[0].plot(self.times_tf, self.activity_beta_band,  label='beta [12-30 Hz]')
        # ax_bands[1].plot(self.times_tf, self.activity_alpha_band, label='alpha [8-12 Hz]')
        # ax_bands[2].plot(self.times_tf, self.activity_theta_band, label='theta [4-8 Hz]')

        ax[0].set_ylim([-15,15])

        ax[0].legend(loc='upper right')
        ax[1].legend(loc='upper right')
        # ax_bands[2].legend(loc='upper right')

        ax[0].set_xlabel('Time [s]')
        ax[1].set_xlabel('Time [s]')

        ax[0].set_ylabel(f'dB change from baseline')
        
        ax[0].set_title(f'low beta')
        ax[1].set_title(f'high beta')

        fig.suptitle(f'{self.label_seg}-{self.id_seg}')
        # ax[0].set_title(f'Power({ch_label}) -- dB change from baseline [frequency bands]')

        ## plot annotations bad segments
        for time_ann in self.time_bad_segs:
            print(f"time bad annot: {time_ann}")
            ax[0].fill_between(time, 0, 1, where=((time >= time_ann[0])&(time < time_ann[1])), color='tab:red', alpha=0.25, transform=ax[0].get_xaxis_transform())
            ax[1].fill_between(time, 0, 1, where=((time >= time_ann[0])&(time < time_ann[1])), color='tab:red', alpha=0.25, transform=ax[1].get_xaxis_transform())

        # for ann in self.bad_annot_list:
        #     if ann['description'].startswith('bad'):
        #         onset = ann['onset']
        #         duration = ann['duration']
        #         ## bad segment in the plot each band 
        #         for id in np.arange(len(ax_bands)):
        #             ax_bands[id].fill_between(self.times_tf, 0, 1, where=((self.times_tf >= onset)&(self.times_tf < (onset+duration))), color='tab:red', alpha=0.25, transform=ax_bands[id].get_xaxis_transform())
        #         ##

        # fig_bands.savefig(f"{self.tf_cu_filename}_{ch_label}.png")
        
        return 0

    ###############################
    def plot_curves_bands(self, ch_label):
        ##
        fig_bands, ax_bands = plt.subplots(nrows=3, ncols=1, figsize=(9,6), sharey=True, sharex=True)
        ax_bands[0].plot(self.times_tf, self.activity_beta_band,  label='beta [12-30 Hz]')
        ax_bands[1].plot(self.times_tf, self.activity_alpha_band, label='alpha [8-12 Hz]')
        ax_bands[2].plot(self.times_tf, self.activity_theta_band, label='theta [4-8 Hz]')

        ax_bands[0].set_ylim([-15,15])
        ax_bands[0].legend(loc='upper right')
        ax_bands[1].legend(loc='upper right')
        ax_bands[2].legend(loc='upper right')

        ax_bands[-1].set_xlabel('Time [s]')
        ax_bands[0].set_title(f'Power({ch_label}) -- dB change from baseline [frequency bands]')

        ## plot annotations bad segments
        for ann in self.bad_annot_list:
            if ann['description'].startswith('bad'):
                onset = ann['onset']
                duration = ann['duration']
                ## bad segment in the plot each band 
                for id in np.arange(len(ax_bands)):
                    ax_bands[id].fill_between(self.times_tf, 0, 1, where=((self.times_tf >= onset)&(self.times_tf < (onset+duration))), color='tab:red', alpha=0.25, transform=ax_bands[id].get_xaxis_transform())
                ##

        fig_bands.savefig(f"{self.tf_cu_filename}_{ch_label}.png")
        
        return 0

###############################
    def boxplots_beta_bands(self, ch_list):
        print(f"ch list: {ch_list}")
        print(f"{self.label_seg}-{self.id_seg}")
        ## dataframe of frequency bands to create boxplots
        print(f"df_ch_bands.columns: {self.df_ch_bands.columns}")

        df_masked = self.df_ch_bands.loc[self.df_ch_bands['mask']==0]


        # data_dict = {
        #     'theta': self.activity_theta_band,
        #     'alpha': self.activity_alpha_band,
        #     'beta' : self.activity_beta_band,
        # }
        # df_bands = pd.DataFrame(data_dict)
        ## adding mask bad segments
        ############################
        # ## create a mask to avoid data of bad segments
        # mask_ann = np.zeros(len(self.times_tf))
        # ## annotations bad segments
        # for ann in self.bad_annot_list:
        #     if ann['description'].startswith('bad'):
        #         onset = ann['onset']
        #         duration = ann['duration']
        #         print(f"bad segment:\nonset: {onset}\nduration: {duration}")
        #         ## identify samples inside the bad segment
        #         idx_times = np.nonzero((self.times_tf>=onset) & (self.times_tf<(onset+duration)))
        #         ## initial (t0) and final (t1) samples of the bad segment
        #         t0 = idx_times[0][0]
        #         t1 = idx_times[0][-1]
        #         print(f"samples bad segment (t0, t1): ({t0,t1})")
        #         ## a mask for all channels, all frequencies, and same range in time (between t0 and t1)
        #         mask_ann[t0:t1]=1
        # ############################
        # df_bands['mask'] = mask_ann
        ##
        ## boxplots
        # print(f"df_bands:\n{df_bands}")
        # df_masked = df_bands.loc[df_bands['mask']==0]
        # print(f"df_masked:\n{df_masked}")

        fig_box, ax_box = plt.subplots(nrows=1, ncols=1, figsize=(9,6), sharey=True,)
        df_masked.boxplot(['theta','alpha','beta'], showfliers=False, ax=ax_box)

        ax_box.set_title(f"Power ({ch_list}) -- boxplots frequency bands")
        ax_box.set_ylabel(f"dB change from baseline")
        
        # fig_box, ax_box = plt.subplots(nrows=1, ncols=3, figsize=(9,6), sharey=True,)
        # df_bands_all.boxplot(['a_closed_eyes_theta','b_closed_eyes_theta'], showfliers=False, ax=ax_box[0])
        # df_bands_all.boxplot(['a_closed_eyes_alpha','b_closed_eyes_alpha'], showfliers=False, ax=ax_box[1])
        # df_bands_all.boxplot(['a_closed_eyes_beta' ,'b_closed_eyes_beta'], showfliers=False, ax=ax_box[2])

        # ax_box[0].set_xticks([1, 2], ['a', 'b',])
        # ax_box[1].set_xticks([1, 2], ['a', 'b',])
        # ax_box[2].set_xticks([1, 2], ['a', 'b',])

        # ax_box[0].set_title(f"theta band")
        # ax_box[1].set_title(f"alpha band")
        # ax_box[2].set_title(f"beta band")

        # fig_box.savefig(f"{self.tf_bp_filename}_{ch_label}.png")

        return 0

    ###############################
    def plot_boxplots_bands(self, ch_label):
        ## dataframe of frequency bands to create boxplots
        data_dict = {
            'theta': self.activity_theta_band,
            'alpha': self.activity_alpha_band,
            'beta' : self.activity_beta_band,
        }
        df_bands = pd.DataFrame(data_dict)
        ## adding mask bad segments
        ############################
        ## create a mask to avoid data of bad segments
        mask_ann = np.zeros(len(self.times_tf))
        ## annotations bad segments
        for ann in self.bad_annot_list:
            if ann['description'].startswith('bad'):
                onset = ann['onset']
                duration = ann['duration']
                print(f"bad segment:\nonset: {onset}\nduration: {duration}")
                ## identify samples inside the bad segment
                idx_times = np.nonzero((self.times_tf>=onset) & (self.times_tf<(onset+duration)))
                ## initial (t0) and final (t1) samples of the bad segment
                t0 = idx_times[0][0]
                t1 = idx_times[0][-1]
                print(f"samples bad segment (t0, t1): ({t0,t1})")
                ## a mask for all channels, all frequencies, and same range in time (between t0 and t1)
                mask_ann[t0:t1]=1
        ############################
        df_bands['mask'] = mask_ann
        ##
        ## boxplots
        # print(f"df_bands:\n{df_bands}")
        df_masked = df_bands.loc[df_bands['mask']==0]
        # print(f"df_masked:\n{df_masked}")

        fig_box, ax_box = plt.subplots(nrows=1, ncols=1, figsize=(9,6), sharey=True,)
        df_masked.boxplot(['theta','alpha','beta'], showfliers=False, ax=ax_box)

        ax_box.set_title(f"Power ({ch_label}) -- boxplots frequency bands")
        ax_box.set_ylabel(f"dB change from baseline")
        
        # fig_box, ax_box = plt.subplots(nrows=1, ncols=3, figsize=(9,6), sharey=True,)
        # df_bands_all.boxplot(['a_closed_eyes_theta','b_closed_eyes_theta'], showfliers=False, ax=ax_box[0])
        # df_bands_all.boxplot(['a_closed_eyes_alpha','b_closed_eyes_alpha'], showfliers=False, ax=ax_box[1])
        # df_bands_all.boxplot(['a_closed_eyes_beta' ,'b_closed_eyes_beta'], showfliers=False, ax=ax_box[2])

        # ax_box[0].set_xticks([1, 2], ['a', 'b',])
        # ax_box[1].set_xticks([1, 2], ['a', 'b',])
        # ax_box[2].set_xticks([1, 2], ['a', 'b',])

        # ax_box[0].set_title(f"theta band")
        # ax_box[1].set_title(f"alpha band")
        # ax_box[2].set_title(f"beta band")

        fig_box.savefig(f"{self.tf_bp_filename}_{ch_label}.png")

        return 0

    ###############################
    def get_ch_equivalent(self, ch_name, eeg_system):
        if eeg_system == 'geodesic':
            if ch_name == 'C3':
                ch_label = 'E36'
            elif ch_name == 'C4':
                ch_label = 'E104'
            elif ch_name == 'Cz':
                ch_label = 'VREF'
            else:
                print(f"{ch_name} is not in the table of equivalences.")
                ch_label = ch_name
        else:
            ch_label = ch_name
        
        return ch_label
    
    ###############################
    def tf_plot(self, ch_name, eeg_system, flag_norm=True):

        ## 128-channel geodesic to 10-10 equivalent names
        ch_label = self.get_ch_equivalent(ch_name, eeg_system)

        ## visualization time-frequency plots
        fig_tf, ax_tf = plt.subplots(nrows=1, ncols=1, figsize=(16,4), sharey=True, sharex=True)
        ## plot data
        if flag_norm:
            range = (-12,12)
            self.tfr_seg_norm.plot(picks=[ch_name], title=f"Power ({ch_label})-- dB change from baseline", yscale='auto', vlim=range, axes=ax_tf, show=True)
        else:
            self.tfr_seg.plot(picks=[ch_name], title=f"Power ({ch_label})", yscale='auto', axes=ax_tf, show=True)

        ## plot annotations bad segments
        for ann in self.bad_annot_list:
            if ann['description'].startswith('bad'):
                onset = ann['onset']
                duration = ann['duration']
                ## bad segment in the plot
                ax_tf.fill_between(self.times_tf, 0, 1, where=((self.times_tf >= onset)&(self.times_tf < (onset+duration))), color='tab:pink', alpha=0.5, transform=ax_tf.get_xaxis_transform())
                ##

        fig_tf.savefig(f"{self.tf_ch_filename}_{ch_label}_{self.id_seg}.png")
        return 0

    ###############################
    def tfr_norm_plot(self, ch_name, eeg_system,):

        ## 128-channel geodesic to 10-10 equivalent names
        ch_label = self.get_ch_equivalent(ch_name, eeg_system)

        ## data tf results
        data_tf, times_tf, freqs_tf = self.tfr_seg.get_data(picks=[ch_label], return_times=True, return_freqs=True)

        ## visualization time-frequency plots
        fig_tf, ax_tf = plt.subplots(nrows=1, ncols=1, figsize=(16,4), sharey=True, sharex=True)

        ## masks
        arr_mask = self.df_ch_bands['mask'].to_numpy()
        # print(f"arr_mask shape: {arr_mask.shape}")
        arr_mask = np.logical_not(arr_mask).reshape(1,-1)
        arr_mask = np.repeat(arr_mask, len(freqs_tf),axis=0)

        range = (-12,12)
        self.tfr_seg.plot(picks=[ch_label], mask=arr_mask, mask_alpha=0.5, title=f"Time-frequency power, channel: {ch_name}, {self.title_fig} {self.id_seg}\ndB change from baseline", yscale='auto', vlim=range, axes=ax_tf, show=False)

            # self.tfr_seg.plot(picks=[ch_label], title=f"Power ({ch_name})", yscale='auto', axes=ax_tf, show=False)

        ## plot annotations bad segments
        # for ann in self.bad_annot_list:
        #     if ann['description'].startswith('bad'):
        #         onset = ann['onset']
        #         duration = ann['duration']
        #         ## bad segment in the plot
        #         ax_tf.fill_between(self.times_tf, 0, 1, where=((self.times_tf >= onset)&(self.times_tf < (onset+duration))), color='tab:pink', alpha=0.5, transform=ax_tf.get_xaxis_transform())
        #         ##

        fig_tf.savefig(f"{self.tf_ch_filename}_{ch_name}.png")
        plt.close(fig_tf)

        return 0

    ##################################
    def plot_time_series(self, label, title):
        data = []
        if label=='csd':
            data = self.raw_seg_csd
        else:
            print(f"plot error: data not found")
            return 0

        mne.viz.plot_raw(data, start=0, duration=240, scalings=self.scale_dict, highpass=None, lowpass=None, filtorder=4, title=title, block=True)

        return 0
    
    ##################################
    def get_label(self):
        return self.label_seg
    
    def get_label_simple(self):
        return self.label
    
    ##################################
    def get_id(self):
        return self.id_seg
    
    ##################################
    def get_bad_channels(self):
        return self.bad_ch_list
    
    ##################################
    def get_bad_annot(self):
        return self.bad_annot_list
    
    ##################################
    def get_ica_exclude(self):

        return self.ica_exclude

    ##################################
    def set_annotations_freq_bands(self):
        
        data_tf, times_tf, freqs_tf = self.tfr_seg.get_data(return_times=True, return_freqs=True)
        ## include a column of time in the dataframe
        self.df_ch_bands['time'] = times_tf
        # ## adding mask bad segments
        ############################
        ## create a mask to avoid data of bad segments
        mask_ann = np.zeros(len(times_tf))
        ## annotations bad segments
        for ann in self.bad_annot_list:
            if ann['description'].startswith('bad'):
                onset = ann['onset']
                duration = ann['duration']
                # print(f"bad segment:\nonset: {onset}\nduration: {duration}")
                ## identify samples inside the bad segment
                idx_times = np.nonzero((times_tf>=onset) & (times_tf<(onset+duration)))
                ## initial (t0) and final (t1) samples of the bad segment
                t0 = idx_times[0][0]
                t1 = idx_times[0][-1]

                self.time_bad_segs.append([onset, onset+duration])
                # print(f"samples bad segment (t0, t1): ({t0,t1})")
                ## a mask for all channels, all frequencies, and same range in time (between t0 and t1)
                mask_ann[t0:t1]=1
        ############################
        ## include a column of mask to identify sections of bad annotations (value=1)
        self.df_ch_bands['mask'] = mask_ann

        return 0
    
    ##################################
    def get_df_ch_bands(self):
        ## Median values over time from each freq bands (theta, alpha, beta, beta_low, beta_high)
        ## for a list of selected channels (C3, Cz, C4)
        return self.df_ch_bands
    
    ##################################
    def get_raw_seg(self):
        return self.raw_seg
    
    def get_sfreq(self):
        return self.sampling_rate
    
    def set_selected_flag(self):
        self.flag_selection = True

    def get_selected_flag(self):
        return self.flag_selection