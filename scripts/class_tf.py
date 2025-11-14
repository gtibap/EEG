#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import mne
from mne.preprocessing import EOGRegression, ICA, corrmap, create_ecg_epochs, create_eog_epochs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import make_splrep


class TF_components:

    def __init__(self, raw_seg, filt_seg, label_seg, id_seg):
        ## EEG data per segment
        self.raw_seg = raw_seg
        self.filt_seg = filt_seg
        self.label_seg = label_seg
        self.id_seg = id_seg

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

        ## ica parameters to calculate ICA components
        self.ica = ICA(n_components= 0.99, method='picard', max_iter="auto", random_state=97)

        ## bad channels and bad annotations lists
        self.bad_annot_list = []
        self.bad_ch_list = []
        self.ica_exclude = []

        self.sampling_rate = raw_seg.info['sfreq']
        ## scale selection for visualization raw data with annotations
        self.scale_dict = dict(mag=1e-12, grad=4e-11, eeg=100e-6, eog=150e-6, ecg=400e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)

    ##################################
    def selection_bads(self):
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
            mne.viz.plot_raw(self.raw_seg, picks=['eeg','ecg'], start=0, duration=240, scalings=self.scale_dict, highpass=1.0, lowpass=45.0, title=f"Time-series signals EEG {self.label_seg} -- Please select bad segments and bad channels interactively", block=True)
            ##
            ## power spectral density (psd)
            ##
            ## interactively, identify channels that are out of the tendency (ouliers) as bad channels
            ## 
            fig_psd, ax_psd = plt.subplots(nrows=1, ncols=1, figsize=(9,4), sharey=True, sharex=True)
            mne.viz.plot_raw_psd(self.raw_seg, picks=['eeg'], exclude=['VREF'], ax=ax_psd, fmin=0.9, fmax=101, xscale='log',)
            ##
            ax_psd.set_title('EEG power spectral density (baseline)')
            ##
            flag_bad_ch = input('Include more bad channels? (1 (True), 0 (False)): ')
            flag_bad_ch = 0 if (flag_bad_ch == '') else int(flag_bad_ch)
            # print(f"flag bad_ch: {flag_bad_ch}")

        ## bad channels
        self.bad_ch_list = self.raw_seg.info['bads']      
        ## bad annotations
        interactive_annot = self.raw_seg.annotations
        time_offset = self.raw_seg.first_samp / self.sampling_rate  ## in seconds
        self.bad_annot_list = self.ann_remove_offset(interactive_annot, time_offset)

        ## setting same bad channels and bad segments for the filtered data
        self.filt_seg.set_annotations(self.bad_annot_list)
        self.filt_seg.info['bads'] = self.bad_ch_list

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
    def ica_components(self):
        ## ica works better with clean (denoised) EEG signals with 0 offset (a high pass filter with a 1 Hz cutoff frequency could improve that condition, that is why we use the filtered version of the data [self.filt_seg])
        ##
        ## ICA fitting model to the filtered raw data
        self.ica.fit(self.filt_seg, reject_by_annotation=True)
        ## interactive selection of ica components to exclude
        self.ica.plot_components(inst=self.raw_seg, contours=0,)
        self.ica.plot_sources(self.raw_seg, start=0, stop=240, show_scrollbars=False, block=True)
        ## selected ica components to exclude
        self.ica_exclude = self.ica.exclude
        ## apply ica
        self.raw_seg_ica = self.raw_seg.copy()
        ## ica in place
        self.ica.apply(self.raw_seg_ica)
        return 0

    ##################################
    def bads_interpolation(self):
        print("Bad channels interpolation")
        self.raw_seg_ica.interpolate_bads()
        return 0
    
    ##################################
    def apply_csd(self):
        self.raw_seg_csd = mne.preprocessing.compute_current_source_density(self.raw_seg_ica)
        return 0
    
    ##################################
    def tf_calculation(self):
        ## Time-frequency (tf) decomposition from each EEG channel
        ## logarithmic scale frequencies
        start= 0.60 # 10^start,  
        stop = 1.50 # 10^stop
        num  = 100 # samplesq
        freqs = np.logspace(start, stop, num=num,)
        # print(f'log freqs: {freqs}')

        # tfr_bl = raw_seg_ica.compute_tfr('morlet',freqs, picks=['eeg'])
        # data_bl, times_bl, freqs_bl = tfr_bl.get_data(picks=['eeg'],return_times=True, return_freqs=True)
        # print(f"Time-frequency analysis (Morlet wavelet)")
        self.tfr_seg = self.raw_seg_csd.compute_tfr('morlet', freqs, reject_by_annotation=False)
        ## time-frequency data
        self.data_tf, self.times_tf, self.freqs_tf = self.tfr_seg.get_data(return_times=True, return_freqs=True)
        print(f"data tf analysis shape: {self.data_tf.shape}")
        print(f"times tf analysis shape: {self.times_tf.shape}")
        print(f"freqs tf analysis shape: {self.freqs_tf.shape}")
        print(f"times tf analysis: {self.times_tf}")
        print(f"freqs tf analysis: {self.freqs_tf}")
        
        return 0

    ###############################
    def get_tf_baseline(self):
        ## create a mask to avoid data of bad segments
        self.mask_data_tf = np.zeros(self.data_tf.shape)
        self.mask_ann = np.zeros(len(self.times_tf))
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
                self.mask_data_tf[:,:,t0:t1]=1
                self.mask_ann[t0:t1]=1
        ## including a mask to avoid bad segments in the mean calculation
        data_tf_masked = np.ma.array(self.data_tf, mask=self.mask_data_tf)
        ## mean values per frequency per channel (ref for baseline normalization)
        self.baseline_tf = data_tf_masked.mean(axis=2)
        self.baseline_tf = self.baseline_tf.data

        # print(f"ref baseline data shape: {self.baseline_tf.shape}")
        # print(f"ref baseline data:\n{self.baseline_tf}")
        return self.baseline_tf
    
    ###############################
    def tf_normalization(self, tf_ref):
        ## time-frequency power normalization for each channel
        dim_ch, dim_fr, dim_t = self.data_tf.shape
        # print(f"bl dim_ch, dim_fr, dim_t: {dim_ch, dim_fr, dim_t}")
        
        ## initialization new array for baseline normalization
        data_tf_norm = np.zeros(self.data_tf.shape)

        id_ch=0
        # print(f"normalization ch:")
        for mean_ch, arr_num in zip(tf_ref, self.data_tf):
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
        self.tfr_seg_norm = self.tfr_seg.copy()
        self.tfr_seg_norm._data = data_tf_norm

        return 0
    
    ###############################
    def channel_bands_power(self, ch_label):
        ## separate components frequency bands theta, alpha, and beta
        data_ch, times_ch, freqs_ch = self.tfr_seg_norm.get_data(picks=[ch_label],return_times=True, return_freqs=True)
        print(f"Channel {ch_label} data shape: {data_ch.shape}")
        print(f"times shape: {times_ch.shape}")
        print(f"freqs shape: {freqs_ch.shape}")

        # plot matrix selected channel (as an image)
        # fig_tf, ax_tf = plt.subplots(nrows=3, ncols=1, figsize=(16,4), sharey=True, sharex=True)
        # ax_tf[0].imshow(data_ch[0], aspect='auto', cmap='coolwarm', vmin=-12, vmax=12)

        ## normalized tf matrix to dataframe [VREF]
        df_tf = pd.DataFrame(data_ch[0])
        # rows-->freqs (from the lowest to highest freqs), columns-->times
        df_tf['freq'] = freqs_ch
        # print(f"df_tf:\n{df_tf}")
        ##
        ## accumulative power per band per every time sample
        ##
        ## theta (4-8 Hz), alpha (8-12 Hz), beta (12-30 Hz)
        # selecting group of rows based on frequency band values
        df_theta = df_tf.loc[(df_tf['freq'] >= 4)  & (df_tf['freq'] < 8)]
        df_alpha = df_tf.loc[(df_tf['freq'] >= 8)  & (df_tf['freq'] < 12)]
        df_beta  = df_tf.loc[(df_tf['freq'] >= 12) & (df_tf['freq'] < 30)]

        # print(f"df_theta  shape: {df_theta.shape}")
        # print(f"df_alpha  shape: {df_alpha.shape}")
        # print(f"df_beta shape:   {df_beta.shape}")

        # print(f"df_theta:\n{df_theta}")
        # print(f"df_theta shape:\n{df_theta.shape}")
        ##
        ## exclude column freq
        df_theta = df_theta.loc[:,df_theta.columns != 'freq']
        df_alpha = df_alpha.loc[:,df_alpha.columns != 'freq']
        df_beta  =  df_beta.loc[:,df_beta.columns  != 'freq']

        ## calculate median value for each time sample for each freq. band
        self.activity_theta_band = df_theta.median(axis=0).to_numpy()
        self.activity_alpha_band = df_alpha.median(axis=0).to_numpy()
        self.activity_beta_band  = df_beta.median(axis=0).to_numpy()

        # print(f"theta median shape: {self.activity_theta_band.shape}")
        # print(f"alpha median shape: {self.activity_alpha_band.shape}")
        # print(f"beta median shape:  {self.activity_beta_band.shape}")

        ## plot curves frequency bands
        self.plot_curves_bands()
        self.plot_boxplots_bands()

        return 0
    
    ###############################
    def plot_curves_bands(self):
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
        ax_bands[0].set_title(f'Power({self.label_seg} {self.id_seg}) -- dB change from baseline [median]')

        ## plot annotations bad segments
        for ann in self.bad_annot_list:
            if ann['description'].startswith('bad'):
                onset = ann['onset']
                duration = ann['duration']
                ## bad segment in the plot each band 
                for id in np.arange(len(ax_bands)):
                    ax_bands[id].fill_between(self.times_tf, 0, 1, where=((self.times_tf >= onset)&(self.times_tf < (onset+duration))), color='tab:red', alpha=0.25, transform=ax_bands[id].get_xaxis_transform())
                ##

        return 0

    ###############################
    def plot_boxplots_bands(self):
        ## dataframe of frequency bands to create boxplots
        data_dict = {
            'theta': self.activity_theta_band,
            'alpha': self.activity_alpha_band,
            'beta' : self.activity_beta_band,
        }
        df_bands = pd.DataFrame(data_dict)
        ## adding mask bad segments
        df_bands['mask'] = self.mask_ann
        ##
        ## boxplots
        # print(f"df_bands:\n{df_bands}")
        df_masked = df_bands.loc[df_bands['mask']==0]
        # print(f"df_masked:\n{df_masked}")

        fig_box, ax_box = plt.subplots(nrows=1, ncols=1, figsize=(9,6), sharey=True,)
        df_masked.boxplot(['theta','alpha','beta'], showfliers=False, ax=ax_box)
        
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

        return 0

    ###############################
    def tf_plot(self, ch_name, flag_norm=True):
        ## visualization time-frequency plots
        fig_tf, ax_tf = plt.subplots(nrows=1, ncols=1, figsize=(16,4), sharey=True, sharex=True)
        ## plot data
        if flag_norm:
            range = (-12,12)
            self.tfr_seg_norm.plot(picks=[ch_name], title='auto', yscale='auto', vlim=range, axes=ax_tf, show=True)
        else:
            self.tfr_seg.plot(picks=[ch_name], title='auto', yscale='auto', axes=ax_tf, show=True)

        ## plot annotations bad segments
        for ann in self.bad_annot_list:
            if ann['description'].startswith('bad'):
                onset = ann['onset']
                duration = ann['duration']
                ## bad segment in the plot
                ax_tf.fill_between(self.times_tf, 0, 1, where=((self.times_tf >= onset)&(self.times_tf < (onset+duration))), color='tab:red', alpha=0.25, transform=ax_tf.get_xaxis_transform())
                ##

        return 0

    ##################################
    def plot_time_series(self, label):
        data = []
        if label=='csd':
            data = self.raw_seg_csd
        else:
            print(f"plot error: data not found")
            return 0

        mne.viz.plot_raw(data, start=0, duration=240, scalings=self.scale_dict, highpass=None, lowpass=None, filtorder=4, title=f'baseline after Surface Laplacian', block=True)

        return 0
    
    ##################################
    def get_label(self):
        return self.label_seg
    
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
