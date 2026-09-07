import numpy as np
import pandas as pd
# Import the FOOOF object
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic
from fooof.plts.spectra import plot_spectra


class PSD_Epochs_Class:

    def __init__(self, label):
        print(f"obj label: {label}")

        if label == 'a_closed_eyes':
            self.label = 'a_ce'
            self.title_ax =  'resting (before cycling)'
            self.title_fig = 'resting (before cycling) closed-eyes'
        elif label == 'a_opened_eyes':
            self.label = 'a_oe'
            self.title_ax = 'resting (before cycling)'
            self.title_fig = 'resting (before cycling) open-eyes'
        elif label == 'b_closed_eyes':
            self.label = 'b_ce'
            self.title_ax = 'cycling (passive)'
            self.title_fig = 'passive cycling closed-eyes'
        elif label == 'b_opened_eyes':
            self.label = 'b_oe'
            self.title_ax  = 'cycling (passive)'
            self.title_fig = 'passive cycling open-eyes'
        elif label == 'c_closed_eyes':
            self.label = 'c_ce'
            self.title_ax  = 'resting (after cycling)'
            self.title_fig = 'resting (after cycling) closed-eyes'
        elif label == 'c_opened_eyes':
            self.label = 'c_oe'
            self.title_ax  = 'resting (after cycling)'
            self.title_fig = 'resting (after cycling) open-eyes'
        else:
            self.label = ''

        ######
        self.psd_dict = {}
        self.quantiles_dict = {}
        self.range_freqs_dict = {'central_left':[], 'central_right':[]}
        self.thr_peaks_dict = {'central_left':[], 'central_right':[]}
        self.fm_dict = {'central_left':[], 'central_right':[]}


    ########################################
    def calculate_average_psd_model(self, region, ax):
        ## power spectral density (PSD) from epochs of selected channels
        psd_epochs = self.get_psd(region)

        # ## mean values of PSD from epochs
        # psd_epochs_avg, freqs = psd_epochs.average(method='median').get_data(return_freqs=True)
        psd_epochs_avg, freqs = psd_epochs.average(method='mean').get_data(return_freqs=True)

        
        psd_channels_q1 = 10*np.log10(np.quantile(1e12*psd_epochs_avg, q=0.25, axis=0))
        psd_channels_q2 = 10*np.log10(np.quantile(1e12*psd_epochs_avg, q=0.50, axis=0))
        psd_channels_q3 = 10*np.log10(np.quantile(1e12*psd_epochs_avg, q=0.75, axis=0))

        df_psd_quantiles = pd.DataFrame()
        df_psd_quantiles['freqs'] = freqs
        df_psd_quantiles['psd_q1'] = psd_channels_q1
        df_psd_quantiles['psd_q2'] = psd_channels_q2
        df_psd_quantiles['psd_q3'] = psd_channels_q3
       
        ## label: left or right regions
        self.quantiles_dict[region] = df_psd_quantiles
        
        ## plot region between the q1 and q1 quantiles, i.e. the region where 25% and 75% data is located
        # ax.fill_between(freqs, psd_channels_q1, psd_channels_q3, alpha=0.5, color='tab:gray')

        ## mean values of PSD from epochs, i.e. average curves from epochs for each selected channel
        # self.psd_epochs_mean_dict[region] = psd_epochs.average(method='mean')
        # self.psd_epochs_mean_dict[region].plot(axes=ax)
        psd_epochs.average(method='mean').plot(axes=ax)

        ax.set_ylabel(f"Power (dB $\mu$V$^2$/Hz)")
        ax.set_title(self.title_ax)

        # self.ax_copy = np.copy(ax)
        return 0

    #############
    def fit_fooof(self, label, range_freqs, thr_peaks, ax):
        # Set whether to plot in log-log space
        self.range_freqs_dict[label] = range_freqs
        self.thr_peaks_dict[label] = thr_peaks

        plt_log = False
        print(f"FOOOF: aperiodic and periodic components' estimation")
        # df_psd_global
        # fm = FOOOF(aperiodic_mode='fixed', peak_width_limits=[0.5, 12], max_n_peaks=5, min_peak_height=1.0)
        fm = FOOOF(aperiodic_mode='fixed', peak_width_limits=[1.0, 15.0], max_n_peaks=7, min_peak_height=thr_peaks)
        
        ## label: left or right side electrodes
        df_psd_quantiles = self.quantiles_dict[label]
        # print(f"df psd quantiles:\n{df_psd_quantiles}")

        # fm.add_data(df_psd_quantiles['freqs'].to_numpy(), 10**(df_psd_quantiles['psd_q2'].to_numpy()), range_freqs)
        # fm.fit(df_psd_global['freqs'].to_numpy(), 10**(df_psd_global['psd_q2'].to_numpy()), range_freqs)

        ## 10^(psd_q2) because the fit function apply log10 to the data
        fm.fit(df_psd_quantiles['freqs'].to_numpy(), 10**(df_psd_quantiles['psd_q2'].to_numpy()), range_freqs)

        init_ap_fit = gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
        init_flat_spec = fm.power_spectrum - init_ap_fit

        # # Plot the flattened the power spectrum
        plot_spectra(fm.freqs, init_flat_spec, plt_log, label='Flattened Spectrum', color='tab:blue', ax=ax)
        plot_spectra(fm.freqs, fm._peak_fit, plt_log, label='peak fit', color='tab:red', ax=ax)

        self.fm_dict[label] = fm
    
        return 0
    
    ########################################
    # def get_average_psd_model(self, channels, freq_range, label, ax):
    def plot_psd_quantiles(self, region, ax):
        ## power spectral density (PSD) from epochs of selected channels
        # psd_epochs = self.epochs.compute_psd(picks=channels, exclude='bads',fmin=freq_range[0], fmax=freq_range[1])
        psd_epochs = self.get_psd(region)

        df_psd_quantiles = self.quantiles_dict[region]

        print(f"quantiles:\n{df_psd_quantiles}")

        freqs = df_psd_quantiles['freqs']
        psd_channels_q1 = df_psd_quantiles['psd_q1']
        psd_channels_q2 = df_psd_quantiles['psd_q2']
        psd_channels_q3 = df_psd_quantiles['psd_q3']

        print(f"inside plot psd quantiles...")
        ## plot region between the q1 and q1 quantiles, i.e. the region where 25% and 75% data is located
        ax.fill_between(freqs, psd_channels_q1, psd_channels_q3, alpha=0.5, color='tab:gray')

        ## mean values of PSD from epochs, i.e. average curves from epochs for each selected channel
        ax.plot(freqs, psd_channels_q2)

        ax.set_ylabel(f"Power (dB $\mu$V$^2$/Hz)")

        ax.set_title(self.title_ax)
        ax.grid(ls=':',lw=0.5)

        return ax

    #############
    def set_psd(self, psd_data, region):
        self.psd_dict[region] = psd_data
        return 0

    #############
    def get_psd(self, region):
        return self.psd_dict[region]

    #############
    def get_label_simple(self):
        return self.label

    #############
    def get_fooof_model(self, label):
        return self.fm_dict[label]
