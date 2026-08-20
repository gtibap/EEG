import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator

# Import the FOOOF object
from fooof import FOOOF

class FOOOF_class:
    ################
    def __init__(self, label, freqs, thres, df):
        self.label = label
        self.freq_range = freqs
        self.thres = thres
        self.df = df
        self.df_per = pd.DataFrame()
        self.df_fooof = pd.DataFrame()
        self.fm = []
        self.flag_fooof = False
        self.peak_value = []
        self.peak_freq = []
        self.bands_avg = {}

    ################
    def set_quantiles(self):
        return 0

    ####################
    def fit_fooof(self):
        ## fooof initial parametres
        if self.thres != []:
            self.fm = FOOOF(aperiodic_mode='fixed', peak_width_limits=[1.0, 15.0], max_n_peaks=7, min_peak_height=self.thres)
            ## fit model, we include 10** to neutralize the log10 that is applied by the function before fitting
            self.fm.fit(self.df['freqs'].to_numpy(), 10**(self.df['psd_q2'].to_numpy()), self.freq_range)
            ## obtain the periodic component (psd - aperiodic fit)
            self.df_fooof['freqs'] = self.fm.freqs
            self.df_fooof['periodic'] = self.fm.power_spectrum - self.fm._ap_fit
            self.df_fooof['psd'] = self.fm.power_spectrum
            self.df_fooof['aperiodic'] = self.fm._ap_fit
        else:
            print(f"Warning: No fooof fitted model for {self.label}")

        return 0

    ###############
    def calculate_mean_psd_band(self, label_band, freq_left, freq_right):
        if self.fm != []:
            df = self.df_fooof.copy()
            df = df.loc[(df['freqs']>=freq_left) & (df['freqs']<freq_right)]
            self.bands_avg[label_band] = df['periodic'].mean()
        else:
            self.bands_avg[label_band] = np.nan
        return self.bands_avg[label_band]

    ##########
    def get_mean_psd_band(self, label_band):
        return self.bands_avg[label_band]
    
    ##############################
    def get_freq_peak_value(self, f0, f1):
        if self.fm != []:
            ## a fooof model was fitted, hence the aperiodic component was removed and we work with the periodic one
            df = self.df_fooof.copy()
            df = df.loc[(df['freqs']>=f0) & (df['freqs']<f1)]
            x = df['freqs'].to_numpy()
            y = df['periodic'].to_numpy()
            ## interpolation to smooth the curve
            aki = Akima1DInterpolator(x, y)
            x_data = np.arange(np.min(x), np.max(x), 0.01)
            y_data = aki(x_data)
            # plt.plot(x_data,y_data)
            ## max value in the data
            max_value = np.max(y_data)
            ## where the maximum is located in the array
            max_id = np.argmax(y_data)
            ## that location is the same for freq max
            freq_max = x_data[max_id]
            # print(f"{self.label} (freq, max_value):\n{np.round(freq_max,1), np.round(max_value,1)}")
            self.peak_value = max_value
            self.peak_freq  = freq_max 

            return self.peak_freq
        else:
            return np.nan
        
    ################
    def plot_psd_fooof(self, ax, color, flags):

        if self.fm != []:
            ## psd
            ax.plot(self.fm.freqs, self.fm.power_spectrum, label=self.label, color=color)
            ## aperiodic
            ax.plot(self.fm.freqs, self.fm._ap_fit, label=self.label, color=color, alpha=0.5, linestyle='--')
            ## to generate legend of figures
            if color == 'tab:blue':
                flags[0] = 1
            elif color == 'tab:orange':
                flags[1] = 1
            elif color == 'tab:green':
                flags[2] = 1
            return flags
        else:
            print(f'Noting to plot for: {self.label}')
            return flags

    ################
    def plot_per_fooof(self, ax, color):

        if self.fm != []:
            psd_minus_aperiodic = self.fm.power_spectrum - self.fm._ap_fit
            ax.plot(self.fm.freqs, psd_minus_aperiodic, label=self.label, color=color)
            self.df_per['freqs'] = self.fm.freqs
            self.df_per['values'] = psd_minus_aperiodic
            return 0
        else:
            print(f'Noting to plot for: {self.label}')
            return 0

    ################
    def get_label(self):
        return self.label

    ################
    def get_thres(self):
        return self.thres

    ################
    def get_df_per(self):
        return self.df_per

    def get_fooof_peaks_params(self):
        if self.fm != []:
            return self.fm.peak_params_
        else:
            return []

    ###################
    def get_fooof_model(self):
        return self.fm

    def get_fooof_data(self):
        if self.fm != []:
            return self.df_fooof
        else:
            return np.nan
    

