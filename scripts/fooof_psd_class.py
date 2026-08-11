import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the FOOOF object
from fooof import FOOOF
# from fooof.sim.gen import gen_aperiodic
# from fooof.plts.spectra import plot_spectra
# from fooof.plts.annotate import plot_annotated_peak_search


class FOOOF_class:
    ################
    def __init__(self, label, freqs, thres, df):
        self.label = label
        self.freqs = freqs
        self.thres = thres
        self.df = df
        self.df_per = pd.DataFrame()
        self.fm = []
        self.flag_fooof = False
        pass

    ################
    def set_quantiles(self):
        return 0

    ################
    def fit_fooof(self):
        ## fooof initial parametres
        if self.thres != []:
            self.fm = FOOOF(aperiodic_mode='fixed', peak_width_limits=[1.0, 15.0], max_n_peaks=7, min_peak_height=self.thres)
            ## fit model
            self.fm.fit(self.df['freqs'].to_numpy(), 10**(self.df['psd_q2'].to_numpy()), self.freqs)
        else:
            print(f"Warning: No fooof fitted model for {self.label}")

        return 0

    ################
    def plot_psd_fooof(self, ax, color, flags):

        if self.fm != []:
            ax.plot(self.fm.freqs, self.fm.power_spectrum, label=self.label, color=color)
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
    

