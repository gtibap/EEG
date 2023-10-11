# import kineticstoolkit.lab as ktk
import scipy.io
from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

class EEG_components:

    def __init__(self, signals, sampling_rate):
        self.signals = signals
        self.sampling_rate = sampling_rate
        
    def freq_components(self,ids):
        
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7),sharex=True, sharey=True, squeeze=True)
        ax = ax.reshape(-1)
        
        f1=[[]]*len(self.signals)
        S1=[[]]*len(self.signals)
        
        i=0
        for signal in self.signals:
            (f1[i], S1[i])= scipy.signal.welch(signal[ids[0]:ids[1]], self.sampling_rate, nperseg=2*1024, scaling='density')
            ax[i].axes.plot(f1[i], np.sqrt(S1[i]))
            i=i+1
            
        ax[0].set_xlim([0,50])
        
        return 0
