import numpy as np
import matplotlib.pyplot as plt
import pickle
import mne

def main(args):
    print(f'baseline wavelet !')

    path = "../data/oct06_Taha/"
    ## load baseline
    with open(path + 'baseline.pkl', 'rb') as file:
        eeg_baseline = pickle.load(file)

    print(f"type(eeg_baseline): {type(eeg_baseline[0])}")
    ## scale selection for visualization raw data with annotations
    # scale_dict = dict(mag=1e-12, grad=4e-11, eeg=100e-6, eog=150e-6, ecg=300e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)
    ## signals visualization (channels' voltage vs time)
    # mne.viz.plot_raw(eeg_baseline[0], start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title='baseline', block=True)

    ## wavelets
    freqs = np.arange(4.0, 45.0, 1.0)
    pw = mne.time_frequency.tfr_morlet(eeg_baseline[0],freqs=freqs, n_cycles=4)
    # pw = eeg_baseline[0].compute_tfr(method="morlet")
    print(f'power: {pw}')

            
    plt.show()
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
