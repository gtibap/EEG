import numpy as np
import matplotlib.pyplot as plt
import pickle
import mne

def main(args):
    print(f'baseline wavelet !')

    path = "../../data/oct06_Taha/"
    eeg_baseline=[]
    ## load baseline
    # with open(path + 'baseline.pkl', 'rb') as file:
    #     eeg_baseline = pickle.load(file)

    eeg_baseline = mne.io.read_raw_fif(path + 'baseline.fif.gz',)
    # eeg_data_dict['baseline'] = mne.io.read_raw_fif(path + 'baseline.fif',)

    # print(f"type(eeg_baseline): {type(eeg_baseline[0])}")
    print(f"eeg_baseline: {eeg_baseline}")
    print(f"eeg_baseline: {eeg_baseline.info}")

    # return 0
    ## scale selection for visualization raw data with annotations
    scale_dict = dict(mag=1e-12, grad=4e-11, eeg=100e-6, eog=150e-6, ecg=300e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)
    # signals visualization (channels' voltage vs time)
    # mne.viz.plot_raw(eeg_baseline, start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title='baseline', block=False)

    ## wavelets
    freqs = np.arange(4.0, 45.0, 1)

    tfr_bl = eeg_baseline.compute_tfr('morlet',freqs,)
    print(f"type(tfr_power): {type(tfr_bl)}")
    print(tfr_bl)

    data, times, freqs = tfr_bl.get_data(picks=['all'],return_times=True, return_freqs=True)
    # picks=['AFz','Cz','POz']
    print(f"data:\n{data.shape}")
    # print(f"times:\n{times}")
    # print(f"freqs:\n{freqs}")

    tfr_bl.plot(picks=['Cz'], title='auto', yscale='linear', show=False)
    # 'AFz','Cz','POz'

    ## mean along time samples
    mean_data = np.mean(data, axis=2)
    print(f"mean data: {mean_data.shape}")

    dim_ch, dim_fr, dim_t = data.shape
    print(f"dim_ch, dim_fr, dim_t: {dim_ch, dim_fr, dim_t}")
    id_ch=0
    for mean_ch, arr_num in zip(mean_data, data):
        # mean for each frequency per channel
        ## mean_ch is an array with a number of elements equal to the number of evaluated frequencies
        ## each element of the array represents the mean value of time samples per each frequency
        arr_den = np.repeat(mean_ch, dim_t ,axis=0).reshape((len(mean_ch),-1))
        arr_dB = 10*np.log10(arr_num / arr_den)
        # print(f"arr_res: {arr_dB.shape}")
        data[id_ch] = arr_dB
        id_ch+=1

    ## baseline scaling
    ## dB = 10*log10( matrix_time_freq / mean_for_each_freq_baseline )
    # data_2 = data*20
    tfr_bl._data = data

    vlim = (-10,10)
    tfr_bl.plot(picks=['Cz'], title='auto', yscale='linear', vlim=vlim)


    plt.show(block=True)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
