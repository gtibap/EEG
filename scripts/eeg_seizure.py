import numpy as np
import sys
import mne
import matplotlib.pyplot as plt

mne.set_log_level('error')

def read_data(filename):

    # read data
    raw = mne.io.read_raw_edf(filename)
    raw.load_data()

    # data info and channels' names
    print(f'data:\n{raw.info}')
    print('channels: ', raw.info['ch_names'])

    return raw


def frequency_components(raw, t, nsec, nseg):

    # selected range of frequencies 
    freq_min =  0.0
    freq_max = 25.0
    # frequency spectrum calculation
    spectrum = raw.compute_psd(tmin=t, tmax=t+nsec, fmin=freq_min, fmax=freq_max)
    # spectrum' magnitude (psds) and frequencies (freqs) from selected channels
    psds, freqs = spectrum.get_data(return_freqs=True)
    # Normalize the PSDs of each channel
    psds = psds / np.sum(psds, axis=1, keepdims=True)

    # spectrum segmentation being nseg number of segments (partitions)
    seg_len = (freq_max-freq_min) / nseg

    X = []
    # take each segment and calculate the mean value from all selected channels
    for i in np.arange(nseg):
        # segment's begining
        f1 = seg_len*i + freq_min
        # segment's end
        f2 = seg_len*(i+1) + freq_min
        print(f'f1, f2: {f1}, {f2}')
        # extraction of selected segment from each channel
        seg_freq = psds[:, (freqs>=f1) & (freqs<f2)]
        mean_values = seg_freq.mean(axis=1)
        # print(f'mean_values: {mean_values}')
        X.append(mean_values)

    X = np.transpose(X)
    # print(f'{X}\nX {X.shape}')

    # frequency spectrum plotting from selected eeg signals
    spectrum.plot(average=True, picks="data", exclude="bads", amplitude=False,)

    # plot eeg signals
    raw.plot(start=t, duration=nsec, scalings=dict(eeg=1e-4),)

    return X


if __name__ == "__main__":

    # data location
    dir_data = '../data/'
    filename_eeg = 'chb01_03.edf'

    # eeg data reading
    raw = read_data(dir_data+filename_eeg)

    # frequency spectrum calculation from a selected segment of eeg signals
    
    # beginning of a selected eeg segment in seconds
    t = 2900
    # eeg segment length in seconds
    nsec = 5
    # number of segments from the frequency spectrum
    nseg = 8
    # frequency components selected segments from selected channels
    feature_vector = frequency_components(raw, t, nsec, nseg)
    
    feature_vector = feature_vector.reshape(1,-1)
    print(f'feature vector:\n{feature_vector}')
    
    plt.show()

