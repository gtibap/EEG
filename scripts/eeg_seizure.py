import numpy as np
import sys
import mne
import matplotlib.pyplot as plt

mne.set_log_level('error')

def read_data(flag_plot):

    dir_data = '../data/'
    filename_eeg = 'chb01_03.edf'

    raw = mne.io.read_raw_edf(dir_data+filename_eeg)
    # raw.crop(tmax=60).load_data()
    raw.load_data()

    print(f'data:\n{raw.info}')
    print('channels: ', raw.info['ch_names'])

    if flag_plot:
        raw.plot(
            start=0,
            duration=60,
            scalings=dict(eeg=1e-4, resp=1e3, eog=1e-4, emg=1e-7, misc=1e-1),
        )
    else:
        pass

    return raw


def spectral_density(raw):

    spectrum = raw.compute_psd()
    spectrum.plot(average=True, picks="data", exclude="bads", amplitude=False)

    return 0


def segments_raw():

    # ann_data = mne.read_annotations(dir_data+filename_ann)
    # print(f'ann: {ann_data}')

    freq = raw.info['sfreq']
    print('freq: ', freq, ' Hz')

    fp1_f7 = raw.get_data(picks='FP1-F7',tmin=2990, tmax=3050)[0]
    print(fp1_f7.shape)

    # number of seconds per segment
    nsec = 4

    # number of samples per segment
    nsam = int(freq * nsec)

    # number of segments per signal
    nseg = len(fp1_f7) // nsam

    # selected interval in seconds
    ta=2990
    tb=3050

    # specific frequency bands
    FREQ_BANDS = {
        "delta": [0.5, 4.5],
        "theta": [4.5, 8.5],
        "alpha": [8.5, 11.5],
        "sigma": [11.5, 15.5],
        "beta": [15.5, 30],
    }

    # for i in np.arange(nseg):
    for t in np.arange(ta,tb,nsec):
        # samples = fp1_f7[nsam*i : nsam*(i+1)]
        # print(f'{i}:{samples}')
        spectrum = raw_data.compute_psd(picks='FP1-F7',tmin=t, tmax=t+nsec, fmin=0.0, fmax=25.0)
        # print(f'spectrum {t}: {spectrum}')
        # raw_data.compute_psd(picks='FP1-F7',tmin=nsec*i, tmax=nsec*(i+1))
        psds, freqs = spectrum.get_data(return_freqs=True)
        # print(f'spectrum: {psds}')
        # Normalize the PSDs
        psds /= np.sum(psds, axis=-1, keepdims=True)
        # print(f'normaliz: {psds}')

        X = []
        for fmin, fmax in FREQ_BANDS.values():
            psds_band = psds[:, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
            X.append(psds_band.reshape(len(psds), -1))

        # print(f'X={X}')

    fig, ax = plt.subplots(nrows=1, ncols=1)
    spectrum.plot(
                ci=None,
                axes=ax,
                show=False,
                average=True,
                amplitude=False,
                spatial_colors=False,
                picks="data",
                exclude="bads",
            )

    return 0



    # plt.plot(fp1_f7)
    # raw_data.plot()
    # plt.show()


if __name__ == "__main__":

    flag_plot=True
    raw = read_data(flag_plot)
    spectral_density(raw)
    
    plt.show()

