import mne
import numpy as np
import matplotlib.pyplot as plt

from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from list_participants import participants_list
from bad_channels import bad_channels_dict

## global variables
# subject = 0

def data_segmentation(raw_data):
    ############################################
    ## cropping data according to annotations
    ## usually segments of each label have different duration

    count1 = 0
    ## we could add padding at the begining and end of each segment if required
    tpad = 0 # seconds

    ## a for resting and b for biking
    a_closed_eyes_list = []
    a_opened_eyes_list = []
    b_closed_eyes_list = []
    b_opened_eyes_list = []

    eeg_data_dict={}

    for ann in raw_data.annotations:
        # print(f'ann:\n{ann}')
        label = ann["description"]
        duration = ann["duration"]
        onset = ann["onset"]
        
        # print(f'annotation:{count1, onset, duration, label}')

        t1 = onset - tpad
        t2 = onset + duration + tpad
        if label == 'a_closed_eyes':
            # print('a closed eyes')
            a_closed_eyes_list.append(raw_data.copy().crop(tmin=t1, tmax=t2,))

        elif label == 'a_opened_eyes':
            # print('a opened eyes')
            a_opened_eyes_list.append(raw_data.copy().crop(tmin=t1, tmax=t2,))

        elif label == 'b_closed_eyes':
            # print('a opened eyes')
            b_closed_eyes_list.append(raw_data.copy().crop(tmin=t1, tmax=t2,))

        elif label == 'b_opened_eyes':
            # print('a opened eyes')
            b_opened_eyes_list.append(raw_data.copy().crop(tmin=t1, tmax=t2,))

        else:
            pass
        count1+=1

    print(f'size list a_closed_eyes: {len(a_closed_eyes_list)}')
    print(f'size list a_opened_eyes: {len(a_opened_eyes_list)}')
    print(f'size list b_closed_eyes: {len(b_closed_eyes_list)}')
    print(f'size list b_opened_eyes: {len(b_opened_eyes_list)}')

    ## eeg data to a dictionary
    eeg_data_dict['a_closed_eyes'] = a_closed_eyes_list
    eeg_data_dict['a_opened_eyes'] = a_opened_eyes_list
    eeg_data_dict['b_closed_eyes'] = b_closed_eyes_list
    eeg_data_dict['b_opened_eyes'] = b_opened_eyes_list

    return eeg_data_dict

############################
## Bad channels identification
def bad_channels_interp(raw_data, subject, session):
    ## include bad channels previously identified
    raw_data.info["bads"] = bad_channels_dict[subject]['session_'+str(session)]['general']
    print(f'bad channels: {raw_data.info["bads"]}')
    ## interpolate only selected bad channels
    raw_data.interpolate_bads()
    return raw_data

def main(args):
    # global subject

    ## run matplotlib in interactive mode
    plt.ion()

    print(f'folder location: {args[1]}') ## folder location
    print(f'subject: {args[2]}') ## subject = {0:Mme Chen, 1:Taha, 2:Carlie, 3:Iulia, 4:A. Caron}
    print(f'session: {args[3]}') ## session = {0:initial, 1:three months}
    print(f'ABT: {args[4]}') ## ABT = {0:resting, 1:biking}
    
    path=args[1]
    subject= int(args[2])
    session= int(args[3])
    abt= int(args[4])

    fn_in=''
    t0=0
    t1=0

    #########################
    ## new path, eeg filename (fn_in), annotations filename (fn_csv), eeg raw data (raw_data)
    path, fn_in, fn_csv, raw_data, fig_title, rows_plot = participants_list(path, subject, session, abt)
    if fn_csv == '':
        print(f'It could not find the selected subject. Please check the path, and the selected subject number in the list of participants.')
        return 0
    else:
        pass

    ##########################
    # printing basic information from data
    print(f'raw data filename: {fn_in}')
    print(f'annotations filename: {fn_csv}')
    print(f'raw data info:\n{raw_data.info}')
    print(f'title: {fig_title}')
    print(f'Only resting (1), resting and ABT (2): {rows_plot}')
    # printing basic information from data
    ############################
    ## global variable sampling rate
    sampling_rate = raw_data.info['sfreq']
    ############################
    ## read annotations (.csv file)
    # print(f'CSV file: {fn_csv}')
    my_annot = mne.read_annotations(path + fn_csv)
    print(f'annotations:\n{my_annot}')
    ## read annotations (.csv file)
    ############################
    ## adding annotations to raw data
    raw_data.set_annotations(my_annot)
    # print(raw_data.annotations)
    
    ############################
    ## high-pass filter
    raw_data.filter(l_freq=0.1, h_freq=None)

    ############################
    ## identify bad channels
    ## bad channels interpolation
    raw_data = bad_channels_interp(raw_data, subject, session)


    ## crop segments of raw data according to annotations
    eeg_data_dict = data_segmentation(raw_data)

    # print(f'eeg_data_dict:\n{eeg_data_dict}')

    ## scale selection
    scale_dict = dict(mag=1e-12, grad=4e-11, eeg=200e-6, eog=150e-6, ecg=300e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)
    # ############################
    ## signals visualization
    ## band pass filter (0.3 - 45 Hz) only for visualization
    a_opened_eyes_list = eeg_data_dict['a_opened_eyes']
    a_closed_eyes_list = eeg_data_dict['a_closed_eyes']
    b_opened_eyes_list = eeg_data_dict['b_opened_eyes']
    b_closed_eyes_list = eeg_data_dict['b_closed_eyes']

    topo_dict = {'contours':0}

    # signal filtering high-pass 1 Hz
    # filt_raw=[[]]*len(a_opened_eyes_list)
    # for id, raw in enumerate(a_opened_eyes_list):
    #     filt_raw[id] = raw.copy().filter(l_freq=1.0, h_freq=None)

    raw = a_opened_eyes_list[0]
    # raw = a_closed_eyes_list[0]
    filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)

    # visualization time vs channels
    mne.viz.plot_raw(raw, start=0, duration=240, scalings=scale_dict, highpass=None, lowpass=45.0, title='resting open eyes', block=False)

    ## ICA components
    ica = ICA(n_components= 0.99, method='fastica', max_iter="auto", random_state=97)
    ica.fit(filt_raw)
    print(f'ica:\n{ica}')

    explained_var_ratio = ica.get_explained_variance_ratio(filt_raw)
    for channel_type, ratio in explained_var_ratio.items():
        print(f"Fraction of {channel_type} variance explained by all components: {ratio}")
    
    # raw.load_data()
    ica.plot_sources(raw, show_scrollbars=False, block=False)
    ica.plot_components(inst=raw, contours=0)
    # blinks
    # ica.plot_overlay(raw, exclude=[2], picks="eeg")

    # plt.show(block=True)
    # return 0
    
    
    ica.exclude = [2]  # indices chosen based on various plots above

    # ica.apply() changes the Raw object in-place, so let's make a copy first:
    reconst_raw = raw.copy()
    ica.apply(reconst_raw)


    # raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)
    # reconst_raw.plot(order=artifact_picks, n_channels=len(artifact_picks), show_scrollbars=False)
    # del reconst_raw
    # visualization time vs channels
    mne.viz.plot_raw(reconst_raw, start=0, duration=240, scalings=scale_dict, highpass=None, lowpass=45.0, title='ica resting open eyes', block=False)

    # power spectrum density visualization
    # useful for bad-electrodes identification
    fig_psd, ax_psd = plt.subplots(2, 1, sharex=True, sharey=True)
    fig_psd.suptitle(fig_title)
    ax_psd = ax_psd.ravel()
    ax_psd[0].set_ylim([-30, 45])
    ## psd power spectral density
    ## spectrum closed eyes resting
    mne.viz.plot_raw_psd(raw, exclude=['VREF'], ax=ax_psd[0], fmax=180)
    mne.viz.plot_raw_psd(reconst_raw, exclude=['VREF'], ax=ax_psd[1], fmax=180)
    
    plt.show(block=True)
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
