from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np
import mne
mne.set_log_level('error')

import math
from bad_channels import bad_channels_dict
from list_participants import participants_list
from selected_sequences import selected_sequences_dict

## global variables
sampling_rate = 1
raw_data_list = []
data_eeg = [[]]*2
cbar_ax = [[]]*2
im = [[]]*2
# y_limit = [0.4e-3, 0.4]
y_limit = [(None, None),(None, None)]
ax_topoplot = []
# channels=['EEG','Current source density']
channels=['eeg', 'eeg']
subject = 0


############################################
def data_segmentation(raw_data):
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


#############################
## plot power spectral density
def plot_spectrum(raw_data, ax_psd):
    ## cropping data according to annotations
    ## usually segments of each label have different duration
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

    ##########################
    # # power spectrum density visualization
    # # useful for bad-electrodes identification
    # fig_psd, ax_psd = plt.subplots(rows_plot, 2, sharex=True, sharey=True)
    # fig_psd.suptitle(fig_title)
    # ax_psd = ax_psd.ravel()
    # ax_psd[0].set_ylim([-30, 45])

    ax_number = 0
    ##########################
    if len(a_closed_eyes_list) > 0 and len(a_opened_eyes_list) > 0:
        # pre-processing selected segment: resting, closed- and opened-eyes
        print(f'subject: {subject}')
        section = 'a_closed_eyes'
        sequence = selected_sequences_dict[subject][section]
        print(f'sequence: {sequence}')
        # eeg_data_dict = set_bad_channels(eeg_data_dict, subject, section, sequence)

        ## spectrum closed eyes resting
        mne.viz.plot_raw_psd(eeg_data_dict[section][sequence], exclude=['VREF'], ax=ax_psd[ax_number], fmax=180)
        ax_psd[ax_number].set_title('rest closed eyes')
        ax_number+=1

        section = 'a_opened_eyes'
        sequence = selected_sequences_dict[subject][section]
        # eeg_data_dict = set_bad_channels(eeg_data_dict, subject, section, sequence)        

        ## spectrum opened eyes resting
        mne.viz.plot_raw_psd(eeg_data_dict[section][sequence], exclude=['VREF'], ax=ax_psd[ax_number], fmax=180)
        ax_psd[ax_number].set_title('rest opened eyes')
        ax_number+=1

    else:
        pass

    if len(b_closed_eyes_list) > 0 and len(b_opened_eyes_list) > 0:
        # pre-processing selected segment: biking, closed- and opened-eyes

        section = 'b_closed_eyes'
        sequence = selected_sequences_dict[subject][section]
        # eeg_data_dict = set_bad_channels(eeg_data_dict, subject, section, sequence)

        ## spectrum closed eyes ABT
        mne.viz.plot_raw_psd(eeg_data_dict[section][sequence], exclude=['VREF'], ax=ax_psd[ax_number], fmax=180)
        ax_psd[ax_number].set_title('ABT closed eyes')
        ax_number+=1

        section = 'b_opened_eyes'
        sequence = selected_sequences_dict[subject][section]
        # eeg_data_dict = set_bad_channels(eeg_data_dict, subject, section, sequence)

        ## spectrum opened eyes ABT 
        mne.viz.plot_raw_psd(eeg_data_dict[section][sequence], exclude=['VREF'], ax=ax_psd[ax_number], fmax=180)
        ax_psd[ax_number].set_title('ABT opened eyes')
        
    else:
        pass

    return 0


#############################
## Bad channels identification
def bad_channels_interp(raw_data, subject, session):
    ## include bad channels previously identified
    raw_data.info["bads"] = bad_channels_dict[subject]['session_'+str(session)]['general']
    print(f'bad channels: {raw_data.info["bads"]}')
    ## interpolate bad channels
    raw_data.interpolate_bads()
    return raw_data

#############################
## topographic views
def plot_topographic_view():
    global frame_slider, data_eeg, axfreq, cbar_ax, fig_topoplot, im
    ## spatial visualization (topographical maps)

    # Passband filter parameters
    low_cut =    0.3
    hi_cut  =   45.0

    init_frame = 0

    for id, raw_data in enumerate(raw_data_list):
        ## band-pass filter
        data = raw_data.copy().filter(l_freq=low_cut, h_freq=hi_cut, )
        print(f'data info:\n{data.info}')

        data_eeg[id] = data.get_data()
        print(f'data eeg: {data_eeg[id].shape}')

        im[id], cn = mne.viz.plot_topomap(data_eeg[id][:,init_frame], data.info, vlim=y_limit[id], contours=0, axes=ax_topoplot[id], cmap='magma')

    # Make room to place a horizontal slider that controls frame number of topographical maps
    axfreq = fig_topoplot.add_axes([0.25, 0.1, 0.65, 0.03])

    # Make colorbar for each topomap (before and after Surface Laplacian)
    cbar_ax[0] = fig_topoplot.add_axes([0.05, 0.25, 0.03, 0.65])
    cbar_ax[1] = fig_topoplot.add_axes([0.90, 0.25, 0.03, 0.65])

    ## colors for each bar
    fig_topoplot.colorbar(im[0], cax=cbar_ax[0])
    fig_topoplot.colorbar(im[1], cax=cbar_ax[1])

    ## max number of seconds (total number of samples of one channel [channel 0] divided by number of samples per second [sampling rate])
    max_val = len(data_eeg[id][0])/sampling_rate
    print(f'max val: {max_val}')
    frame_slider = Slider( ax=axfreq, label='Time [s]', valmin=0, valmax=max_val, valinit=int(init_frame/sampling_rate), )

    # register the update function with each slider
    frame_slider.on_changed(update)
    return 0


#############################
# The function to be called anytime a slider's value changes
def update(val):
    global fig_topoplot, im

    frame = math.floor(frame_slider.val*sampling_rate)
    print(f'Time: {frame_slider.val}, Sampling rate: {sampling_rate}, frame: {frame} !!!')
    
    for id, raw_data in enumerate(raw_data_list):
        ## update both topographical maps: raw data and current source density
        im[id], cn = mne.viz.plot_topomap(data_eeg[id][:,frame], raw_data.info, vlim=y_limit[id], contours=0, axes=ax_topoplot[id], cmap='magma')
        # colorbar
        fig_topoplot.colorbar(im[id], cax=cbar_ax[id])

    fig_topoplot.canvas.draw_idle()
    return 0


def main(args):
    global raw_data_list, fig_topoplot, ax_topoplot, sampling_rate, subject

    print(f'arg {args[1]}') ## folder location
    print(f'arg {args[2]}') ## subject = {0:Mme Chen, 1:Taha, 2:Carlie, 3:Iulia, 4:A. Caron}
    print(f'arg {args[3]}') ## session = {1:time zero, 2: three months}
    print(f'arg {args[4]}') ## ABT = {0:resting, 1:biking}
    
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

    raw_data = raw_data.pick(picks='eeg')

    ################################
    ## Stage 1: high pass filter (in place)
    #################################
    low_cut =    0.1
    hi_cut  =   None
    raw_data.filter(l_freq=low_cut, h_freq=hi_cut, picks='eeg')

    ################################
    ## Stage 2: bad channels interpolation
    #################################
    raw_data = bad_channels_interp(raw_data, subject, session)

    ################################
    ## Stage 3: data epochs based on annotations
    #################################
    ## crop segments of raw data according to annotations
    eeg_data_dict = data_segmentation(raw_data)

    a_opened_eyes_list = eeg_data_dict['a_opened_eyes']
    a_closed_eyes_list = eeg_data_dict['a_closed_eyes']
    b_opened_eyes_list = eeg_data_dict['b_opened_eyes']
    b_closed_eyes_list = eeg_data_dict['b_closed_eyes']

    ################################
    ## Stage 3: current source density calculation (surface Laplacian)
    #################################
    raw_csd = mne.preprocessing.compute_current_source_density(raw_data)
    
    ## run matplotlib in interactive mode
    plt.ion()

    # ############################
    ## signals visualization
    ## scale selection for visualization
    scale_dict = dict(mag=1e-12, grad=4e-11, eeg=200e-6, eog=150e-6, ecg=300e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)
    ## signals visualization    
    ## band pass filter (0.3 - 45 Hz) only for visualization
    mne.viz.plot_raw(raw_data, start=0, duration=240, scalings=scale_dict, highpass=0.3, lowpass=45.0, title=fig_title+' : Raw data', block=False)
    ## signals visualization    
    ## band pass filter (0.3 - 45 Hz) only for visualization
    mne.viz.plot_raw(raw_csd, start=0, duration=240, scalings=scale_dict, highpass=0.3, lowpass=45.0, title=fig_title+' : Surface Laplacian', block=False)

    ###########################
    # topographical maps to compare eeg voltages and eeg current source density frame by frame
    fig_topoplot = plt.figure(fig_title, figsize=(12, 5))
    ax_topoplot = fig_topoplot.subplots(1, 2, sharex=True, sharey=True)
    fig_topoplot.subplots_adjust(bottom=0.25)
    
    ax_topoplot[0].set_title('Raw data')
    ax_topoplot[1].set_title('Surface Laplacian')

    raw_data_list.extend([raw_data, raw_csd])
    print(f'raw data list: {len(raw_data_list)}')
    print(f'raw_data info:\n{raw_data_list[0].info}')
    print(f'raw_csd info:\n{raw_data_list[1].info}')
    
    ## topographical map; we apply band pass filter (0.3 - 45 Hz) only for visualization 
    plot_topographic_view()

    ###########################
    ## power spectral density
    fig_psd = [[]]*2
    ax_psd = [[]]*2
    fig_title_psd = ['Power spectral density : Raw data', 'Power spectral density : Surface Laplacian']
    for id, raw_data in enumerate(raw_data_list):
        # power spectrum density visualization
        # useful for data comparison
        fig_psd[id] = plt.figure(fig_title_psd[id])
        ax_psd[id] = fig_psd[id].subplots(rows_plot, 2, sharex=True, sharey=True)
        ## ravel: to transform a matrix of two indices to a vector of only one index
        ax_psd[id] = ax_psd[id].ravel()
        ax_psd[id][0].set_ylim([-30, 45])

        plot_spectrum(raw_data, ax_psd[id])

    


    plt.show(block=True)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
