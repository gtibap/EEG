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

#############################
## Bad channels identification
def set_bad_channels(data_dict, subject, section, sequence):
    ## EEG signals of selected section (a_opened_eyes, a_closed_eyes, b_opened_eyes, b_closed_eyes)
    raw_cropped = data_dict[section][sequence]
    ## include bad channels previously identified
    raw_cropped.info["bads"] = bad_channels_dict[subject][section][sequence]
    ## interpolate only selected bad channels; exclude bad channels that are in the extremes or those who do not have good number of surounding good channels
    ch_excl_interp = bad_channels_dict[subject][section+'_excl'][sequence]
    raw_cropped.interpolate_bads(exclude=ch_excl_interp)
    ## re-referencing average (this technique is good for dense EEG)
    # data_dict[section][sequence] = raw_cropped.set_eeg_reference("average",ch_type='eeg',)
    data_dict[section][sequence] = raw_cropped
    return data_dict

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
        print(f'id: {id}')
        print(f'raw_data: {raw_data.pick_types}')
        data = raw_data.copy().filter(l_freq=low_cut, h_freq=hi_cut, )
        print(f'data info:\n{data.info}')
        # picks='eeg'
        data_eeg[id] = data.get_data()
        print(f'data eeg: {data_eeg[id].shape}')
        # picks=['eeg']
        # df_eeg = data.to_data_frame(picks=['eeg'], index='time')
        # print(f'shape data:\n{data_eeg.shape}\n{data_eeg}')
        # print(f'dataframe data:\n{df_eeg}')

        # vlim=(-y_limit[id], y_limit[id]),
        im[id], cn = mne.viz.plot_topomap(data_eeg[id][:,init_frame], data.info, vlim=y_limit[id], contours=0, axes=ax_topoplot[id], cmap='magma')

    # Make a horizontal slider to control the frequency.
    axfreq = fig_topoplot.add_axes([0.25, 0.1, 0.65, 0.03])

    # Make colorbar
    cbar_ax[0] = fig_topoplot.add_axes([0.05, 0.25, 0.03, 0.65])
    cbar_ax[1] = fig_topoplot.add_axes([0.90, 0.25, 0.03, 0.65])

    fig_topoplot.colorbar(im[0], cax=cbar_ax[0])
    fig_topoplot.colorbar(im[1], cax=cbar_ax[1])
    # clb.ax.set_title("topographic view",fontsize=16) # title on top of colorbar
    # fig_topoplot.add_axes([0.25, 0.1, 0.65, 0.03])
    # valmin=0, valmax=len(df_eeg)/sampling_rate,
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
    print(f'!!! update frame: {frame_slider.val}, {sampling_rate}, {frame} !!!')
    # print(f'!!! data_eeg: {data_eeg.shape} !!!')
    # y_limit = 0.4e-3
    
    for id, raw_data in enumerate(raw_data_list):
        # vlim=(-y_limit[id], y_limit[id]),
        im[id], cn = mne.viz.plot_topomap(data_eeg[id][:,frame], raw_data.info, vlim=y_limit[id], contours=0, axes=ax_topoplot[id], cmap='magma')
        # colorbar
        fig_topoplot.colorbar(im[id], cax=cbar_ax[id])

    fig_topoplot.canvas.draw_idle()
    return 0


def main(args):
    global raw_data_list, fig_topoplot, ax_topoplot, sampling_rate

    print(f'arg {args[1]}') ## folder location
    print(f'arg {args[2]}') ## subject = {0:Mme Chen, 1:Taha, 2:Carlie, 3:Iulia, 4:A. Caron}
    print(f'arg {args[3]}') ## ABT = {0:resting, 1:biking}
    
    path=args[1]
    subject= int(args[2])
    abt= int(args[3])

    fn_in=''
    t0=0
    t1=0

    #########################
    ## new path, eeg filename (fn_in), annotations filename (fn_csv), eeg raw data (raw_data)
    path, fn_in, fn_csv, raw_data, fig_title, rows_plot = participants_list(path, subject, abt)
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

    # Passband filter in place
    low_cut =    0.3
    hi_cut  =   None
    raw_data.filter(l_freq=low_cut, h_freq=hi_cut, picks='eeg')

    ## set bad channels
    # section = 'a_closed_eyes'
    # sequence = selected_sequences_dict[subject][section]
    raw_data.info["bads"] = ['E119']
    raw_data.interpolate_bads()
    # eeg_data_dict = set_bad_channels(eeg_data_dict, subject, section, sequence)

    raw_csd = mne.preprocessing.compute_current_source_density(raw_data)
    
    ## run matplotlib in interactive mode
    plt.ion()

    ############################
    # ############################
    ## signals visualization
    ## band pass filter (0.3 - 45 Hz) only for visualization

    ## scale selection
    scale_dict = dict(mag=1e-12, grad=4e-11, eeg=200e-6, eog=150e-6, ecg=300e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)
    ## signals visualization    
    mne.viz.plot_raw(raw_data, start=0, duration=240, scalings=scale_dict, highpass=0.3, lowpass=45.0, title=fig_title+' : Raw data', block=False)

    ## signals visualization    
    mne.viz.plot_raw(raw_csd, start=0, duration=240, scalings=scale_dict, highpass=0.3, lowpass=45.0, title=fig_title+' : Surface Laplacian', block=False)


    # adjust the main plot to make room for the sliders
    fig_topoplot = plt.figure(fig_title, figsize=(12, 5))
    ax_topoplot = fig_topoplot.subplots(1, 2, sharex=True, sharey=True)
    # fig_topoplot, ax_topoplot = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    fig_topoplot.subplots_adjust(bottom=0.25)
    # fig_topoplot.suptitle(fig_title)
    
    
    ax_topoplot[0].set_title('Raw data')
    ax_topoplot[1].set_title('Surface Laplacian')

    sampling_rate = raw_data.info['sfreq']

    raw_data_list.extend([raw_data, raw_csd])
    print(f'raw data list: {len(raw_data_list)}')
    print(f'raw_data info:\n{raw_data_list[0].info}')
    print(f'raw_csd info:\n{raw_data_list[1].info}')
    ###########################
    ## topographical map; we apply band pass filter (0.3 - 45 Hz) only for visualization 
    plot_topographic_view()



    plt.show(block=True)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
