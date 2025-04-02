import mne
import numpy as np
import matplotlib.pyplot as plt

from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
from list_participants import participants_list

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
        
        print(f'annotation:{count1, onset, duration, label}')

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

def main(args):

    print(f'folder location: {args[1]}') ## folder location
    print(f'subject: {args[2]}') ## subject = {0:Mme Chen, 1:Taha, 2:Carlie, 3:Iulia, 4:A. Caron}
    print(f'ABT: {args[3]}') ## ABT = {0:resting, 1:biking}
    
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
    ## crop segments of raw data according to annotations
    eeg_data_dict = data_segmentation(raw_data)

    print(f'eeg_data_dict:\n{eeg_data_dict}')

    ## scale selection
    scale_dict = dict(mag=1e-12, grad=4e-11, eeg=200e-6, eog=150e-6, ecg=300e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)
    # ############################
    ## signals visualization
    ## band pass filter (0.3 - 45 Hz) only for visualization
    a_opened_eyes_list = eeg_data_dict['a_opened_eyes']

    topo_dict = {'contours':0}

    # eog_evoked = create_eog_epochs(a_opened_eyes_list[0], ch_name=['E8','E14','E17','E21','E25'],).average()
    # eog_evoked.apply_baseline(baseline=(None, -0.2))
    # eog_evoked.plot_joint(topomap_args=topo_dict)

    ecg_evoked = create_ecg_epochs(a_opened_eyes_list[0], ch_name='ECG').average()
    ecg_evoked.apply_baseline(baseline=(None, -0.2))
    ecg_evoked.plot_joint(topomap_args=topo_dict)

    mne.viz.plot_raw(a_opened_eyes_list[0], start=0, duration=240, scalings=scale_dict, highpass=1.0, lowpass=45.0, title=fig_title, block=True)

    plt.show()

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
