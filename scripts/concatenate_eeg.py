import mne
import numpy as np
import sys

## include modules from another directory
sys.path.insert(0, '../../scripts')
from bad_channels import bad_channels_dict

def main(args):
    ## two eeg recoreded data same patient 
    ## time axis concatenation
    print(f'folder location: {args[1]}\n') ## folder location
    print(f'filename first raw eeg data: {args[2]}') ## file name .mff
    print(f'filename first annotations {args[3]}\n') ## file name raw_annotations.csv (first time) or annotations.fif (already edited the raw annotations)
    print(f'filename second raw eeg data: {args[4]}') ## file name .mff
    print(f'filename second annotations {args[5]}\n') ## file name raw_annotations.csv (first time) or annotations.fif (already edited the raw annotations)
    
    path=args[1]
    fn_raw_0 =args[2]
    fn_ann_0 =args[3]
    fn_raw_1 =args[4]
    fn_ann_1 =args[5]

    # print(f"path:{path}\nraw_data_0:{fn_raw_0}\nannotations_0:{fn_ann_0}\nraw_data_1:{fn_raw_1}\nannotations_1:{fn_ann_1}")
    ## open files raw eeg data
    raw_0 = mne.io.read_raw_egi(path + fn_raw_0, preload=True)
    raw_1 = mne.io.read_raw_egi(path + fn_raw_1, preload=True)
    ## open annotations
    annot_0 = mne.read_annotations(path + fn_ann_0)
    annot_1 = mne.read_annotations(path + fn_ann_1)

    ## adding annotations to eeg raw data
    raw_0.set_annotations(annot_0)
    raw_1.set_annotations(annot_1)

    ## removing event/annotations channels
    bads_list = ['__01','__02','__03','__04','__05']
    raw_0.info["bads"] = bads_list
    raw_0.drop_channels(raw_0.info['bads'])

    bads_list = ['__01','__02','__05']
    raw_1.info["bads"] = bads_list
    raw_1.drop_channels(raw_1.info['bads'])

    # print(f"channels 0: {raw_0.info['nchan']}")
    # print(f"channels 1: {raw_1.info['nchan']}")

    # print(f"channels 0: {raw_0.info['ch_names']}\n")
    # print(f"channels 1: {raw_1.info['ch_names']}\n")

    ##########################
    ## concatenation raw data
    mne.concatenate_raws([raw_0,raw_1],)

    ## visualization
    ############################
    ## visualization scale
    scale_dict = dict(mag=1e-12, grad=4e-11, eeg=100e-6, eog=150e-6, ecg=150e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)

    ## signals visualization (channels' voltage vs time)
    filt_raw_data = raw_0.copy().filter(l_freq=1.0, h_freq=45.0)
    # highpass=1.0, lowpass=45.0,
    fig = filt_raw_data.plot(start=0, duration=240, n_channels=32, scalings=scale_dict, block=True)
    ############################
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
