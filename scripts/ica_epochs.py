import mne 
from mne.preprocessing import ICA
import matplotlib.pyplot as plt

#####################
## global variables
## scale selection for visualization raw data with annotations
scale_dict = dict(mag=1e-12, grad=4e-11, eeg=100e-6, eog=150e-6, ecg=400e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)

##################################
def read_ica_model(self):
    try:
        print(f"loading pre-calculated ICA model... ", end='')
        self.ica = mne.preprocessing.read_ica(self.filename_ica, verbose=None)
        print(f"done.")
        read_ica_flag = True
    except:
        print(f'Pre-calculated ICA was not found.')
        read_ica_flag = False

    return read_ica_flag

##################################
def display_psd_epochs2x(epochs_before, epochs_after):
        
        ## original epochs
        # epochs.plot(n_epochs=12, events=True, block=False, n_channels=36, scalings=self.scale_dict, title=f"Epochs {self.label_seg}_{self.id_seg} EEG time series",)

        # # ## we exclude VREF because it is the reference (0 volts all the time)
        fig_psd, ax_psd = plt.subplots(nrows=2, ncols=1, figsize=(9,4), sharey=True, sharex=True)
        # bads_list = self.raw_seg.info['bads']
        # bads_list = bads_list + epochs_before.info['bads']
        bads_list = epochs_before.info['bads']
        bads_list.append('VREF')
        epochs_before.plot_psd(picks=['eeg'], exclude=bads_list, ax=ax_psd[0], fmin=0, fmax=80, xscale='linear',)
        epochs_after.plot_psd(picks=['eeg'], exclude=bads_list, ax=ax_psd[1], fmin=0, fmax=80, xscale='linear',)
        ax_psd[0].set_title(f"PSD before ICA")
        ax_psd[1].set_title(f"PSD after ICA")
        
        return 0

##################################################
def ica_epochs_interactive(epochs, label):

    ## ica parameters to calculate ICA components
    ica = ICA(n_components= 0.99, method='picard', max_iter="auto", random_state=97)

    ## copy of raw data
    copy_epochs = epochs.copy()

    ## ica works better with clean (denoised) EEG signals with 0 offset (a high pass filter with a 1 Hz cutoff frequency could improve that condition, that is why we use the filtered version of the data [self.filt_seg])
    ################################
    ## Stage 1: passband and notch filters, and resampling
    low_cut =    1.0
    hi_cut  =   45.0

    # filter applied in place
    print(f"Band-pass filter before ICA {low_cut, hi_cut} Hz...")
    copy_epochs.filter(l_freq=low_cut, h_freq=hi_cut, picks='eeg')

    flag_ica = 1

    ## re-calculate ICA components    
    while flag_ica==1 :
        ## ICA fitting model to the filtered raw data
        
        print(f"creating an ICA model...")
        ica.fit(copy_epochs, reject_by_annotation=True)
        ica.exclude = []
        
        # interactive selection of ICA components to exclude
        ica.plot_sources(epochs, start=None, stop=None, show_scrollbars=False, show=True, title=f"{label} -- ICA components", block=False)

        print(f"Ploting ICA components...")
        ## plot_components shows 2D-topomaps of the ICA components
        ica.plot_components(inst=epochs, contours=0, show=True, title=f"epochs {label} -- ICA components")
        # self.save_fig_ica_comp(fig_ica_comp)
        print(f"ica excluded components: {ica.exclude}")

        
        ## save selected ICA components to exclude
        # self.save_ica_excluded_comp()

        ############
        ## visual comparison before and after ICA
        ## data visualization EEG         
        ## psd
        epochs_before_ica = epochs.copy()
        ## apply ICA to a copy of the original epochs
        ica.apply(epochs)

        display_psd_epochs2x(epochs_before_ica, epochs)
        ## original epochs
        epochs_before_ica.plot(n_epochs=12, events=True, block=False, n_channels=36, scalings=scale_dict, title=f"Epochs {label} before ICA",)
        
        ## results of ICA after components exclusion
        epochs.plot(n_epochs=12, events=True, block=True, n_channels=36, scalings=scale_dict, title=f"Epochs {label} after ICA",)
        
        
        ## visual comparison before and after ICA
        ############

        ## update ica calculation flag
        option_ica = int(input(f"0: Save the current model\n1: Modify list of exclusion ICA components\n ?: "))
        # option_ica = 0 if (flag_ica == '') else int(flag_ica)
        if option_ica==0:
            ## choosing zero the loop is finished to apply the ICA model to the epochs in place
            ## save ICA model and excluded components
            # self.save_ica_model()
            ## break the loop
            flag_ica = 0
        else:
            ## choosing one: revisiting ICA components for any modification
            recal_ica_flag=False
            ## keep in the loop
            flag_ica = 1
        
        print(f"continuous loop: {flag_ica}")


    ## Applying ICA to epochs in place
    ica.apply(epochs)

    return epochs

