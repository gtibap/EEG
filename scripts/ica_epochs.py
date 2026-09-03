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
        epochs_before.plot_psd(picks=['eeg'], exclude=bads_list, ax=ax_psd[0], fmin=0, fmax=80, xscale='log',)
        epochs_after.plot_psd(picks=['eeg'], exclude=bads_list, ax=ax_psd[1], fmin=0, fmax=80, xscale='log',)
        ax_psd[0].set_title(f"PSD before ICA")
        ax_psd[1].set_title(f"PSD after ICA")
        
        return 0

##################################################
def ica_epochs_interactive(epochs, label, flag_update_ica):

    ## ica parameters to calculate ICA components
    ica = ICA(n_components= 0.99, method='picard', max_iter="auto", random_state=97)

    # recal_ica_flag = int(input(f"Do you want to (re)calculate an ICA model (yes 1, no 0) ?: "))
    flag_update_ica = 1
    recal_ica_flag = 1

    if flag_update_ica:
        flag_ica = 1
        ## re-calculate ICA components    
        while flag_ica==1 :
            ## copy of raw data
            copy_epochs = epochs.copy()
            ## display psd eeg raw data
            # self.display_psd_epochs(epochs)
            
                            
            if recal_ica_flag==False:
                print(f"reading the previous ICA model...")
                # self.read_ica_model()
                # self.read_ica_excluded_comp()
            else:
                ## ica works better with clean (denoised) EEG signals with 0 offset (a high pass filter with a 1 Hz cutoff frequency could improve that condition, that is why we use the filtered version of the data [self.filt_seg])
                ## ICA fitting model to the filtered raw data
                print(f"creating an ICA model...")
                ica.fit(epochs, reject_by_annotation=True)
                ica.exclude = []
                ## save ICA model and excluded components
                # self.save_ica_model()
            
            print(f"Ploting ICA components...")
            ## plot_components shows 2D-topomaps of the ICA components
            fig_ica_comp = ica.plot_components(inst=epochs, contours=0, show=True, title=f"epochs {label} -- ICA components")
            # self.save_fig_ica_comp(fig_ica_comp)

            # interactive selection of ICA components to exclude
            ica.plot_sources(epochs, start=None, stop=None, show_scrollbars=False, show=True, title=f"{label} -- ICA components", block=True)
            print(f"ica excluded components: {ica.exclude}")
            ## save selected ICA components to exclude
            # self.save_ica_excluded_comp()

            ############
            ## visual comparison before and after ICA
            ## data visualization EEG         
            ## psd
            ## apply ICA to a copy of the original epochs
            ica.apply(copy_epochs)
            display_psd_epochs2x(epochs, copy_epochs)
            ## original epochs
            epochs.plot(n_epochs=12, events=True, block=False, n_channels=36, scalings=scale_dict, title=f"Epochs {label} (EEG) Before ICA",)
            ## results of ICA after components exclusion
            copy_epochs.plot(n_epochs=12, events=True, block=True, n_channels=36, scalings=scale_dict, title=f"Epochs {label} (EEG) After ICA",)
            
            ## visual comparison before and after ICA
            ############

            # ## update ica calculation flag
            # option_ica = int(input(f"0: Save the current model\n1: Re-calculate ICA components\n2: Redefine list of exclusion ICA components\n ?: "))
            # # option_ica = 0 if (flag_ica == '') else int(flag_ica)
            # if option_ica==1:
            #     ## update bad channels interactively
            #     fig_raw = epochs.plot(n_epochs=21, events=True, block=False, n_channels=36, scalings=scale_dict, title=f"Epochs {label} (EEG) Before ICA",)
            #     update_channels_bads()
            #     ## save on disk bad channels
            #     self.save_channels_bads()
            #     ## activate flag to recalculate ICA components
            #     recal_ica_flag=True
            #     ## keep in the loop
            #     flag_ica = 1
            # elif option_ica==2:
            #     ## same ICA model but choosing other components to exclude
            #     ## does not recalculate ICA components
            #     recal_ica_flag=False
            #     ## keep in the loop
            #     flag_ica = 1
            # else:
            #     ## choosing zero le loop is finished to apply the ICA model to the epochs in place
            #     ## break the loop
            #     flag_ica = 0

            # print(f"continuous loop: {flag_ica}")
    else:
        print(f"ending...")
        return 0
        print(f"reading the previous ICA model...")
        self.read_ica_model()
        self.read_ica_excluded_comp()
        ## selected ICA components to exclude
        self.ica_exclude = self.ica.exclude
        print(f"ica excluded components: {self.ica_exclude}")

    ## Applying ICA to epochs in place
    # ica.apply(epochs)

    ## save plot ica sources
    # self.save_plot_ica_sources_epochs()

    return 0
## ica epochs components_interactive()
