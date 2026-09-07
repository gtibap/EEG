import mne 
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
import json

#####################
## global variables
## scale selection for visualization raw data with annotations
scale_dict = dict(mag=1e-12, grad=4e-11, eeg=100e-6, eog=150e-6, ecg=400e-6, emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4, whitened=1e2)

##################################
def display_psd_epochs(epochs):
        
        ## original epochs
        # epochs.plot(n_epochs=12, events=True, block=False, n_channels=36, scalings=self.scale_dict, title=f"Epochs {self.label_seg}_{self.id_seg} EEG time series",)

        # # ## we exclude VREF because it is the reference (0 volts all the time)
        fig_psd, ax_psd = plt.subplots(nrows=1, ncols=1, figsize=(6,4), sharey=True, sharex=True)
        # bads_list = self.raw_seg.info['bads']
        # bads_list = bads_list + epochs_before.info['bads']
        bads_list = epochs.info['bads']
        bads_list.append('VREF')
        epochs.plot_psd(picks=['eeg'], exclude=bads_list, ax=ax_psd, fmin=0.1, fmax=65, xscale='linear', dB=True, estimate='power',)
        ax_psd.set_title(f"PSD before ICA")
        ax_psd.set_xlabel(f"frequency (Hz)")
        
        return 0

##################################
def display_psd_epochs2x(epochs_before, epochs_after):
        
        ## original epochs
        # epochs.plot(n_epochs=12, events=True, block=False, n_channels=36, scalings=self.scale_dict, title=f"Epochs {self.label_seg}_{self.id_seg} EEG time series",)

        # # ## we exclude VREF because it is the reference (0 volts all the time)
        fig_psd, ax_psd = plt.subplots(nrows=2, ncols=1, figsize=(6,8), sharey=True, sharex=True)
        # bads_list = self.raw_seg.info['bads']
        # bads_list = bads_list + epochs_before.info['bads']
        bads_list = epochs_before.info['bads']
        bads_list.append('VREF')
        epochs_before.plot_psd(picks=['eeg'], exclude=bads_list, ax=ax_psd[0], fmin=0.1, fmax=65, xscale='linear', dB=True, estimate='power',)
        epochs_after.plot_psd(picks=['eeg'], exclude=bads_list, ax=ax_psd[1], fmin=0.1, fmax=65, xscale='linear', dB=True, estimate='power',)
        ax_psd[0].set_title(f"PSD before ICA")
        ax_psd[1].set_title(f"PSD after ICA")
        ax_psd[1].set_xlabel(f"frequency (Hz)")
        
        return 0

##################################################
def ica_epochs_interactive(epochs, label, root_filename):
    ## ICA fitting model to the filtered epochs data

    ## copy of epochs 
    copy_epochs = epochs.copy()

    ## ica parameters to calculate ICA components
    ica = ICA(n_components= 0.99, method='picard', max_iter="auto", random_state=97)

    ## ica works better with clean (denoised) EEG signals with 0 offset (a high pass filter with a 1 Hz cutoff frequency could improve that condition, that is why we use the filtered version of the data [self.filt_seg])
    ################################
    ## Stage 1: passband and notch filters, and resampling
    low_cut =    1.0
    hi_cut  =   45.0
    # filter applied in place
    print(f"Band-pass filter before ICA {low_cut, hi_cut} Hz...")
    copy_epochs.filter(l_freq=low_cut, h_freq=hi_cut, picks='eeg')

    display_psd_epochs(copy_epochs)

    print(f"creating an ICA model...")
    ica.fit(copy_epochs, reject_by_annotation=True)

    ## loop to re-evaluate excluded epochs 
    flag_ica = 1
    while flag_ica==1 :

        # print(f"Ploting ICA components...")
        # ## plot_components shows 2D-topomaps of the ICA components
        # ica.plot_components(picks=None, inst=copy_epochs, contours=0, show=True, title=f"epochs {label} -- ICA components")

        # interactive selection of ICA components to exclude
        print(f"Ploting ICA sources...")
        ica.plot_sources(copy_epochs, picks=None, start=0, stop=36, show_scrollbars=False, show=True, title=f"{label} -- ICA components", block=True)

        # self.save_fig_ica_comp(fig_ica_comp)
        print(f"ica excluded components: {ica.exclude}")

        ############
        ## visual comparison before and after ICA
        epochs_before_ica = copy_epochs.copy()
        epochs_after_ica  = copy_epochs.copy()
        ## apply ICA to a copy of the original epochs to observe ICA effects in place
        ica.apply(epochs_after_ica)

        ## psd comparison before and after ICA
        display_psd_epochs2x(epochs_before_ica, epochs_after_ica)
        ## time-series comparison before and after ICA
        epochs_before_ica.plot(n_epochs=36, events=True, block=False, n_channels=36, scalings=scale_dict, title=f"Epochs {label} before ICA",)
        
        ## results of ICA after components exclusion
        epochs_after_ica.plot(n_epochs=36, events=True, block=True, n_channels=36, scalings=scale_dict, title=f"Epochs {label} after ICA",)
        
        
        ## visual comparison before and after ICA
        ############

        ## update ica calculation flag
        option_ica = int(input(f"0: Save the current model\n1: Modify list of exclusion ICA components\n ?: "))
        # option_ica = 0 if (flag_ica == '') else int(flag_ica)
        if option_ica==0:
            ## choosing zero the loop is finished to apply the ICA model to the epochs in place
            ## break the loop
            flag_ica = 0
        else:
            ## choosing one: revisiting ICA components and apply modifications if necessary
            ## keep in the loop
            flag_ica = 1
        
    ## Applying ICA to epochs in place
    ica.apply(epochs)
    ## save ICA model
    ica.save(f"{root_filename}-ica.fif.gz", overwrite=True)
    ## save excluded ICA components
    with open(f"{root_filename}-ica_excluded_comp.json", "w") as f:
        json.dump(ica.exclude, f)

    return epochs

##################################
def read_ica_model(epochs, label, root_filename):

    ## copy of epochs 
    copy_epochs = epochs.copy()

    print(f"loading pre-calculated ICA model... ", end='')
    ica = mne.preprocessing.read_ica(f"{root_filename}-ica.fif.gz", verbose=None)
    with open(f"{root_filename}-ica_excluded_comp.json", "r") as f:
        ica_excl_list = json.load(f)
    ica.exclude = ica_excl_list
    print(f"done.")

    ############
    ## visual comparison before and after ICA
    epochs_before_ica = copy_epochs.copy()
    epochs_after_ica  = copy_epochs.copy()
    ## apply ICA to a copy of the original epochs to observe ICA effects in place
    ica.apply(epochs_after_ica)
    ## psd comparison before and after ICA
    display_psd_epochs2x(epochs_before_ica, epochs_after_ica)

    flag_ica = int(input(f"Update list of excluded ICA components? (0/1): "))
    ## loop to re-evaluate excluded epochs 
    while flag_ica:

        # print(f"Ploting ICA components...")
        # ## plot_components shows 2D-topomaps of the ICA components
        # ica.plot_components(picks=None, inst=copy_epochs, contours=0, show=True, title=f"epochs {label} -- ICA components")

        # interactive selection of ICA components to exclude
        print(f"Ploting ICA sources...")
        ica.plot_sources(copy_epochs, picks=None, start=0, stop=36, show_scrollbars=False, show=True, title=f"{label} -- ICA components", block=True)

        # self.save_fig_ica_comp(fig_ica_comp)
        print(f"ica excluded components: {ica.exclude}")

        ############
        ## visual comparison before and after ICA
        epochs_before_ica = copy_epochs.copy()
        epochs_after_ica  = copy_epochs.copy()
        ## apply ICA to a copy of the original epochs to observe ICA effects in place
        ica.apply(epochs_after_ica)

        ## psd comparison before and after ICA
        display_psd_epochs2x(epochs_before_ica, epochs_after_ica)
        ## time-series comparison before and after ICA
        epochs_before_ica.plot(n_epochs=36, events=True, block=False, n_channels=36, scalings=scale_dict, title=f"Epochs {label} before ICA",)
        
        ## results of ICA after components exclusion
        epochs_after_ica.plot(n_epochs=36, events=True, block=True, n_channels=36, scalings=scale_dict, title=f"Epochs {label} after ICA",)
        
        
        ## visual comparison before and after ICA
        ############

        ## update ica calculation flag
        option_ica = int(input(f"0: Save the current model\n1: Modify list of exclusion ICA components\n ?: "))
        # option_ica = 0 if (flag_ica == '') else int(flag_ica)
        if option_ica==0:
            ## choosing zero the loop is finished to apply the ICA model to the epochs in place
            ## save excluded ICA components
            with open(f"{root_filename}-ica_excluded_comp.json", "w") as f:
                json.dump(ica.exclude, f)
            ## break the loop
            flag_ica = 0
        else:
            ## choosing one: revisiting ICA components and apply modifications if necessary
            ## keep in the loop
            flag_ica = 1

    ## Applying ICA to epochs in place
    ica.apply(epochs)

    return epochs


