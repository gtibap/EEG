from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.widgets as mwidgets
import numpy as np
import pandas as pd
import sys
import json

##### global variables
obj_list = []
freq_range = [1.0, 30.0]
fig_a = []
fig_ap = []
fig_ce = []
fig_oe = []
fig_mea = []
ax_a = []
ax_ap = []
ax_mea = []
selectors = []
flag_peaks_global = False
thr_peaks_global = 1.0
region_global = ''
f0_global =  0.5  # Hz
f1_global = 30.0  # Hz
ylim = [0, 40] # dB
event_list_ce = ['a_ce','b_ce','c_ce']
event_list_oe = ['a_oe','b_oe','c_oe']
path=''
info_p =''
obj_global = []
region_global = ''

##############################
def psd_regions_visualization(obj_list_ref, info_pt_ref, path_root, flag_save):
    global obj_list, freq_range, ax_ce_global, fig_ce_global, flag_eyes_closed, fig_ce, fig_oe, fig_mea, ax_ce, ax_oe, ax_mea, path, info_p
    ##
    obj_list = obj_list_ref.copy()
    path = path_root
    info_p = info_pt_ref

    ## ids to define a subplot order for ax_ce and ax_oe
    ax_ce_dict = {'a_ce':0, 'b_ce':2, 'c_ce':4}
    ax_oe_dict = {'a_oe':0, 'b_oe':2, 'c_oe':4}

    ## how many rows in the figures depends of how many segments were recorded
    ## for closed and open eyes
    sum_ce=0
    sum_oe=0
    ## count number of selected segments [a_ce, a_oe, ...]
    for obj in obj_list:
        ## find the selected segment for each label
        if ('ce' in obj.get_label_simple()):
            sum_ce+=1
        elif ('oe' in obj.get_label_simple()):
            sum_oe+=1
        else:
            pass
    
    ## the figures would include one or more of the following: resting before cycling, cycling, and resting after cycling. The columns would present responses of the left and right sides
    fig_ce, ax_ce = plt.subplots(sum_ce, 2, sharex=True, sharey=True, figsize=(12,6))
    fig_oe, ax_oe = plt.subplots(sum_oe, 2, sharex=True, sharey=True, figsize=(12,6))
    ax_ce = ax_ce.flatten()
    ax_oe = ax_oe.flatten()

    for obj in obj_list:
        ## get the label of the selected object [a_ce, a_oe, ...]
        label_eyes = obj.get_label_simple()
        ## separate closed eyes and open eyes
        if 'ce' in label_eyes:
            ## closed eyes [a_ce, b_ce, c_ce]
            id_ax = ax_ce_dict[label_eyes]
            region ='central_left'
            obj.calculate_average_psd_model(region, ax_ce[id_ax])
            region ='central_right'
            obj.calculate_average_psd_model(region, ax_ce[id_ax+1])
        else:
            ## open eyes [a_oe, b_oe, c_oe]
            id_ax = ax_oe_dict[label_eyes]
            region ='central_left'
            obj.calculate_average_psd_model(region, ax_oe[id_ax])
            region ='central_right'
            obj.calculate_average_psd_model(region, ax_oe[id_ax+1])

    
    ## ax limits, closed eyes, open eyes
    ax_ce[0].set_ylim(ylim[0], ylim[1])
    ax_oe[0].set_ylim(ylim[0], ylim[1])

    ax_ce[0].set_xlim(freq_range[0]-1, freq_range[1]+1)
    ax_oe[0].set_xlim(freq_range[0]-1, freq_range[1]+1)

    ## labels x axes
    ax_ce[-2].set_xlabel(f'frequency [Hz]')
    ax_ce[-1].set_xlabel(f'frequency [Hz]')
    ax_oe[-2].set_xlabel(f'frequency [Hz]')
    ax_oe[-1].set_xlabel(f'frequency [Hz]')


    fig_ce.suptitle(f'{info_p}\nEYES CLOSED')
    fig_oe.suptitle(f'{info_p}\nEYES OPEN')

    if flag_save:
        fig_ce.savefig(path+'psd/psd_ce.png',bbox_inches='tight')
        fig_oe.savefig(path+'psd/psd_oe.png',bbox_inches='tight')
    else:
        pass

    ###############
    ## mouse, and keyboard interactions with figures and plots
    ## run actions described on on_click once the mouse's left-click is pressed over the figure EYES CLOSED
    # fig_ce_global.canvas.mpl_connect('button_press_event', on_click)
    fig_ce.canvas.mpl_connect('button_press_event', on_click)
    fig_oe.canvas.mpl_connect('button_press_event', on_click)

    return 0

######################################
def on_click(event):
    global ax_index, flag_eyes_closed, fig_title_mea, ax_mea

    # ax_copy = np.copy(ax_ce_global)

    # print(f"onclick event.inaxes: {event.inaxes}")
    ## is the mouse left-button pressed ?
    if (event.button is MouseButton.LEFT):
        # print(f"button left")
        ##
        # print(f"event.canvas.figure: {event.canvas.figure}")
        # print(f"fig_ce: {fig_ce}")
        # print(f"fig_oe: {fig_oe}")

        if event.canvas.figure == fig_ce:
            flag_eyes_closed = True
            ax_copy = np.copy(ax_ce)
            fig_title_mea = f"EYES CLOSED"
            print(f"flag_eyes_closed: {flag_eyes_closed}")
        elif event.canvas.figure == fig_oe:
            flag_eyes_closed = False
            ax_copy = np.copy(ax_oe)
            fig_title_mea = f"EYES OPEN"
            print(f"flag_eyes_closed: {flag_eyes_closed}")
        else:
            print(f"flag_eyes_closed: not found")
            return 0
    
        # print(f"event.inaxes: {event.inaxes}")
        ## is the mouse over any subplot?
        if event.inaxes in ax_copy:
            ## which subplot?
            ax_index = np.argwhere(event.inaxes == ax_copy)[0][0]
            ## subplot index
            print(f"selected ax: {ax_index}")
            ## open a new window with the signals of the selected subplot
            # signal_measurements(ax_index, flag_eyes_closed, fig_title_mea)
            plot_quantiles(ax_index, flag_eyes_closed, fig_title_mea)
            ax_mea[1].cla()
            fit_fooof_psd()
        else:
            pass
            # print(f"event.inaxes out of ax")
    else:
        pass
        # print(f"other button")

    return 0

##############################################
def onselect(vmin, vmax):
    global f0_global, f1_global
    print(vmin, vmax)

    f0_global = vmin
    f1_global = vmax

    return 0

####################################
# def signal_measurements(ax_index, flag_eyes_closed, fig_title):
def plot_quantiles(ax_index, flag_eyes_closed, fig_title):
    ## open a new figure and plot graphical info of the selected subplot 
    global fig_mea, ax_mea, emg_list, selectors, df_psd_global, obj_global, region_global

    ## legend's components
    gray_patch = mpatches.Patch(color='tab:gray', alpha=0.5, label=f'Q3-Q1\ninterquantil\nrange')
    q2_line  = mlines.Line2D([], [], color='tab:blue', label='Q2 (median)')

    ## create a figure (first time) or clean it to update it
    if fig_mea == []:
        ## creates a figure to plot the selected stimulation responses 
        print(f"creates a figure to plot the selected stimulation responses...")
        n_rows = 2
        n_cols = 1
        fig_mea, ax_mea = plt.subplots(n_rows, n_cols, sharex=True, figsize=(9*n_cols, 4*n_rows))
        ## interactive selection freq range
        span = mwidgets.SpanSelector(ax_mea[0], onselect, 'horizontal', interactive=True, useblit=True, props=dict(facecolor='blue', alpha=0.2))
        selectors.append(span)
        ## keyboard interaction
        fig_mea.canvas.mpl_connect('key_press_event', on_press)
        # Creating legend with color box
    else:
        print(f"Cleaning subplot fig. quantiles...")
        ax_mea[0].cla()

    ## subplots graphics order; each row one state: a_, b_, or c_; left columns for channels left side, right columns for channels right side
    ax_ce_dict_global = {0:'a_ce', 1:'a_ce', 2:'b_ce', 3:'b_ce', 4:'c_ce', 5:'c_ce'}
    ax_oe_dict_global = {0:'a_oe', 1:'a_oe', 2:'b_oe', 3:'b_oe', 4:'c_oe', 5:'c_oe'}

    ## selected subplot state
    if flag_eyes_closed:
        sel_label = ax_ce_dict_global[ax_index]
    else:
        sel_label = ax_oe_dict_global[ax_index]
    ## selected region
    ## odd or even (left or right head side)
    if ax_index % 2 == 0:
        # even
        region ='central_left'
    else:
        # odd
        region ='central_right'

    print(f"region: {region}")

    fig_title = fig_title + ' ' + region + ' side'
    
    ## ax limits, closed eyes, open eyes
    # ax_mea[0].set_ylim(ylim_global[0], ylim_global[1])
    ax_mea[0].set_xlim(freq_range[0]-1, freq_range[1]+1)
    fig_mea.suptitle(f"{fig_title}")

    ## search obj according to the selected plot    
    for obj in obj_list:
        ## get the label of the selected object [a_ce, a_oe, ...]
        label_eyes = obj.get_label_simple()
        ## separate closed eyes and open eyes
        # print(f"sel_label, label_eyes: {sel_label, label_eyes}")
        if sel_label in label_eyes:
            ## closed eyes [a_ce, b_ce, c_ce]
            ## plot a graphical representation of the PSD of the selected subplot
            print("figure quantiles...")
            ax_mea[0] = obj.plot_psd_quantiles(region, ax_mea[0])
            ## obj global
            obj_global = obj
            region_global = region
            break

    fig_mea.legend(handles=[gray_patch, q2_line], loc="upper right") ## loc="outside right upper"

    return 0

###################
def fit_fooof_psd():
    global ax_seg, fig_seg, fig_mea, ax_mea, fig_peaks, ax_peaks, selectors, flag_peaks_global, obj_global
    
    ## fit a fooof model: periodic and aperiodic components
    range_freqs = [f0_global, f1_global]

    if flag_peaks_global == False:
        ## peaks' threshold manually selected (mouse interaction)
        span_peaks = mwidgets.SpanSelector(ax_mea[1], onselect_peaks, 'vertical', interactive=True, useblit=True, props=dict(facecolor='tab:green', alpha=0.2))
        selectors.append(span_peaks)
        flag_peaks_global = True
    else:
        ax_mea[1].cla()

    ## fit the fooof model
    print(f"threshold peaks: {thr_peaks_global}")
    obj_global.fit_fooof(region_global, range_freqs, thr_peaks_global, ax_mea[1])

    ## threshold line to include and exclude peaks for the gaussian model fitting (fooof)
    ax_mea[1].axhline(y=thr_peaks_global, xmin=-10, xmax=100, ls='--', lw=1.0, color='tab:blue')

    # (Re)Plot curves of quantiles from the PSD of the selected channels
    # signal_measurements(ax_index, flag_eyes_closed, fig_title_mea)
    plot_quantiles(ax_index, flag_eyes_closed, fig_title_mea)
    
    fm = obj_global.get_fooof_model(region_global)
    # plot_spectra(fm.freqs, fm.fooofed_spectrum_, plt_log, label='Full Model', color='tab:red', ax=ax_mea[0])
    # plot_spectra(fm.freqs, fm._ap_fit, plt_log, label='Final Aperiodic Fit', color='blue', alpha=0.5, linestyle='dashed', ax=ax_mea[0])
    ## aperiodic fit over quantiles plot
    ax_mea[0].plot(fm.freqs, fm._ap_fit, color='tab:red', linestyle='dashed', alpha=0.5)
    
    
    ax_mea[0].axvline(x=range_freqs[0], ymin=-10, ymax=100, ls='--', lw=1.0, color='tab:blue')
    ax_mea[0].axvline(x=range_freqs[1], ymin=-10, ymax=100, ls='--', lw=1.0, color='tab:blue')
    ax_mea[1].axvline(x=range_freqs[0], ymin=-10, ymax=100, ls='--', lw=1.0, color='tab:blue')
    ax_mea[1].axvline(x=range_freqs[1], ymin=-10, ymax=100, ls='--', lw=1.0, color='tab:blue')


    # Print out the model results
    print(f'CF [center frequency], PW [Power], BW [Bandwidth]')
    # fm.print_results()
    print(f"results 01 fooof:\n{fm.peak_params_}")
    # print(f"results 02 fooof:\n{obj_global.get_results_fooof(region_global)}")

    # print(f"fm results:\n{fm.peak_params_}")

    # Plot the full model fit of the power spectrum
    #  The final fit (red), and aperiodic fit (blue), are the same as we plotted above
    # fm.plot(plt_log)
    return 0


#########################
def on_press(event):
    global ax_seg, fig_seg, fig_mea, ax_mea, fig_peaks, ax_peaks, selectors, flag_peaks_global, obj_global
    # print('press', event.key)
    sys.stdout.flush()

    range_freqs = [f0_global, f1_global]

    # Set whether to plot in log-log space
    plt_log = False

    print(f"pressed: {event.key}")
    print(f'freq range: {range_freqs}')

    ## measuring amplitude peak to peak
    if event.key == 'a':
        ## fit a fooof model: periodic and aperiodic components
        if flag_peaks_global == False:
            ## peaks' threshold manually selected (mouse interaction)
            span_peaks = mwidgets.SpanSelector(ax_mea[1], onselect_peaks, 'vertical', interactive=True, useblit=True, props=dict(facecolor='tab:green', alpha=0.2))
            selectors.append(span_peaks)
            flag_peaks_global = True
        else:
            ax_mea[1].cla()

        ## fit the fooof model
        print(f"threshold peaks: {thr_peaks_global}")
        obj_global.fit_fooof(region_global, range_freqs, thr_peaks_global, ax_mea[1])

        ## threshold line to include and exclude peaks for the gaussian model fitting (fooof)
        ax_mea[1].axhline(y=thr_peaks_global, xmin=-10, xmax=100, ls='--', lw=1.0, color='tab:blue')

        # (Re)Plot curves of quantiles from the PSD of the selected channels
        # signal_measurements(ax_index, flag_eyes_closed, fig_title_mea)
        plot_quantiles(ax_index, flag_eyes_closed, fig_title_mea)
        
        fm = obj_global.get_fooof_model(region_global)
        # plot_spectra(fm.freqs, fm.fooofed_spectrum_, plt_log, label='Full Model', color='tab:red', ax=ax_mea[0])
        # plot_spectra(fm.freqs, fm._ap_fit, plt_log, label='Final Aperiodic Fit', color='blue', alpha=0.5, linestyle='dashed', ax=ax_mea[0])
        ## aperiodic fit over quantiles plot
        ax_mea[0].plot(fm.freqs, fm._ap_fit, color='tab:red', linestyle='dashed', alpha=0.5)
        
        
        ax_mea[0].axvline(x=range_freqs[0], ymin=-10, ymax=100, ls='--', lw=1.0, color='tab:blue')
        ax_mea[0].axvline(x=range_freqs[1], ymin=-10, ymax=100, ls='--', lw=1.0, color='tab:blue')
        ax_mea[1].axvline(x=range_freqs[0], ymin=-10, ymax=100, ls='--', lw=1.0, color='tab:blue')
        ax_mea[1].axvline(x=range_freqs[1], ymin=-10, ymax=100, ls='--', lw=1.0, color='tab:blue')


        # Print out the model results
        print(f'CF [center frequency], PW [Power], BW [Bandwidth]')
        # fm.print_results()
        print(f"results 01 fooof:\n{fm.peak_params_}")
        # print(f"results 02 fooof:\n{obj_global.get_results_fooof(region_global)}")

        # print(f"fm results:\n{fm.peak_params_}")

        # Plot the full model fit of the power spectrum
        #  The final fit (red), and aperiodic fit (blue), are the same as we plotted above
        # fm.plot(plt_log)

    # elif event.key == 'u':
    #     ## update plots including fooof model in the psd eyes closed and eyes open
    #     ## print results from all fitted models
    #     flag_save_fig = True
    #     flag_update_plot = True
    #     update_psd_plots(path_fig_psd, flag_update_plot, flag_save_fig)
        
    elif event.key == 'z':
        ## fooof curves comparison
        flag_save_fig = True
        plot_psd_responses_fooof(obj_list, event_list_ce, event_list_oe, info_p, flag_save_fig)

        ## to compare plot responses without aperiodic component
        plot_psd_minus_aperiodic(obj_list, event_list_ce, event_list_oe, info_p, flag_save_fig)
        ## save quantiles from the psd distribution of selected region
        ## save parameters used for fooof fit
        save_psd_quantiles(obj_list, event_list_ce, event_list_oe)


    # elif event.key == 'c':
    #     ## fooof aperiodic component subtraction from psd
    #     flag_save_fig = True
    #     plot_psd_minus_aperiodic(obj_list, event_list_ce, event_list_oe, path_fig_fooof, info_p, flag_save_fig)

    elif event.key == 'q':      
        ## close all windows
        plt.close('all')
    else:
        pass

    plt.show()

    return 0

##############################################
def onselect_peaks(vmin, vmax):
    global thr_peaks_global
    print(vmin, vmax)

    thr_peaks_global = np.max([vmin,vmax])
    print(f"thr peak: {thr_peaks_global}")

    return 0

##################
def plot_psd_responses_fooof(obj_list, event_list_ce, event_list_oe, info_p, flag_save):
    global fig_a, ax_a
    ## comparison aperiodic models among resting-cycling-resting, open-eyes, closed-eyes
    if fig_a == []:
        fig_a, ax_a = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(12,6), layout='constrained')
        ax_a = ax_a.flatten()
    else:
        for ax in ax_a:
            ax.cla()
        pass

    ymin = 5
    ymax = 35

    flags = [0,0,0]

    fooof_curves_dict = {}
    for obj in obj_list:
        ## find the selected segment for each label
        ## At the beginning, one of each condition was selected, i.e. a_ce, a_oe, b_ce, b_oe, c_ce, c_oe
        
        ## color represents resting start (blue), cycling (orange), or resting end (green)
        label = obj.get_label_simple()
        color = get_color(label)
        
        ## a_ce, b_ce, c_ce
        if label in event_list_ce:
            ## closed eyes
            print(f"{obj.get_label()}")
            region ='central_left'
            if obj.get_fooof_model(region) != []:
                freqs, mag = obj.plot_psd_fooof(ax_a[0], region, color)
                flags = label_flags(obj, flags)
                ## to save data of fooof models
                df = pd.DataFrame({'freqs':freqs, 'mag':mag})
                fooof_curves_dict[label+'_left'] = df.to_dict(orient='split',index=False)
            else:
                pass

            region ='central_right'
            if obj.get_fooof_model(region) != []:
                freqs, mag = obj.plot_psd_fooof(ax_a[1], region, color)
                flags = label_flags(obj, flags)
                ## to save data of fooof models
                df = pd.DataFrame({'freqs':freqs, 'mag':mag})
                fooof_curves_dict[label+'_right'] = df.to_dict(orient='split',index=False)
            else:
                pass
            
        ## a_oe, b_oe, c_oe
        elif label in event_list_oe:
            ## open eyes
            print(f"{obj.get_label()}")
            region ='central_left'
            if obj.get_fooof_model(region) != []:
                freqs, mag = obj.plot_psd_fooof(ax_a[2], region, color)
                flags = label_flags(obj, flags)
                ## to save data of fooof models
                df = pd.DataFrame({'freqs':freqs, 'mag':mag})
                fooof_curves_dict[label+'_left'] = df.to_dict(orient='split',index=False)
            else:
                pass
            
            region ='central_right'
            if obj.get_fooof_model(region) != []:
                freqs, mag = obj.plot_psd_fooof(ax_a[3], region, color)
                flags = label_flags(obj, flags)
                ## to save data of fooof models
                df = pd.DataFrame({'freqs':freqs, 'mag':mag})
                fooof_curves_dict[label+'_right'] = df.to_dict(orient='split',index=False)
            else:
                pass
        else:
            pass

    ## x and y limits
    ax_a[0].set_xlim(0, 31.0)
    # ax_a[0].set_ylim(ymin,ymax)

    fig_a = set_legend(fig_a, flags)
    set_labels_ax_4only(ax_a)
    set_title_ax4only(ax_a)
    set_grid_ax4only(ax_a)

    ## save figures
    fig_a.suptitle(f"{info_p}\n",)

    if flag_save:
        fig_a.savefig(path+'fooof/psd_q2_dict.png', bbox_inches ="tight")
        ## save dict with curves (dataframes) from fooof modeling
        with open(path+'fooof/psd_q2_dict.json', 'w') as f:
            f.write(json.dumps(fooof_curves_dict))

    return 0


##################
def plot_psd_minus_aperiodic(obj_list, event_list_ce, event_list_oe, info_p, flag_save):
    global fig_ap, ax_ap
    ## comparison aperiodic models among resting-cycling-resting, open-eyes, closed-eyes
    if fig_ap == []:
        fig_ap, ax_ap = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(12,6), layout='constrained')
        ax_ap = ax_ap.flatten()
    else:
        for ax in ax_ap:
            ax.cla()
        pass

    flags = [0,0,0]

    fooof_curves_dict = {}
    for obj in obj_list:
        ## find the selected segment for each label
        ## At the beginning, one of each condition was selected, i.e. a_ce, a_oe, b_ce, b_oe, c_ce, c_oe
        
        ## color represents resting start (blue), cycling (orange), or resting end (green)
        label = obj.get_label_simple()
        color = get_color(label)
        
        ## a_ce, b_ce, c_ce
        if label in event_list_ce:
            ## closed eyes
            print(f"{obj.get_label()}")
            region ='central_left'
            if obj.get_fooof_model(region) != []:
                freqs, mag = obj.plot_periodic_comp(ax_ap[0], region, color)
                flags = label_flags(obj, flags)
                ## to save data of fooof models
                df = pd.DataFrame({'freqs':freqs, 'mag':mag})
                fooof_curves_dict[label+'_left'] = df.to_dict(orient='split',index=False)
            else:
                pass

            region ='central_right'
            if obj.get_fooof_model(region) != []:
                freqs, mag = obj.plot_periodic_comp(ax_ap[1], region, color)
                flags = label_flags(obj, flags)
                ## to save data of fooof models
                df = pd.DataFrame({'freqs':freqs, 'mag':mag})
                fooof_curves_dict[label+'_right'] = df.to_dict(orient='split',index=False)
            else:
                pass
            
        ## a_oe, b_oe, c_oe
        elif label in event_list_oe:
            ## open eyes
            print(f"{obj.get_label()}")
            region ='central_left'
            if obj.get_fooof_model(region) != []:
                freqs, mag = obj.plot_periodic_comp(ax_ap[2], region, color)
                flags = label_flags(obj, flags)
                ## to save data of fooof models
                df = pd.DataFrame({'freqs':freqs, 'mag':mag})
                fooof_curves_dict[label+'_left'] = df.to_dict(orient='split',index=False)
            else:
                pass
            
            region ='central_right'
            if obj.get_fooof_model(region) != []:
                freqs, mag = obj.plot_periodic_comp(ax_ap[3], region, color)
                flags = label_flags(obj, flags)
                ## to save data of fooof models
                df = pd.DataFrame({'freqs':freqs, 'mag':mag})
                fooof_curves_dict[label+'_right'] = df.to_dict(orient='split',index=False)
            else:
                pass
        else:
            pass

    ## x and y limits
    ax_ap[0].set_xlim(0, 31.0)

    fig_ap = set_legend(fig_ap, flags)
    set_labels_ax_4only(ax_ap)
    set_title_ax4only(ax_ap)
    set_grid_ax4only(ax_ap)

    ## save figures
    fig_ap.suptitle(f"{info_p}\n",)

    if flag_save:
        fig_ap.savefig(path+'fooof/periodic_comp_q2.png', bbox_inches ="tight")
        ## save dict with curves (dataframes) from fooof modeling
        with open(path+'fooof/periodic_comp_q2.json', 'w') as f:
            f.write(json.dumps(fooof_curves_dict))
        
    return 0

#############################
def save_psd_quantiles(obj_list, event_list_ce, event_list_oe):
    global fig_ap, ax_ap

    parameters_fooof_dict = {}
    psd_quantiles_dict = {}
    for obj in obj_list:
        ## find the selected segment for each label
        ## At the beginning, one of each condition was selected, i.e. a_ce, a_oe, b_ce, b_oe, c_ce, c_oe
        
        label = obj.get_label_simple()
        ## a_ce, b_ce, c_ce
        if label in event_list_ce:
            ## closed eyes
            # print(f"{obj.get_label(), obj.get_id()}")
            region ='central_left'
            df = obj.get_psd_quantiles(region)
            psd_quantiles_dict[label+'_left'] = df.to_dict(orient='split',index=False)
            ## parameters fooof fitted
            parameters_fooof_dict[label+'_left'] = obj.get_params_fooof(region)

            region ='central_right'
            df = obj.get_psd_quantiles(region)
            psd_quantiles_dict[label+'_right'] = df.to_dict(orient='split',index=False)
            ## parameters fooof fitted
            parameters_fooof_dict[label+'_right'] = obj.get_params_fooof(region)
            
        ## a_oe, b_oe, c_oe
        elif label in event_list_oe:
            ## open eyes
            # print(f"{obj.get_label(), obj.get_id()}")
            region ='central_left'
            df = obj.get_psd_quantiles(region)
            psd_quantiles_dict[label+'_left'] = df.to_dict(orient='split',index=False)
            ## parameters fooof fitted
            parameters_fooof_dict[label+'_left'] = obj.get_params_fooof(region)
            
            region ='central_right'
            df = obj.get_psd_quantiles(region)
            psd_quantiles_dict[label+'_right'] = df.to_dict(orient='split',index=False)
            ## parameters fooof fitted
            parameters_fooof_dict[label+'_right'] = obj.get_params_fooof(region)
        else:
            pass
    
    ## save dict with curves (dataframes) from fooof modeling
    with open(path+'fooof/psd_quantiles_dict.json', 'w') as f:
        f.write(json.dumps(psd_quantiles_dict))
    ## save dict with parameters from fooof modeling
    with open(path+'fooof/fooof_parameters_dict.json', 'w') as f:
        f.write(json.dumps(parameters_fooof_dict))

    return 0

##################
def get_color(label):
    ## curve color
    if 'a_' in label:
        ## rest start
        color='tab:blue'
    elif 'b_' in label:
        ## cycling
        color='tab:orange'
    elif 'c_' in label:
        ## rest end
        color='tab:green'
    else:
        color='black'
    return color

###################
def label_flags(obj, flags):
    label = obj.get_label_simple()
    if label in ['a_ce','a_oe']:
        flags[0] = 1
    elif label in ['b_ce','b_oe']:
        flags[1] = 1
    elif label in ['c_ce','c_oe']:
        flags[2] = 1
    return flags

################
def set_labels_ax_4only(ax_list,):

    ## remove legends
    for ax in ax_list:
        try:
            ax.get_legend().set_visible(False)
        except:
            print("legend not found")
        
    ## y axis labels
    ax_list[0].set_ylabel(f"Power\n[$dB(mV/m^2)^2/Hz$]", fontsize=11)
    ax_list[1].set_ylabel(f"")
    ax_list[2].set_ylabel(f"Power\n[$dB(mV/m^2)^2/Hz$]", fontsize=11)
    ax_list[3].set_ylabel(f"")
    

    ## x axis labels
    ax_list[0].set_xlabel(f"")
    ax_list[1].set_xlabel(f"")
    ax_list[2].set_xlabel(f"frequency (Hz)", fontsize=11)
    ax_list[3].set_xlabel(f"frequency (Hz)", fontsize=11)

    return 0

###################
def set_legend(fig, flags):
    # blue_line = mlines.Line2D([], [], color='tab:blue', label="\nresting\nbefore\nbiking\n")
    # orange_line = mlines.Line2D([], [], color='tab:orange', label="biking")
    # green_line = mlines.Line2D([], [], color='tab:green', label="\nresting\nafter\nbiking\n")
    blue_line = mlines.Line2D([], [], color='tab:blue', label="rest start")
    orange_line = mlines.Line2D([], [], color='tab:orange', label="biking")
    green_line = mlines.Line2D([], [], color='tab:green', label="rest end")

    print(f"sum flags = {sum(flags)}")
    if sum(flags) == 3:
        handles_list=[blue_line, orange_line, green_line]
    elif sum(flags) == 2:
        handles_list=[blue_line, orange_line,]
    else:
        handles_list=[blue_line,]

    fig.legend(handles=handles_list, loc="outside right upper")

    return fig

###########
def set_title_ax4only(ax,):
    ##
    ax[0].set_title(f"average central left region\nclosed eyes", loc='center')
    ax[1].set_title(f"average central right region\nclosed eyes",loc='center')
    ax[2].set_title(f"open eyes",   loc='center')
    ax[3].set_title(f"open eyes",  loc='center')

    ##
    # ax[0].set_title(f"central left channels")
    # ax[1].set_title(f"central right channels")
    # ax[2].set_title(f"central left channels")
    # ax[3].set_title(f"central right channels")
    return 0

###########
def set_grid_ax4only(ax_list):
    ## hide grid
    for ax in ax_list:
        ax.grid(lw=0.5, ls='--', alpha=0.5)

    return 0





