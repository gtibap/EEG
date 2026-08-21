from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

## global varibles
flag_legend = [0,0,0]

################
def plot_psd_states(obj_list, info_p, path_fig):
    ## plot psd median values
    fig, axs = plt.subplots(2,2, figsize=(12,6), sharex=True, sharey=True)
    axs = axs.flatten()

    for obj in obj_list:
        print(obj.get_label())
        if 'ce_left' in obj.get_label():
            plot_psd_obj(obj, axs[0])

        elif 'ce_right' in obj.get_label():
            plot_psd_obj(obj, axs[1])
    
        elif 'oe_left' in obj.get_label():
            plot_psd_obj(obj, axs[2])
    
        elif 'oe_right' in obj.get_label():
            plot_psd_obj(obj, axs[3])

        else:
            pass

    ## if the courrent source density (csd) was applied or not
    flag_csd = False
    set_labels(axs, flag_csd)
    set_legend(fig)
    set_grid(axs)
    set_title_axs(axs)
    fig.suptitle(f"{info_p}\nAverage PSD + aperiodic component",)
    fig.savefig(path_fig+'psd_median.png', bbox_inches ="tight")
    
    return 0

##################
def plot_psd_obj(obj, ax):
    df = obj.get_fooof_data()
    ## if a fooof has been fitted, df would be a DataFrame, otherwise df would be np.nan (type float)
    if not isinstance(df, float):
        color = get_color(obj.get_label())
        ax.plot(df['freqs'], df['psd'], color=color )
        ## add aperiodic component to the plots
        ax.plot(df['freqs'], df['aperiodic'], color=color, alpha=0.5, linestyle='dashed')
        ## set flags required to include curves in the figure's legend
        set_flag_legend(obj.get_label())
    else:
        pass
    return 0

################
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


################
def set_labels(ax_list, flag_csd):

    ## remove legends
    for ax in ax_list:
        try:
            ax.get_legend().set_visible(False)
        except:
            print("legend not found")
        
    ## y axis labels
    if flag_csd:
        ax_list[0].set_ylabel(f"Power\n[$dB(mV/m^2)^2/Hz$]", fontsize=11)
        ax_list[1].set_ylabel(f"")
        ax_list[2].set_ylabel(f"Power\n[$dB(mV/m^2)^2/Hz$]", fontsize=11)
        ax_list[3].set_ylabel(f"")
    else:
        ax_list[0].set_ylabel(f"Power (dB $\mu$V$^2$/Hz)", fontsize=11)
        ax_list[1].set_ylabel(f"")
        ax_list[2].set_ylabel(f"Power (dB $\mu$V$^2$/Hz)", fontsize=11)
        ax_list[3].set_ylabel(f"")
    

    ## x axis labels
    ax_list[0].set_xlabel(f"")
    ax_list[1].set_xlabel(f"")
    ax_list[2].set_xlabel(f"frequency (Hz)", fontsize=11)
    ax_list[3].set_xlabel(f"frequency (Hz)", fontsize=11)

    return 0

#################
def set_legend(fig):
    global flag_legend

    blue_line = mlines.Line2D([], [], color='tab:blue', label="rest start")
    orange_line = mlines.Line2D([], [], color='tab:orange', label="biking")
    green_line = mlines.Line2D([], [], color='tab:green', label="rest end")

    print(f"sum flags = {sum(flag_legend)}")
    if sum(flag_legend) == 3:
        handles_list=[blue_line, orange_line, green_line]
    elif sum(flag_legend) == 2:
        handles_list=[blue_line, orange_line,]
    else:
        handles_list=[blue_line,]

    fig.legend(handles=handles_list, loc="outside right upper", fontsize=11)

    return fig

################
def set_flag_legend(label):
    global flag_legend

    if 'c_' in label:
        flag_legend[2]=1
    elif 'b_' in label:
        flag_legend[1]=1
    elif 'a_' in label:
        flag_legend[0]=1
    else:
        pass

############
def set_grid(ax_list):
    ## hide grid
    for ax in ax_list:
        ax.grid(lw=0.5, ls='--', alpha=0.5)

    return 0

############
def set_annotated_grid(ax_list, band_markers, ycoord):

    for label_band in band_markers:
        ## alpha and beta labels
        lims_band = band_markers[label_band]
        for ax in ax_list:
            ## to avoid to re-paint same line twice
            if label_band == 'alpha':
                ax.axvline(lims_band[0], linestyle='dashed', lw=1.0, color='black', alpha=0.75)
                ax.axvline(lims_band[1], linestyle='dashed', lw=1.0, color='black', alpha=0.75)
            else:
                ax.axvline(lims_band[1], linestyle='dashed', lw=1.0, color='black', alpha=0.75)
            xcoord =  (lims_band[1] + lims_band[0]) / 2
            ax.annotate(f'$\{label_band}$', xy=(xcoord, ycoord), xycoords='data', xytext=(xcoord, ycoord), textcoords='data', va='top', ha='center',)

    return 0

###########
def set_title_axs(axs):
    ##
    axs[0].set_title(f"eyes closed -- left central side", loc='center')
    axs[1].set_title(f"eyes closed -- right central side",loc='center')
    axs[2].set_title(f"eyes open -- left central side",   loc='center')
    axs[3].set_title(f"eyes open -- right central side",  loc='center')

    return 0

############
def graphics_periodic_comp(obj_list, info_p, band_markers, peak_value, path_fig):
    ## plot psd median values
    fig, axs = plt.subplots(2,2, figsize=(12,6), sharex=True, sharey=True)
    axs = axs.flatten()

    for obj in obj_list:
        print(obj.get_label())
        if 'ce_left' in obj.get_label():
            plot_periodic_obj(obj, axs[0])

        elif 'ce_right' in obj.get_label():
            plot_periodic_obj(obj, axs[1])
    
        elif 'oe_left' in obj.get_label():
            plot_periodic_obj(obj, axs[2])
    
        elif 'oe_right' in obj.get_label():
            plot_periodic_obj(obj, axs[3])

        else:
            pass

    ## ylim scale based on the max peak values alpha band
    axs[0].set_ylim(-3.0, peak_value+(0.2*peak_value))
    ## if the courrent source density (csd) was applied or not
    flag_csd = False
    set_labels(axs, flag_csd)
    set_legend(fig)
    set_annotated_grid(axs, band_markers, peak_value+(0.15*peak_value))
    set_title_axs(axs)
    fig.suptitle(f"{info_p}\n PSD periodic component",)
    fig.savefig(path_fig+'psd_periodic.png', bbox_inches ="tight")

    return 0

#############################
def plot_periodic_obj(obj, ax):
    df = obj.get_fooof_data()
    ## if a fooof has been fitted, df would be a DataFrame, otherwise df would be np.nan (type float)
    if not isinstance(df, float):
        color = get_color(obj.get_label())
        ax.plot(df['freqs'], df['periodic'], color=color )
        ## set flags required to include curves in the figure's legend
        set_flag_legend(obj.get_label())
    else:
        pass
    return 0