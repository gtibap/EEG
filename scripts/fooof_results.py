from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import json
import sys

sys.path.insert(0, '../../scripts')
from list_participants import participants_list

#################
# global variables
df_dict_global = {}
flag_labels_dict = {}
flags = np.array([0,0,0])

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

def set_flags(label):
    global flags
    if 'c_' in label:
        flags[2]=1
    elif 'b_' in label:
        flags[1]=1
    elif 'a_' in label:
        flags[0]=1
    else:
        pass

#############
def plot_curve(dict, label, ax):
    global df_dict_global, flag_labels_dict
    color = get_color(label)
    try:
        df = pd.DataFrame.from_dict(dict[label]['data'], orient='columns')
        df_dict_global[label] = df
        flag_labels_dict[label] = True
        ax.plot(df[0],df[1], color=color)
    except:
        flag_labels_dict[label] = False
        print(f"no data for {label}")

        
    return 0

#############
def calculate_mean(freqs, curve_x, curve_ref, ax, color):

    data_dict = {'freqs':freqs, 'ref':curve_ref, 'x':curve_x}
    df = pd.DataFrame(data_dict)
    # markers = [5,10,15,20,25,30]
    f_ini = freqs[0]
    f_end = freqs[-1]
    num_div = 20
    markers = np.linspace(f_ini, f_end, num=num_div)
    # print(f"dataframe:\n{df}")
    for f0, f1 in zip(markers, markers[1:]):
        df_sel= df.loc[(df['freqs']>=f0) & (df['freqs']<f1)]
        # print(f"df_sel:\n{df_sel}")
        arr_ref = df_sel['ref'].to_numpy()
        arr_x = df_sel['x'].to_numpy()
        avg_ref = np.mean(arr_ref)
        avg_x = np.mean(arr_x)
        # print(f"mean [{f0, f1}] = {avg}")

        ## difference mean values per bin or section
        avg_diff = (avg_x - avg_ref)

        ax.fill_between(freqs, avg_diff, where=(freqs >= f0) & (freqs < f1), facecolor=color, alpha=.5)
        
    

    return 0
#############
def subtract_baseline(labels_list, ax, ax_mean):
    df_diff = pd.DataFrame()
    ## select baseline
    for label in labels_list:
        if 'a_' in label:
            label_ref = label
            df_ref = df_dict_global[label_ref]
            ## find min and max frequencies
            fref_min = df_ref[0].min()
            fref_max = df_ref[0].max()
            set_flags(label)
            break
        else:
            pass
    # print (f"df ref ({label_ref}):\n{df_ref}")
    ## subtract 'psd' of df_ref from other df
    ## first find min and max of freq
    ## then identify common range of freq between two df
    for label in labels_list:
        if ('a_' not in label)  and (flag_labels_dict[label]==True):
            df = df_dict_global[label]
            ## find min and max frequencies
            f_min = df[0].min()
            f_max = df[0].max()

            ## common range of frequencies
            frange_min = np.max([f_min, fref_min ])
            frange_max = np.min([f_max, fref_max ])

            # print(f"{label} freq range min max: {frange_min, frange_max}")

            ## values in the common range of frequencies
            # print(f"df ref:\n{df_ref}")
            ## first from the reference (a_)
            df_a = df_ref.loc[(df_ref[0]>=frange_min) & (df_ref[0]<frange_max)]
            ## column 0: array of frequencies
            ## column 1: array of psd values
            freqs = df_a[0].to_numpy()
            curve_a = df_a[1].to_numpy()

            ## from the second dataframe, taking elements starting from the same frequency value
            df_b = df.loc[(df[0]>=frange_min)]
            # print(f"df_b:\n{df_b}")
            ## and take same number of elements based on the lenght of the freqs array 
            curve_b = df_b[1].to_numpy()
            curve_b = curve_b[:len(freqs)]

            curve_diff = curve_b - curve_a


            color = get_color(label)
            ## horizontal line is the reference, i.e. 'rest start'
            ax.hlines(y=0, xmin=freqs[0],xmax=freqs[-1], color='tab:blue')
            ax.plot(freqs,curve_diff,color=color)

            ## from the difference calculate mean per intervals, example: [5-10, 10-15, 15,20, 20-25, 25-30] Hz
            calculate_mean(freqs, curve_b, curve_a, ax_mean, color)

            
            # print(f"df freqs:\n{df_a}")
            # print(f"f_arr: {f_arr}")
            # print(f"len f_arr: {len(f_arr)}")
            set_flags(label)

        else:
            pass

    return 0
            
################
def set_labels_ax_4only(ax_list, flag_csd):

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
    global flags

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

    fig.legend(handles=handles_list, loc="outside right upper", fontsize=11)

    return fig

###########
def set_title_ax4only(ax,):
    ##
    ax[0].set_title(f"average central left region\nclosed eyes", loc='center')
    ax[1].set_title(f"average central right region\nclosed eyes",loc='center')
    ax[2].set_title(f"open eyes",   loc='center')
    ax[3].set_title(f"open eyes",  loc='center')

    return 0
############
def set_grid_ax4only(ax_list):
    ## hide grid
    for ax in ax_list:
        ax.grid(lw=0.5, ls='--', alpha=0.5)

    return 0

##############
def on_press(event):
    sys.stdout.flush()
    if event.key == 'q':
        plt.close()
    else:
        pass
    return 0


##############
def main(args):

    flag_csd = False
    ## run matplotlib in interactive mode
    plt.ion()

    print(f'arg {args[1]}') ## folder location
    print(f'arg {args[2]}') ## subject = {0:patient 1, 1:patient 2, ...}
    print(f'arg {args[3]}') ## session = {1:time zero, 2:three months, 3:six months}
    print(f'arg {args[4]}') ## ABT = {0:resting, 1:biking}
    print(f'arg {args[5]}') ## rest end = {0:off, 1:on}

    
    path=args[1]
    subject= int(args[2])
    session=int(args[3])
    abt= int(args[4])
    flag_rest_end = int(args[5])

    ## new path, eeg filename (fn_in), annotations filename (fn_csv), eeg raw data (raw_data)
    path, fn_in, fn_csv, raw_data, fig_title, flag_notch, acquisition_system, info_p, Dx, selected_segs_dict, ch_excl_list, ylims, thr_peaks_global = participants_list(path, subject, session, abt)
    if fn_csv == '':
        print(f'It could not find the selected subject. Please check the path, and the selected subject number in the list of participants.')
        return 0
    else:
        pass
    

    # path = '../../data/a_neuroplasticity/n_007/'
    # session = '1'
    path_fig_fooof = path+'session_'+str(session)+f'/figures/fooof/'

    if flag_rest_end:
        labels_ce_right = ['a_ce_right','b_ce_right','c_ce_right']
        labels_oe_right = ['a_oe_right','b_oe_right','c_oe_right']
        labels_ce_left  = ['a_ce_left','b_ce_left','c_ce_left']
        labels_oe_left  = ['a_oe_left','b_oe_left','c_oe_left']
    else:
        labels_ce_right = ['a_ce_right','b_ce_right']
        labels_oe_right = ['a_oe_right','b_oe_right']
        labels_ce_left  = ['a_ce_left','b_ce_left']
        labels_oe_left  = ['a_oe_left','b_oe_left']

    
    if abt:
        ## median of psd curves minus its aperiodic component
        filename = path_fig_fooof+'fooof_curves_dict.json'
    else:
        ## approx. of the median psd response using fooof (gaussian) model
        filename = path_fig_fooof+'psd_minus_aperiodic_dict.json'

    # fooof_curves_dict = np.load(path_fig_fooof+'fooof_curves_dict.npy', allow_pickle=False)
    with open(filename) as f:
        fooof_curves_dict = json.loads(f.read())
    # print(f'dict:\n{fooof_curves_dict}')

    ###
    # plot curves
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(12,6), layout='constrained')
    ax = ax.flatten()
    fig.canvas.mpl_connect('key_press_event', on_press)
    set_labels_ax_4only(ax, flag_csd)
    
    for label in labels_ce_left:
        plot_curve(fooof_curves_dict, label, ax[0])
    for label in labels_ce_right:
        plot_curve(fooof_curves_dict, label, ax[1])
    for label in labels_oe_left:
        plot_curve(fooof_curves_dict, label, ax[2])
    for label in labels_oe_right:
        plot_curve(fooof_curves_dict, label, ax[3])

    ###########
    ## subtract baseline
    ## for example for the sequence: a_ce_right, b_ce_right, and c_ce_right, the baseline is a_ce_right
    ## hence, (b_ce_right - a_ce_right), and (c_ce_right - a_ce_right)
    ## however, the arrays could cover a different range of frequencies
    ## therefore, we select a common frequency range for the subtraction
    # 
    fig_diff, ax_diff = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(12,6), layout='constrained')
    ax_diff = ax_diff.flatten()
    fig_diff.canvas.mpl_connect('key_press_event', on_press)

    fig_mean, ax_mean = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(12,6), layout='constrained')
    ax_mean = ax_mean.flatten()
    fig_mean.canvas.mpl_connect('key_press_event', on_press)

    subtract_baseline(labels_ce_left,  ax_diff[0], ax_mean[0])
    subtract_baseline(labels_ce_right, ax_diff[1], ax_mean[1])
    subtract_baseline(labels_oe_left,  ax_diff[2], ax_mean[2])
    subtract_baseline(labels_oe_right, ax_diff[3], ax_mean[3])

    ax_diff[0].set_ylim([-7,7])
    ax_mean[0].set_ylim([-7,7])
    set_labels_ax_4only(ax_diff, flag_csd)
    set_labels_ax_4only(ax_mean, flag_csd)

    fig = set_legend(fig)
    fig_diff = set_legend(fig_diff)
    fig_mean = set_legend(fig_mean)

    set_title_ax4only(ax_diff)
    set_grid_ax4only(ax_diff)
    fig_diff.suptitle(f"{info_p}\n",)

    set_title_ax4only(ax_mean)
    set_grid_ax4only(ax_mean)
    fig_mean.suptitle(f"{info_p}\n",)

    set_title_ax4only(ax)
    set_grid_ax4only(ax)
    fig.suptitle(f"{info_p}\n",)

    flag_save = True
    ## save figures
    if flag_save:
        fig_diff.savefig(path_fig_fooof+'psd_diff.png', bbox_inches ="tight")
        fig_mean.savefig(path_fig_fooof+'psd_diff_mean.png', bbox_inches ="tight")

    plt.show(block=True)

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
