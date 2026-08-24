import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import numpy as np

###### global variables ######
flag_legend = [0,0,0]

##################
def plot_eeg_bands(sessions_values_dict, info_patient_dict, fig_filename, flag_save, title_val, title_ref):

    fig = plt.figure(figsize=(12, 6),) ## layout='constrained'
    axs = create_axes()

    bar_colors = ['tab:purple', 'tab:olive', 'tab:cyan', 'tab:pink', 'tab:brown']

    # axs[0].bar(colors, counts, label=bar_labels, color=bar_colors)
    # axs[0].set_ylabel('color supply')
    # axs[0].legend(title='Fruit color')

    for label_values in sessions_values_dict:

        values_dict = sessions_values_dict[label_values]
        print(f"values_dict[{label_values}]:{values_dict}")

        info_pt = info_patient_dict[label_values]
        title_id_pt = info_pt['id']
        # print(f"info pt: {info_pt}")
        label_bar = " ".join([info_pt['days'], f"[{info_pt['ais_nli']}]"])

        ## alpha band components
        alpha_dict = values_dict['alpha']
        ## beta band components
        beta_dict = values_dict['beta']

        # print(f"labels alpha: {alpha_dict}")
        # print(f"labels beta: {beta_dict}")

        ## session id
        id_s = label_values
        # print(f"id_s: {id_s}")
        for alpha_label in alpha_dict:
            if 'ce_left' in alpha_label:
                if not np.isnan(alpha_dict[alpha_label]):
                    ## to avoid include label for nan values 
                    axs[0].bar(id_s, alpha_dict[alpha_label], label=label_bar, color=bar_colors[id_s])
            elif 'ce_right' in alpha_label:
                axs[1].bar(id_s, alpha_dict[alpha_label], color=bar_colors[id_s])
            elif 'oe_left' in alpha_label:
                axs[4].bar(id_s, alpha_dict[alpha_label], color=bar_colors[id_s])
            elif 'oe_right' in alpha_label:
                axs[5].bar(id_s, alpha_dict[alpha_label], color=bar_colors[id_s])
            else:
                pass

        for beta_label in beta_dict:
            if 'ce_left' in beta_label:
                axs[2].bar(id_s, beta_dict[beta_label], color=bar_colors[id_s])
            elif 'ce_right' in beta_label:
                axs[3].bar(id_s, beta_dict[beta_label], color=bar_colors[id_s])
            elif 'oe_left' in beta_label:
                axs[6].bar(id_s, beta_dict[beta_label], color=bar_colors[id_s])
            elif 'oe_right' in beta_label:
                axs[7].bar(id_s, beta_dict[beta_label], color=bar_colors[id_s])
            else:
                pass
                

    loc = 'outside right upper'
    fig.legend(loc=loc, title='Days after trauma\n[AIS-NLI]')
    
    fig.suptitle(f"({title_val} $-$ {title_ref}) response\n{title_id_pt}")
    # plt.tight_layout()
    if flag_save:
        fig.savefig(fig_filename, bbox_inches ="tight")
    else:
        pass

    plt.show(block=True)

    return 0

##################
def create_axes():

    axs = []

    outer = gridspec.GridSpec(nrows=2, ncols=2)
    for row in range(2):
        for col in range(2):
            inner = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=outer[row, col], wspace=0)
            axs += [plt.subplot(cell) for cell in inner]

    for id, ax in enumerate(axs):
        ax.hlines(0, xmin=-1, xmax=4, color='black', lw=0.5)

        if id in [1,3,5,7]:
            ax.set_yticks([])
        else:
            ax.set_ylabel(f'normalized power difference')
        ax.set_xticks([])
        ax.set_ylim(-1.0, 1.0)
        # ax.set_xlim(-2,5)


        if id in [0,2]:
            ax.annotate(f'left central side\neyes closed', xy=(0, 0), xycoords='data', xytext=(0.01, .99), textcoords='axes fraction', va='top', ha='left',)
        elif id in [1,3]:
            ax.annotate(f'right central side\neyes closed', xy=(0, 0), xycoords='data', xytext=(0.01, .99), textcoords='axes fraction', va='top', ha='left',)
        elif id in [4,6]:
            ax.annotate(f'left central side\neyes open', xy=(0, 0), xycoords='data', xytext=(0.01, .99), textcoords='axes fraction', va='top', ha='left',)
        elif id in [5,7]:
            ax.annotate(f'right central side\neyes open', xy=(0, 0), xycoords='data', xytext=(0.01, .99), textcoords='axes fraction', va='top', ha='left',)
        else:
            pass

        if id in [0,1]:
            trans = ax.get_xaxis_transform() # x in data untis, y in axes fraction
            ann = ax.annotate(f'alpha band', xy=(0, 1.05 ), xycoords=trans) ## left central side
        elif id in [2,3]:
            trans = ax.get_xaxis_transform() # x in data untis, y in axes fraction
            ann = ax.annotate(f'beta band', xy=(0, 1.05 ), xycoords=trans) ## right central side
        elif id in [4,5]:
            trans = ax.get_xaxis_transform() # x in data untis, y in axes fraction
            ann = ax.annotate(f'alpha band', xy=(0, 1.05 ), xycoords=trans) ## left central side
        elif id in [6,7]:
            trans = ax.get_xaxis_transform() # x in data untis, y in axes fraction
            ann = ax.annotate(f'beta band', xy=(0, 1.05 ), xycoords=trans) ## right central side

    return axs


#################
################
def get_color(label):
    ## curve color
    if 'a_' in label:
        ## rest start
        color='tab:blue'
        id = 0
    elif 'b_' in label:
        ## cycling
        color='tab:orange'
        id = 1
    elif 'c_' in label:
        ## rest end
        color='tab:green'
        id = 2
    else:
        color='black'
        id = -1
    return id, color

#################
def plot_normalized_mean_bands(normalized_mean_bands_dict, info_p, path_fig):

    fig = plt.figure(figsize=(12, 6),) ## layout='constrained'
    axs = create_axes_normalized()

    bar_colors = ['tab:blue', 'tab:orange', 'tab:green',]

    for label_band in normalized_mean_bands_dict:
        ## label_band: alpha or beta
        if label_band == 'alpha':
            # print(f"label band: {label_band}")
            for label_state in normalized_mean_bands_dict['alpha']:
                ## label_state: a_ce_left, a_ce_right, b_ce_left, ...
                # print(f"label state: {label_state}")
                id_s, color = get_color(label_state)
                set_flag_legend(label_state)

                if 'ce_left' in label_state:
                    axs[0].bar(id_s, normalized_mean_bands_dict[label_band][label_state], color=color)
                    axs[0].set_ylim(-0.5, 4)

                elif 'ce_right' in label_state:
                    axs[1].bar(id_s, normalized_mean_bands_dict[label_band][label_state], color=color)
                    axs[1].set_ylim(-0.5, 4)

                elif 'oe_left' in label_state:
                    axs[4].bar(id_s, normalized_mean_bands_dict[label_band][label_state], color=color)
                    axs[4].set_ylim(-0.5, 4)

                elif 'oe_right' in label_state:
                    axs[5].bar(id_s, normalized_mean_bands_dict[label_band][label_state], color=color)
                    axs[5].set_ylim(-0.5, 4)

                else:
                    pass
            
        elif label_band == 'beta':
            # print(f"label band: {label_band}")
            for label_state in normalized_mean_bands_dict['beta']:
                ## label_state: a_ce_left, a_ce_right, b_ce_left, ...
                # print(f"label state: {label_state}")
                id_s, color = get_color(label_state)
                if 'ce_left' in label_state:
                    axs[2].bar(id_s, normalized_mean_bands_dict[label_band][label_state], color=color)
                    axs[2].set_ylim(-0.25, 1)

                elif 'ce_right' in label_state:
                    axs[3].bar(id_s, normalized_mean_bands_dict[label_band][label_state], color=color)
                    axs[3].set_ylim(-0.25, 1)

                elif 'oe_left' in label_state:
                    axs[6].bar(id_s, normalized_mean_bands_dict[label_band][label_state], color=color)
                    axs[6].set_ylim(-0.25, 1)

                elif 'oe_right' in label_state:
                    axs[7].bar(id_s, normalized_mean_bands_dict[label_band][label_state], color=color)
                    axs[7].set_ylim(-0.25, 1)

                else:
                    pass

        else:
            pass

    loc = 'outside right upper'
    fig.suptitle(f"{info_p}\n PSD periodic component",)
    set_legend(fig)
    plt.tight_layout()
    fig_filename = path_fig + 'psd_bands_bars.png'
    fig.savefig(fig_filename, bbox_inches ="tight")

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

############################
def create_axes_normalized():

    axs = []

    outer = gridspec.GridSpec(nrows=2, ncols=2)
    for row in range(2):
        for col in range(2):
            inner = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=outer[row, col], wspace=0)
            axs += [plt.subplot(cell) for cell in inner]

    for id, ax in enumerate(axs):
        ax.hlines(0, xmin=-1, xmax=3, color='black', lw=0.5)

        if id in [1,3,5,7]:
            ax.set_yticks([])
        else:
            ax.set_ylabel('normalized power')
        ax.set_xticks([])
        # ax.set_ylim(-1, 4)
        # ax.set_xlim(-2,5)


        if id in [0,2]:
            ax.annotate(f'left central side\neyes closed', xy=(0, 0), xycoords='data', xytext=(0.01, .99), textcoords='axes fraction', va='top', ha='left',)
        elif id in [1,3]:
            ax.annotate(f'right central side\neyes closed', xy=(0, 0), xycoords='data', xytext=(0.01, .99), textcoords='axes fraction', va='top', ha='left',)
        elif id in [4,6]:
            ax.annotate(f'left central side\neyes open', xy=(0, 0), xycoords='data', xytext=(0.01, .99), textcoords='axes fraction', va='top', ha='left',)
        elif id in [5,7]:
            ax.annotate(f'right central side\neyes open', xy=(0, 0), xycoords='data', xytext=(0.01, .99), textcoords='axes fraction', va='top', ha='left',)
        else:
            pass

        if id in [0,1]:
            trans = ax.get_xaxis_transform() # x in data untis, y in axes fraction
            ann = ax.annotate(f'alpha band', xy=(0, 1.05 ), xycoords=trans) ## left central side
        elif id in [2,3]:
            trans = ax.get_xaxis_transform() # x in data untis, y in axes fraction
            ann = ax.annotate(f'beta band', xy=(0, 1.05 ), xycoords=trans) ## right central side
        elif id in [4,5]:
            trans = ax.get_xaxis_transform() # x in data untis, y in axes fraction
            ann = ax.annotate(f'alpha band', xy=(0, 1.05 ), xycoords=trans) ## left central side
        elif id in [6,7]:
            trans = ax.get_xaxis_transform() # x in data untis, y in axes fraction
            ann = ax.annotate(f'beta band', xy=(0, 1.05 ), xycoords=trans) ## right central side

    return axs




