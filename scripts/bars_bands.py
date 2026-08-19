# Source - https://stackoverflow.com/a/67094486
# Posted by JohanC, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-18, License - CC BY-SA 4.0

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_eeg_bands(sessions_values_dict, info_patient_dict, fig_filename):

    fig = plt.figure(figsize=(12, 6),) ## layout='constrained'
    axs = create_axes()
    ## each ax subplot is ready to include information of alpha and beta bands power
    # colors = ['purple','olive','cyan']
    # counts = [4, 1, 3]
    # bar_labels = ['10 (B-C6)', '56 (C-C6)', '256 (D-C7)']
    bar_colors = ['tab:purple', 'tab:olive', 'tab:cyan',]

    # axs[0].bar(colors, counts, label=bar_labels, color=bar_colors)
    # axs[0].set_ylabel('color supply')
    # axs[0].legend(title='Fruit color')

    for label_values in sessions_values_dict:

        values_dict = sessions_values_dict[label_values]
        info_pt = info_patient_dict[label_values]
        title_id_pt = info_pt['id']
        print(f"info pt: {info_pt}")
        label_bar = " ".join([info_pt['days'], f"({info_pt['ais_nli']})"])

        ## alpha band components
        alpha_dict = values_dict['alpha']
        ## beta band components
        beta_dict = values_dict['beta']

        print(f"labels alpha: {alpha_dict}")
        print(f"labels beta: {beta_dict}")

        ## session id
        id_s = label_values
        print(f"id_s: {id_s}")
        for alpha_label in alpha_dict:
            if 'ce_left' in alpha_label:
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
    fig.legend(loc=loc, title='Days after trauma\n(AIS-NLI)')
    
    fig.suptitle(f"Normalized PSD response from EEG during passive cycling minus baseline (resting)\n{title_id_pt}")
    # plt.tight_layout()
    fig.savefig(fig_filename, bbox_inches ="tight")

    plt.show(block=True)

    return 0

def create_axes():

    axs = []

    outer = gridspec.GridSpec(nrows=2, ncols=2)
    for row in range(2):
        for col in range(2):
            inner = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=outer[row, col], wspace=0)
            axs += [plt.subplot(cell) for cell in inner]

    for id, ax in enumerate(axs):
        ax.hlines(0, xmin=-1, xmax=4, color='black', lw=1)

        if id in [1,3,5,7]:
            ax.set_yticks([])
        else:
            ax.set_ylabel('power difference [%]')
        ax.set_xticks([])
        ax.set_ylim(-100,100)
        ax.set_xlim(-2,5)


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



