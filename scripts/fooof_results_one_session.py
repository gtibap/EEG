from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import json
import sys

sys.path.insert(0, '../../scripts')
from list_participants import participants_list

## import class
from fooof_psd_class import FOOOF_class
from graphics_psd_fooof import plot_psd_states, graphics_periodic_comp
from bars_bands import plot_normalized_mean_bands

##################
def calculate_band_attenuation(mean_psd_bands_dict, mean_psd_ref, label_band, label_val, label_ref):

    diff_bands_dict = {}
    ## difference between cycling and baseline (rest start)
    ## closed eyes

    ## differences between biking and rest start left and right closed eyes
    for eyes in ['ce', 'oe']:

        ref = mean_psd_bands_dict[label_band][f'{label_ref}_{eyes}_left']
        val = mean_psd_bands_dict[label_band][f'{label_val}_{eyes}_left']
        nor = mean_psd_ref[f'{label_ref}_{eyes}_left']

        diff = 100*(val-ref)/nor
        diff_bands_dict[f'{label_val}_{label_ref}_{eyes}_left'] = diff

        ref = mean_psd_bands_dict[label_band][f'{label_ref}_{eyes}_right']
        val = mean_psd_bands_dict[label_band][f'{label_val}_{eyes}_right']
        nor = mean_psd_ref[f'{label_ref}_{eyes}_right']

        diff = 100*(val-ref)/nor
        diff_bands_dict[f'{label_val}_{label_ref}_{eyes}_right'] = diff

    # print(f"diff_bands_dict:\n{diff_bands_dict}")

    return diff_bands_dict

##################
def calculate_band_diff(mean_psd_bands_dict, label_band, label_val, label_ref):

    diff_bands_dict = {}
    ## difference between cycling and baseline (rest start)
    ## closed eyes

    ## differences between biking and rest start left and right closed eyes
    for eyes in ['ce', 'oe']:
        try:
            ref = mean_psd_bands_dict[label_band][f'{label_ref}_{eyes}_left']
            val = mean_psd_bands_dict[label_band][f'{label_val}_{eyes}_left']

            diff = val-ref
            diff_bands_dict[f'{label_val}_{label_ref}_{eyes}_left'] = diff

            ref = mean_psd_bands_dict[label_band][f'{label_ref}_{eyes}_right']
            val = mean_psd_bands_dict[label_band][f'{label_val}_{eyes}_right']

            diff = val-ref
            diff_bands_dict[f'{label_val}_{label_ref}_{eyes}_right'] = diff
        except:
            diff_bands_dict[f'{label_val}_{label_ref}_{eyes}_left'] = np.nan
            diff_bands_dict[f'{label_val}_{label_ref}_{eyes}_right'] = np.nan


    # print(f"diff_bands_dict:\n{diff_bands_dict}")

    return diff_bands_dict

##################
def calculate_normalized_mean_bands(mean_psd_bands_dict, mean_psd_ref, label_ref):

    normalized_bands_dict = {}

    for label_band in mean_psd_bands_dict:
        ## label_band: alpha or beta
        # print(f"label band: {label_band}")
        values_dict = {} 
        for label_state in mean_psd_bands_dict[label_band]:
            ## label_state: a_ce_left, a_ce_right, b_ce_left, ...
            # print(f"label state: {label_state}")

            if 'ce_left' in label_state:
                values_dict[label_state] = mean_psd_bands_dict[label_band][label_state] / mean_psd_ref[f'{label_ref}_ce_left']

            elif 'ce_right' in label_state:
                values_dict[label_state] = mean_psd_bands_dict[label_band][label_state] / mean_psd_ref[f'{label_ref}_ce_right']

            elif 'oe_left' in label_state:
                values_dict[label_state] = mean_psd_bands_dict[label_band][label_state] / mean_psd_ref[f'{label_ref}_oe_left']

            elif 'oe_right' in label_state:
                values_dict[label_state] = mean_psd_bands_dict[label_band][label_state] / mean_psd_ref[f'{label_ref}_oe_right']

            else:
                pass
        
        normalized_bands_dict[label_band] = values_dict

    return normalized_bands_dict


######################
def get_info_split(text):

    text = text.split()
    # print(f"text: {text}")
    id_pt = " ".join(text[:7])
    # print(f"{id_pt}")
    ais_nli = "-".join([text[9],text[12]])
    # print(f"{ais_nli}")
    days = text[14]
    # print(f"days: {days}")
    return id_pt, ais_nli, days

##############
## parameters of the fooof model here were defined in a previous step (tf_selected_freq_notch.py)
## hence, here those parameters are readed and the fooof model is reproduced
def get_fooof_results(path, subject, session, abt, flag_rest_end,  label_val, label_ref):
    # global flags_global
    flags_global = np.array([0,0,0])

    flag_csd = False
    ## run matplotlib in interactive mode
    plt.ion()

    ## reading info related to the patient
    path, fn_in, fn_csv, raw_data, fig_title, flag_notch, acquisition_system, info_p, Dx, selected_segs_dict, ch_excl_list, ylims, thr_peaks_global = participants_list(path, subject, session, abt)
    if fn_csv == '':
        print(f'It could not find the selected subject. Please check the path, and the selected subject number in the list of participants.')
        return 0
    else:
        pass

    ## patient info to include in figures (plots)
    # print (f"info_p: {info_p}")
    id_pt, ais_nli, days = get_info_split(info_p)
    info_pt_dict={}
    info_pt_dict['id'] = id_pt
    info_pt_dict['ais_nli'] = ais_nli
    info_pt_dict['days'] = days

    ## path to save figures
    path_fig_fooof = path+'session_'+str(session)+f'/figures/fooof/'
    path_fig_psd   = path+'session_'+str(session)+f'/figures/psd/'

    if flag_rest_end:
        ## to include rest start, cycling, and rest after cycling in the figures
        labels_ce_right = ['a_ce_right','b_ce_right','c_ce_right']
        labels_oe_right = ['a_oe_right','b_oe_right','c_oe_right']
        labels_ce_left  = ['a_ce_left','b_ce_left','c_ce_left']
        labels_oe_left  = ['a_oe_left','b_oe_left','c_oe_left']
    else:
        ## to include rest start and cycling in the figures
        labels_ce_right = ['a_ce_right','b_ce_right']
        labels_oe_right = ['a_oe_right','b_oe_right']
        labels_ce_left  = ['a_ce_left','b_ce_left']
        labels_oe_left  = ['a_oe_left','b_oe_left']

    obj_list = []
    peak_freqs_list = []
    peak_power_list = []
    labels_list = []

    ## read parameters to fit fooof model
    filename_params = path_fig_fooof+'fooof_parameters_dict.json'
    with open(filename_params, "r") as file:
        data_params = json.load(file)

    ## read psd quantiles from selected channels from each case, i.e. a_oe_left, a_oe_right, b_oe_left, ...
    filename_quantiles = path_fig_fooof+'psd_quantiles_dict.json'
    with open(filename_quantiles, "r") as file:
        data_quantiles = json.load(file)

    ## create obj for each case, i.e. a_oe_left, a_oe_right, b_oe_left, ...
    for label in data_params:
        labels_list.append(label)
        # print(f"datum:\n{label}")
        # print(f"freqs: {data[label]['range_freqs']}")
        # print(f"thres: {data[label]['thr_peaks']}")
        ## create objs: name, range of frequencies, and threshold for fooof fitting
        freqs = data_params[label]['range_freqs']
        thres = data_params[label]['thr_peaks']

        ## dict quantiles
        quantiles = data_quantiles[label]
        ## to dataframe
        df_quantiles = pd.DataFrame(data=quantiles['data'], columns=quantiles['columns'])
        # print(f"df_quantiles\n{df_quantiles}")
        ## obj initialization
        obj = FOOOF_class(label, freqs, thres, df_quantiles)
        ## fit fooof for each case and uses the aperiodic fit to calculates the PSD periodic component
        obj.fit_fooof()
        ## finding peak alpha in the 5- 13Hz frequency range only if fooof model was fitted
        if obj.get_fooof_model() != []:
            ## only for those who have the periodic component
            ## list of freq of psd peaks in the selected freq range [5-13 Hz] to search the peak of alpha band
            freq_peak, value_peak = obj.get_freq_peak_value(5,13)
            peak_freqs_list.append(freq_peak)
            peak_power_list.append(value_peak)
        ## save objs in a list to iterate later on
        obj_list.append(obj)

    # print(f"peak_freqs_dict:\n{peak_freqs_list}")
    ## central frequency value (median)
    alpha_freq_central = np.median(peak_freqs_list)
    ## 5 Hz range for alpha band
    alpha_freq_left  = alpha_freq_central - 2.5 ## Hz
    alpha_freq_right = alpha_freq_central + 2.5 ## Hz
    # print(f"alpha freqs band (left, central, right):\n{alpha_freq_left, alpha_freq_central, alpha_freq_right}")

    ## beta band starts at the end of alpha band and with a 10 Hz range
    beta_freq_left  = alpha_freq_right ## Hz
    beta_freq_right = alpha_freq_right + 10 ## Hz
    # print(f"beta freqs band (left and right):\n{beta_freq_left, beta_freq_right}")

    ## limits freqs bands, which define frequency ranges
    lim_freqs_bands={}
    lim_freqs_bands['alpha'] = [alpha_freq_left, alpha_freq_right]
    lim_freqs_bands['beta']  = [beta_freq_left, beta_freq_right]

    ## calculate mean values of psd limited by the previously defined frequencies
    mean_psd_bands_dict = {}
    for label_band in ['alpha', 'beta']:
        ##  freq limits for each band
        lim_freqs = lim_freqs_bands[label_band]
        # mean_psd_bands_dict[obj.get_label()] = []
        values_dict = {}
        for obj in obj_list:
            ## calculate mean values from the periodic component of the fooof decomposition
            if obj.get_fooof_model() != []:
                ## mean value from periodic component for label_band (alpha or beta)
                mean_bands = obj.calculate_mean_psd_band(label_band, lim_freqs[0], lim_freqs[1])
                ## get calculated values
                # mean_bands = obj.get_mean_psd_band(label_band)
                # print(f"mean_bands: {mean_bands}")
                values_dict[obj.get_label()] = mean_bands
            else:
                print(f"No periodic component for {obj.get_label()}. No values to calculate.")

        # print(f"values dict:\n{values_dict}")
        mean_psd_bands_dict[label_band] = values_dict
    ## results mean values per frequency band
    print(f"mean_bands dict:\n{mean_psd_bands_dict}")

    ## to normalize values and facilitate comparison between sessions, a mean value of psd that includes alpha and theta band is calculated for each state, only for rest start (reference1), i.e. a_ce_left, a_oe_right ...
    labels_ref = [f'{label_ref}_ce_left',f'{label_ref}_ce_right',f'{label_ref}_oe_left',f'{label_ref}_oe_right']
    ## psd average from the reference in the range including alpha and beta for normalization
    label_band = 'alpha+beta'
    lim_freqs_bands_all = [alpha_freq_left, beta_freq_right]

    values_ref_dict = {}
    for obj in obj_list:
        ## calculate mean values from the periodic component of the fooof decomposition
        if (obj.get_fooof_model() != []) and (obj.get_label() in labels_ref):
            ## mean value from periodic component of the reference of each state (rest start, i.e. labels that start with a_)
            # print(f"baseline {obj.get_label()}")
            obj.calculate_mean_psd_band(label_band, lim_freqs_bands_all[0], lim_freqs_bands_all[1])
            ## get calculated values
            mean_bands = obj.get_mean_psd_band(label_band)
            # print(f"mean_bands: {mean_bands}")
            values_ref_dict[obj.get_label()] = mean_bands
        else:
            pass

    print(f"values ref:\n{values_ref_dict}")

    # # calculate_band_attenuation(mean_psd_bands_dict) between two conditions: 
    # # cycling (b) and rest start (a)
    # # rest end (c) and rest start (a)
    # diff_bands_dict = {}
    # for label_band in ['alpha','beta']:
    #     diff_bands_dict[label_band] = calculate_band_attenuation(mean_psd_bands_dict, values_ref_dict, label_band, 'b', 'a')
    
    ## calculate normalized mean values for each state, alpha and beta bands
    normalized_mean_bands_dict = calculate_normalized_mean_bands(mean_psd_bands_dict, values_ref_dict, label_ref)
    print(f"normalized_mean_bands_dict:\n{normalized_mean_bands_dict}")

    plot_normalized_mean_bands(normalized_mean_bands_dict, info_p, path_fig_psd)

    # calculate_band_attenuation(subtraction) between two conditions: 
    # cycling (b) and rest start (a)
    # rest end (c) and rest start (a)

    comp_dict = {}

    print(f"val-ref: {label_val, label_ref}")

    diff_bands_dict = {}
    for label_band in ['alpha','beta']:
        diff_bands_dict[label_band] = calculate_band_diff(normalized_mean_bands_dict, label_band, label_val, label_ref)
        # calculate_band_diff(mean_psd_bands_dict, label_band, label_val, label_ref)
        ## calculate normalized mean values for each state, alpha and beta bands
    # print(f"diff bands:\n{diff_bands_dict}")
    # comp_dict[f'{label_val}_{label_ref}_diff'] = diff_bands_dict

    # print(f"{comp_dict}")


    ###########
    ## this section is dedicated to generate figures that could be used for a presentation
    ## first figure includes PSD from the states that have a fooof fitted model
    # plot_psd_states(obj_list, info_p, path_fig_psd)
    ## plot PSD periodic components, i.e. PSD - aperiodic component
    max_peak_value = np.max(peak_power_list)
    graphics_periodic_comp(obj_list, info_p, lim_freqs_bands, max_peak_value, path_fig_psd)


    ###########

    return diff_bands_dict, info_pt_dict, path

