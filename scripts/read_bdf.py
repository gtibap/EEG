#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  read_bdf.py
#  

import mne
mne.set_log_level('error')

import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd

from class_read_bdf import EEG_components

def labels_activity(ax, pos_y):
    
    ax.axvline(x = 4, color = 'tab:gray', alpha=0.5)
    ax.axvline(x = 8, color = 'tab:gray', alpha=0.5)
    ax.axvline(x = 13,color = 'tab:gray', alpha=0.5)
    
    ax.annotate('Delta', xy=(0.5, pos_y),
                    color='blue',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                    )
    ax.annotate('Theta', xy=(4.3, pos_y),
                    color='blue',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                    )
    ax.annotate('Alpha', xy=(8.7, pos_y),
                    color='blue',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                    )
    ax.annotate('Beta', xy=(15, pos_y),
                    color='blue',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                    )
    return ax


def main(args):
    
    print(f'arg {args[1]}')
    print(f'arg {args[2]}')
    print(f'arg {args[3]}')
    
    title=args[3]

    # Filter settings
    low_cut = 0.1
    hi_cut  = 40    
   
    raw_data = mne.io.read_raw_bdf(args[1], preload=True)
    raw_filt = raw_data.copy().filter(l_freq=low_cut, h_freq=hi_cut)
    
    list_date =[]
    list_time=[]
    list_action=[]
    list_section=[]
    
    with open(args[2]) as f:
        # Read the contents of the file into a variable
        # names = f.read()
        for num, line in enumerate(f):
            list_line = line.split()
            
            ## converting hh:mm:ss to seconds
            arr_time = np.array(list_line[1].split(':')).astype(float)
            t_s = arr_time[0]*3600 + arr_time[1]*60 + arr_time[2]
            
            list_date.append(list_line[0])
            list_time.append(t_s)
            list_action.append(list_line[2])
            list_section.append(list_line[4])
            
            # print(f'{list_line}')
            # print(f'line {num}, {line}') 
    df_events = pd.DataFrame()
    
    df_events['date']=list_date
    df_events['time']=list_time
    df_events['section']=list_section
    df_events['action']=list_action
    # df_events = df_events.reset_index()  # make sure indexes pair with number of rows
    
    print(f'{df_events}')
    
    ## DataFrame for closed_eyes and opened_eyes
    df_c_eyes=pd.DataFrame()
    df_o_eyes=pd.DataFrame()

        
    # print(df_events.loc[df_events['action'].isin(['closed_eyes_start','opened_eyes_start'])])
    # df_cs = df_events.loc[(df_events['action']=='closed_eyes_start') & (df_events['section']=='A')]
    # df_os = df_events.loc[(df_events['action']=='opened_eyes_start') & (df_events['section']=='A')]
    # df_ps = df_events.loc[(df_events['action']=='pause_start') & (df_events['section']=='A')]
    df_cs = df_events.loc[(df_events['action']=='closed_eyes_start')]
    df_os = df_events.loc[(df_events['action']=='opened_eyes_start')]
    df_ps = df_events.loc[(df_events['action']=='pause_start')]
    
    time_ce_a=[]
    time_ce_b=[]
    print(df_cs)
    for id_cs, row in df_cs.iterrows():
        id_os = id_cs+1
        # print(f'id_os:{id_os}')
        if id_os in df_os.index:
            # print(f'{df_os.loc[[id_os]]}')
            if df_cs._get_value(id_cs,'section') == df_os._get_value(id_os,'section') and df_os._get_value(id_os,'section') == 'A':
                time_ce_a.append([df_cs._get_value(id_cs,'time'),df_os._get_value(id_os,'time')])
            elif df_cs._get_value(id_cs,'section') == df_os._get_value(id_os,'section') and df_os._get_value(id_os,'section') == 'B':
                time_ce_b.append([df_cs._get_value(id_cs,'time'),df_os._get_value(id_os,'time')])
            else:
                pass
        else:
            pass
    
    print(f'time closed eyes A: {time_ce_a}')
    print(f'time closed eyes B: {time_ce_b}')

    # print(type(raw_data.info))
    print(f'{type(raw_data.info)}, {raw_data.info}')
    print(f"sfreq: {type(raw_data.info['sfreq'])}, {raw_data.info['sfreq']}")
    print(f"date: {raw_data.info['meas_date'].hour}, {raw_data.info['meas_date'].minute}, {raw_data.info['meas_date'].second}")
    
    sfreq = raw_data.info['sfreq']
    meas_date = raw_data.info['meas_date']
    
    start_eeg = meas_date.hour*3600 + meas_date.minute*60 + meas_date.second
    print(f'start eeg: {start_eeg} s')
    
    
    
    
    # data_dict = raw_data.__dict__
    # print(data_dict)
    # raw_dict  = data_dict["_raw_extras"][0]
    # print(f'extras: {raw_dict}')
    
    # print(type(data_dict))
    # print(raw_dict["ch_names"])
    # print(raw_dict["ch_names"][64])
    # print(raw_dict["ch_names"][64+63])
    # print(type())
    
    
    # raw_filt.plot_psd(picks=['Fp1'], fmax=10);
    # raw_filt.plot_psd(picks=['Fp1'], fmax=10);
    
    # raw_data.plot(picks=['Fp1'])
    # raw_filt.plot(picks=['Fp1'])
    # raw_data.plot_psd(picks=['Fp1'], fmax=400);
    
    
    # data_2 = raw_data.get_data(picks=['Fp1'])
    # print(data_2.shape)
    # data_2 = raw_data.get_data(picks=['Fp1','O2'],tmin=0,tmax=60*5)
    # data_2 = raw_data.get_data(picks=['OCU3','ECG5'],tmin=0,tmax=60*5)
    # plt.plot(data_2[1])
    
    offset=-0.0001

    label_signals=['P3','P4','O1','O2']
    signals = raw_filt.get_data(picks=label_signals)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,5), squeeze=False)
    ax = ax.reshape(-1)
    # ax[0].plot(signals[1]+offset, label='Fpz')
    # ax[0].plot(signals[0], label='Cz')
    ax[0].plot(signals[0]+offset*0, label=label_signals[0])
    ax[0].plot(signals[1]+offset*1, label=label_signals[1])
    ax[0].plot(signals[2]+offset*2, label=label_signals[2])
    ax[0].plot(signals[3]+offset*3, label=label_signals[3])
    
    # print()
    for ta,section,action in zip(df_events['time'].tolist(), df_events['section'].tolist(), df_events['action'].tolist()):
        ts = int((ta - start_eeg)*sfreq)
        ax[0].axvline(x = ts, color = 'b', alpha=0.5)
        print(f'{ts}, {section}, {action}')
    
    fig.canvas.draw()
    ax[0].set_ylim([-0.0004, 0.0001])
    
    ## Subject 1
    # ax[0].set_xlim([195000, 326000])
    # pos_xlabel1=222000
    # pos_xlabel2=285000

    ## Subject 2
    ax[0].set_xlim([105000, 175000])
    pos_xlabel1=120000
    pos_xlabel2=155000    
    
    
    ## Subject 3
    # ax[0].set_xlim([65000, 140000])
    # pos_xlabel1=80500
    # pos_xlabel2=115500
    
    x_labels = [item.get_text() for item in ax[0].get_xticklabels()]
    x_labels = (np.array(x_labels).astype(int)/(sfreq)).astype(int)
    print(f'x_labels {x_labels}')
    
    ax[0].set_xticklabels(x_labels)
    
    ax[0].set_yticks([offset*0,offset*1,offset*2,offset*3])
    ax[0].set_yticklabels(label_signals)
    
    # ax[0].set_ylabel('amplitude [uV]')
    ax[0].set_xlabel(f'time (s)')
    ax[0].annotate('eyes-closed', xy=(pos_xlabel1, -offset*0.5),
                    color='blue',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                    )
    ax[0].annotate('eyes-opened', xy=(pos_xlabel2, -offset*0.5),
                    color='blue',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                    )
    # plt.legend(loc='lower right')
    ax[0].set_title(f'{title}')
    
    plt.savefig(f'figures/{title}.png', bbox_inches='tight')
    # plt.suptitle(f'{title}')
    # raw_data.set_montage('standard_1005')
    # raw_data.plot_sensors()
    
    # print(data_2.shape)
    # print(data_2)
    
    time_ce_a=np.array(time_ce_a)
    time_ce_b=np.array(time_ce_b)
    
    time_ce_a = ((time_ce_a - start_eeg)*sfreq).astype(int)
    time_ce_b = ((time_ce_b - start_eeg)*sfreq).astype(int)
    
    print(f'{time_ce_a}')
    print(f'{time_ce_b}')
    ## frequency components signals segments closed eyes and open eyes
    obj_signals = EEG_components(signals, sfreq)
    
    fig1, ax1 = plt.subplots(nrows=2, ncols=2, figsize=(10, 7),sharex=True, sharey=True)
    ax1 = ax1.reshape(-1)
    
     # ## vertical lines
    # ax1[0].axvline(x = 4, color = 'tab:gray', alpha=0.5)
    # ax1[0].axvline(x = 8, color = 'tab:gray', alpha=0.5)
    # ax1[0].axvline(x = 13,color = 'tab:gray', alpha=0.5)
    
    # ax1[1].axvline(x = 4, color = 'tab:gray', alpha=0.5)
    # ax1[1].axvline(x = 8, color = 'tab:gray', alpha=0.5)
    # ax1[1].axvline(x = 13,color = 'tab:gray', alpha=0.5)
    
    # ax1[2].axvline(x = 4, color = 'tab:gray', alpha=0.5)
    # ax1[2].axvline(x = 8, color = 'tab:gray', alpha=0.5)
    # ax1[2].axvline(x = 13,color = 'tab:gray', alpha=0.5)
    
    # ax1[3].axvline(x = 4, color = 'tab:gray', alpha=0.5)
    # ax1[3].axvline(x = 8, color = 'tab:gray', alpha=0.5)
    # ax1[3].axvline(x = 13,color = 'tab:gray', alpha=0.5)
    
    arr = np.array(time_ce_a[0])
    arr[arr<0]=0
    ax1 = obj_signals.freq_components(arr, ax1,'0')
    
    arr = np.array(time_ce_a[1])
    arr[arr<0]=0
    ax1 = obj_signals.freq_components(arr, ax1,'1')
    
    arr = np.array(time_ce_a[2])
    arr[arr<0]=0
    ax1 = obj_signals.freq_components(arr, ax1,'2')
    
    # pos_y=4.0e-6
    pos_y=1.6e-5
    # label_signals=['P3','P4','O1','O2']
    ax1[0] = labels_activity(ax1[0], pos_y)
    ax1[1] = labels_activity(ax1[1], pos_y)
    ax1[2] = labels_activity(ax1[2], pos_y)
    ax1[3] = labels_activity(ax1[3], pos_y)
                    
    ax1[0].set_title(label_signals[0])
    ax1[1].set_title(label_signals[1])
    ax1[2].set_title(label_signals[2])
    ax1[3].set_title(label_signals[3])
    
    ax1[2].set_xlabel('frequency (Hz)')
    ax1[3].set_xlabel('frequency (Hz)')
    
    ax1[0].set_ylabel('PSD (V**2/Hz)')
    ax1[2].set_ylabel('PSD (V**2/Hz)')
    
    plt.suptitle(f'{title}__eyes-closed')
    
    plt.savefig(f'figures/{title}_freq.png', bbox_inches='tight')

    # plt.legend(loc='lower right')
    
    
    # ax1 = obj_signals.freq_components(arr, ax1)
    
    # obj_signals.freq_components(time_ce_a[3])
    # obj_signals.freq_components(time_ce_b[2])
    
    
    
    # print(raw_data['cal'])
    # raw_data.plot()
    # data = raw_data.get_data()
    # print(type(data))
    # print(data.shape)
    # plt.plot(data[133])
    
    # raw_data.plot_psd(fmax=100)
    
    plt.show()
    
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
