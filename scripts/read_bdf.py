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

def main(args):
    
    print(f'arg {args[1]}')
    print(f'arg {args[2]}')
    print(f'arg {args[3]}')
    
    title=args[3]

    # Filter settings
    low_cut = 0.1
    hi_cut  = 40    
    # raw_data = mne.io.read_raw_bdf('../001SCNT_TPD_T3.bdf')
    # raw_data = mne.io.read_raw_bdf('../data/eeg_test-p2_s1.bdf')
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

    # for index, row in df_events.iterrows():
        # print(row['section'], row['action'])
        # if row['action'] == 'closed_eyes_start':
            # id_ce = index
        # elif row['action'] == 'opened_eyes_start':
            # id_oe = index
        # else:
            # pass
        # print(f'')
        
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
        # print(f'index: {index}, row: {row}')
        # print(row['section'], row['action'])
        # if row['action'] == 'closed_eyes_start':
            # id_ce = index
        # elif row['action'] == 'opened_eyes_start':
            # id_oe = index
        # else:
            # pass
        # print(f'')
    
    # print(df_cs)
    # print(f'df_cs.index: {df_cs.index}')
    # print(f'{8} in df_cs ? {8 in df_cs.index}')
    # print(f'{9} in df_cs ? {9 in df_cs.index}')
    # print(df_os)
    # print(df_ps)
    
    
    
    
    
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
    
    offset=0.002
    
    # signals = raw_filt.get_data(picks=['Cz','Fpz','Oz'])
    signals = raw_filt.get_data(picks=['P3','P4','O1','O2'])
    # data_2 = raw_filt.get_data(picks=['Fp1','O2'],tmin=0,tmax=60*5)
    # data_2 = raw_filt.get_data(picks=['OCU3','ECG5'],tmin=0,tmax=60*5)
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,5), squeeze=False)
    ax = ax.reshape(-1)
    # ax[0].plot(signals[1]+offset, label='Fpz')
    # ax[0].plot(signals[0], label='Cz')
    ax[0].plot(signals[0], label='P3')
    
    # print()
    for ta,section,action in zip(df_events['time'].tolist(), df_events['section'].tolist(), df_events['action'].tolist()):
        ts = int((ta - start_eeg)*sfreq)
        ax[0].axvline(x = ts, color = 'b', alpha=0.5)
        print(f'{ts}, {section}, {action}')
    
    ax[0].set_ylim([-0.0005, 0.0005])
    ax[0].set_xlim([0, 150000])
    # ax[0].set_xlim([0, len(signals[0])])
    
    ax[0].set_ylabel('amplitude [uV]')
    ax[0].set_xlabel(f'samples ({sfreq} Hz)')
    
    plt.legend(loc='lower right')
    plt.suptitle(f'{title}')
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
    obj_signals.freq_components(time_ce_b[1])
    obj_signals.freq_components(time_ce_b[2])
    
    
    
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
