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
    
    print(f'{df_events}')
    
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
    
    signals = raw_filt.get_data(picks=['Cz','Fpz','Oz'])
    # data_2 = raw_filt.get_data(picks=['Fp1','O2'],tmin=0,tmax=60*5)
    # data_2 = raw_filt.get_data(picks=['OCU3','ECG5'],tmin=0,tmax=60*5)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,5), squeeze=False)
    ax = ax.reshape(-1)
    ax[0].plot(signals[1]+offset, label='Fpz')
    ax[0].plot(signals[0], label='Cz')
    ax[0].plot(signals[2]-offset, label='Oz')
    
    print()
    for ta in df_events['time'].tolist():
        ts = int((ta - start_eeg)*sfreq)
        ax[0].axvline(x = ts, color = 'b', alpha=0.5)
    
    ax[0].set_ylim([-0.005, 0.005])
    ax[0].set_xlim([0, len(signals[0])])
    
    ax[0].set_ylabel('amplitude [uV]')
    ax[0].set_xlabel('time [s]')
    
    plt.legend(loc='lower right')
    plt.suptitle(f'P002 - {title}')
    # raw_data.set_montage('standard_1005')
    # raw_data.plot_sensors()
    
    # print(data_2.shape)
    # print(data_2)
    
    
    
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
