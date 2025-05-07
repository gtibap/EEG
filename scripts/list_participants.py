import numpy as np
import mne
mne.set_log_level('error')

def participants_list(path, subject, session, abt):
        # Mme Chen
    if subject == 100:
        # print(f'subject in:{subject}')
        path = path + 'aug04_MsChen/'
        fn_in = 'eeg_test-p3-chen_s01.bdf'
        fn_csv = ['annotations.csv','new_annotations.csv']
        title = 'P_'+str(subject)+'_rest_and_ABT_biosemi64'
        rows_plot = 2

        ## read raw data
        raw_data = mne.io.read_raw_bdf(path + fn_in, preload=True)
        ## select channels
        sel_ch = np.arange(64,128)
        raw_data.pick(sel_ch)
        ## rename channels
        maps_dict = {'C1-1':'C1', 'C2-1':'C2', 'C3-1':'C3', 'C4-1':'C4', 'C5-1':'C5', 'C6-1':'C6'}
        mne.rename_channels(raw_data.info, maps_dict)
        ## electrodes montage
        raw_data.set_montage("biosemi64")
        # fig = raw_data.plot_sensors(show_names=True, sphere='eeglab')
        # print(f'raw_data.info:\n{raw_data.info}')
        
    ############################
    # Mr Taha
    elif subject == 101:
        path = path + 'oct06_Taha/'
        if abt == 0: # resting
            fn_in = 'eeg_taha_test_rest.bdf'
            fn_csv = ['annotations_rest.csv', 'new_annotations_rest']
            title = 'P_'+str(subject)+'_rest_biosemi64'
        else:
            fn_in = 'eeg_taha_test_velo.bdf'
            fn_csv = ['annotations_velo.csv','new_annotations_velo']
            title = 'P_'+str(subject)+'_ABT_biosemi64'
        rows_plot = 1
        ## read raw data
        raw_data = mne.io.read_raw_bdf(path + fn_in, preload=True)
        ## select channels
        sel_ch = np.arange(64,128)
        raw_data.pick(sel_ch)
        ## electrodes montage
        raw_data.set_montage("biosemi64")
        # fig = raw_data.plot_sensors(show_names=True, sphere='eeglab')

    ############################
    # Mr Peltier (only resting) EEG data very noisy
    elif subject == 102:
        path = path + 'oct24_Peltier/'
        
        fn_in = 'eeg_pat_oct24.bdf'
        fn_csv = ['annotations.csv', 'new_annotations.csv']
        title = 'P_'+str(subject)+'_rest_biosemi64'
        rows_plot = 1

        ## read raw data
        raw_data = mne.io.read_raw_bdf(path + fn_in, preload=True)
        ## select channels
        sel_ch = np.arange(0,64)
        raw_data.pick(sel_ch)
        ## electrodes montage
        raw_data.set_montage("biosemi64")
        # fig = raw_data.plot_sensors(show_names=True, sphere='eeglab')

    ############################
    # Mr Eric feb20
    elif subject == 103:
        path = path + 'feb20_Eric/'
        
        if abt == 0: 
            # resting
            fn_in = 'eeg_001_session1_rest.bdf'
            fn_csv = ['annotations_rest.csv', '_rest']
            title = 'P_'+str(subject)+'_rest_biosemi64'
        else:
            # active-based therapy (ABT)
            fn_in = 'eeg_001_session1_velo.bdf'
            fn_csv = ['annotations_velo.csv', '_velo']
            title = 'P_'+str(subject)+'_ABT_biosemi64'
        rows_plot = 1
        ## read raw data
        raw_data = mne.io.read_raw_bdf(path + fn_in, preload=True)
        ## select channels
        sel_ch = np.arange(0,64)
        raw_data.pick(sel_ch)
        ## electrodes montage
        raw_data.set_montage("biosemi64")
        # fig = raw_data.plot_sensors(show_names=True, sphere='eeglab')

    ############################
    # feb22
    elif subject == 104:
        path = path + 'feb22/'
        
        if abt == 0: 
            # resting
            fn_in = 'eeg_002_session1_rest.bdf'
            fn_csv = 'annotations_rest_1.csv'
            title = 'P_'+str(subject)+'_rest_biosemi64'
        elif abt == 1:
            # active-based therapy (ABT) 1
            fn_in = 'eeg_002_session1_velo.bdf'
            fn_csv = 'annotations_velo_1.csv'
            title = 'P_'+str(subject)+'_ABT_biosemi64'
        else:
            # active-based therapy (ABT) 2
            fn_in = 'eeg_002_session1_velo2.bdf'
            fn_csv = 'annotations_velo_2.csv'
            title = 'P_'+str(subject)+'_ABT_biosemi64'
        rows_plot = 1
        ## read raw data
        raw_data = mne.io.read_raw_bdf(path + fn_in, preload=True)
        ## select channels
        sel_ch = np.arange(0,64)
        raw_data.pick(sel_ch)
        ## electrodes montage
        raw_data.set_montage("biosemi64")
        # fig = raw_data.plot_sensors(show_names=True, sphere='eeglab')

    ############################
    # Ms Cristina
    elif subject == 105:
        path = path + 'aug11_Cristina/'
        if abt == 0: # resting
            fn_in = 'eeg_test_p4_s1_rest.bdf'
            fn_csv = ['annotations_rest.csv','_rest']
            title = 'P_'+str(subject)+'_rest_biosemi64'
        else:
            fn_in = 'eeg_test_p4_s1_bike.bdf'
            fn_csv = ['annotations_velo.csv','_velo']
            title = 'P_'+str(subject)+'_ABT_biosemi64'
        rows_plot = 1
        ## read raw data
        raw_data = mne.io.read_raw_bdf(path + fn_in, preload=True)
        ## select channels
        sel_ch = np.arange(64,128)
        raw_data.pick(sel_ch)
        ## electrodes montage
        raw_data.set_montage("biosemi64")
        # fig = raw_data.plot_sensors(show_names=True, sphere='eeglab')
    ############################        

    # Mme Carlie
    elif subject == 200:
        path = path + 'apic_data/initial_testing/p01/'
        fn_in = 'APIC_TEST_CM_20241205_023522.mff'
        fn_csv = 'saved-annotations.csv'
        title = 'P_'+str(subject)+'_rest_and_ABT_biosemi64'
        rows_plot = 2
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        # raw_data.plot_sensors(show_names=True,)
        
    ############################
    # Mme Iulia
    elif subject == 201:
        path = path + 'apic_data/initial_testing/p02/'
        fn_in = 'APIC_TEST_IULIA_20241217_011900.mff'
        fn_csv = 'saved-annotations.csv'
        title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'
        rows_plot = 2
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        # fig = raw_data.plot_sensors(show_names=True,)

    ############################
    # Mme Dafne
    elif subject == 202:
        path = path + 'neuroplasticity/control_test/'
        fn_in = 'Control_001_20230107_063228.mff'
        fn_csv = 'annotations.csv'
        title = 'P_'+str(subject)+'_rest_geodesic_net_128'
        rows_plot = 1
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        # fig = raw_data.plot_sensors(show_names=True,)

    ############################
    # neuro_001
    elif subject == 1:
        path = path + 'neuroplasticity/n_001/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'Neuro001_session1_20250113_111350.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.csv', 'session_'+str(session)+'/'+'new_annotations']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            rows_plot = 2
        elif session==1:
            fn_in = 'session_'+str(session)+'/'+'Neuro_001_3M_20230101_082244.mff'
            fn_csv = 'session_'+str(session)+'/'+'annotations.csv'
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            rows_plot = 2
        else:
            pass
        # fn_out = 'neuro_001_ann'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        
    ############################
    ############################
    # neuro_002
    elif subject == 2:
        path = path + 'neuroplasticity/n_002/'
        fn_in = 'Neuro_002_20250117_110033.mff'
        fn_csv = 'annotations.csv'
        title = 'P_'+str(subject)+'_rest_geodesic_net_128'
        rows_plot = 1
        # fn_out = 'neuro_002_ann'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        
    ############################
    ############################
    # neuro_003
    elif subject == 3:
        path = path + 'neuroplasticity/n_003/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'neuro_003_20221231_080823.mff'
            fn_csv = 'session_'+str(session)+'/'+'annotations.csv'
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            rows_plot = 2
        elif session==1:
            fn_in = 'session_'+str(session)+'/'+'Neuro_003_3M_20221231_090044.mff'
            fn_csv = 'session_'+str(session)+'/'+'annotations.csv'
            title = 'P_'+str(subject)+'_rest_geodesic_net_128'+' session_'+str(session)
            rows_plot = 1
        else:
            pass
        # fn_out = 'neuro_003_ann'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
    ############################
    # neuro_004
    elif subject == 4:
        path = path + 'neuroplasticity/n_004/'
        fn_in = 'neuro_004_20230102_063924.mff'
        fn_csv = 'annotations.csv'
        title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'
        rows_plot = 2
        # fn_out = 'neuro_004_ann'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
    ############################
    # neuro_005
    elif subject == 5:
        path = path + 'neuroplasticity/n_005/'
        fn_in = 'Neuro_005_20250106_111519.mff'
        fn_csv = 'annotations.csv'
        title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'
        rows_plot = 2
        # fn_out = 'neuro_005_ann'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
    ############################
    # neuro_006
    elif subject == 6:
        path = path + 'neuroplasticity/n_006/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'NEURO_006_20250111_113255.mff'
            fn_csv = 'session_'+str(session)+'/'+'annotations.csv'
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            rows_plot = 2
        elif session==1:
            fn_in = 'session_'+str(session)+'/'+'neuro006_3m_20230107_082740.mff'
            fn_csv = 'session_'+str(session)+'/'+'annotations.csv'
            title = 'P_'+str(subject)+'_rest_geodesic_net_128'+' session_'+str(session)
            rows_plot = 1
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        
    ############################
    ############################
    # neuro_007
    elif subject == 7:
        path = path + 'neuroplasticity/n_007/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'neuro007_S1_20221231_100552.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.csv','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            rows_plot = 2
        elif session==1:
            fn_in = 'session_'+str(session)+'/'+'neuro_007_3m_20230108_081936.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.csv','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            rows_plot = 2
        # fn_out = 'neuro_007_ann'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        
    ############################
    else:
        fn_in = ''
        fn_csv = ''
        raw_data = np.NaN
    

    return path, fn_in, fn_csv, raw_data, title, rows_plot