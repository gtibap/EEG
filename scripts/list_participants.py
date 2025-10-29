import numpy as np
import mne
mne.set_log_level('error')

def participants_list(path, subject, session, abt):
        # Mme Chen
    if subject == 100:
        info_p = 'F, '
        path = path + 'aug04_MsChen/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'eeg_test-p3-chen_s01.bdf'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif', '']
            title = 'P_'+str(subject)+'_rest_and_ABT_biosemi64'+' session_'+str(session)
            Dx = ''
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
            acquisition_system = 'biosemi'
        else:
            print(f"data of session {session} do not found")
            exit()
        
    ############################
    # Mr Taha
    elif subject == 101:
        info_p = 'M, 20 y'
        Dx = 'Neuro-intact (control)'
        path = path + 'oct06_Taha/'
        print(f"path: {path}")
        if session==0:
            if abt == 0: # resting
                fn_in = 'session_'+str(session)+'/'+'eeg_taha_test_rest.bdf'
                fn_csv = ['session_'+str(session)+'/'+'annotations_rest.fif','']
                title = 'P_'+str(subject)+'_rest_biosemi64'
            else:
                fn_in = 'session_'+str(session)+'/'+'eeg_taha_test_velo.bdf'
                fn_csv = ['session_'+str(session)+'/'+'annotations_velo.fif','']
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
            acquisition_system = 'biosemi'
        else:
            print(f"data of session {session} do not found")
            exit()


    ############################
    # Mr Peltier (only resting) EEG data very noisy
    elif subject == 102:
        info_p = 'M, 50 - 70 y'
        Dx = ''

        path = path + 'oct24_Peltier/'
        
        fn_in = 'eeg_pat_oct24.bdf'
        fn_csv = ['annotations.csv', '']
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
        info_p = 'M, 50 - 70 y'
        Dx = ''

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

        info_p = ''
        Dx = ''
        
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
        info_p = 'F, 20 - 30 y'
        Dx = 'Neuro-intact (control)'
        path = path + 'aug11_Cristina/'
        if abt == 0: # resting
            fn_in = 'eeg_test_p4_s1_rest.bdf'
            fn_csv = ['annotations_rest.fif','']
            title = 'P_'+str(subject)+'_rest_biosemi64'
        else:
            fn_in = 'eeg_test_p4_s1_bike.bdf'
            fn_csv = ['annotations_velo.fif','']
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
        acquisition_system = 'biosemi'
    ############################        

    # Mme Carlie
    elif subject == 200:
        info_p = 'F, 20 - 30 y'
        Dx = 'Neuro-intact (control)'
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
        info_p = 'F, 20 - 30 y'
        Dx = 'Neuro-intact (control)'
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
        info_p = 'F, 20 - 30 y'
        Dx = 'Neuro-intact (control)'
        path = path + 'neuroplasticity/control_test/'
        fn_in = 'Control_001_20230107_063228.mff'
        fn_csv = 'annotations.csv'
        title = 'P_'+str(subject)+'_rest_geodesic_net_128'
        rows_plot = 1
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
    ############################
    # Mr Gerardo
    elif subject == 203:
        info_p = 'H, 44 y'
        Dx = 'Neuro-intact (control)'
        path = path + 'neuroplasticity/control_test/Gerardo/'
        fn_in = 'Control_Gerardo_20250526_161206.mff'
        fn_csv = ['annotations.fif','']
        title = 'P_'+str(subject)+'_rest_abt_geodesic_net_128'
        rows_plot = 2
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
    ############################
    # Mr Hamza
    elif subject == 204:
        info_p = 'H, ~20 y'
        Dx = 'Neuro-intact (control)'
        path = path + 'neuroplasticity/control_01/'
        fn_in = 'neuro_C01_ha_20230107_192613.mff'
        fn_csv = ['annotations.fif','']
        title = 'P_'+str(subject)+'_rest_abt_geodesic_net_128'
        rows_plot = 2
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
    ############################
    # Mr Phillipe
    elif subject == 205:
        info_p = 'H, ~20 y'
        Dx = 'Neuro-intact (control)'
        path = path + 'neuroplasticity/control_02/'
        fn_in = 'neuro_control_02_phi_20230115_213158.mff'
        fn_csv = ['annotations.fif','']
        title = 'P_'+str(subject)+'_rest_abt_geodesic_net_128'
        rows_plot = 2
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
    ############################
    # Mr Oussama
    elif subject == 206:
        info_p = 'H, 20 - 30 y'
        Dx = 'Neuro-intact (control)'
        path = path + 'neuroplasticity/control_03/'
        fn_in = 'neuro_control_003_Ou_20230115_233355.mff'
        fn_csv = ['annotations.fif','']
        title = 'P_'+str(subject)+'_rest_abt_geodesic_net_128'
        rows_plot = 2
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
    ############################
    # Mr Karim
    elif subject == 207:
        info_p = 'H, ~20 y'
        Dx = 'Neuro-intact (control)'
        path = path + 'neuroplasticity/control_04/'
        fn_in = 'neuro_control_04_ka_20250724_171515.mff'
        fn_csv = ['annotations.fif','']
        title = 'P_'+str(subject)+'_rest_abt_geodesic_net_128'
        rows_plot = 2
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
     ############################
    # Mr Luis Alejandro
    elif subject == 208:
        info_p = 'H, ~20 y'
        Dx = 'Neuro-intact (control)'
        path = path + 'neuroplasticity/control_05/'
        fn_in = 'neuro_control_05_luis_20250730_152919.mff'
        fn_csv = ['annotations.fif','']
        title = 'P_'+str(subject)+'_rest_abt_geodesic_net_128'
        rows_plot = 2
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'

    ############################
    ############################

    # neuro_001
    elif subject == 1:
        info_p = 'M, 58 y'
        path = path + 'neuroplasticity/n_001/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'Neuro001_session1_20250113_111350.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif', '']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = 'C - T11'
            rows_plot = 2
        elif session==1:
            fn_in = 'session_'+str(session)+'/'+'Neuro_001_3M_20230101_082244.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif', '']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = ''
            rows_plot = 2
        elif session==2:
            fn_in = 'session_'+str(session)+'/'+'neuro_001_6m_20230108_192933.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif', '']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = ''
            rows_plot = 2
        else:
            pass
        # fn_out = 'neuro_001_ann'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
        
    ############################
    ############################
    # neuro_002
    elif subject == 2:
        info_p = 'M, 64 y'
        path = path + 'neuroplasticity/n_002/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'Neuro_002_20250117_110033.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif', '']
            title = 'P_'+str(subject)+'_rest_geodesic_net_128'
            Dx = 'A - L4'
            rows_plot = 1
            # fn_out = 'neuro_002_ann'
            ## read raw data
        elif session==1:
            fn_in = 'session_'+str(session)+'/'+'neuro_002_suivi_3m_20250522_160123.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif', '']
            title = 'P_'+str(subject)+'_rest_geodesic_net_128'
            Dx = ''
            rows_plot = 1
            # fn_out = 'neuro_002_ann'
            ## read raw data
        elif session==2:
            ## data collected in the lab with Oussama during the morning
            ## patient able to walk (with a cane)
            fn_in = 'session_'+str(session)+'/'+'neuro_002_6m_suivi_20250811_104052.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif', '']
            title = 'P_'+str(subject)+'_rest_geodesic_net_128'
            Dx = ''
            rows_plot = 1
            # fn_out = 'neuro_002_ann'
            ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
        
    ############################
    ############################
    # neuro_003
    elif subject == 3:
        info_p = 'F, 61 y'
        path = path + 'neuroplasticity/n_003/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'neuro_003_20221231_080823.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = 'C - C5'
            rows_plot = 2
        elif session==1:
            fn_in = 'session_'+str(session)+'/'+'Neuro_003_3M_20221231_090044.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_geodesic_net_128'+' session_'+str(session)
            Dx = ''
            rows_plot = 1
        elif session==2:
            ## date: Oct 22, 2025, 13h30, in the lab. We tried cycling but not possible... patient was not feeling well during cycling... too heavy legs for the bike
            fn_in = 'session_'+str(session)+'/'+'neuro_003_6mois_20251022_020453.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_geodesic_net_128'+' session_'+str(session)
            Dx = ''
            rows_plot = 1
        else:
            pass
        # fn_out = 'neuro_003_ann'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
    ############################
    # neuro_004
    elif subject == 4:
        info_p = 'M, 19 y'
        if session==0:
            path = path + 'neuroplasticity/n_004/'
            fn_in = 'session_'+str(session)+'/'+'neuro_004_20230102_063924.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'
            Dx = 'D - T3'
            rows_plot = 2
        elif session==1:
            path = path + 'neuroplasticity/n_004/'
            fn_in = 'session_'+str(session)+'/'+'neuro_004_suivi_3m_20250523_113939.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'
            Dx = ''
            rows_plot = 2
        # fn_out = 'neuro_004_ann'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
    ############################
    # neuro_005
    elif subject == 5:
        info_p = 'M, 18 y'
        if session==0:
            path = path + 'neuroplasticity/n_005/'
            fn_in = 'session_'+str(session)+'/'+'Neuro_005_20250106_111519.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'
            Dx = 'C - C6'
            rows_plot = 2
        elif session==1:
            path = path + 'neuroplasticity/n_005/'
            fn_in = 'session_'+str(session)+'/'+'neuro_5_suivi_3m_20250602_111411.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'
            Dx = ''
            rows_plot = 2
        # fn_out = 'neuro_005_ann'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
    ############################
    # neuro_006
    elif subject == 6:
        info_p = 'M, 55 y'
        path = path + 'neuroplasticity/n_006/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'NEURO_006_20250111_113255.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = 'D - C5'
            rows_plot = 2
        elif session==1:
            fn_in = 'session_'+str(session)+'/'+'neuro006_3m_20230107_082740.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_geodesic_net_128'+' session_'+str(session)
            Dx = ''
            rows_plot = 1
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
        
    ############################
    ############################
    # neuro_007
    elif subject == 7:
        info_p = 'M, 59 y'
        path = path + 'neuroplasticity/n_007/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'neuro007_S1_20221231_100552.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = 'B - C6'
            rows_plot = 2
        elif session==1:
            fn_in = 'session_'+str(session)+'/'+'neuro_007_3m_20230108_081936.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = ''
            rows_plot = 2
        # fn_out = 'neuro_007_ann'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
        
    ############################
    ############################
    # neuro_008
    elif subject == 8:
        info_p = 'M, 40 y'
        path = path + 'neuroplasticity/n_008/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'neuro_008_session_0_20250507_115913.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = 'D - C5'
            rows_plot = 2
        elif session==1:
            fn_in = ''
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = ''
            rows_plot = 2
        # fn_out = 'neuro_007_ann'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
    
    ############################
    ############################
    # neuro_009
    elif subject == 9:
        info_p = 'M, 63 y'
        path = path + 'neuroplasticity/n_009/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'Neuro_009_20250509_142653.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = 'D - C1'
            rows_plot = 2
        elif session==1:
            fn_in = 'session_'+str(session)+'/'+'neuro_009_suivi_3m_20250813_102909.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = 'D - C1'
            rows_plot = 2
        elif session==2:
            ## in the lab with the iMac
            fn_in = 'session_'+str(session)+'/'+'Neuro-009- session suivi 6 mois _20251020_101357.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = ''
            rows_plot = 2
        # fn_out = 'neuro_007_ann'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'

    ############################
    # neuro_010
    elif subject == 10:
        info_p = 'F, '
        path = path + 'neuroplasticity/n_010/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'abc001_20250618_104348.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = 'A - T3'
            rows_plot = 2
        elif session==1:
            fn_in = ''
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = ''
            rows_plot = 2
        else:
            print(f"Data from session {session} did not find.")
            return 0
        # fn_out = 'neuro_007_ann'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'

    ############################
    # neuro_011
    elif subject == 11:
        info_p = 'M, 20 y'
        path = path + 'neuroplasticity/n_011/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'neuro_011_20250620_150332.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = 'A - T5'
            rows_plot = 2
        elif session==1:
            fn_in = ''
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = ''
            rows_plot = 2
        else:
            print(f"Data from session {session} did not find.")
            return 0
        # fn_out = 'neuro_007_ann'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
    ############################
    ############################
    # neuro_012
    elif subject == 12:
        info_p = 'M, 56 y'
        path = path + 'neuroplasticity/n_012/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'Neuro_012_20230103_225313.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = 'C - C1'
            rows_plot = 2
        elif session==1:
            fn_in = ''
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = ''
            rows_plot = 2
        else:
            print(f"Data from session {session} did not find.")
            return 0
        # fn_out = 'neuro_007_ann'
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
    ############################
    ############################
    # neuro_013
    elif subject == 13:
        info_p = 'M, 64 y'
        path = path + 'neuroplasticity/n_013/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'neuro_013_20230114_193005.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = 'D - C5'
            rows_plot = 2
        elif session==1:
            ## M Oct 29, 2025, 10am, in the lab with the macbook
            fn_in = 'session_'+str(session)+'/'+'Neuro_013_3mois_20251029_102413.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = 'D - C5'
            rows_plot = 2
        else:
            print(f"Data from session {session} did not find.")
            return 0
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
    ############################
    ############################
    # neuro_014
    elif subject == 14:
        info_p = 'F, 51 y'
        path = path + 'neuroplasticity/n_014/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'neuro_014_20250725_141440.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = 'D - '
            rows_plot = 2
        elif session==1:
            ## data collected friday, Oct 17, 2025 dans le lab with iMac
            fn_in = 'session_'+str(session)+'/'+'Neuro_014-3mois_20251017_112508.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = 'D - '
            rows_plot = 2

        else:
            print(f"Data from session {session} did not find.")
            return 0
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
    ############################
    ############################
    # neuro_015
    elif subject == 15:
        info_p = 'M, 69 y'
        path = path + 'neuroplasticity/n_015/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'neuro_15_go_20250804_114337.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = 'B - C7'
            rows_plot = 2
        else:
            print(f"Data from session {session} did not find.")
            return 0
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
    ############################
    ############################
    # neuro_016
    elif subject == 16:
        info_p = 'M, '
        path = path + 'neuroplasticity/n_016/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'neuro_016_Gi_20250818_114905.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = 'A - '
            rows_plot = 2
        else:
            print(f"Data from session {session} did not find.")
            return 0
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
    ############################
    ############################
    # neuro_017
    elif subject == 17:
        info_p = ' '
        path = path + 'neuroplasticity/n_017/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'Neuro_0017_2sem_20250912_103330.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = ' '
            rows_plot = 2
        else:
            print(f"Data from session {session} did not find.")
            return 0
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
    ############################
    ############################
    # neuro_018
    elif subject == 18:
        info_p = ' '
        path = path + 'neuroplasticity/n_018/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'Neuro_018_20250925_141202.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = ' '
            rows_plot = 2
        else:
            print(f"Data from session {session} did not find.")
            return 0
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
    ############################
    ############################
    # neuro_019
    elif subject == 19:
        info_p = ' '
        path = path + 'neuroplasticity/n_019/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'Neuro_019_20250923_115932.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = ' '
            rows_plot = 2
        else:
            print(f"Data from session {session} did not find.")
            return 0
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
    ############################
    ############################
    # neuro_020
    elif subject == 20:
        info_p = ' '
        path = path + 'neuroplasticity/n_020/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'Neuro_020_20251002_122042.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = ' '
            rows_plot = 2
        else:
            print(f"Data from session {session} did not find.")
            return 0
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
    ############################
    ############################
    # neuro_021
    elif subject == 21:
        info_p = ' '
        path = path + 'neuroplasticity/n_021/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'Neuro 21_20251006_103615.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = ' '
            rows_plot = 2
        else:
            print(f"Data from session {session} did not find.")
            return 0
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
    ############################
    ############################
    # neuro_022
    elif subject == 22:
        info_p = ' '
        path = path + 'neuroplasticity/n_022/'
        if session==0:
            fn_in = 'session_'+str(session)+'/'+'Neuro_022_20251016_125438.mff'
            fn_csv = ['session_'+str(session)+'/'+'annotations.fif','']
            title = 'P_'+str(subject)+'_rest_and_ABT_geodesic_net_128'+' session_'+str(session)
            Dx = ' '
            rows_plot = 2
        else:
            print(f"Data from session {session} did not find.")
            return 0
        ## read raw data
        raw_data = mne.io.read_raw_egi(path + fn_in, preload=True)
        acquisition_system = 'geodesic'
    
    
    else:
        fn_in = ''
        fn_csv = ''
        raw_data = np.NaN
    

    return path, fn_in, fn_csv, raw_data, title, rows_plot, acquisition_system, info_p, Dx