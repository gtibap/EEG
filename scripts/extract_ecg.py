import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from list_participants import participants_list

def main(args):
    
    # filename_in = args[1]
    # filename_out = args[2]

    print(f'arg {args[1]}') ## folder location
    print(f'arg {args[2]}') ## subject = {0:Mme Chen, 1:Taha, 2:Carlie, 3:Iulia, 4:A. Caron}
    print(f'arg {args[3]}') ## session = {1:time zero, 2:three months, 3:six months}
    print(f'arg {args[4]}') ## ABT = {0:resting, 1:biking}
    
    path=args[1]
    subject= int(args[2])
    session=int(args[3])
    abt= int(args[4])

    # path, fn_in, fn_csv, raw, fig_title, rows_plot, acquisition_system = participants_list(path, subject, session, abt)
    path, fn_in, fn_csv, raw, fig_title, rows_plot, acquisition_system, info_p, Dx = participants_list(path, subject, session, 0)

    print(f'filename: {fn_in}') ## folder location
    
    # raw = mne.io.read_raw_egi(filename_in, preload=True)
    ecg_data, times = raw.get_data(picks=['ecg'], return_times=True)
    print(f"ecg size: {ecg_data.shape}")
    # print(f"times: {times}")
    # print(f"ecg: {ecg_data}")

    d = {'time (s)':times, 'ecg (V)':ecg_data[0]}
    df=pd.DataFrame(data=d)
    print(f"df:\n{df}")

    filename_out = f"ecg/n{str(subject).zfill(2)}_s{session}.csv"
    df.to_csv(filename_out, index=False)

    fig, ax = plt.subplots(1,1)
    ax.plot(df['time (s)'],df['ecg (V)'])
    plt.show(block=True)

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))