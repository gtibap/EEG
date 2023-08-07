#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  read_bdf.py
#  
#  Copyright 2023 Gerardo <gerardo@CNMTLO-MX2074SBP>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
import mne
mne.set_log_level('error')

import numpy as np
import matplotlib.pyplot as plt
import glob

def main(args):
    
    print(f'arg {args[1]}')
    
    # raw_data = mne.io.read_raw_bdf('../001SCNT_TPD_T3.bdf')
    # raw_data = mne.io.read_raw_bdf('../data/eeg_test-p2_s1.bdf')
    raw_data = mne.io.read_raw_bdf(args[1])
    # print(type(raw_data.info))
    print(raw_data.info)
    data_dict = raw_data.__dict__
    raw_dict  = data_dict["_raw_extras"][0]
    
    # print(type(data_dict))
    print(raw_dict["ch_names"])
    # print(type())
    
    # print(raw_data['cal'])
    # raw_data.plot()
    # data = raw_data.get_data()
    # print(type(data))
    # print(data.shape)
    # plt.plot(data[128])
    # plt.show()
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
