import numpy as np
from datetime import datetime
import pytz

time_str = '2024-12-04T08:34:16.526096-05:00'
# date_s, time_s = time_str.split('T')
# # print(f'a_str: {a_str}')
# hhmmss, gmt = time_s.split('-')
# # print(f'a_str: {a_str}')
# hhmmss = hhmmss.split(':')
# # print(f'hh, mm, ss: {hh, mm, ss}')
# gmt = gmt.split(':')
# # print(f'hh_gmt, mm_gmt: {hh_gmt, mm_gmt}')

# hhmmss = np.array(hhmmss).astype(np.float)
# gmt = np.array(gmt).astype(np.float)

# # for a, b in zip(hhmmss, gmt):
#     # a+b

date_time = datetime.fromisoformat(time_str)
print(date_time)
print(f'time: {date_time.hour, date_time.minute, date_time.second, date_time.microsecond, date_time.tzname()}')
start_eeg = date_time.hour*3600 + date_time.minute*60 + date_time.second

# Convert to Asia/Kolkata time zone
# now_asia = date_time.astimezone(pytz.timezone('Asia/Kolkata'))
# now_asia = date_time.astimezone(pytz.timezone('America/Montreal'))
now_gmt = date_time.astimezone(pytz.timezone('Europe/London'))
print(f'time: {now_gmt.hour, now_gmt.minute, now_gmt.second, now_gmt.microsecond, now_gmt.tzname()}')

# print('the supported timezones by the pytz module:',
    #   pytz.all_timezones, '\n')
