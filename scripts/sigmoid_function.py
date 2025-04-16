import numpy as np
import matplotlib.pyplot as plt

freq = 500 # sampling rate (samples per second)
tmin=-10 # seconds
tmax= 10 # seconds
step = 1.0 / freq # time between two samples

t = np.arange(tmin,tmax, step)

freq_sin = 10
fsin = np.sin(2*np.pi*freq_sin*t)

t0=0.4
s0=20
f_ini = 1 / (1 + np.exp(-s0*(t-(tmin+t0))))
f_end = 1 / (1 + np.exp(s0*(t-(tmax-t0))))

f_ref = f_ini*f_end

# rmul = f*fsin

# plt.plot(t,f, t,fsin, t,rmul)
# plt.plot(t,rmul)
plt.plot(t,f_ref*fsin)
plt.show()

