from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from scipy.signal import firwin, freqz, kaiserord
from grid_cell_stimuli import get_nyquist_rate


def antialias_and_downsample(v, dt, ripple_attenuation, transition_width, cutoff_freq, dt_new_max):
    nyq_rate = get_nyquist_rate(dt / 1000)
    N, beta = kaiserord(ripple_attenuation, transition_width / nyq_rate)
    assert N < len(v)  # filter not bigger than data to filter
    filter = firwin(N + 1, cutoff_freq / nyq_rate, window=('kaiser', beta),
                    pass_zero=True)  # pass_zeros True for low pass filter
    v_antialiased = np.convolve(v, filter, mode='same')
    t_antialiased = np.arange(0, len(v_antialiased) * dt, dt)

    downsample_rate = int(np.floor(dt_new_max / dt))
    v_downsampled = v_antialiased[::downsample_rate]
    t_downsampled = t_antialiased[::downsample_rate]
    return v_downsampled, t_downsampled, filter


def plot_v_downsampled(v, t, v_downsampled, t_downsampled, save_dir):
    idx_cut = int(np.ceil((len(t) - len(v_downsampled)) / 2.0))

    pl.figure()
    pl.plot(t, v, 'b', label='$V_{APs\ removed}$')
    pl.plot(t_downsampled + t[idx_cut], v_downsampled, 'g', label='$V_{downsampled}$')
    pl.ylabel('Membrane potential (mV)', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir, 'v_downsampled.svg'))
    pl.show()


def plot_filter(filter, dt, save_dir):
    nyq_rate = get_nyquist_rate(dt / 1000)

    pl.figure()
    w, h = freqz(filter, worN=int(round(nyq_rate / 0.1)))
    pl.plot((w / np.pi) * nyq_rate, np.absolute(h))
    pl.xlabel('Frequency (Hz)', fontsize=16)
    pl.ylabel('Gain', fontsize=16)
    pl.ylim(-0.05, 1.05)
    pl.xlim(0, 10000)
    pl.savefig(os.path.join(save_dir, 'gain_filter.svg'))
    pl.show()