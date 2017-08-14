from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from scipy.signal import firwin, freqz, kaiserord
from grid_cell_stimuli import compute_fft


def nyq_from_dt(dt_sec):
    sample_rate = 1.0 / dt_sec
    nyq_rate = sample_rate / 2.0
    return nyq_rate


def get_ramp_and_theta(v, dt, ripple_attenuation, transition_width, cutoff_ramp, cutoff_theta_low, cutoff_theta_high):
    dt_sec = dt / 1000
    nyq_rate = nyq_from_dt(dt_sec)
    N, beta = kaiserord(ripple_attenuation, transition_width / nyq_rate)
    assert N < len(v)  # filter not bigger than data to filter
    filter_ramp = firwin(N + 1, cutoff_ramp / nyq_rate, window=('kaiser', beta), pass_zero=True)
    filter_theta = firwin(N + 1, [cutoff_theta_low / nyq_rate, cutoff_theta_high / nyq_rate], window=('kaiser', beta),
                          pass_zero=False)  # pass_zero seems to flip from bandpass to bandstop
    ramp = np.convolve(v, filter_ramp, mode='same')
    theta = np.convolve(v, filter_theta, mode='same')
    t_ramp_theta = np.arange(0, len(ramp) * dt, dt)
    return ramp, theta, t_ramp_theta, filter_ramp, filter_theta


def plot_filter(filter_ramp, filter_theta, dt, save_dir):
    nyq_rate = nyq_from_dt(dt / 1000)

    pl.figure()
    w, h = freqz(filter_ramp, worN=int(round(nyq_rate / 0.01)))
    pl.plot((w / np.pi) * nyq_rate, np.absolute(h), 'g', label='Ramp')
    w, h = freqz(filter_theta, worN=int(round(nyq_rate / 0.01)))
    pl.plot((w / np.pi) * nyq_rate, np.absolute(h), 'b', label='Theta')
    pl.xlabel('Frequency (Hz)', fontsize=16)
    pl.ylabel('Gain', fontsize=16)
    pl.ylim(-0.05, 1.05)
    pl.xlim(0, 20)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir, 'gain_filter.svg'))
    pl.show()


def plot_spectrum(v, ramp, theta, dt, save_dir):
    dt_sec = dt / 1000

    pl.figure()
    v_fft, freqs = compute_fft(v, dt_sec)
    pl.plot(freqs, np.abs(v_fft) ** 2, 'k', label='V')
    ramp_fft, freqs = compute_fft(ramp, dt_sec)
    pl.plot(freqs, np.abs(ramp_fft) ** 2, 'g', label='Ramp')
    theta_fft, freqs = compute_fft(theta, dt_sec)
    pl.plot(freqs, np.abs(theta_fft) ** 2, 'b', label='Theta')
    pl.xlabel('Frequency', fontsize=16)
    pl.ylabel('Power', fontsize=16)
    pl.xlim(0, 50)
    pl.ylim(0, 1e10)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir, 'power_spectrum.svg'))
    pl.show()


def plot_v_ramp_theta(v, t, ramp, theta, t_ramp_theta, save_dir):
    idx_cut = int(np.ceil((len(t) - len(ramp)) / 2.0))

    pl.figure()
    pl.plot(t, v, 'k', label='Membrane potential')
    pl.plot(t_ramp_theta + t[idx_cut], ramp, 'g', linewidth=2, label='Ramp')
    pl.plot(t_ramp_theta + t[idx_cut], theta - 75, 'b', linewidth=2, label='Theta')
    pl.xlabel('t')
    pl.xlim(t_ramp_theta[0] + t[idx_cut], t_ramp_theta[-1] + t[idx_cut])
    pl.ylabel('Voltage (mV)', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir, 'ramp_and_theta.svg'))
    pl.show()