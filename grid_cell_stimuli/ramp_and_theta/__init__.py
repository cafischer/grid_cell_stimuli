from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from scipy.signal import firwin, freqz, kaiserord
from grid_cell_stimuli import compute_fft, get_nyquist_rate
import copy
pl.style.use('paper')


def get_ramp_and_theta(v, dt, ripple_attenuation, transition_width, cutoff_ramp, cutoff_theta_low, cutoff_theta_high,
                       pad_if_to_short=False):
    dt_sec = dt / 1000.
    nyq_rate = get_nyquist_rate(dt_sec)
    N, beta = kaiserord(ripple_attenuation, transition_width / nyq_rate)
    v_true = copy.copy(v)
    if pad_if_to_short and N > len(v):
        pad_len = int(np.ceil((N - len(v)) / 2.)) - 1
        v = np.pad(v, pad_len, mode='edge')
    else:
        assert N <= len(v)  # filter not bigger than data to filter
    filter_ramp = firwin(N + 1, cutoff_ramp / nyq_rate, window=('kaiser', beta), pass_zero=True)
    filter_theta = firwin(N + 1, [cutoff_theta_low / nyq_rate, cutoff_theta_high / nyq_rate], window=('kaiser', beta),
                          pass_zero=False)  # pass_zero seems to flip from bandpass to bandstop

    v_ramp_pad = np.pad(v, (int(np.floor(len(filter_ramp) / 2.)), int(np.ceil(len(filter_ramp) / 2.)) - 1), mode='edge')
    v_theta_pad = np.pad(v, (int(np.floor(len(filter_ramp) / 2.)), int(np.ceil(len(filter_ramp) / 2.)) - 1), mode='edge')
    ramp = np.convolve(v_ramp_pad, filter_ramp, mode='valid')
    theta = np.convolve(v_theta_pad, filter_theta, mode='valid')
    if pad_if_to_short and N > len(v_true):
        ramp = ramp[pad_len: -pad_len]
        theta = theta[pad_len: -pad_len]
    assert len(ramp) == len(theta) == len(v_true)
    t_ramp_theta = np.arange(len(ramp)) * dt
    return ramp, theta, t_ramp_theta, filter_ramp, filter_theta


def plot_filter(filter_ramp, filter_theta, dt, save_dir, show=False):
    nyq_rate = get_nyquist_rate(dt / 1000)

    pl.figure()
    w, h = freqz(filter_ramp, worN=int(round(nyq_rate / 0.01)))
    pl.plot((w / np.pi) * nyq_rate, np.absolute(h), 'g', label='Ramp')
    w, h = freqz(filter_theta, worN=int(round(nyq_rate / 0.01)))
    pl.plot((w / np.pi) * nyq_rate, np.absolute(h), 'darkorange', label='Theta')
    pl.xlabel('Frequency (Hz)')
    pl.ylabel('Gain')
    pl.ylim(-0.05, 1.05)
    pl.xlim(0, 20)
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir, 'gain_filter.png'))

    pl.figure()
    w, h = freqz(filter_ramp, worN=int(round(nyq_rate / 0.01)))
    pl.plot((w / np.pi) * nyq_rate, 20 * np.log10(abs(h)), 'g', label='Ramp')
    w, h = freqz(filter_theta, worN=int(round(nyq_rate / 0.01)))
    pl.plot((w / np.pi) * nyq_rate, 20 * np.log10(abs(h)), 'darkorange', label='Theta')
    pl.xlabel('Frequency (Hz)')
    pl.ylabel('Amplitude (dB)')
    pl.ylim(-100, 0.1)
    pl.xlim(0, 20)
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir, 'amplitude_filter.png'))
    if show:
        pl.show()


def plot_spectrum(v, ramp, theta, dt, save_dir, show=False):
    dt_sec = dt / 1000

    pl.figure()
    v_fft, freqs = compute_fft(v, dt_sec)
    pl.plot(freqs, np.abs(v_fft) ** 2, 'k', label='Membrane Potential')
    ramp_fft, freqs = compute_fft(ramp, dt_sec)
    pl.plot(freqs, np.abs(ramp_fft) ** 2, 'g', label='Ramp')
    theta_fft, freqs = compute_fft(theta, dt_sec)
    pl.plot(freqs, np.abs(theta_fft) ** 2, 'darkorange', label='Theta')
    pl.xlabel('Frequency')
    pl.ylabel('Power')
    pl.xlim(0, 50)
    pl.ylim(0, np.max(np.abs(theta_fft) ** 2))
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir, 'power_spectrum.png'))
    if show:
        pl.show()


def plot_v_ramp_theta(v, t, ramp, theta, t_ramp_theta, save_dir=None, show=False):
    pl.figure()
    pl.plot(t, v, 'k', label='Membrane Potential')
    pl.plot(t_ramp_theta, ramp, 'g', linewidth=2, label='Ramp')
    pl.plot(t_ramp_theta, theta + v[0], 'darkorange', linewidth=2, label='Theta')
    pl.xlabel('t')
    pl.ylabel('Voltage (mV)')
    pl.xlabel('Time (ms)')
    pl.legend()
    pl.tight_layout()
    if save_dir is not None:
        pl.savefig(os.path.join(save_dir, 'ramp_and_theta.png'))
    if show:
        pl.show()