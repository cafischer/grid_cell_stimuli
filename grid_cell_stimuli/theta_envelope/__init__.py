import numpy as np
import matplotlib.pyplot as pl
from scipy.signal import hilbert
import os


def compute_envelope(x):
    return np.abs(hilbert(x))


def plot_envelope(theta, theta_envelope, t, save_dir=None):
    pl.figure()
    pl.plot(t, theta, 'darkorange', label='Theta')
    pl.plot(t, theta_envelope, 'r', label='Theta Envelope')
    pl.ylabel('Voltage (mV)', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.legend(fontsize=16)
    if save_dir is not None:
        pl.savefig(os.path.join(save_dir, 'theta_envelope.svg'))