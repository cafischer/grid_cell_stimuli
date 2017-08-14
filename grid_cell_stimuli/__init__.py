import numpy as np


def compute_fft(y, dt):
    """
    Compute FFT on y.

    :param y: Input array.
    :type y: array
    :param dt: time step in sec.
    :type dt: float
    :return: FFT of y, associated frequencies
    """
    fft_y = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(y), d=dt)  # dt in sec

    idx = np.argsort(freqs)
    freqs = freqs[idx]
    fft_y = fft_y[idx]
    return fft_y, freqs