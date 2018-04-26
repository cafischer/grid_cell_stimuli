import numpy as np
from cell_characteristics.analyze_APs import get_AP_onset_idxs, get_AP_max_idx


def get_spike_idxs(v, AP_threshold, dt, interval=2, v_diff_onset_max=5):
    """
    Get all indices of the AP maxima.
    :param v: Membrane Potential (mV).
    :param AP_threshold: Threshold (mV) for AP onset.
    :param dt: Duration of one time step.
    :param interval: Maximal interval (ms) in which to search for the AP.
    :param v_diff_onset_max: Minimal difference between v at the onset of an AP and at the maximum of an AP.
    :return: AP_max_idxs: Indices of all AP maxima.
    """
    AP_onsets = get_AP_onset_idxs(v, AP_threshold)
    AP_onsets = np.concatenate((AP_onsets, np.array([len(v)])))
    AP_max_idxs_ = np.array([get_AP_max_idx(v, AP_onsets[i], AP_onsets[i + 1], interval=int(round(interval / dt)),
                                            v_diff_onset_max=v_diff_onset_max)
                             for i in range(len(AP_onsets) - 1)])
    idxs_not_none = ~np.array([x is None for x in AP_max_idxs_], dtype=bool)
    AP_max_idxs = np.array(AP_max_idxs_[idxs_not_none], dtype=int)

    # # for testing:
    # t = np.arange(len(v)) * dt
    # pl.figure()
    # pl.plot(t, v, 'k')
    # pl.plot(t[AP_onsets[:-1]], v[AP_onsets[:-1]], 'or')
    # pl.plot(t[AP_onsets[:-1][~idxs_not_none]], v[AP_onsets[:-1][~idxs_not_none]], 'ob')
    # pl.show()
    return AP_max_idxs


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


def get_nyquist_rate(dt_sec):
    sample_rate = 1.0 / dt_sec
    nyq_rate = sample_rate / 2.0
    return nyq_rate