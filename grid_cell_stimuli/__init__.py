import numpy as np
from cell_characteristics.analyze_APs import get_AP_onset_idxs, get_AP_max_idx


def get_AP_max_idxs(v, AP_threshold, dt, interval=2, v_diff_onset_max=None):
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


def find_all_AP_traces(v, before_AP_idx, after_AP_idx, AP_max_idxs, AP_max_idxs_all=None):
    """
    Find the window around all AP_max_idxs where the window contains no other AP.
    :param v: Membrane potential.
    :param before_AP_idx: Length of the window (index) before the AP_max_idx.
    :param after_AP_idx: Length of the window (index) after the AP_max_idx.
    :param AP_max_idxs: Indices of the AP maximum (can be a selected subset).
    :param AP_max_idxs_all: Indices of all AP maxima (to filter out other APs in the window). If None no APs will be
           filtered out.
    :return: Matrix of voltage traces containing the window around each AP_max_idx (row: voltage trace (time),
    column: different APs) or None if no window could be found.
    """
    v_APs = []
    for i, AP_max_idx in enumerate(AP_max_idxs):
        if AP_max_idx - before_AP_idx >= 0 and AP_max_idx + after_AP_idx < len(v):  # able to draw window
            v_AP = v[AP_max_idx - before_AP_idx:AP_max_idx + after_AP_idx + 1]

            if AP_max_idxs_all is not None:  # check that there are no other APs in the window
                AP_max_idxs_window = AP_max_idxs_all[np.logical_and(AP_max_idxs_all > AP_max_idx - before_AP_idx,
                                                                AP_max_idxs_all < AP_max_idx + after_AP_idx + 1)]
                AP_max_idxs_window = filter(lambda x: x != AP_max_idx, AP_max_idxs_window)  # remove the AP that should be in the window
                if len(AP_max_idxs_window) == 0:  # check no other APs in the window
                    v_APs.append(v_AP)
            else:
                v_APs.append(v_AP)
    if len(v_APs) > 0:
        v_APs = np.vstack(v_APs)
        return v_APs
    else:
        return None


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