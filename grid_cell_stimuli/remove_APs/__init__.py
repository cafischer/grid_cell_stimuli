from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import copy
import os
from cell_characteristics.analyze_APs import get_AP_onset_idxs


def remove_APs(v, t, AP_threshold, t_before, t_after):
    dt = t[1] - t[0]
    AP_onsets = get_AP_onset_idxs(v, threshold=AP_threshold)
    v_APs_removed = copy.copy(v)
    idx_before = int(round(t_before / dt))
    idx_after = int(round(t_after / dt))

    diff_onsets = np.diff(AP_onsets)
    diff_to_small = (diff_onsets <= idx_before + idx_after).astype(int)
    diff_to_small = np.concatenate((diff_to_small, np.array([0])))

    start_burst_idxs = np.where(np.diff(diff_to_small) == 1)[0] + 1
    end_burst_idxs = np.where(np.diff(diff_to_small) == -1)[0] + 1
    diff_to_small[end_burst_idxs] = 1  # do not mistake burst ends as single spikes
    single_spike_idxs = np.where(diff_to_small == 0)[0]
    start_selector = np.concatenate((start_burst_idxs,
                                     single_spike_idxs))
    end_selector = np.concatenate((end_burst_idxs,
                                   single_spike_idxs))
    idx_start = AP_onsets[start_selector]
    idx_end = AP_onsets[end_selector]

    idx_start -= idx_before
    idx_end += idx_after

    idx_start[idx_start < 0] = 0
    idx_end[idx_end >= len(v_APs_removed)] = len(v_APs_removed) - 1

    for s, e in zip(idx_start, idx_end):
        slope = (v_APs_removed[s] - v_APs_removed[e]) / (t[s] - t[e])
        v_APs_removed[s:e + 1] = slope * np.arange(0, e - s + 1, 1) * dt + v_APs_removed[s]
    return v_APs_removed


def plot_v_APs_removed(v_APs_removed, v, t, save_dir):
    pl.figure()
    pl.plot(t, v, 'k', label='$V$')
    pl.plot(t, v_APs_removed, 'b', label='$V_{APs\ removed}$')
    pl.ylabel('Membrane potential (mV)', fontsize=16)
    pl.xlabel('Time (ms)', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir, 'v_APs_removed.svg'))
    pl.show()
