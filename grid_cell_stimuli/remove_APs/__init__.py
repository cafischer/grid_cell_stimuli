from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import copy
import os
from grid_cell_stimuli import get_spike_idxs
pl.style.use('paper')


def remove_APs(v, t, AP_threshold, t_before, t_after):
    dt = t[1] - t[0]
    v_APs_removed = copy.copy(v)
    idx_before = int(round(t_before / dt))
    idx_after = int(round(t_after / dt))

    AP_max_idxs = get_spike_idxs(v, AP_threshold, dt)
    if len(AP_max_idxs) == 0:
        return v
    start_idxs, end_idxs = get_start_end_indices(AP_max_idxs, idx_after, idx_before, v_APs_removed)

    for s, e in zip(start_idxs, end_idxs):
        slope = (v_APs_removed[s] - v_APs_removed[e]) / (t[s] - t[e])
        v_APs_removed[s:e + 1] = slope * np.arange(0, e - s + 1, 1) * dt + v_APs_removed[s]
    return v_APs_removed


def get_start_end_indices(AP_onsets, idx_after, idx_before, v_APs_removed):
    diff_onsets = np.diff(AP_onsets)
    diff_to_small = (diff_onsets <= idx_before + idx_after).astype(int)
    diff_to_small = np.concatenate((diff_to_small, np.array([0])))
    start_burst_idxs = np.where(np.diff(np.insert(diff_to_small, 0, 0)) == 1)[0]
    end_burst_idxs = np.where(np.diff(diff_to_small) == -1)[0] + 1
    diff_to_small[end_burst_idxs] = 1  # do not mistake burst ends as single spikes
    single_spike_idxs = np.where(diff_to_small == 0)[0]
    start_selector = np.concatenate((start_burst_idxs,
                                     single_spike_idxs))
    end_selector = np.concatenate((end_burst_idxs,
                                   single_spike_idxs))
    start_idxs = AP_onsets[start_selector]
    end_idxs = AP_onsets[end_selector]
    start_idxs -= idx_before
    end_idxs += idx_after
    start_idxs[start_idxs < 0] = 0
    end_idxs[end_idxs >= len(v_APs_removed)] = len(v_APs_removed) - 1
    return start_idxs, end_idxs


def plot_v_APs_removed(v_APs_removed, v, t, save_dir, show=False):
    pl.figure()
    pl.plot(t, v, 'k', label='$V$')
    pl.plot(t, v_APs_removed, 'b', label='$V_{APs\ removed}$')
    pl.ylabel('Membrane potential (mV)')
    pl.xlabel('Time (ms)')
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir, 'v_APs_removed.png'))
    if show:
        pl.show()
