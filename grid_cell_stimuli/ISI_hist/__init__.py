from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from cell_characteristics.analyze_APs import get_AP_onset_idxs


def get_ISI_hist(v, t, AP_threshold, bins):
    AP_onsets = get_AP_onset_idxs(v, threshold=AP_threshold)
    ISIs = np.diff(t[AP_onsets])
    ISI_hist, bin_edges = np.histogram(ISIs, bins=bins)
    return ISI_hist, ISIs


def get_ISI_hists_into_outof_field(v, t, AP_threshold, bins, field_pos_idxs):
    field_between = np.array([(f_j - f_i) / 2 + f_i for f_i, f_j in zip(field_pos_idxs[:-1], field_pos_idxs[1:])],
                             dtype=int)
    into_idx = [(f_b, f_p) for f_b, f_p in zip(np.concatenate((np.array([0]), field_between)), field_pos_idxs)]
    outof_idx = [(f_p, f_b) for f_b, f_p in
                 zip(np.concatenate((field_between, np.array([len(v) - 1]))), field_pos_idxs)]
    ISI_hist_into = get_ISI_hist_for_intervals(v, t, AP_threshold, bins, into_idx)
    ISI_hist_outof = get_ISI_hist_for_intervals(v, t, AP_threshold, bins, outof_idx)
    return ISI_hist_into, ISI_hist_outof


def get_ISI_hist_for_intervals(v, t, AP_threshold, bins, indices):
    ISI_hist = np.zeros((len(indices), len(bins) - 1))
    for i, (s, e) in enumerate(indices):
        AP_onsets_ = get_AP_onset_idxs(v[s:e], threshold=AP_threshold)
        ISIs_ = np.diff(t[AP_onsets_])
        ISI_hist[i, :], bin_edges = np.histogram(ISIs_, bins=bins)
    return np.sum(ISI_hist, 0)


def plot_ISI_hist(ISI_hist, bins, save_dir):
    width = bins[1] - bins[0]

    pl.figure()
    pl.bar(bins[:-1], ISI_hist, width, color='0.5')
    pl.xlabel('ISI (ms)', fontsize=16)
    pl.ylabel('Count', fontsize=16)
    pl.savefig(os.path.join(save_dir, 'ISI_hist.svg'))
    pl.show()


def plot_ISI_hist_into_outof_field(ISI_hist_into, ISI_hist_outof, bins, save_dir):
    width = bins[1] - bins[0]

    pl.figure()
    pl.bar(bins[:-1], ISI_hist_into, width, color='r', alpha=0.5, label='into')
    pl.bar(bins[:-1], ISI_hist_outof, width, color='b', alpha=0.5, label='outof')
    pl.xlabel('ISI (ms)', fontsize=16)
    pl.ylabel('Count', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir, 'ISI_hist.svg'))
    pl.show()