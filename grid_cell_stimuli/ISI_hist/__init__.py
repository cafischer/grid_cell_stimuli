from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from cell_characteristics.analyze_APs import get_AP_onset_idxs
pl.style.use('paper')


def get_ISIs(AP_idxs, t):
    ISIs = np.diff(t[AP_idxs])
    return ISIs

# def get_ISIs(v, t, AP_threshold):
#     AP_onsets = get_AP_onset_idxs(v, threshold=AP_threshold)
#     ISIs = np.diff(t[AP_onsets])
#
#     # pl.figure()
#     # pl.plot(t, v, 'k')
#     # pl.plot(t[AP_onsets], v[AP_onsets], 'or')
#     # pl.hlines(AP_threshold, t[0], t[-1], 'r')
#     # pl.show()
#     return ISIs


def get_ISI_hist(ISIs, bins):
    ISI_hist, bin_edges = np.histogram(ISIs, bins=bins)
    return ISI_hist


def get_cumulative_ISI_hist(ISIs, upper_bound=None):
    if upper_bound is not None:
        ISIs = ISIs[ISIs <= upper_bound]
    ISIs_sorted = np.sort(ISIs)
    cum_ISI_hist_x, cum_ISI_hist_y = np.unique(ISIs_sorted, return_counts=True)
    cum_ISI_hist_y = np.cumsum(cum_ISI_hist_y) / len(ISIs)
    cum_ISI_hist_x = np.insert(cum_ISI_hist_x, 0, 0)
    cum_ISI_hist_y = np.insert(cum_ISI_hist_y, 0, 0)
    # pl.figure()
    # pl.plot(cum_ISI_hist_x, cum_ISI_hist_y)
    # pl.show()
    return cum_ISI_hist_y, cum_ISI_hist_x


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
        ISIs_ = get_ISIs(v[s:e], t, AP_threshold)
        ISI_hist[i, :], bin_edges = np.histogram(ISIs_, bins=bins)
    return np.sum(ISI_hist, 0)


def plot_ISI_hist(ISI_hist, bins, save_dir=None, title=None, show=False):
    pl.figure()
    pl.bar(bins[:-1], ISI_hist, bins[1] - bins[0], color='0.5')
    pl.xlabel('ISI (ms)')
    pl.ylabel('Count')
    if title is not None:
        pl.title(title)
    pl.tight_layout()
    if save_dir is not None:
        pl.savefig(os.path.join(save_dir, 'ISI_hist.png'))
    if show:
        pl.show()


def plot_ISI_hist_into_outof_field(ISI_hist_into, ISI_hist_outof, bins, save_dir, show=False):
    width = bins[1] - bins[0]

    pl.figure()
    pl.bar(bins[:-1], ISI_hist_into, width, color='r', alpha=0.5, label='into')
    pl.bar(bins[:-1], ISI_hist_outof, width, color='b', alpha=0.5, label='outof')
    pl.xlabel('ISI (ms)')
    pl.ylabel('Count')
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir, 'ISI_hist_into_outof.png'))
    if show:
        pl.show()


def plot_cumulative_ISI_hist(cum_ISI_hist_x, cum_ISI_hist_y, xlim=(None, None), save_dir=None, title=None, show=False):
    pl.figure()
    pl.plot(cum_ISI_hist_x, cum_ISI_hist_y, color='0.5')
    pl.xlabel('ISI (ms)')
    pl.ylabel('CDF')
    pl.xlim(xlim)
    if title is not None:
        pl.title(title)
    pl.tight_layout()
    if save_dir is not None:
        pl.savefig(os.path.join(save_dir, 'cum_ISI_hist.png'))
    if show:
        pl.show()


def plot_cumulative_ISI_hist_into_outof(cum_ISI_hist_into, cum_ISI_hist_outof, bins, save_dir, show=False):
    width = bins[1] - bins[0]

    pl.figure()
    pl.plot(bins[:-1], cum_ISI_hist_into, width, color='r', label='into')
    pl.plot(bins[:-1], cum_ISI_hist_outof, width, color='b', label='outof')
    pl.xlabel('ISI (ms)')
    pl.ylabel('Count')
    pl.legend()
    pl.tight_layout()
    if save_dir is not None:
        pl.savefig(os.path.join(save_dir, 'cum_ISI_hist_into_outof.png'))
    if show:
        pl.show()