from __future__ import division
import matplotlib.pyplot as pl
import numpy as np
import os
from itertools import combinations, product
from analyze_in_vivo.load.load_domnisoru import get_celltype
from cell_characteristics.analyze_APs import get_AP_onset_idxs
from matplotlib.colors import to_rgb
from cell_fitting.util import change_color_brightness
pl.style.use('paper_subplots')


def get_ISIs(AP_idxs, t):
    ISIs = np.diff(t[AP_idxs])
    return ISIs


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
        AP_idxs = get_AP_onset_idxs(v[s:e], AP_threshold)
        ISIs_ = get_ISIs(AP_idxs, t)
        ISI_hist[i, :], bin_edges = np.histogram(ISIs_, bins=bins)
    return np.sum(ISI_hist, 0)


def plot_ISI_hist(ISI_hist, bins, save_dir=None, title=None):
    pl.figure()
    pl.bar(bins[:-1], ISI_hist, bins[1] - bins[0], color='0.5', align='edge')
    pl.xlabel('ISI (ms)')
    pl.ylabel('Count')
    if title is not None:
        pl.title(title)
    pl.tight_layout()
    if save_dir is not None:
        pl.savefig(os.path.join(save_dir, 'ISI_hist.png'))


def plot_ISI_hist_into_outof_field(ISI_hist_into, ISI_hist_outof, bins, save_dir):
    width = bins[1] - bins[0]

    pl.figure()
    pl.bar(bins[:-1], ISI_hist_into, width, color='r', alpha=0.5, label='into', align='edge')
    pl.bar(bins[:-1], ISI_hist_outof, width, color='b', alpha=0.5, label='outof', align='edge')
    pl.xlabel('ISI (ms)')
    pl.ylabel('Count')
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir, 'ISI_hist_into_outof.png'))


def plot_cumulative_ISI_hist(cum_ISI_hist_x, cum_ISI_hist_y, xlim=(None, None), save_dir=None, title=None):
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


def plot_cumulative_ISI_hist_all_cells(cum_ISI_hist_y, cum_ISI_hist_x, cum_ISI_hist_y_avg, cum_ISI_hist_x_avg,
                                       cell_ids, max_x=200, save_dir=None):
    save_dir_data = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'
    cm = pl.cm.get_cmap('plasma')
    colors = cm(np.linspace(0, 1, len(cell_ids)))
    cum_ISI_hist_x_avg_with_end = np.insert(cum_ISI_hist_x_avg, len(cum_ISI_hist_x_avg), max_x)
    cum_ISI_hist_y_avg_with_end = np.insert(cum_ISI_hist_y_avg, len(cum_ISI_hist_y_avg), 1.0)
    pl.figure()
    pl.plot(cum_ISI_hist_x_avg_with_end, cum_ISI_hist_y_avg_with_end, label='all',
            drawstyle='steps-post', linewidth=2.0, color='0.4')
    for i, cell_id in enumerate(cell_ids):
        if get_celltype(cell_id, save_dir_data) == 'stellate':
            label = cell_id + ' ' + u'\u2605'
        elif get_celltype(cell_id, save_dir_data) == 'pyramidal':
            label = cell_id + ' ' + u'\u25B4'
        else:
            label = cell_id
        cum_ISI_hist_x_with_end = np.insert(cum_ISI_hist_x[i], len(cum_ISI_hist_x[i]), max_x)
        cum_ISI_hist_y_with_end = np.insert(cum_ISI_hist_y[i], len(cum_ISI_hist_y[i]), 1.0)

        ISI_burst = 10.0  # ms
        cross_burst = np.where(cum_ISI_hist_x_with_end >= ISI_burst)[0][0]
        cross_burst_y = cum_ISI_hist_y_with_end[cross_burst]
        if cross_burst_y < 0.1:
            label = '* ' + label

        pl.plot(cum_ISI_hist_x_with_end, cum_ISI_hist_y_with_end, label=label, color=colors[i],
                drawstyle='steps-post')
    pl.xlabel('ISI (ms)')
    pl.ylabel('CDF')
    pl.xlim(0, max_x)
    if len(cell_ids) > 6:
        pl.legend(fontsize=7, loc='lower right')
    else:
        pl.legend(fontsize=10, loc='lower right')
    pl.tight_layout()
    if save_dir is not None:
        pl.savefig(os.path.join(save_dir, 'cum_ISI_hist.png'))


def plot_cumulative_ISI_hist_all_cells_with_bursty(cum_ISI_hist_y, cum_ISI_hist_x,
                                                   cum_ISI_hist_y_avg_bursty, cum_ISI_hist_x_avg_bursty,
                                                   cum_ISI_hist_y_avg_nonbursty, cum_ISI_hist_x_avg_nonbursty,
                                                   cell_ids, burst_label, max_x=200, save_dir=None):
    save_dir_data = '/home/cf/Phd/programming/projects/analyze_in_vivo/analyze_in_vivo/data/domnisoru'

    cum_ISI_hist_x_avg_bursty = np.insert(cum_ISI_hist_x_avg_bursty, len(cum_ISI_hist_x_avg_bursty), max_x)
    cum_ISI_hist_x_avg_nonbursty = np.insert(cum_ISI_hist_x_avg_nonbursty, len(cum_ISI_hist_x_avg_nonbursty), max_x)
    cum_ISI_hist_y_avg_bursty = np.insert(cum_ISI_hist_y_avg_bursty, len(cum_ISI_hist_y_avg_bursty), 1.0)
    cum_ISI_hist_y_avg_nonbursty = np.insert(cum_ISI_hist_y_avg_nonbursty, len(cum_ISI_hist_y_avg_nonbursty), 1.0)

    pl.figure()
    pl.plot(cum_ISI_hist_x_avg_nonbursty, cum_ISI_hist_y_avg_nonbursty, label='Non-bursty',
            drawstyle='steps-post', linewidth=2.0, color='b')
    pl.plot(cum_ISI_hist_x_avg_bursty, cum_ISI_hist_y_avg_bursty, label='Bursty',
            drawstyle='steps-post', linewidth=2.0, color='r')

    for i, cell_id in enumerate(cell_ids):
        cum_ISI_hist_x_with_end = np.insert(cum_ISI_hist_x[i], len(cum_ISI_hist_x[i]), max_x)
        cum_ISI_hist_y_with_end = np.insert(cum_ISI_hist_y[i], len(cum_ISI_hist_y[i]), 1.0)

        # cross_burst = np.where(cum_ISI_hist_x_with_end >= burst_ISI)[0][0]
        # cross_burst_y = cum_ISI_hist_y_with_end[cross_burst]
        # if cross_burst_y < 0.1:
        #     bursty = False
        # else:
        #     bursty = True

        color = 'r' if burst_label[i] else 'b'
        pl.plot(cum_ISI_hist_x_with_end, cum_ISI_hist_y_with_end,
                color=change_color_brightness(to_rgb(color), 30, 'brighter'), alpha=0.5,
                drawstyle='steps-post')
    pl.xlabel('ISI (ms)')
    pl.ylabel('CDF')
    pl.xlim(0, max_x)
    pl.legend(loc='lower right')
    pl.tight_layout()
    if save_dir is not None:
        pl.savefig(os.path.join(save_dir, 'cum_ISI_hist.png'))


def plot_cumulative_ISI_hist_into_outof(cum_ISI_hist_into, cum_ISI_hist_outof, bins, save_dir):
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


def plot_cumulative_comparison_all_cells(cum_ISI_hist_x, cum_ISI_hist_y, cell_ids, p_val_dict, save_dir):
    fig, ax = pl.subplots(len(cell_ids) - 1, len(cell_ids) - 1, figsize=(10, 10))
    cm = pl.cm.get_cmap('plasma')
    colors = cm(np.linspace(0, 1, len(cell_ids)))
    for i1, i2 in product(range(len(cell_ids) - 1), repeat=2):
        ax[i2 - 1, i1].spines['left'].set_visible(False)
        ax[i2 - 1, i1].spines['bottom'].set_visible(False)
        ax[i2 - 1, i1].set_xticks([])
        ax[i2 - 1, i1].set_yticks([])
    for i1, i2 in combinations(range(len(cell_ids)), 2):
        ax[i2 - 1, i1].set_title('p-value: %.3f' % p_val_dict[(i1, i2)])
        ax[i2 - 1, i1].plot(cum_ISI_hist_x[i1], cum_ISI_hist_y[i1], color=colors[i1])
        ax[i2 - 1, i1].plot(cum_ISI_hist_x[i2], cum_ISI_hist_y[i2], color=colors[i2])
        ax[i2 - 1, i1].set_xlim(0, 200)
        ax[i2 - 1, i1].set_xticks([0, 100, 200])
        ax[i2 - 1, i1].set_yticks([0, 1])
    for i1, i2 in combinations(range(len(cell_ids)), 2):
        ax[i2 - 1, i1].set(xlabel=cell_ids[i1], ylabel=cell_ids[i2])
        ax[i2 - 1, i1].label_outer()
        ax[i2 - 1, i1].spines['left'].set_visible(True)
        ax[i2 - 1, i1].spines['bottom'].set_visible(True)
    pl.tight_layout()
    pl.savefig(save_dir)