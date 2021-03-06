from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from scipy.signal import argrelmax, argrelmin
from scipy.stats import linregress
from scipy.stats import circmean, circstd
from circular_statistics import circ_cmtest, circ_median
from cell_characteristics import to_idx
pl.style.use('paper')


def get_spike_phases(AP_idxs, t, theta, order, dist_to_AP):
    phases = np.zeros(len(AP_idxs))
    for i, AP_idx in enumerate(AP_idxs):
        if AP_idx - dist_to_AP < 0 or AP_idx + dist_to_AP >= len(theta):
            phases[i] = np.nan
            continue
        max_before = argrelmax(theta[AP_idx - dist_to_AP:AP_idx], order=order)[0]
        max_after = argrelmax(theta[AP_idx:AP_idx + dist_to_AP], order=order)[0]
        if len(max_before) == 0 or len(max_after) == 0:
            phases[i] = np.nan
            continue
        peak_before_idx = max_before[-1] + AP_idx - dist_to_AP
        peak_after_idx = max_after[0] + AP_idx
        phases[i] = 360 * (t[AP_idx] - t[peak_before_idx]) / (t[peak_after_idx] - t[peak_before_idx])

        # print phases[i]
        # pl.figure()
        # pl.plot(t[int(peak_before_idx-10):int(peak_after_idx+10)],
        #         theta[int(peak_before_idx-10):int(peak_after_idx+10)], 'b')
        # pl.plot(t[peak_before_idx], theta[peak_before_idx], 'og', label='start')
        # pl.plot(t[peak_after_idx], theta[peak_after_idx], 'or', label='end')
        # pl.plot(t[AP_idx], theta[AP_idx], 'oy', label='AP')
        # pl.legend()
        # pl.show()
    return phases


def get_spike_phases_by_min(AP_idxs, t, theta, order, dist_to_AP, mode='lowest'):
    phases = np.zeros(len(AP_idxs))
    for i, AP_idx in enumerate(AP_idxs):
        if AP_idx - dist_to_AP < 0 or AP_idx + dist_to_AP >= len(theta):
            phases[i] = np.nan
            continue
        min_before = argrelmin(theta[AP_idx - dist_to_AP:AP_idx], order=order)[0]
        min_after = argrelmin(theta[AP_idx:AP_idx + dist_to_AP], order=order)[0]
        if len(min_before) == 0 or len(min_after) == 0:
            phases[i] = np.nan
            continue

        trough_before_idx = min_before + AP_idx - dist_to_AP
        trough_after_idx = min_after + AP_idx
        if mode == 'lowest':  # if several minima take the lowest
            trough_before_idx = trough_before_idx[np.argmin(theta[trough_before_idx])]
            trough_after_idx = trough_after_idx[np.argmin(theta[trough_after_idx])]
        elif mode == 'nearest':  # if several minima take the nearest to the AP
            trough_before_idx = trough_before_idx[-1]
            trough_after_idx = trough_after_idx[0]
        phases[i] = 360 * (t[AP_idx] - t[trough_before_idx]) / (t[trough_after_idx] - t[trough_before_idx])

        # # plot for testing
        # print phases[i]
        # pl.figure()
        # dt = t[1] - t[0]
        # pl.plot(t[int(trough_before_idx-to_idx(10, dt)):int(trough_after_idx+to_idx(10, dt))],
        #         theta[int(trough_before_idx-to_idx(10, dt)):int(trough_after_idx+to_idx(10, dt))], 'b')
        # pl.plot(t[trough_before_idx], theta[trough_before_idx], 'og', label='start')
        # pl.plot(t[trough_after_idx], theta[trough_after_idx], 'or', label='end')
        # pl.plot(t[AP_idx], theta[AP_idx], 'oy', label='AP')
        # pl.legend()
        # pl.show()
    return phases


def compute_phase_precession(phases, phases_pos):
    error = np.zeros(360)
    for d in range(360):
        phases_shifted = (phases + d) % 360
        slope, intercept, r_val, p_val, stderr = linregress(phases_pos, phases_shifted)
        error[d] = np.mean((phases_shifted - (slope * phases_pos + intercept)) ** 2)
    best_shift = np.argmin(error)
    phases_best_shift = (phases + best_shift) % 360
    slope, intercept, r_val, p_val, stderr = linregress(phases_pos, phases_best_shift)
    return slope, intercept, best_shift


def plot_phase_hist(phases, mean_phase=None, std_phase=None, title=None, color_hist='0.5', color_mean='r',
                    save_dir_img=None):
    pl.figure()
    pl.hist(phases, bins=np.arange(0, 360 + 10, 10), color=color_hist)
    pl.xlabel('Phase ($^{\circ}$)')
    pl.ylabel('Count')
    pl.xlim(0, 360)
    tick_locs = tick_labels = [0, 90, 180, 270, 360]
    if mean_phase is not None:
        pl.axvline(mean_phase, color=color_mean, linewidth=2)
        if std_phase is not None:
            pl.axvline((mean_phase-std_phase) % 360, color=color_mean, linestyle='--', linewidth=2)
            pl.axvline((mean_phase+std_phase) % 360, color=color_mean, linestyle='--', linewidth=2)
            pl.xticks(list(tick_locs) + [mean_phase, (mean_phase-std_phase) % 360, (mean_phase+std_phase) % 360],
                      list(tick_labels) + ['$\mu$', '$-\sigma$', '$+\sigma$'])
    [t.set_color(i) for (i, t) in zip(['k'] * len(tick_locs) + ['r'] * 3, pl.gca().xaxis.get_ticklabels())]
    if title is not None:
        pl.title(title)
    pl.tight_layout()
    if save_dir_img is not None:
        pl.savefig(os.path.join(save_dir_img, 'phase_hist.png'))


def plot_phase_hist_on_axes(ax, cell_idx, phases, title=None, color_hist='0.5', color_lines='r',
                            alpha=0.5, label='', y_max_vline=1, plot_mean=False, plot_median=False, plot_std=False):
    ax.hist(phases[cell_idx], bins=np.arange(0, 360 + 10, 10), color=color_hist, alpha=alpha, label=label)
    ax.set_xlim(0, 360)
    if plot_mean:
        mean_phase = circmean(phases[cell_idx], 360, 0)
        ax.axvline(mean_phase, ymax=y_max_vline, color=color_lines, linewidth=1.5)
    if plot_median:
        median_phase = circ_median(phases[cell_idx] / 360. * 2 * np.pi) * 360. / (2 * np.pi)
        ax.axvline(median_phase, ymax=y_max_vline, color=color_lines, linewidth=1.5)
    if plot_std:
        std_phase = circstd(phases[cell_idx], 360, 0)
        ax.axvline((mean_phase - std_phase) % 360, color=color_lines, linestyle='--', linewidth=1.5)
        ax.axvline((mean_phase + std_phase) % 360, color=color_lines, linestyle='--', linewidth=1.5)
    tick_locs = tick_labels = [0, 180, 360]
    ax.set_xticks(tick_locs)  # + [mean_phase, (mean_phase - std_phase) % 360, (mean_phase + std_phase) % 360])
    ax.set_xticklabels(tick_labels)  # + ['$\mu$', '$-\sigma$', '$+\sigma$'])
    [t.set_color(i) for (i, t) in zip(['k'] * len(tick_locs) + ['r'] * 3, ax.xaxis.get_ticklabels())]
    if title is not None:
        ax.title(title)


def plot_phase_vs_position_per_run(phases, phases_pos, AP_onsets, track_len, run_start_idx, save_dir_img):
    phases_run = [0] * (len(run_start_idx) - 1)
    phases_pos_run = [0] * (len(run_start_idx) - 1)
    for i_run, (run_start, run_end) in enumerate(zip(run_start_idx[:-1], run_start_idx[1:])):
        phases_run[i_run] = phases[np.logical_and(AP_onsets > run_start, AP_onsets < run_end)]
        phases_pos_run[i_run] = phases_pos[np.logical_and(AP_onsets > run_start, AP_onsets < run_end)]

    pl.figure()
    for i_run in range(len(phases_run)):
        pl.plot(phases_pos_run[i_run], phases_run[i_run], 'o')
    pl.xlim(0, track_len)
    pl.ylim(0, 360)
    pl.xlabel('Position (cm)')
    pl.ylabel('Phase ($^{\circ}$)')
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'phase_vs_position.png'))
        

def plot_phase_precession(phases, phases_pos, slope, intercept, best_shift, save_dir_img):
    phases_shifted = (phases + best_shift) % 360
    pl.figure()
    pl.plot(phases_pos, phases_shifted, 'ok', label='Spike Phase')
    pl.plot(phases_pos, slope * phases_pos + intercept, 'r', label='Linear Fit')
    pl.ylim(0, 360)
    pl.xlabel('Position (cm)', fontsize=16)
    pl.ylabel('Phase ($^{\circ}$)')
    pl.legend()
    pl.tight_layout()
    pl.savefig(os.path.join(save_dir_img, 'phase_precession.png'))