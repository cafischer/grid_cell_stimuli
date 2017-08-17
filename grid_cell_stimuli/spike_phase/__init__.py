from __future__ import division
import numpy as np
import matplotlib.pyplot as pl
import os
from scipy.signal import argrelmax
from scipy.stats import linregress


def get_spike_phases(AP_onsets, t, theta, order, dist_to_AP):
    phases = np.zeros(len(AP_onsets))
    for i, AP_idx in enumerate(AP_onsets):
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
        # pl.plot(t[peak_before_idx], theta[peak_before_idx], 'og')
        # pl.plot(t[peak_after_idx], theta[peak_after_idx], 'or')
        # pl.plot(t[AP_idx], theta[AP_idx], 'oy')
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


def plot_phase_hist(phases, save_dir, show=False):
    pl.figure()
    pl.hist(phases, bins=np.arange(0, 360 + 10, 10), weights=np.ones(len(phases)) / len(phases), color='0.5')
    pl.xlabel('Phase ($^{\circ}$)', fontsize=16)
    pl.ylabel('Normalized Count', fontsize=16)
    pl.xlim(0, 360)
    pl.savefig(os.path.join(save_dir, 'phase_hist.svg'))
    if show:
        pl.show()


def plot_phase_vs_position_per_run(phases, phases_pos, AP_onsets, track_len, run_start_idx, save_dir, show=False):
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
    pl.xlabel('Position (cm)', fontsize=16)
    pl.ylabel('Phase ($^{\circ}$)', fontsize=16)
    pl.savefig(os.path.join(save_dir, 'phase_vs_position.svg'))
    if show:
        pl.show()
        

def plot_phase_precession(phases, phases_pos, slope, intercept, best_shift, save_dir, show=False):
    phases_shifted = (phases + best_shift) % 360
    pl.figure()
    pl.plot(phases_pos, phases_shifted, 'ok', label='Spike Phase')
    pl.plot(phases_pos, slope * phases_pos + intercept, 'r', label='Linear Fit')
    pl.ylim(0, 360)
    pl.xlabel('Position (cm)', fontsize=16)
    pl.ylabel('Phase ($^{\circ}$)', fontsize=16)
    pl.legend(fontsize=16)
    pl.savefig(os.path.join(save_dir, 'phase_precession.svg'))
    if show:
        pl.show()