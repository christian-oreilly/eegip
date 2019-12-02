import mne
import numpy as np
import os
from scipy.stats import entropy
import mne.externals.pymatreader
from .config import eegip_config


loadmat = mne.externals.pymatreader.read_mat


def load_montage(raw=None):
    assert(len(raw.ch_names) == 129)
    montage = mne.channels.make_standard_montage(eegip_config['eeg']['montage'])
    if raw is not None:
        raw.set_montage(montage)
        # raw.rename_channels(chan_mapping)
        # raw.set_channel_types({ch: "eog" for ch in eog_mapping})
        # raw.set_channel_types({ch: "misc" for ch in misc_mapping})
        # raw.drop_channels("Cz")

    return montage


def remove_baseline_jumps(raw):
    # Correction for jumps introduced at the epoch transition (every 500 samples)
    # which will cause problem when applying any signal processing steps to the
    # raw signals, such as filtering.
    data = raw.get_data(picks=["eeg", "eog"])
    inds = np.arange(0, data.shape[1], 500)
    jumps = data[:, inds] - data[:, inds - 1]
    corrected_jumps = ((data[:, inds - 1] - data[:, inds - 2]) + (data[:, inds + 1] - data[:, inds])) / 2.0
    cum_jumps = np.zeros_like(data)
    cum_jumps[:, inds] = jumps - corrected_jumps
    cum_jumps = np.cumsum(cum_jumps, axis=1)
    raw._data[mne.pick_types(raw.info, meg=False, eeg=True, eog=True), :] = data - cum_jumps


def remove_other_jumps(raw):
    data = raw.get_data(picks=["eeg", "eog"])
    win_size = 3
    x = np.abs(data[:, win_size::win_size]-data[:, :-win_size:win_size])
    threshold = np.median(x) + 2*np.percentile(x, 99)
    inds = np.where(x > threshold)
    inds = (inds[0], inds[1]*win_size)
    cum_jumps = np.zeros_like(data)
    for i in range(1, win_size+1):
        cum_jumps[inds[0], inds[1]+i] = data[inds[0], inds[1]+i] - data[inds[0], inds[1]+i-1]
    cum_jumps = np.cumsum(cum_jumps, axis=1)
    raw._data[mne.pick_types(raw.info, meg=False, eeg=True, eog=True), :] = data - cum_jumps


def events_from_time(t_events, event_id, fs=500):
    if not isinstance(t_events, np.ndarray):
        if isinstance(t_events, list):
            t_events = np.array(t_events)
        else:
            raise TypeError
    return np.vstack([np.round(t_events*fs), [0]*len(t_events), [event_id]*len(t_events)]).transpose().astype(int)



def differential_entropy2(u, n_bins=None):
    if n_bins is None:
        n_bins = min(500, int(np.round(np.sqrt(u.shape[1]))))

    h_u = []
    for u_vect in u:
        h = np.histogram(u_vect, n_bins)
        # probability of bins
        p = h[0].astype(float)/h[0].sum()
        h_u.append(entropy(p) + np.log(h[1][1]-h[1][0]))
        
    return np.array(h_u)


def differential_entropy(u, n_bins=None):
    # Calculate nx1 marginal entropies of components of u.
    #
    # Inputs:
    #           u       Matrix (n by N) of nu time series.
    #           n_bins  Number of bins to use in computing pdfs. Default is
    #                   min(100,sqrt(N)).
    #
    # Outputs:
    #           H_u      Vector n by 1 differential entropies of rows of u.                   
    #           delta_u  Vector n by 1 of bin deltas of rows of u pdfs.
    #           
    n_u = u.shape[1]

    if n_bins is None:
        n_bins = min(100, int(np.round(np.sqrt(n_u))))

    h_u = []
    delta_u = []
    
    for u_vect in u:
        u_max = np.max(u_vect)
        u_min = np.min(u_vect)
        delta_u.append((u_max-u_min)/n_bins)
        u_vect = 1 + np.round((n_bins - 1) * (u_vect - u_min) / (u_max - u_min))
        pmfr = np.diff(np.concatenate([[0], np.where(np.diff(sorted(u_vect)))[0]+1, [n_u]]))/n_u
        h_u.append(-np.sum(pmfr*np.log(pmfr)) + np.log(delta_u[-1]))
    return np.array(h_u)


def mutual_entropy_reduction(data, transformed_data, tranform_mat):
    h0 = differential_entropy(data)
    h = differential_entropy(transformed_data)

    return np.sum(h0) - np.sum(h) + np.sum(np.log(np.abs(np.linalg.eig(tranform_mat)[0])))
