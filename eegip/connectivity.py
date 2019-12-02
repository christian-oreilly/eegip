import numpy as np
import pandas as pd
import mne
import os
import tqdm
import xarray as xr
from warnings import warn
import scipy
from itertools import product
from glob import glob
from scipy.cluster import hierarchy
from mne.connectivity import spectral_connectivity

from .config import eegip_config, load_additionnal_config
from .path import get_path, parse_pattern


def get_matrices(band, event_type, dataset, con_type, notebook=False, **kwargs):
    con_matrices = {}

    var_names = np.concatenate([["dataset", "event_type", "con_type", "fmin", "fmax"], list(kwargs.keys())])

    if notebook:
        progress = tqdm.tqdm_notebook
    else:
        progress = tqdm.tqdm
    for item in progress(list(product(dataset, event_type, con_type, band, *kwargs.values()))):
        # unpacking the band
        item = tuple(np.concatenate([item[:3], item[3], item[4:]]))

        kw_items = dict(zip(var_names, item))
        pattern = get_path("connectivity_matrix", **kw_items)

        con_matrices[item] = []

        file_names = sorted(glob(pattern))
        if len(file_names) == 0:
            raise ValueError("No file found for pattern {}.".format(pattern))
        for file_name in file_names:

            df = pd.read_csv(file_name, index_col=0)
            array = xr.DataArray(df, dims=('signal1', 'signal2'),
                                 coords={'signal1': df.index, 'signal2': df.columns})

            con_matrices[item].append(array)
    return con_matrices


def get_cov_matrix(raw, epochs,  dataset):
  
    if dataset == "london06":
        cov_event_id = {"noise": 3}
        tmax_ = eegip_config["tmax"][dataset]
    elif dataset == "london12":
        cov_event_id = {"base": 0}
        tmax_ = eegip_config["tmax"][dataset]
    elif dataset == "washington":
        cov_event_id = eegip_config["event_id"]["washington"]["cov"]
        tmax_ = eegip_config["tmax"][dataset]["videos"]
    else:
        raise ValueError

    if list(cov_event_id.values())[0] in epochs.events[:, 2]:
        # From the doc: "you should use baseline correction when constructing
        # the epochs. Otherwise the covariance matrix will be inaccurate.
        epochs_cov = mne.Epochs(raw, epochs.events, cov_event_id, tmin=0.0, tmax=tmax_-1./raw.info["sfreq"],
                                baseline=(None, None), verbose='error')
    else:
        epochs_cov = epochs.copy()

    # This method also attenuates any activity in your source estimates that resemble the baseline
    return mne.compute_covariance(epochs_cov, tmax=None)


def get_map(con_matrices, center=None, stat="mean"):
    if len(con_matrices) == 0:
        return None, None
    c = xr.concat(con_matrices, dim='recording')
    if stat == "mean":
        dat = pd.DataFrame(c.mean(dim='recording').data + c.mean(dim='recording').data.T,
                           index=c.signal1, columns=c.signal2)
        if center is None:
            center = True

    elif stat == "std":
        dat = pd.DataFrame(c.std(dim='recording').data + c.std(dim='recording').data.T,
                           index=c.signal1, columns=c.signal2)
        if center is None:
            center = False

    #elif stat == "z-score":
    #    dat = (c - c.mean(dim='recording'))/c.std(dim='recording')
    #    dat = pd.DataFrame(dat.data + dat.data.T, index=c.signal1, columns=c.signal2)
    #
    #    if center is None:
    #        center = False

    else:
        raise ValueError

    for chan_del in eegip_config["eeg"]["eog_channels"]:
        if chan_del in dat:
            del dat[chan_del]

    dat = dat.T
    for chan_del in eegip_config["eeg"]["eog_channels"]:
        if chan_del in dat:
            del dat[chan_del]        

    mask = np.isnan(dat.to_numpy()) + np.isinf(dat.to_numpy()) + np.diag(np.ones(dat.shape[0]))
    mask = mask.astype(bool)
    dat[np.isnan(dat)] = 0.0
    dat[np.isinf(dat)] = 0.0

    if center:
        mean_val = np.mean(np.mean(dat[dat != 0]))
        mean_vals = mean_val*np.ones(dat.shape)
        mean_vals[mask] = 0.0
        return dat-mean_vals, mask
    return dat, mask


def get_cluster_t(linkage, start=0.5, stop=1.5, step=0.1, step_min=0.000001, n_clusters_min=10):
    if step < step_min:
        return start
    
    ts = np.arange(start, stop, step)
    nb_clusters = np.array([np.max(scipy.cluster.hierarchy.fcluster(linkage, t)) for t in ts])
    try:
        # There is a threshold at which there is sharp drop in number of cluster, typically
        # from above 10 (n_clusters_min) to 1 (using 3 to leave a bit of margin).
        drop_check = (nb_clusters[:-1] > n_clusters_min)*(nb_clusters[1:] < 3)
        new_start = ts[:-1][drop_check][0]
        new_stop = ts[1:][drop_check][0]
    except IndexError:
        return start
    return get_cluster_t(linkage, new_start, new_stop, step/10.0)


def compute_connectivity_matrices(dataset, con_type, sfreq=None, method=None, mode='multitaper', faverage=True,
                                  mt_adaptive=True, n_jobs=1, notebook=False, small=False, resume=True, config=None,
                                  **kwargs):

    def compute_connectivity_subfct():
        file_name_out = get_path("connectivity_matrix", dataset, path_pattern_type="to_fill",
                                 path_fill_kwargs=path_kwargs, **file_kwargs)

        if resume:
            if os.path.exists(file_name_out):
                return

        con, freqs, times, n_epochs, n_tapers = spectral_connectivity(
            signals, method=method, mode=mode, sfreq=sfreq_subject,
            fmin=fmin, fmax=fmax, faverage=faverage,
            mt_adaptive=mt_adaptive, n_jobs=n_jobs)

        con = pd.DataFrame(con[:, :, 0])

        con.columns = label_names
        con.index = label_names

        con.to_csv(file_name_out)

    # ###########
    if config is not None:
        load_additionnal_config(config)

    if dataset == "washington":
        kwargs["task"] = "*"

    if con_type == "sources":
        path_pattern = get_path("sources", dataset, event_type="*", con_type=con_type, **kwargs)
    elif con_type == "scalp":
        path_pattern = get_path("epoch", dataset, **kwargs)
    else:
        raise ValueError("Invalid value for the con_type argument.")

    paths = sorted(glob(path_pattern))
    if len(paths) == 0:
        raise FileNotFoundError("No file were found to process. We tried with {}.".format(path_pattern))

    if small:
        paths = paths[:1]

    if notebook:
        process = tqdm.tqdm_notebook
    else:
        process = tqdm.tqdm

    for file_name_signals in process(paths):

        if con_type == "sources":
            path_kwargs, file_kwargs = parse_pattern("sources", dataset, file_name_signals)
            if sfreq is None:
                file_kwargs_preprocessed = file_kwargs.copy()
                if "event_type" in file_kwargs_preprocessed:
                    del file_kwargs_preprocessed["event_type"]
                if "con_type" in file_kwargs_preprocessed:
                    del file_kwargs_preprocessed["con_type"]

                preprocessed_file_name = get_path("preprocessed_recording", dataset, path_pattern_type="to_fill",
                                                  path_fill_kwargs=path_kwargs, **file_kwargs_preprocessed)
                raw = mne.io.read_raw_fif(preprocessed_file_name, preload=False)
                sfreq_subject = raw.info["sfreq"]
            else:
                sfreq_subject = sfreq

            file_name_labels = get_path("labels", dataset, path_pattern_type="to_fill",
                                        path_fill_kwargs=path_kwargs, **file_kwargs)
            signals = np.load(file_name_signals)
            label_names = np.load(file_name_labels)

        elif con_type == "scalp":
            path_kwargs, file_kwargs = parse_pattern("epoch", dataset, file_name_signals)

            epochs = mne.read_epochs(file_name_signals, preload=True)
            label_names = epochs.info["ch_names"]
            if sfreq is None:
                sfreq_subject = epochs.info["sfreq"]
            else:
                sfreq_subject = sfreq

        if method is not None:
            file_kwargs["method"] = method
        else:
            method = "wpli"

        for fmin, fmax in eegip_config["analysis"]["band"]:

            file_kwargs["fmax"] = fmax
            file_kwargs["fmin"] = fmin

            if con_type == "sources":
                compute_connectivity_subfct()
            elif con_type == "scalp":
                file_kwargs["con_type"] = con_type
                for event_type in epochs.event_id:
                    file_kwargs["event_type"] = event_type
                    signals = epochs[event_type]
                    if len(signals) == 0:
                        warn("No epochs found for event_id {}. Skipping connectivity computation for this event_id."
                             .format(event_type))
                    else:
                        compute_connectivity_subfct()
