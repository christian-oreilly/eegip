from collections import OrderedDict
from glob import glob
import numpy as np
import os
import tqdm

import mne
from mne.io.eeglab.eeglab import _check_load_mat, _get_info
from mne.preprocessing import read_ica_eeglab
from mne.io import read_raw_eeglab

from .config import eegip_config, load_additionnal_config
from .utils import load_montage
from .path import get_path, parse_pattern

from slurppy import Config


def mark_bad_channels(raw, file_name, mark_to_remove=("manual", "rank")):
    raw_eeg = _check_load_mat(file_name, None)
    info, _, _ = _get_info(raw_eeg)
    chan_info = raw_eeg.marks["chan_info"]

    mat_chans = np.array(info["ch_names"])
    assert(len(chan_info["flags"][0]) == len(mat_chans))

    if len(np.array(chan_info["flags"]).shape) > 1:
        ind_chan_to_drop = np.unique(np.concatenate([np.where(flags)[0] for flags, label in zip(chan_info["flags"],
                                                                                                chan_info["label"])
                                                     if label in mark_to_remove]))
    else:
        ind_chan_to_drop = np.where(chan_info["flags"])[0]

    bad_chan = [chan for chan in mat_chans[ind_chan_to_drop]]

    raw.info['bads'].extend(bad_chan)


def channel_rejection(raw, file_name, mark_to_remove=("manual", "rank")):
    mark_bad_channels(raw, file_name, mark_to_remove)
    raw.drop_channels(raw.info['bads'])


def add_bad_segment_annot(raw, file_name, mark_to_remove=("manual",)):

    raw_eeg = _check_load_mat(file_name, None)
    time_info = raw_eeg.marks["time_info"]
    
    if len(np.array(time_info["flags"]).shape) > 1:
        ind_time_to_drop = np.unique(np.concatenate([np.where(flags)[0] for flags, label in zip(time_info["flags"],
                                                                                                time_info["label"])
                                                     if label in mark_to_remove]))
    else:
        ind_time_to_drop = np.where(time_info["flags"])[0]

    ind_starts = np.concatenate([[ind_time_to_drop[0]], ind_time_to_drop[np.where(np.diff(ind_time_to_drop) > 1)[0]+1]])
    ind_ends = np.concatenate([ind_time_to_drop[np.where(np.diff(ind_time_to_drop) > 1)[0]], [ind_time_to_drop[-1]]])
    durations = (ind_ends+1 - ind_starts)/raw.info["sfreq"]
    onsets = ind_starts/raw.info["sfreq"]

    for onset, duration in zip(onsets, durations):
        raw.annotations.append(onset, duration, description="bad_lossless_qc")
    

def remove_rejected_ica_components(raw, file_name, inplace=True):
    raw_eeg = _check_load_mat(file_name, None)
    mark_to_remove = ["manual"]
    comp_info = raw_eeg.marks["comp_info"]

    if len(np.array(comp_info["flags"]).shape) > 1:
        ind_comp_to_drop = np.unique(np.concatenate([np.where(flags)[0] for flags, label in zip(comp_info["flags"],
                                                                                                comp_info["label"])
                                                     if label in mark_to_remove]))
    else:
        ind_comp_to_drop = np.where(comp_info["flags"])[0]

    if inplace:
        read_ica_eeglab(file_name).apply(raw, exclude=ind_comp_to_drop)
    else:
        read_ica_eeglab(file_name).apply(raw.copy(), exclude=ind_comp_to_drop)


def get_events(raw, dataset):
    annot_sample = []
    annot_id = []
    freq = raw.info["sfreq"]

    if dataset == "london06":
        annot_map = {'sta1': 'direct',
                     'sta2': 'indirect', 'sta3': 'indirect',
                     'dyn1': 'direct',
                     'dyn2': 'indirect', 'dyn3': 'indirect',
                     'Nois': 'noise'}

        for a in raw.annotations:
            if a["description"] in annot_map:
                annot_sample.append(int(a["onset"]*freq))
                annot_id.append(eegip_config['event_id'][dataset][annot_map[a["description"]]])

    elif dataset == "london12":
        annots = [OrderedDict((("onset", 0), ("duration", 0), ("description", "base"), ('orig_time', None)))]
        annots.extend([a for a in raw.annotations if a["description"] in ["eeg1", "eeg2", "eeg3"]])
        annots.append(OrderedDict((("onset", annots[-1]["onset"]+50.), ("duration", 0),
                                   ("description", "end"), ('orig_time', None))))

        for annot, next_annot in zip(annots[:-1], annots[1:]):
            annot_sample.append(np.arange(int(annot["onset"]*freq),
                                          int(next_annot["onset"]*freq),
                                          int(eegip_config['tmax'][dataset]*freq)))
            annot_id.extend(eegip_config['event_id'][dataset][annot["description"]]*np.ones(len(annot_sample[-1])))

        annot_sample = np.concatenate(annot_sample)

    elif dataset == "washington":        
        
        # COV AND VIDEOS
        annots = [OrderedDict((("onset", 0), ("duration", 0), ("description", "cov"), ('orig_time', None)))]
        annots.extend([a for a in raw.annotations if a["description"] in ["Toys", "EndM", "Socl"]])
        annots.append(OrderedDict((("onset", annots[-1]["onset"]+50.), ("duration", 0),
                                   ("description", "end"), ('orig_time', None))))
        
        for annot, next_annot in zip(annots[:-1], annots[1:]):
            if annot["description"] == "EndM":
                continue
            
            annot_sample.append(np.arange(int(annot["onset"]*freq),
                                          int(next_annot["onset"]*freq),
                                          int(eegip_config['tmax'][dataset]["videos"]*freq)))
            id_ = eegip_config['event_id'][dataset]["videos"][annot["description"]]
            annot_id.extend(id_*np.ones(len(annot_sample[-1])))

        # IMAGES
        id_map = {"base": "base", "stm+": "stmp", "stm-": "stmn"}
        annots = [a for a in raw.annotations if a["description"] in ['base', 'stm+', 'stm-']]
        onset = [a["onset"] for a in annots]
        annot_sample.append(np.array(onset)*freq)
        annot_id.extend([eegip_config['event_id'][dataset]["images"][id_map[a["description"]]] for a in annots])
            
        annot_sample = np.concatenate(annot_sample)        

    return np.array([annot_sample, [0]*len(annot_sample), annot_id], dtype=int).T


def get_epochs(raw, events, dataset):

    freq = raw.info["sfreq"]    
    
    if dataset in ["london06", "london12"]:
        filtered_event_id = {key: val for key, val in eegip_config['event_id'][dataset].items() if val in events[:, 2]}
        if len(filtered_event_id):
            # "tmax = tmax[dataset] - 1.0 / freq" because MNE is inclusive on the last point and we don't want that
            # "baseline=None" because the baseline is corrected by a 1Hz high-pass on the raw data
            return mne.Epochs(raw, events, filtered_event_id, tmin=0.0,
                              tmax=eegip_config['tmax'][dataset]-1.0/freq,  baseline=None,
                              preload=True, reject_by_annotation=True)
        return None

    if dataset == "washington": 
        filtered_event_id = {key: val for key, val in eegip_config['event_id'][dataset]["videos"].items()
                             if val in events[:, 2]}
        if len(filtered_event_id):
            epochs_videos = mne.Epochs(raw, events, filtered_event_id, tmin=0.0,
                                       tmax=eegip_config['tmax'][dataset]["videos"]-1.0/freq,
                                       baseline=None, preload=True, reject_by_annotation=True)
        else:
            epochs_videos = None
            
        filtered_event_id = {key: val for key, val in eegip_config['event_id'][dataset]["images"].items()
                             if val in events[:, 2]}
        if len(filtered_event_id):
            epochs_images = mne.Epochs(raw, events, filtered_event_id, tmin=0.0,
                                       tmax=eegip_config['tmax'][dataset]["images"]-1.0/freq,
                                       baseline=None,  preload=True, reject_by_annotation=True)
        else:
            epochs_images = None
            
        return epochs_videos, epochs_images


def get_evoked(epochs, return_std=False):
    evoked_dic = {}
    evoked_std_dic = {}

    for stim in epochs.event_id:
        data_tmp = epochs[stim].copy()
        evoked_dic[stim] = data_tmp.average()
        if return_std:
            evoked_std_dic[stim] = data_tmp.standard_error()
    
    if return_std:
        return evoked_dic, evoked_std_dic
    return evoked_dic


def preprocess(raw, notch_width=None, line_freq=50.0):

    # raw_data.filter(1., 40., l_trans_bandwidth=0.5, n_jobs=1, verbose='INFO')

    if notch_width is None:
        notch_width = np.array([1.0, 0.1, 0.01, 0.1])

    notch_freqs = np.arange(line_freq, raw.info["sfreq"]/2.0, line_freq)
    raw.notch_filter(notch_freqs, picks=["eeg", "eog"], fir_design='firwin',
                     notch_widths=notch_width[:len(notch_freqs)], verbose=None)



def preprocess_dataset(dataset, resume=True, notebook=False, small=False, config=None):

    analysis = Analysis()

    analysis.config = Config
    analysis.dataset = dataset

    analysis.preprocess_dataset()

    if config is not None:
        load_additionnal_config(config)

    paths = sorted(glob(get_path("eegip_recording", dataset)))

    if small:
        paths = paths[:1]

    if notebook:
        progress = tqdm.tqdm_notebook
    else:
        progress = tqdm.tqdm

    for file_name in progress(paths):
        preprocess_a_file(file_name, dataset, resume)


def preprocess_a_file(file_name, dataset, resume=True):

    # READING AND PREPROCESSING RAW DATA
    path_kwargs, file_kwargs = parse_pattern("eegip_recording", dataset, file_name)

    raw_file_name = get_path("preprocessed_recording", dataset, path_pattern_type="to_fill",
                             path_fill_kwargs=path_kwargs, **file_kwargs)

    if resume:
        if os.path.exists(raw_file_name):
            print("Skipping " + file_name + " because this file has already been treated...")
            return

    raw = read_raw_eeglab(file_name, preload=True, verbose=None)

    # WOULD FAIL FOR BOSTON, WHICH HAS SOME LOWER RESOLUTION GRID
    assert(len(raw.ch_names) == 129)

    load_montage(raw=raw)

    if dataset in ["london06", "london12"]:
        preprocess(raw, line_freq=50.0, notch_width=np.array([0.0, 0.1, 0.01, 0.1]))
    elif dataset == "washington":
        preprocess(raw, line_freq=60.0, notch_width=np.array([0.0, 0.1, 0.01, 0.1]))

    raw = raw.filter(1, None, fir_design='firwin', verbose=False)

    mark_bad_channels(raw, file_name)
    add_bad_segment_annot(raw, file_name)

    remove_rejected_ica_components(raw, file_name, inplace=True)

    raw = raw.interpolate_bads(reset_bads=True, verbose=False)

    raw.rename_channels({ch: ch2 for ch, ch2 in eegip_config["eeg"]["chan_mapping"].items()
                         if ch in raw.ch_names})
    raw.set_channel_types({ch: "eog" for ch in eegip_config["eeg"]["eog_channels"]
                           if ch in raw.ch_names})

    # GETTING EVENTS
    events = get_events(raw, dataset)
    if not len(events):
        message = "Skipping " + file_name + " because it contains no events for epoching."
        mne.utils.warn(message)
        return

    # CREATING OUTPUT DIRECTIONS IF NECESSARY
    dir_name = os.path.dirname(raw_file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # GETTING EPOCHS AND SAVING EPOCHS AND EVOKED
    if dataset in ["london06", "london12"]:
        epochs = get_epochs(raw, events, dataset)
        if epochs is not None:
            if len(epochs.events):

                epoch_file_name = get_path("epoch", dataset, path_pattern_type="to_fill",
                                           path_fill_kwargs=path_kwargs, **file_kwargs)
                epochs.save(epoch_file_name, overwrite=True)

                if dataset == "london06":
                    evoked = get_evoked(epochs)
                    evoked_file_name = get_path("evoked", dataset, path_pattern_type="to_fill",
                                                path_fill_kwargs=path_kwargs, **file_kwargs)
                    mne.write_evokeds(evoked_file_name, list(evoked.values()))

    elif dataset == "washington":
        epochs_videos, epochs_images = get_epochs(raw, events, dataset)
        if epochs_videos is not None:
            if len(epochs_videos.events):
                file_kwargs["task"] = "videos"
                epoch_file_name = get_path("epoch", dataset, path_pattern_type="to_fill",
                                           path_fill_kwargs=path_kwargs, **file_kwargs)
                epochs_videos.save(epoch_file_name, overwrite=True)

        if epochs_images is not None:
            if len(epochs_images.events):
                file_kwargs["task"] = "images"
                epoch_file_name = get_path("epoch", dataset, path_pattern_type="to_fill",
                                           path_fill_kwargs=path_kwargs, **file_kwargs)
                epochs_images.save(epoch_file_name, overwrite=True)

                evoked_file_name = get_path("evoked", dataset, path_pattern_type="to_fill",
                                            path_fill_kwargs=path_kwargs, **file_kwargs)
                evoked = get_evoked(epochs_images)
                mne.write_evokeds(evoked_file_name, list(evoked.values()))
    else:
        raise ValueError

    # SAVING RAW
    raw.save(raw_file_name, overwrite=True)
