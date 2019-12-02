import numpy as np
import tqdm
import re
import os
import pandas as pd
import warnings
from scipy.spatial import distance
from mne.channels import make_standard_montage
from glob import glob
from parse import parse
from slurppy import Config
import typing

from .atlas import center_of_masses
#from .config import eegip_config, load_additionnal_config
from .path import get_path
from .dataset import DatasetMng

inv_chan_mapping = {ch2: ch1 for ch1, ch2 in eegip_config["eeg"]["chan_mapping"].items()}
montage = make_standard_montage(eegip_config["eeg"]["montage"])

class Analysis

    def __init__(self,
                 config: typing.Optional[Config] = None,
                 dataset: typing.Optional[str] = None):

        self.config = Config.get_config(config_paths=config, load_default=False)
        self.dataset = dataset

    @property
    def dataset(self):
        return self._dataset_mng.current_dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset_mng = DatasetMng(config, dataset)


    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config: typing.Optional[Config]):
        if config is None:
            return
        if not isinstance(config, Config):
            raise TypeError("config must be of type Config. Received type: {}".format(type(config)))

        self._config = config
        for step in self.processing_steps.values():
            step.config = config


    def preprocess_dataset(self, resume=True, notebook=False, small=False):
        if notebook:
            progress = tqdm.tqdm_notebook
        else:
            progress = tqdm.tqdm

        for recording in progress(paths):
            recording.preprocess(resume)
            if small:
                break
























def format_factors(model, show_ref_level=False):
    if show_ref_level:
        format_string = "{0}[{2}-{1}]"
    else:
        format_string = "{0}[{2}]"

    xnames = []
    for xname in model.data.xnames:
        factors = []
        for factor in xname.split(":"):
            if "Treatment" in factor:
                factors.append(format_string.format(*parse('C({}, Treatment("{}"))[T.{}]', factor)))
            else:
                factors.append(factor)
        xnames.append(" : ".join(factors))
    model.data.xnames = xnames
    model.data.param_names[:len(xnames)] = xnames


def get_scalp_dist(ch_name1, ch_name2, montage_=montage):
    if ch_name1 in inv_chan_mapping:
        ch_name1 = inv_chan_mapping[ch_name1]
    if ch_name2 in inv_chan_mapping:
        ch_name2 = inv_chan_mapping[ch_name2]
    return distance.euclidean(montage_._get_ch_pos()[ch_name1], montage_._get_ch_pos()[ch_name2])


def get_event_types(dataset):
    if dataset == "washington":
        return list(eegip_config["event_id"][dataset]["all"].keys())
    return list(eegip_config["event_id"][dataset].keys())


def get_subjects_info(dataset):

    if dataset == "washington":
        data_subjects = pd.read_csv(os.path.join(eegip_config["paths"]["demo_var_dir"], 'washington_participants.csv'),
                                    index_col=0, names=["risk", "asd", "gender"])

        warnings.warn('Setting "no_asd_18m" to "no_asd".')
        warnings.warn('Setting "unknown" to "no_asd".')
        warnings.warn('Setting "asd_18m" to "asd".')
        data_subjects.loc[data_subjects["asd"] == "no_asd_18m", "asd"] = "no_asd"
        data_subjects.loc[data_subjects["asd"] == "asd_18m", "asd"] = "asd"
        data_subjects.loc[data_subjects["asd"] == "unknown", "asd"] = "no_asd"    

        return data_subjects
    
    if dataset in ["london06", "london12"]:
        # VERIFY THIS CORRESPONDENCE
        gender_dict = {0: "M", 1: "F"}
        risk_dict = {0: "LRC", 1: "HRA"}
        asd_dict = {0: "no_asd", 1: "no_asd", 2: "asd"}
        data_subjects = pd.read_excel(os.path.join(eegip_config["paths"]["demo_var_dir"], 'london_ERPdata.xls'),
                                      index_col=0, usecols=[0, 21, 22, 23],  header=0)

        data_subjects["gender"] = [gender_dict[g] for g in data_subjects["Gender"]]
        del data_subjects["Gender"]

        data_subjects["risk"] = [risk_dict[g] for g in data_subjects["Group"]]
        del data_subjects["Group"]

        data_subjects["asd"] = ["no_asd" if np.isnan(g) else asd_dict[g] for g in data_subjects["Subgroup"]]
        del data_subjects["Subgroup"]

        return data_subjects

    raise ValueError


def get_subject_id(file_name):
    return int(re.search(r'\d{3}', file_name).group(0))


def get_sources_dist(source1, source2, center_of_masses_=center_of_masses):
    if source1 not in center_of_masses_.index:
        return np.nan
    if source2 not in center_of_masses_.index:
        return np.nan
    
    return distance.euclidean(center_of_masses_.loc[source1].to_numpy(),
                              center_of_masses_.loc[source2].to_numpy())


def get_signal_distance(con_type, signal1, signal2, **kwargs):
    if con_type == "sources":
        return get_sources_dist(signal1, signal2, **kwargs)
    elif con_type == "scalp":
        return get_scalp_dist(signal1, signal2, **kwargs)
    else:
        raise ValueError("Unknown type of connections.")


def compute_connectivity_aggregate(dataset, con_type, small=False, resume=True, config=None, notebook=False,  **kwargs):

    if config is not None:
        load_additionnal_config(config)

    if dataset == "washington":
        kwargs["task"] = "*"

    out_file_name = get_path("connectivity_aggregate", dataset=dataset, con_type=con_type, **kwargs)
    if resume and os.path.exists(out_file_name):
        return

    if notebook:
        progress = tqdm.tqdm_notebook
    else:
        progress = tqdm.tqdm

    data_subjects = get_subjects_info(dataset)
    event_types = get_event_types(dataset)
    bands = eegip_config["analysis"]["band"]
    if small:
        event_types = event_types[:1]
        bands = bands[:1]
    if config is not None:
        load_additionnal_config(config)

    if "con_type" in kwargs:
        con_type = kwargs["con_type"]
    else:
        if "inv_method" in kwargs:
            con_type = "sources"
        else:
            con_type = "scalp"

    con_matrices = []
    for event_type in progress(event_types, desc="event types"):
        for fmin, fmax in progress(bands, desc="frequency bands", leave=False):
            path_pattern = get_path("connectivity_matrix", dataset, event_type=event_type,
                                    con_type=con_type, fmin=fmin, fmax=fmax, **kwargs)
            file_names = sorted(glob(path_pattern))
            if len(file_names) == 0:
                raise FileNotFoundError("No file were found to process. We tried with {}.".format(path_pattern))

            if small:
                file_names = file_names[:2]
            for file_name in progress(file_names, desc="files", leave=False):
                subject = get_subject_id(file_name)
                if subject not in data_subjects.index:
                    print("No info on subject", subject, "Skipping.")
                    continue

                try:
                    con_unstack = pd.read_csv(file_name, index_col=0).unstack().reset_index()
                except pd.errors.EmptyDataError:
                    print("There has been an error reading the file {}.".format(file_name))
                    raise

                con_unstack.columns = ["signal1", "signal2", "pli"]
                con_unstack = con_unstack[con_unstack["pli"] != 0.0]
                con_unstack["event_type"] = event_type
                con_unstack["fmin"] = fmin
                con_unstack["fmax"] = fmax
                con_unstack["asd"] = data_subjects.loc[subject, "asd"]
                con_unstack["risk"] = data_subjects.loc[subject, "risk"]
                con_unstack["gender"] = data_subjects.loc[subject, "gender"]
                con_unstack["subject"] = subject
                con_unstack["dist"] = [get_signal_distance(con_type, source1, source2) for source1, source2 in
                                       zip(con_unstack["signal1"], con_unstack["signal2"])]
                if dataset == "washington":
                    for i in range(3):
                        if "t{}task".format(i+1) in file_name:
                            con_unstack["task"] = "t{}task".format(i+1) 
                    for age in ["m06", "m12", "m18"]:
                        if age in file_name:
                            con_unstack["age"] = int(age[1:])

                for key in kwargs:
                    con_unstack[key] = kwargs[key]

                con_matrices.append(con_unstack)
                
    con_matrix = pd.concat(con_matrices, ignore_index=True)
    del con_matrices

    # CREATING OUTPUT DIRECTIONS IF NECESSARY
    dir_name = os.path.dirname(out_file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    con_matrix.to_pickle(out_file_name)


def read_connectivity_aggregate(dataset, con_type, **kwargs):
    path = get_path("connectivity_aggregate", dataset, con_type=con_type, **kwargs)
    try:
        return pd.read_pickle(path)
    except FileNotFoundError:
        print("The arguments you passed resolved to the following file name: ", path)
        print("This file was not found. Here are similar files you may are looking for and which exists: ")
        print(glob(get_path("connectivity_aggregate", dataset, con_type=con_type,
                            file_pattern_type="glob_patterns", **kwargs)))
        print("If one of these files is what you are looking for, you need to adjust the arguments passed to " +
              "read_connectivity_aggregate().")
        raise


def clean_con_matrix(dataset, con_matrix, con_type):

    if con_type == "sources":
        # There are no source in entorhinal-lh so it produces NaNs.
        con_matrix = con_matrix[(con_matrix["signal1"] != "entorhinal-lh") & (con_matrix["signal2"] != "entorhinal-lh")]
        con_matrix = con_matrix[(con_matrix["pli"] < 1.0) & (con_matrix["pli"] > 0.02)]
        
    elif con_type == "scalp":
        con_matrix = con_matrix[(con_matrix["pli"] < 1.0) & (con_matrix["pli"] > 0.04)]
        
    if dataset == "washington":
        con_matrix.loc[con_matrix["asd"] == "asd_18m", "asd"] = "asd"

    elif dataset == "london06":
        cond = con_matrix["event_type"].values.copy()
        cond[cond != "noise"] = "face"
        con_matrix.insert(len(con_matrix.columns), "cond", cond)
    
    con_matrix.insert(len(con_matrix.columns), "log_con", np.log(con_matrix["pli"]))

    bins = np.percentile(con_matrix["dist"], np.arange(0, 101, 2))
    bins[-1] += 0.00001
    dist_bins = (bins[1:] + bins[:-1])/2.0
    con_matrix.insert(len(con_matrix.columns), "dist_bin", dist_bins[np.digitize(con_matrix["dist"], bins)-1])

    return con_matrix
    
    
def get_dist_binned_con_matrix(dataset, con_matrix):
    if dataset == "washington":
        data = con_matrix.groupby(["fmin", "subject", "dist_bin", "event_type",
                                   "asd", "risk", "gender", "age"]).median().reset_index()
        return data[data["event_type"] != "cov"]

    if dataset == "london06": 
        return con_matrix.groupby(["fmin", "subject", "dist_bin", "event_type",
                                   "asd", "risk", "gender", "cond"]).median().reset_index()

    if dataset == "london12":
        return con_matrix.groupby(["fmin", "subject", "dist_bin", "event_type",
                                   "asd", "risk", "gender"]).median().reset_index()
