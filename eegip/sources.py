import os
from glob import glob

from mne import make_forward_solution
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs

from .config import eegip_config, load_additionnal_config
from .path import get_path, parse_pattern
from .recording import Recording

import mne
import xarray as xr
import numpy as np
import tqdm


def get_cov_matrix(raw, epochs,  dataset, method="auto"):

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
    return mne.compute_covariance(epochs_cov, tmax=None, method=method)


def compute_sources(dataset, raw_data, epochs_all, noise_cov, src, solution, file_name_epo, resume,
                    labels_parc, fs_subject, atlas_dir, snr=1.0,  # use smaller SNR for raw data
                    inv_method=None, lambda2=None):

    path_kwargs, file_kwargs = parse_pattern("epoch", dataset, file_name_epo)
    if inv_method is not None:
        file_kwargs["inv_method"] = inv_method

    filtered_event_id = {key: val for key, val in epochs_all.event_id.items() if val in epochs_all.events[:, 2]}
    if resume:
        event_id = sorted(filtered_event_id)[-1]

        file_kwargs["event_type"] = event_id
        file_name_out = get_path("sources", dataset, con_type="sources", path_pattern_type="to_fill",
                                 path_fill_kwargs=path_kwargs, **file_kwargs)

        if os.path.exists(file_name_out):
            print(file_name_out + " aleady exists. Skipping.")
            return    
    
    if lambda2 is None:
        lambda2 = 1.0 / snr ** 2    
    
    if dataset == "london06":
        block_size = 10
    elif dataset == "london12":
        block_size = 3
    elif dataset == "washington":
        block_size = 3
    else:
        raise ValueError("The dataset " + dataset + " is not recognized.")

    # compute the fwd matrix
    fname_trans = None  # Should investigate how to create co-registration in the context of our atlas

    fwd = make_forward_solution(raw_data.info, fname_trans, src, solution,
                                mindist=5.0)  # ignore sources<=5mm from innerskull

    # Compute inverse operator
    inverse_operator = make_inverse_operator(epochs_all.info, fwd, noise_cov, depth=None, fixed=False)
    inv_src = inverse_operator['src']   

    for event_id in sorted(filtered_event_id):

        file_kwargs["event_type"] = event_id
        file_name_out = get_path("sources", dataset, con_type="sources", path_pattern_type="to_fill",
                                 path_fill_kwargs=path_kwargs, **file_kwargs)
        file_name_labels = get_path("labels", dataset, con_type="sources", path_pattern_type="to_fill",
                                    path_fill_kwargs=path_kwargs, **file_kwargs)

        # if resume:
        #    if os.path.exists(file_name_out):
        #        print(file_name_out + " aleady exists. Skipping.")
        #        continue

        epochs = epochs_all[event_id]
        if len(epochs) == 0:
            print("No events " + event_id + " found. Skipping.")
            continue      

        inds = np.linspace(0, len(epochs), int(len(epochs)/block_size)+2, endpoint=True, dtype=int)
        label_ts = []
        for start, stop in zip(inds[:-1], inds[1:]):
            stcs = apply_inverse_epochs(epochs[start:stop], inverse_operator, lambda2, method=inv_method,
                                        pick_ori=None, return_generator=False)

            # Average the source estimates within each label of the cortical parcellation
            # and each sub structures contained in the src space
            # If mode = 'mean_flip' this option is used only for the cortical label

            label_ts.append(mne.extract_label_time_course(stcs, labels_parc, inv_src, 
                                                          mode='mean_flip', allow_empty=True,
                                                          return_generator=False))

        labels_aseg = mne.get_volume_labels_from_src(inv_src, fs_subject, atlas_dir)
        labels = labels_parc + labels_aseg
        label_names = [label.name for label in labels]            
        np.save(file_name_labels, np.array(label_names))

        np.save(file_name_out, np.vstack(np.array(label_ts)))    


def compute_dataset_sources(dataset, fname_src=None, fname_bem=None, fs_subject="",
                            parc='aparc', surf_name='pial', inv_method=None,
                            resume=True, notebook=False, small=False, config=None):

    if config is not None:
        load_additionnal_config(config)

    atlas_dir = eegip_config["paths"]["atlas_dir"]
    if fname_src is None:
        fname_src = os.path.join(atlas_dir, fs_subject, 'infant_atlas-src.fif')
    if fname_bem is None:
        fname_bem = os.path.join(atlas_dir, fs_subject, 'infant_atlas-bem')

    atlas_dir = eegip_config['paths']['atlas_dir']

    # Load forward modeling artifacts
    src = mne.read_source_spaces(fname_src)
    solution = mne.read_bem_solution(fname_bem)
    labels_parc = mne.read_labels_from_annot(fs_subject, parc=parc, surf_name=surf_name,
                                             subjects_dir=atlas_dir)

    paths = sorted(glob(get_path("preprocessed_recording", dataset)))

    if small:
        paths = paths[:1]

    if notebook:
        progress = tqdm.tqdm_notebook
    else:
        progress = tqdm.tqdm

    for file_name in progress(paths):
        raw_data = mne.io.read_raw_fif(file_name)
        path_kwargs, file_kwargs = parse_pattern("preprocessed_recording", dataset, file_name)

        epochs_lst = []
        file_name_epo_lst = []
        noise_cov = None
        if dataset == "washington":
            file_kwargs["task"] = "videos"
            file_name_videos = get_path("epoch", dataset, path_pattern_type="to_fill",
                                        path_fill_kwargs=path_kwargs, **file_kwargs)
            if os.path.exists(file_name_videos):
                epochs_videos = mne.read_epochs(file_name_videos)
                noise_cov = get_cov_matrix(raw_data, epochs_videos, dataset)
                epochs_lst.append(epochs_videos)
                file_name_epo_lst.append(file_name_videos)

            file_kwargs["task"] = "images"
            file_name_images = get_path("epoch", dataset, path_pattern_type="to_fill",
                                        path_fill_kwargs=path_kwargs, **file_kwargs)
            if os.path.exists(file_name_images):
                epochs_images = mne.read_epochs(file_name_images)
                if noise_cov is None:
                    noise_cov = get_cov_matrix(raw_data, epochs_images, dataset)
                epochs_lst.append(epochs_images)
                file_name_epo_lst.append(file_name_images)

            if len(epochs_lst) == 0:
                continue

        else:
            file_name_epo = get_path("epoch", dataset, path_pattern_type="to_fill",
                                     path_fill_kwargs=path_kwargs, **file_kwargs)
            if not os.path.exists(file_name_epo):
                continue

            epochs_all = mne.read_epochs(file_name_epo)
            noise_cov = get_cov_matrix(raw_data, epochs_all, dataset)
            epochs_lst = [epochs_all]
            file_name_epo_lst.append(file_name_epo)

        for epochs_all, file_name_epo in zip(epochs_lst, file_name_epo_lst):
            compute_sources(dataset, raw_data, epochs_all,
                            noise_cov, src, solution,
                            file_name_epo, resume,
                            labels_parc, fs_subject, atlas_dir,
                            inv_method=inv_method)


class Sources:
    # ##################### FLIPPING ###################################
    # ##################### type all, labels ###########################

    def __init__(self, model):
        self.model = model
        self.subject = model.subject
        self.recordings = self.subject.recordings

    def _check_recording(self, recording):
        if isinstance(recording, Recording):
            return recording
        if isinstance(recording, str):
            return self.recordings[recording]
        raise TypeError

    def get_data(self, recording, no_epochs=None, grouping=None, type_="all", recompute=False):

        recording = self._check_recording(recording)

        # Returning average
        if grouping is not None:
            grouping_str, _ = self._check_grouping(grouping)
            if type_ == "all":
                self.compute_grouped_sources(recording, no_epochs=no_epochs, grouping=grouping, recompute=recompute)
                file_name = self.model.get_path("sources", "nc", acquisition=recording.name, grouping=grouping_str)
                return xr.open_mfdataset(str(file_name), concat_dim="epochs", combine='by_coords')
            if type_ == "labels":
                self.compute_grouped_labels(recording, no_epochs=no_epochs, grouping=grouping, recompute=recompute)
                file_name = self.model.get_path("labels-sources", "nc", acquisition=recording.name,
                                                grouping=grouping_str)
                return xr.open_mfdataset(str(file_name), concat_dim="epochs", combine='by_coords')

        # Return individual epochs sources
        if type_ == "all":
            self.compute(recording=recording, recompute=recompute)

            if no_epochs is None:
                file_names = str(self.model.get_path("sources", "nc", epoch="*", acquisition=recording.name))
            else:
                file_names = [str(self.model.get_path("sources", "nc", epoch=no_epoch, acquisition=recording.name))
                              for no_epoch in no_epochs]

        # Return individual epochs labels
        elif type_ == "labels":
            self.compute_labels(recording=recording, recompute=recompute)

            if no_epochs is None:
                file_names = str(self.model.get_path("labels-sources", "nc", epoch="*", acquisition=recording.name))
            else:
                file_names = [str(self.model.get_path("labels-sources", "nc",
                                                      epoch=no_epoch, acquisition=recording.name))
                              for no_epoch in no_epochs]
        else:
            raise ValueError

        return xr.open_mfdataset(file_names, concat_dim='epochs', combine='by_coords')

    @staticmethod
    def _check_grouping(grouping):
        if isinstance(grouping, dict):
            if len(grouping) == 1:
                return list(grouping.keys())[0], list(grouping.values())[0]
        if isinstance(grouping, str):
            if grouping == "mean":
                return grouping, np.mean
            if grouping == "std":
                return grouping, np.std
        raise ValueError("Invalid argument. Grouping passed was of type {} and value {}."
                         .format(type(grouping), grouping))

    def compute_grouped_labels(self, recording, no_epochs=None, grouping="mean", recompute=False):
        grouping_str, grouping_func = self._check_grouping(grouping)
        file_name = self.model.get_path("labels-sources", "nc", acquisition=recording.name, grouping=grouping_str)
        if not file_name.exists() or recompute:
            dataset = self.get_data(recording, no_epochs=no_epochs,
                                    recompute=(recompute in ["labels", "sources"]), type_="labels")
            dataset = dataset.reduce(grouping_func, "epochs").compute()
            dataset.to_netcdf(file_name)

    def compute_grouped_sources(self, recording, no_epochs=None, grouping="mean", recompute=False):
        grouping_str, grouping_func = self._check_grouping(grouping)
        file_name = self.model.get_path("sources", "nc", acquisition=recording.name, grouping=grouping_str)
        if not file_name.exists() or recompute:
            dataset = self.get_data(recording, no_epochs=no_epochs, recompute=(recompute == "sources"), type_="all")
            # dataset = dataset.chunk({"vertices": 100}).map_blocks(xr.Dataset.reduce, [grouping_func, "epochs"])
            dataset.reduce(grouping_func, "epochs", allow_lazy=True).compute().to_netcdf(file_name)

    def compute_labels(self, recording, recompute=False, no_epochs=None, **kwargs):
        if isinstance(recording, str):
            if recording not in self.recordings:
                raise TypeError("When Source does not already contain a recording, the recording attribute cannot" +
                                " be a string with the name of the recording. It must be a Recording object.")
            recording = self.recordings[recording]
        elif isinstance(recording, Recording):
            self.recordings[recording.name] = recording
        else:
            raise TypeError("recording must be a string or a Recording object.")

        paths = []
        if no_epochs is None:
            recording.epochs.drop_bad()
            no_epochs = np.arange(len(recording.epochs))

        no_epochs_to_compute = []
        for no_epoch in no_epochs:
            file_name = self.model.get_path("labels-sources", "nc", epoch=no_epoch, acquisition=recording.name)
            if recompute or not file_name.exists():
                no_epochs_to_compute.append(no_epoch)
                paths.append(file_name)

        if len(no_epochs_to_compute) == 0:
            return

        dataset = self.get_data(recording, no_epochs=no_epochs_to_compute,
                                recompute=(recompute == "sources"), **kwargs)

        offset = 0
        vertno = []
        offset_dict = {}
        for source_space in self.model.surface_src + self.model.volume_src:
            vertno.extend((source_space["vertno"] + offset).tolist())
            if source_space["type"] == "vol":
                offset_dict[source_space["seg_name"]] = offset
            offset += source_space["np"]

        ind_map = {no: ind for ind, no in enumerate(vertno)}
        inv_ind_map = {ind: no for ind, no in enumerate(vertno)}

        inv_labels_inds = {}
        for label in self.model.labels_surf:
            if label.hemi == "rh":
                offset = self.model.surface_src[0]["np"]
            else:
                offset = 0
            inv_labels_inds.update({ind_map[v]: label.name for v in label.vertices + offset if v in ind_map})

        for source_space in self.model.volume_src:
            offset = offset_dict[source_space["seg_name"]]
            inv_labels_inds.update({ind_map[v]: source_space["seg_name"] for v in source_space["vertno"] + offset
                                    if v in ind_map})

        label_inds, labels_coord = zip(*sorted(inv_labels_inds.items()))

        dataset = dataset.isel(vertices=np.array(label_inds))
        assert (np.all(dataset.coords["vertices"].data == np.array([inv_ind_map[ind] for ind in label_inds])))
        label_groups = dataset.rename({"vertices": "labels"}).assign_coords(labels=list(labels_coord)).groupby("labels")

        # mne.label.label_sign_flip(model.labels_surf[0],  model.surface_src)
        dataset = label_groups.mean()
        for no_epoch, file_name in zip(no_epochs_to_compute, paths):
            dataset.sel(epochs=[no_epoch]).to_netcdf(file_name)

    def compute(self, recording, snr=1.0, lambda2=None, method="dSPM", recompute=False, notebook=False, no_epochs=None):

        if isinstance(recording, str):
            if recording not in self.recordings:
                raise TypeError("When Source does not already contain a recording, the recording attribute cannot" +
                                " be a string with the name of the recording. It must be a Recording object.")
            recording = self.recordings[recording]
        elif isinstance(recording, Recording):
            self.recordings[recording.name] = recording
        else:
            raise TypeError("recording must be a string or a Recording object.")

        if lambda2 is None:
            lambda2 = 1.0 / snr ** 2

        if notebook:
            process = tqdm.tqdm_notebook
        else:
            process = tqdm.tqdm

        if self.model.inverse_operator is None:
            self.model.make_inverse_operator(recording)

        paths = []
        if no_epochs is None:
            recording.epochs.drop_bad()
            no_epochs = np.arange(len(recording.epochs))

        no_epochs_to_compute = []
        for no_epoch in no_epochs:
            file_name = self.model.get_path("sources", "nc", epoch=no_epoch, acquisition=recording.name)
            if recompute or not file_name.exists():
                no_epochs_to_compute.append(no_epoch)
                paths.append(file_name)

        if len(no_epochs_to_compute) == 0:
            return

        sources = mne.minimum_norm.apply_inverse_epochs(recording.epochs[no_epochs_to_compute],
                                                        self.model.inverse_operator, lambda2,
                                                        method=method, return_generator=True)

        source_spaces = self.model.surface_src + self.model.volume_src
        offset = 0
        vertices = []
        for no_epoch, file_name, source in process(zip(no_epochs_to_compute, paths, sources)):
            if len(vertices) == 0:
                for source_space, vert in zip(source_spaces, source.vertices):
                    vertices.extend((vert + offset).tolist())
                    offset += source_space["np"]

            array = xr.DataArray(source.data[:, :, None],
                                 dims=("vertices", "times", "epochs"),
                                 coords={'vertices': vertices,
                                         "times": source.times,
                                         "epochs": [no_epoch]})
            array.to_netcdf(file_name)

        # TODO: Save a config file.
