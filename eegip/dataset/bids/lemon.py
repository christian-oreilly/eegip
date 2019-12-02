import shutil
from pathlib import Path
from distutils.dir_util import copy_tree
import csv
import json
from mne.externals.pymatreader import read_mat
from bids import BIDSLayout
import numpy as np
from bids.layout.utils import write_derivative_description
import tqdm

bd_root = Path("/media/christian/ElementsSE/MPI-Leipzig_Mind-Brain-Body-LEMON/")
bids_root = bd_root / "BIDS_LEMON"
path_montage = bd_root / "EEG_MPILMBB_LEMON/EEG_Localizer_BIDS_ID/"
parse_pattern_montage = "sub-{subject}/sub-{subject}.mat"
session = "01"

path_pattern_stem = "sub-{subject}[/ses-{session}]/eeg/sub-{subject}[_ses-{session}][_acq-{acquisition}][_run-{run}][_proc-{proc}]_"
path_patterns = [
    path_pattern_stem + "{suffix<channels|electrodes|coordsystem>}.{extension<tsv|json>|tsv}"
]


def convert_LEMON_to_BIDS(subject):
    write_derivative_description(bids_root, "preprocessed", bids_version='1.1.1', exist_ok=True)
    convert_raw_mri([subject])
    convert_preprocessed_eeg([subject])
    convert_channel_file([subject])

def convert_raw_mri(subjects):
    for subject in subjects:
        src = bd_root / "MRI_MPILMBB_LEMON" / "MRI_Raw" / "sub-{}".format(subject)
        dst = bids_root / "sub-{}".format(subject)
        dst.parent.mkdir(parents=True, exist_ok=True)
        copy_tree(str(src), str(dst))  # , symlinks=False, ignore=None)


def convert_channel_file(subjects):
    bids_layout = BIDSLayout(bids_root)

    for subject in subjects:
        file_name = path_montage / parse_pattern_montage.format(subject=subject)

        montage_mat_file = read_mat(file_name)

        head_points = {}
        for ch_name in np.unique(montage_mat_file["HeadPoints"]["Label"]):
            inds = np.where(np.array(montage_mat_file["HeadPoints"]["Label"]) == ch_name)[0]
            head_points[ch_name] = montage_mat_file["HeadPoints"]["Loc"][:, inds].mean(1)

        ch_names = [ch_name.split("_")[-1] for ch_name in montage_mat_file["Channel"]["Name"]]
        ch_names = [ch_name if ch_name[:2] != "FP" else "Fp" + ch_name[2:] for ch_name in ch_names]

        entity = {'subject': subject, 'suffix': "electrodes", "extension": "tsv", "session": session}
        label_file = bids_root / "derivatives" / "preprocessed" / bids_layout.build_path(entity, path_patterns)
        label_file.parent.mkdir(parents=True, exist_ok=True)
        with open(label_file, 'w', encoding='utf8', newline='') as tsv_file:
            tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
            tsv_writer.writerow(["name", "x", "y", "z"])
            for ch_name, (x, y, z) in zip(ch_names, montage_mat_file["Channel"]["Loc"]):
                tsv_writer.writerow([ch_name, x, y, z])

        coord_dict = {
            "AnatomicalLandmarkCoordinates": {
                "LPA": head_points["LPA"].tolist(),
                "NAS": head_points["NA"].tolist(),
                "RPA": head_points["RPA"].tolist(),
            }
        }
        entity = {'subject': subject, 'suffix': "coordsystem", "extension": "json", "session": session}
        coord_file = bids_root / "derivatives" / "preprocessed" / bids_layout.build_path(entity, path_patterns)
        with (coord_file).open('w') as fobj:
            json.dump(coord_dict, fobj, indent=4)


def convert_preprocessed_eeg(subjects, tasks=("EC", "EO"), extensions=("set", "fdt")):
    for subject in subjects:
        for task in tasks:
            for extension in extensions:
                fname_in = bd_root / "EEG_MPILMBB_LEMON" / "EEG_Preprocessed_BIDS_ID" / "EEG_Preprocessed"
                fname_in = fname_in / "sub-{}_{}.{}".format(subject, task, extension)

                fname_out = bids_root / "derivatives" / "preprocessed" / "sub-{}".format(subject)
                fname_out = fname_out / "ses-{}".format(session) / "eeg"
                fname_out.mkdir(parents=True, exist_ok=True)

                fname_out /= "sub-{}_ses-{}_task-{}_eeg.{}".format(subject, session, task, extension)

                shutil.copy(str(fname_in), str(fname_out))


def convert_dataset(notebook=False):

    if notebook:
        process = tqdm.tqdm_notebook
    else:
        process = tqdm.tqdm

    path = bd_root / "MRI_MPILMBB_LEMON" / "MRI_Raw"
    subjects_mri = [p.name[4:] for p in list(path.glob("*"))]

    path = bd_root / "EEG_MPILMBB_LEMON" / "EEG_Preprocessed_BIDS_ID" / "EEG_Preprocessed"
    subjects_EC = [p.name[4:10] for p in list(path.glob("*_EC.fdt"))]
    subjects_EO = [p.name[4:10] for p in list(path.glob("*_EO.fdt"))]

    subject_montage = [p.name[4:] for p in path_montage.glob("*")]

    subjects, counts = np.unique(np.concatenate([subject_montage, subjects_EC, subjects_EO, subjects_mri]),
                                 return_counts=True)

    for subject in process(subjects[counts == 4]):
        convert_LEMON_to_BIDS(subject)

def get_layout():
    return BIDSLayout(bids_root)