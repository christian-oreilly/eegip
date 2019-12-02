import mne
from bids import BIDSLayout
import numpy as np
import csv
import json

from .bids import path_patterns

class Recording:
    pass


class RecordingLEMON(Recording):

    event_ids = {"EO": 1, "EC": 2}

    def __init__(self, name, subject, session="01", derivatives=("preprocessed",), task=None):
        self.name = name
        self.subject = subject
        self.montage = None
        self._raw = None
        self._events = None
        self._epochs = None
        self.task = task
        self.session = session
        self.path = self.subject.bids_root
        for derivative in derivatives:
            self.path = self.path / "derivatives" / derivative
        self.bids_layout = BIDSLayout(self.path)

    @property
    def raw(self):
        if self._raw is None:
            self.load_raw()
        return self._raw

    @property
    def epochs(self):
        if self._epochs is None:
            self.load_epochs()
        return self._epochs

    @property
    def events(self):
        if self._events is None:
            self.load_events()
        return self._events

    def get_path(self, suffix, extension=None, **kwargs):
        entity = {'subject': self.subject.name, 'suffix': suffix}
        if extension is not None:
            entity["extension"] = extension
        if self.session is not None:
            entity["session"] = self.session

        entity.update(kwargs)
        return self.path / self.bids_layout.build_path(entity, path_patterns)

    def load_montage(self):
        with open(self.get_path("electrodes", "tsv"), 'r') as tsv_file:
            reader = csv.reader(tsv_file, delimiter='\t', lineterminator='\n')
            ch_pos = {ch_name: [float(x), float(y), float(z)] for ch_name, x, y, z in reader if ch_name != "name"}
        with open(self.get_path("coordsystem", "json"), 'r') as fobj:
            coord_dict = json.load(fobj)["AnatomicalLandmarkCoordinates"]
        self.montage = mne.channels.make_dig_montage(ch_pos=ch_pos,
                                                     nasion=coord_dict["NAS"],
                                                     lpa=coord_dict["LPA"],
                                                     rpa=coord_dict["RPA"])
        try:
            self._raw.set_montage(self.montage)
        except ValueError:
            print("Channels in montage not present in raw: ",
                  np.array(self.montage.ch_names)[np.logical_not(np.in1d(self.montage.ch_names, self._raw.ch_names))])
            print("Channels in raw not present in montage: ",
                  np.array(self._raw.ch_names)[np.logical_not(np.in1d(self._raw.ch_names, self.montage.ch_names))])
            raise

    def load_raw(self):
        path = self.get_path("eeg", "set", task=self.task)
        self._raw = mne.io.read_raw_eeglab(str(path))
        self._raw._filenames[0] = str(path.with_suffix(".fdt"))
        if self.subject.name == '010063':
            self._raw.load_data()
            self._raw = self._raw.drop_channels(["Oz"])

        mne.set_eeg_reference(self._raw, ref_channels='average', copy=False, projection=True, ch_type='eeg')

    def load_events(self):
        annot_sample = []
        annot_id = []
        freq = self.raw.info["sfreq"]

        annot_map = {'2': "EO", '3': 'EO', '4': 'EC'}

        for a in self.raw.annotations:
            if a["description"] in annot_map:
                annot_sample.append(int(a["onset"] * freq))
                annot_id.append(RecordingLEMON.event_ids[annot_map[a["description"]]])

        self._events = np.array([annot_sample, [0] * len(annot_sample), annot_id], dtype=int).T

        if self.subject.name in ['010165', '010233', '010262', '010263', '010269', '010271']:
            self._events = self._events[1:, :]
        elif self.subject.name in ['010275', '010284', '010268', '010258'] and self.name == "EC":
            self._events = self._events[1:, :]
        elif self.subject.name in ['010260', '010311', '010315'] and self.name == "EO":
            self._events = self._events[1:, :]

    def load_epochs(self, tmin=-0.2, tmax=2.2, baseline=(None, 0)):
        self.load_montage()
        self.load_events()
        self._epochs = mne.Epochs(self.raw, self.events, tmin=tmin, tmax=tmax, baseline=baseline)
