
from .dataset.datasetmng import DatasetMng
from .sourcemodel import SourceModel
from .recording import RecordingLEMON


class Subject:

    def __init__(self, name):
        self.name = name
        self.dataset_mng = DatasetMng()


class SubjectLEMON(Subject):

    def __init__(self, **kwargs):
        super(SubjectLEMON, self).__init__(**kwargs)
        self.models = {}
        self.recordings = {}
        self.dataset_mng.set_dataset("lemon")

    def add_model(self, name, type_="individual", **kwargs):
        self.models[name] = SourceModel(name=name, subject=self, type_=type_, **kwargs)
        return self.models[name]

    def add_recording(self, name, **kwargs):
        self.recordings[name] = RecordingLEMON(name, self, **kwargs)
        return self.recordings[name]

