import mne
from bids.layout.utils import write_derivative_description
from bids import BIDSLayout
import warnings

from .subject import Subject
from .sources import Sources
from .bids import path_patterns



class SourceModel:

    def __init__(self,
                 name: str,
                 subject: Subject,
                 fs_subject: str = None,
                 type_: str = "individual",
                 session: str = "01"):

        self.subject = subject
        self.name = name
        self.type_ = type_

        self._surface_src = None
        self._volume_src = None
        self._bem_model = None
        self._bem_solution = None
        self._labels_surf = None

        self.inverse_operator = None
        self.sources = Sources(self)
        self.fname_trans = None
        self.forward_solution = None
        self.noise_cov = None
        self.labels_vol = []
        self.session = session

        write_derivative_description(subject.bids_root, "model_" + name, bids_version='1.1.1', exist_ok=True)
        self.path = subject.bids_root / "derivatives" / ("model_" + name) / "sub-{}".format(subject.name)
        self.path.mkdir(parents=True, exist_ok=True)
        self.bids_layout = BIDSLayout(self.path.parent)

        if fs_subject is None:
            if self.type_ == "average":
                self.fs_subject = self.name
            else:
                self.fs_subject = self.subject.name
        else:
            self.fs_subject = fs_subject

    def get_path(self, suffix, extension=None, **kwargs):

        if self.type_ == "average" and suffix in ["bem", "bem-surf", "vol-src", "src"]:
            path = self.path.parent / (self.name + "_" + suffix)
            if extension is not None:
                path = path.with_suffix("." + extension)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

        entity = {'subject': self.subject.name, 'session': self.session, 'suffix': suffix}
        if extension is not None:
            entity["extension"] = extension
        entity.update(kwargs)
        path = self.path.parent / self.bids_layout.build_path(entity, path_patterns, validate=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def bem_model(self):
        if self._bem_model is None:
            self.make_bem_model()
        return self._bem_model

    @property
    def bem_solution(self):
        if self._bem_solution is None:
            self.make_bem_solution()
        return self._bem_solution

    @property
    def labels_surf(self):
        if self._labels_surf is None:
            self.read_labels()
        return self._labels_surf

    @property
    def volume_src(self):
        if self._volume_src is None:
            self.setup_volume_source_space()
        return self._volume_src

    @property
    def surface_src(self):
        if self._surface_src is None:
            self.setup_surface_source_space()
        return self._surface_src

    def make_bem_model(self, recompute=False, ico=4):
        file_name = self.get_path("bem-surf")
        if recompute or not file_name.exists():
            self._bem_model = mne.make_bem_model(subject=self.fs_subject, ico=ico)
            mne.write_bem_surfaces(file_name, self.bem_model)
        else:
            self._bem_model = mne.read_bem_surfaces(file_name)

    def make_bem_solution(self, recompute=False):
        file_name = self.get_path("bem")
        if recompute or not file_name.exists():
            self._bem_solution = mne.make_bem_solution(self.bem_model)
            mne.write_bem_solution(file_name, self._bem_solution)
        else:
            self._bem_solution = mne.read_bem_solution(file_name)

    def read_labels(self, parc="aparc"):
        self._labels_surf = mne.read_labels_from_annot(self.fs_subject, parc=parc)

    def setup_volume_source_space(self, labels_vol=None, recompute=False):
        file_name = self.get_path("vol-src", "fif")
        if recompute or not file_name.exists():
            # List of sub structures we are interested in. We select only the
            # sub structures we want to include in the source space
            if labels_vol is None:
                self.labels_vol = ['Left-Amygdala',
                                   'Left-Thalamus-Proper',
                                   'Left-Cerebellum-Cortex',
                                   'Brain-Stem',
                                   'Right-Amygdala',
                                   'Right-Thalamus-Proper',
                                   'Right-Cerebellum-Cortex']
            else:
                self.labels_vol = labels_vol

            if self.bem_model is None:
                self.make_bem_solution()
            self._volume_src = mne.setup_volume_source_space(subject=self.fs_subject, bem=self.bem_solution,
                                                             volume_label=self.labels_vol, mri="aseg.mgz")
            self.volume_src.save(file_name)
        else:
            self._volume_src = mne.read_source_spaces(file_name)
            self.labels_vol = [s["seg_name"] for s in self.volume_src]

    def setup_surface_source_space(self, recompute=False):
        file_name = self.get_path("src", "fif")
        if recompute or not file_name.exists():
            self._surface_src = mne.setup_source_space(self.fs_subject)
            self.surface_src.save(file_name)
        else:
            self._surface_src = mne.read_source_spaces(file_name)

    def setup_source_space(self):
        self.read_labels()
        self.setup_surface_source_space()
        self.setup_volume_source_space()

    @property
    def src(self):
        if self.surface_src is None:
            self.setup_surface_source_space()
        if self.volume_src is None:
            self.setup_volume_source_space()

        return self.surface_src + self.volume_src

    def make_forward_solution(self, recording, fname_trans=None, recompute=False):
        file_name = self.get_path("fwd", "fif")
        if recompute or not file_name.exists():
            if fname_trans is None:
                fname_trans = self.get_path("trans", "fif")

                if fname_trans.exists():
                    self.fname_trans = str(fname_trans)
                else:
                    warnings.warn("Using no transform to align the sensors to the head. This is not recommended.")
            else:
                self.fname_trans = fname_trans

            self.forward_solution = mne.make_forward_solution(recording.raw.info, self.fname_trans,
                                                              self.src, self.bem_solution, mindist=0)
            mne.write_forward_solution(file_name, self.forward_solution)
        else:
            self.forward_solution = mne.read_forward_solution(file_name)

    def estimate_cov_variance(self, recording=None, tmax=0, method="auto", recompute=False):
        file_name = self.get_path("cov", "fif")
        if recompute or not file_name.exists():
            if recording is None:
                raise ValueError("A recording is necessary to compute a covariance matrix.")
            self.noise_cov = mne.compute_covariance(recording.epochs, tmax=tmax, method=method)
            mne.write_cov(file_name, self.noise_cov)
        else:
            self.noise_cov = mne.read_cov(file_name)

    def make_inverse_operator(self, recording, recompute=False, **kwargs):
        file_name = self.get_path("inv", "fif")
        if recompute or not file_name.exists():
            if self.noise_cov is None:
                self.estimate_cov_variance(recording, **kwargs)
            if self.forward_solution is None:
                self.make_forward_solution(recording, **kwargs)

            self.inverse_operator = mne.minimum_norm.make_inverse_operator(recording.raw.info, self.forward_solution,
                                                                           self.noise_cov)
            mne.minimum_norm.write_inverse_operator(file_name, self.inverse_operator)
        else:
            self.inverse_operator = mne.minimum_norm.read_inverse_operator(file_name)

    def compute_sources(self, recording, **kwargs):
        self.sources.compute(recording, **kwargs)
        # TODO: Save a config file.

    def compute_labels(self, recording, **kwargs):
        self.sources.compute_labels(recording, **kwargs)
        # TODO: Save a config file.

    def get_sources(self, **kwargs):
        return self.sources.get_data(**kwargs)

    @property
    def region_names(self):

        return [l.name for l in self.labels_surf] + self.labels_vol
        # [s["seg_name"] for s in self.inverse_operator['src'][2:]]
