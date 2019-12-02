from pathlib import Path
from bids import BIDSLayout

from ..config import Config

default_config_path = "~/.eegip_local.config"

class DatasetMng:
    def __init__(self, config=default_config_path, dataset=None):
        #config_path = Path(config_path).expanduser()
        #if not config_path.exists():
        #    with config_path.open("w"):
        #        pass
        self._config = config

        if dataset is not None:
            self.current_dataset = dataset

    @property
    def config_path(self):
        return self._config.path

    def __check_dataset__(self):
        if self.current_dataset is None:
            raise ValueError("You must first set a dataset (e.g., using DatasetMng.set_dataset(dataset). "+
                             "Registered dataset are {}".format(list(self._config.keys())))

    def __check_is_bids__(self):
        self.__check_dataset__()
        if not self.config["is_bids"]:
            ValueError("The dataset must be a BIDS dataset for it to have a layout object.")

    def set_dataset(self, dataset):
        self.current_dataset = dataset

    @property
    def layout(self):
        self.__check_is_bids__()
        return BIDSLayout(self.config["paths"]["root"])

    @property
    def subjects(self):
        return self.layout.entities["subject"].unique()

    @property
    def datasets(self):
        return list(self._config.keys())

    @property
    def config(self):
        self.__check_dataset__()
        if self.current_dataset not in self._config:
            err_msg = ("No configuration found for the dataset {}. Consider adding the configuration " +
                       "information in {}.").format(self.current_dataset, default_config_path)
            raise ValueError(err_msg)

        return self._config[self.current_dataset]

    def get_recordings(self):
        #return a generator for recordings
