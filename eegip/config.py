from yaml import load
from pathlib import Path
import collections

# python 3.8+ compatibility
try:
    collectionsAbc = collections.abc
except ImportError:
    collectionsAbc = collections

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


# Recursive updates. Default dictionnary update is not recursive, which
# cause dict within dict to be simply overwritten rather than merged
def update(d, u):
    for k, v in u.items():
        dv = d.get(k, {})
        if not isinstance(dv, collectionsAbc.Mapping):
            d[k] = v
        elif isinstance(v, collectionsAbc.Mapping):
            d[k] = update(dv, v)
        else:
            d[k] = v
    return d


def join(loader, tag_suffix, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


class EEGIPLoader(Loader):
    pass


EEGIPLoader.add_multi_constructor("!join", join)


class Config(collections.MutableMapping):

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return self.store.__repr__()

    def __contains__(self, item):
        return item in self.store

    @staticmethod
    def get_config(config_paths=(), load_default=True):
        self = Config()

        if load_default:
            self.path = Path(__file__).parent.parent / "configs" / "config.yaml"
            with self.path.open('r') as stream:
                update(self, load(stream, Loader=EEGIPLoader))

        if isinstance(config_paths, str):
            config_paths = [config_paths]
        for config_path in config_paths:
            with Path(config_path).open('r') as stream:
                config_supp = load(stream, Loader=EEGIPLoader)
            if config_supp is not None:
                self = update(self, config_supp)

        return self


eegip_config = Config.get_config()


def load_additionnal_config(config_paths):
    update(eegip_config, Config.get_config(config_paths, load_default=False))


if __name__ == "__main__":
    print(get_config())
