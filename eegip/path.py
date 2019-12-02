import os
from eegip.config import eegip_config
from parse import Parser
from parse_type import TypeBuilder
from warnings import warn


class FormatStr(str):
    def __format__(self, _):
        return self


def format_dict(dic):
    return {key: FormatStr(val) for key, val in dic.items()}


def with_zero_or_one(cls, converter, pattern=None):
    ret_val = cls.with_optional(converter, pattern)

    def convert_optional(text):
        if text:
            text = text.strip()
        if not text:
            return ""
        return converter(text)

    convert_optional.pattern = ret_val.pattern
    convert_optional.regex_group_count = ret_val.regex_group_count
    return convert_optional


parse_str = with_zero_or_one(TypeBuilder, str)


_file_patterns = {
    "eegip_recording": ".set",
    "preprocessed_recording": "-{args:str}-raw.fif",
    "epoch": "-{args:str}-epo.fif",
    "evoked": "-{args:str}-ave.fif",
    "sources": "-{event_type:str}-{args:str}-{con_type:w}.npy",
    "labels": "-{event_type:str}-{args:str}-{con_type:w}-labels.npy",
    "connectivity_matrix": "-{fmin:w}-{fmax:w}-{event_type:str}-{args:str}-{con_type:w}-con.csv",
    "connectivity_aggregate": "/con_matrix-{con_type:w}-{args:str}.pck"
    }


def get_valid_arg_keys(path_type, con_type=None):

    _valid_arg_keys = {
        "eegip_recording": [],
        "preprocessed_recording": [],
        "epoch": ["task"],
        "evoked": ["task"],
    }

    ## At some point con_type should be renamed signal_type troughout eegip...
    _valid_arg_keys_signal_type = {
        "sources": {"sources": ["task", "inv_method"],
                    "labels": ["task", "inv_method"],
                    "connectivity_matrix": ["task", "inv_method", "method"],
                    "connectivity_aggregate": ["task", "inv_method", "method"]},
        "scalp": {"connectivity_matrix": ["task", "method"],
                  "connectivity_aggregate": ["task", "method"]}
    }

    if path_type in _valid_arg_keys:
        return _valid_arg_keys[path_type]

    return _valid_arg_keys_signal_type[con_type][path_type]


# def get_stem_path(file_name, stem_end="_qcr"):
#    if len(file_name.split(stem_end)) != 2:
#        raise ValueError("The stem_end must be present once in file_name.")
#    return file_name.split(stem_end)[0] + stem_end


def _get_path_stem(path_type, dataset):
    if path_type == "eegip_recording":
        return eegip_config["paths"]["data"][dataset]

    if path_type in ["preprocessed_recording", "epoch", "evoked", "sources", "labels",
                     "connectivity_matrix", "connectivity_aggregate"]:
        return eegip_config["paths"]["path_out"][dataset]

    raise ValueError("path_type {} is not recognized. Valid types are {}"
                     .format(path_type, list(_file_patterns.keys())))


def _get_path_pattern(path_type, dataset, filled_path_pattern=None, pattern_type="glob_patterns"):

    if filled_path_pattern is not None:
        pattern = filled_path_pattern
    else:

        pattern = eegip_config[pattern_type][dataset]

    if path_type in ["eegip_recording", "preprocessed_recording", "epoch", "evoked",
                     "sources", "labels", "connectivity_matrix"]:
        return pattern
    if path_type == "connectivity_aggregate":
        return ""

    raise ValueError


def _get_filled_file_pattern(path_type, event_type=None, con_type=None, fmin=None, fmax=None, **kwargs):

    format_kwargs = {}

    for var, name in zip([event_type, con_type, fmin, fmax], ["event_type", "con_type", "fmin", "fmax"]):
        if var is not None:
            if name in _file_patterns[path_type]:
                format_kwargs[name] = var
            else:
                warn("A value has been passed to " + name
                     + " but this variable is not part of the path pattern. It has been ignored.")

    filtered_args = {key:val for key, val in kwargs.items() if key in get_valid_arg_keys(path_type, con_type)}
    format_kwargs["args"] = "-".join(["{}={}".format(key, val) for key, val in sorted(filtered_args.items())])

    try:
        filled_file_pattern = _file_patterns[path_type].format(**format_dict(format_kwargs))
    except KeyError:
        raise KeyError("The argument dictionary format_kwargs ({}) does not match the file pattern ({})"
                       .format(format_kwargs, _file_patterns[path_type]))

    return filled_file_pattern


def get_path(path_type, dataset, filled_path_pattern=None, path_pattern_type="glob_patterns", path_fill_kwargs=None,
             file_pattern_type="to_fill", event_type=None, con_type=None, fmin=None, fmax=None, **kwargs):
    """
     path_pattern_type can take four values:
        glob_patterns : correspond to the value in eegip_config["glob_patterns"]
        parse_patterns : correspond to the value in eegip_config["parse_patterns"]
        to_fill : To fill according to the values of path_fill_kwargs and to the parse_patterns.
    """

    stem = _get_path_stem(path_type, dataset)
    if path_pattern_type == "to_fill":
        if filled_path_pattern is not None:
            ValueError("When passing a value to the filled_pattern argument, " +
                       "the pattern_type should not be set to 'to_fill'.")
        try:
            filled_path_pattern = eegip_config["parse_patterns"][dataset].format(**format_dict(path_fill_kwargs))
        except KeyError:
            raise KeyError("The dictionary path_fill_kwargs ({}) is missing some values to fill the parse_pattern '{}'"
                           .format(str(path_fill_kwargs), eegip_config["parse_patterns"][dataset]))
    path_pattern = _get_path_pattern(path_type, dataset, filled_path_pattern, pattern_type=path_pattern_type)
    if file_pattern_type == "to_fill":
        filled_file_pattern = _get_filled_file_pattern(path_type, event_type=event_type, con_type=con_type,
                                                       fmin=fmin, fmax=fmax, **kwargs)
        return os.path.join(stem,  path_pattern) + filled_file_pattern

    if file_pattern_type == "parse_patterns":
        return os.path.join(stem,  path_pattern) + _file_patterns[path_type]

    if file_pattern_type == "glob_patterns":
        parts = _file_patterns[path_type].split("{")
        return os.path.join(stem,  path_pattern) + "*".join([part.split("}")[1] if no else part
                                                             for no, part in enumerate(parts)])

    raise ValueError("Unrecognized value for the file_pattern_type argument.")


def extract_path_pattern(path_type, path, dataset, event_type=None, con_type=None, fmin=None, fmax=None, **kwargs):

    stem = _get_path_stem(path_type, dataset)
    filled_file_pattern = _get_filled_file_pattern(path_type, event_type, con_type, fmin, fmax, **kwargs)

    schema = os.path.join(stem, "{}") + filled_file_pattern
    parser = Parser(schema, dict(str=parse_str))
    return parser.parse(path)


def parse_pattern(path_type, dataset, path_to_parse):

    # Parsing the path part
    stem = _get_path_stem(path_type, dataset)
    path_pattern = _get_path_pattern(path_type, dataset, pattern_type="parse_patterns")

    schema = os.path.join(stem,  path_pattern) + "{}"
    parser = Parser(schema, dict(str=parse_str))
    result = parser.parse(path_to_parse)

    path_kwargs = result.named
    file_part = result.fixed[-1]

    bck_result = result
    # Parsing the file part
    parser = Parser(_file_patterns[path_type], dict(str=parse_str))
    result = parser.parse(file_part)

    if result is None:
        print(path_kwargs)
        print(file_part)
        print(bck_result)
        print(_file_patterns[path_type])

    file_kwargs = result.named
    if "args" in file_kwargs:
        if len(file_kwargs["args"]):
            try:
                file_kwargs.update(dict([item.split("=") for item in file_kwargs["args"].split("-") if len(item)]))
            except ValueError:
                print(file_kwargs["args"])
                raise
        del file_kwargs["args"]

    return path_kwargs, file_kwargs
