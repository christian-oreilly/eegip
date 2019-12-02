import numpy as np
import xarray as xr
from scipy.stats import t


def compute_diff_exp(model, recordings, alpha=0.05):
    mean_diffs = {}
    ic_widths = {}
    evoked_N = {}
    evoked_source_mean_all = {}
    evoked_source_mean = {}
    evoked_scalp_mean = {}

    evoked_source_std_all = {}
    evoked_source_std = {}
    evoked_scalp_std = {}

    for recording in recordings:
        evoked_source_mean[recording] = model.get_sources(recording=recording, grouping="mean", type_="labels")
        evoked_source_mean[recording] = evoked_source_mean[recording].compute().to_array()
        evoked_source_mean[recording] = evoked_source_mean[recording].rename({"labels": "signals"}).squeeze()
        evoked_source_mean[recording] -= evoked_source_mean[recording].mean("times")

        evoked_source_mean_all[recording] = model.get_sources(recording=recording, grouping="mean", type_="all")
        evoked_source_mean_all[recording] = evoked_source_mean_all[recording].compute().to_array()
        evoked_source_mean_all[recording] = evoked_source_mean_all[recording].squeeze()
        evoked_source_mean_all[recording] -= evoked_source_mean_all[recording].mean("times")

        evoked_source_std[recording] = model.get_sources(recording=recording, grouping="std", type_="labels")
        evoked_source_std[recording] = evoked_source_std[recording].load().to_array().rename(
            {"labels": "signals"}).squeeze()

        evoked_source_std_all[recording] = model.get_sources(recording=recording, grouping="std", type_="all")
        evoked_source_std_all[recording] = evoked_source_std_all[recording].load().to_array().squeeze()

        epochs = model.subject.recordings[recording].epochs
        epochs.drop_bad()
        evoked_scalp_mean[recording] = xr.DataArray(np.mean(epochs.get_data(), axis=0),
                                                    dims=("signals", "times"),
                                                    coords={"signals": epochs.ch_names,
                                                            "times": epochs.times})
        evoked_scalp_mean[recording] -= evoked_scalp_mean[recording].mean("times")
        evoked_scalp_std[recording] = xr.DataArray(np.std(epochs.get_data(), axis=0),
                                                   dims=("signals", "times"),
                                                   coords={"signals": epochs.ch_names,
                                                           "times": epochs.times})
        evoked_N[recording] = len(epochs)

    evoked_mean = {"scalp": evoked_scalp_mean,
                   "sources": evoked_source_mean,
                   "sources_all": evoked_source_mean_all}
    evoked_std = {"scalp": evoked_scalp_std,
                  "sources": evoked_source_std,
                  "sources_all": evoked_source_std_all}

    for signal_type in evoked_mean:
        # channels X times
        ic_width_ec = t.ppf(1 - alpha / 2.0, df=evoked_N["EC"]) * evoked_std[signal_type]["EC"] / np.sqrt(
            evoked_N["EC"])
        ic_width_eo = t.ppf(1 - alpha / 2.0, df=evoked_N["EO"]) * evoked_std[signal_type]["EO"] / np.sqrt(
            evoked_N["EO"])

        ic_widths[signal_type] = np.sqrt(ic_width_ec ** 2 + ic_width_eo ** 2)
        mean_diffs[signal_type] = np.abs(evoked_mean[signal_type]["EC"] - evoked_mean[signal_type]["EO"])

    return mean_diffs, ic_widths
