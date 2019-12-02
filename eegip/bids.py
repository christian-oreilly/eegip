

path_pattern_stem_mod = "sub-{subject}[/ses-{session}]/mod/sub-{subject}[_ses-{session}][_task-{task}]" + \
                         "[_acq-{acquisition}][_ce-{ceagent}][_rec-{reconstruction}]_"
path_pattern_stem_eeg = "sub-{subject}[/ses-{session}]/eeg/sub-{subject}[_ses-{session}][_task-{task}]" + \
                        "[_acq-{acquisition}][_run-{run}][_proc-{proc}]_"

path_patterns = [
    path_pattern_stem_mod + "{suffix<bem-surf|bem>}",
    path_pattern_stem_mod + "{suffix<vol-src|src|fwd|trans|cov|inv>}.{extension<fif>}",
    path_pattern_stem_mod + "[{grouping}-]{suffix<labels-sources|sources>}" +
                            "[_{epoch}].{extension<nc>}",
    path_pattern_stem_eeg + "{suffix<channels|electrodes|coordsystem>}.{extension<tsv|json>|tsv}",
    path_pattern_stem_eeg + "{suffix<eeg>}.{extension<set>}",
]
