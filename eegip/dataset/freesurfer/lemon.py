import os
from pathlib import Path


def recon_all(subject, config, freesurfer_home=None, wsthresh=20):

    if freesurfer_home is None:
        freesurfer_home = os.environ["FREESURFER_HOME"]
    if isinstance(freesurfer_home, str):
        freesurfer_home = Path(freesurfer_home)

    subjects_dir = freesurfer_home / "subjects"
    root_path = Path(config["root_path"])

    t1 = root_path / subject / "sub-{subject}_ses-01_acq-mp2rage_T1w.nii.gz".format(subject=subject)
    flair = root_path / subject / "sub-{subject}_ses-01_acq-highres_FLAIR.nii.gz".format(subject=subject)
    watershed_path = subjects_dir / subject / "bem" / "watershed"

    commands = [
        "recon-all -wsthresh {wsthresh} -subject {subject} -i {t1} -FLAIR {flair} -FLAIRpial -all"
        .format(wsthresh=wsthresh,
                subject=subject,
                t1=t1,
                flair=flair),
        "mri_watershed -useSRAS -less -t 75 -atlas -surf {watershed_path}/{subject} {flair} {watershed_path}/ws"
        .format(watershed_path=watershed_path,
                subject=subject,
                flair=flair)
    ]

    # Names and location needed for pycortex
    for surface in ["inner_skull", "outer_skull", "outer_skin"]:
        target = subjects_dir / subject / "bem" / "watershed"
    target /= "{subject}_{surface}_surface".format(subject=subject, surface=surface)
    ln_path = subjects_dir / subject / "bem" / "{surface}.surf".format(surface=surface)
    ln_path.symlink_to(target=target)

