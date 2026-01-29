"""
BEM Surface Generation for MEG Source Localization.

This script generates BEM (Boundary Element Method) surfaces from FreeSurfer
reconstructions for use in MEG source localization. It creates watershed BEM
surfaces and scalp surfaces for all subjects in each drug condition.

The script processes subjects in parallel using joblib.

Usage:
    python bem_surfaces.py
"""

import mne
from mne.coreg import Coregistration
import os
from joblib import Parallel, delayed

HOMEDIR = "/Brain/private/v20subra/LSD_project"

freesurfer_home = "/Brain/private/v20subra/LSD_project/src_data/Freesurfer/freesurfer"


def process_subjects(subject, subjects_dir):
    """
    Generate BEM surfaces for a single subject.

    Creates watershed BEM surfaces and scalp surfaces from FreeSurfer
    reconstruction output.

    Parameters
    ----------
    subject : str
        Subject identifier (FreeSurfer subject directory name).
    subjects_dir : str
        Path to the FreeSurfer subjects directory.

    Returns
    -------
    None
        BEM surfaces are saved to the subject's FreeSurfer directory.
    """
    print(f"Processing subject {subject}..., from {subjects_dir}")
    
    mne.bem.make_watershed_bem(subject, subjects_dir=subjects_dir, overwrite=True)
    mne.bem.make_scalp_surfaces(subject, subjects_dir=subjects_dir, overwrite=True)


conditions = ['LSD', 'PLA']
for condition in conditions:
    subjects_dir = f"/Brain/private/v20subra/LSD_project/src_data/derivatives/anat/{condition}"
    subjects = os.listdir(f"{HOMEDIR}/src_data/derivatives/anat/{condition}/")
    subjects.remove('fsaverage')
    Parallel(n_jobs=-1)(delayed(process_subjects)(subject, subjects_dir) for subject in subjects)
