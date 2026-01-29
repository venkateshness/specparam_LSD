"""
Forward and Inverse Solver for MEG Source Localization.

This module performs MEG source localization using MNE-Python, including:
- Source space creation (ico5 spacing)
- BEM solution computation
- Forward model creation
- Inverse solution using dSPM
- Morphing to fsaverage template
- Parcellation using HCP-MMP1 atlas

Usage:
    python forward_and_inverse_solver.py -s 001 002 003 -t Music -d LSD
"""

import mne
import argparse
from joblib import Parallel, delayed
import os
import logging
import numpy as np

HOMEDIR = "/Brain/private/v20subra/LSD_project"


def create_source_space(subjects_dir, subject, drug):
    """
    Create or load the source space for a subject.

    Parameters
    ----------
    subjects_dir : str
        Path to the derivatives directory.
    subject : str
        Subject identifier (e.g., 'sub-001').
    drug : str
        Drug condition ('LSD' or 'PLA').

    Returns
    -------
    src : mne.SourceSpaces
        The source space for the subject.
    """
    source_space_file = f"{subjects_dir}/anat/{drug}/{subject}/bem/{subject}-ico5-src.fif"
    subjects_dir = f"{subjects_dir}/anat/{drug}"
    
    
    if os.path.exists(source_space_file):
        print(f"Source space found for subject {subject}. Loading source space...")
        src = mne.read_source_spaces(source_space_file)
        print(f"Source space loaded for subject {subject}.")
        return src
    
    else:
        print(f"Creating source space for subject {subject}...")
        src = mne.setup_source_space(subject, spacing='ico5', subjects_dir=subjects_dir, n_jobs=-1)
        print(f"Source space setup complete for subject {subject}.")
        src_file = f"{subjects_dir}/{subject}/bem/{subject}-ico5-src.fif"
        mne.write_source_spaces(src_file, src, overwrite=True)
        
        print(f"Source space saved at {src_file}.")
        return src


def bem(subjects_dir, subject, drug):
    """
    Create or load the BEM solution for a subject.

    Uses a single-shell model with conductivity 0.3 S/m.

    Parameters
    ----------
    subjects_dir : str
        Path to the derivatives directory.
    subject : str
        Subject identifier (e.g., 'sub-001').
    drug : str
        Drug condition ('LSD' or 'PLA').

    Returns
    -------
    bem_sol : mne.bem.ConductorModel
        The BEM solution for the subject.
    """
    bem_file = f"{subjects_dir}/anat/{drug}/{subject}/bem/{subject}-bem-sol.fif"
    subjects_dir = f"{subjects_dir}/anat/{drug}"
    
    
    if os.path.exists(bem_file):
        print(f"BEM solution found for subject {subject}. Loading BEM solution...")
        bem_sol = mne.read_bem_solution(bem_file)
        print(f"BEM solution loaded for subject {subject}.")
        return bem_sol
    
    else:
        print(f"BEM solution not found for subject {subject}. Creating BEM solution...")
        model = mne.make_bem_model(subject=subject, ico=5, conductivity=(0.3,), subjects_dir=subjects_dir)
        bem_sol = mne.make_bem_solution(model)
        mne.write_bem_solution(bem_file, bem_sol, overwrite=True)
        print(f"BEM solution created and saved at {bem_file}.")
        
        return bem_sol


def forward_model(subjects_dir, subject, epochs, trans, src, bem_sol, drug):
    """
    Create or load the forward model for a subject.

    Parameters
    ----------
    subjects_dir : str
        Path to the derivatives directory.
    subject : str
        Subject identifier (e.g., 'sub-001').
    epochs : mne.Epochs
        The epochs object containing sensor information.
    trans : mne.transforms.Transform
        The head-to-MRI transformation.
    src : mne.SourceSpaces
        The source space.
    bem_sol : mne.bem.ConductorModel
        The BEM solution.
    drug : str
        Drug condition ('LSD' or 'PLA').

    Returns
    -------
    fwd : mne.Forward
        The forward solution.
    """
    fwd_file = f"{subjects_dir}/anat/{drug}/{subject}/bem/{subject}-fwd.fif"
    subjects_dir = f"{subjects_dir}/anat/{drug}"

    if os.path.exists(fwd_file):
        print(f"Forward model found for subject {subject}. Loading forward model...")
        fwd = mne.read_forward_solution(fwd_file)
        print(f"Forward model loaded for subject {subject}.")
        return fwd
    
    else:
        print(f"Forward model not found for subject {subject}. Creating forward model...")
        
        fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src, bem=bem_sol,
                                    meg=True, eeg=False)
        mne.write_forward_solution(fwd_file, fwd, overwrite=True)
        print(f"Forward model created and saved at {fwd_file}.")    
        return fwd


def parcellation(stc):
    """
    Parcellate a source estimate using the HCP-MMP1 atlas.

    Extracts mean time courses for each parcel in the Glasser atlas,
    excluding the unknown regions (indices 0 and 181).

    Parameters
    ----------
    stc : mne.SourceEstimate
        The source estimate to parcellate.

    Returns
    -------
    label_ts : ndarray
        Time courses for each parcel, shape (n_labels, n_times).
    """
    labels=mne.read_labels_from_annot('fsaverage', 'HCPMMP1', sort=False, subjects_dir='/Brain/private/v20subra/LSD_project/src_data/derivatives/anat/LSD')
    src = mne.read_source_spaces(f"{subjects_dir}/anat/LSD/fsaverage/bem/fsaverage-ico-5-src.fif")

    exclude_indices = [0, 181]
    valid_labels = [label for i, label in enumerate(labels) if i not in exclude_indices]
    
    label_ts = mne.extract_label_time_course(
        stc, labels=valid_labels, src=src, mode="mean", allow_empty=True, mri_resolution=False
    )
    return label_ts


def morph_subject_activity_to_fsaverage(stcs, fwd, subject_from, subjects_dir, task, drug):
    """
    Morph source estimates from subject space to fsaverage template.

    Parameters
    ----------
    stcs : list of mne.SourceEstimate
        Source estimates for each epoch.
    fwd : mne.Forward
        The forward solution (used for source space information).
    subject_from : str
        Subject identifier being morphed.
    subjects_dir : str
        Path to the derivatives directory.
    task : str
        Task name.
    drug : str
        Drug condition ('LSD' or 'PLA').

    Returns
    -------
    stc_morphed_all_epochs : list of mne.SourceEstimate
        Morphed source estimates for each epoch.
    """
    subjects_dir_anat = f"{subjects_dir}/anat/{drug}"
    
    src_to = mne.read_source_spaces(f"{subjects_dir_anat}/fsaverage/bem/fsaverage-ico-5-src.fif")    
    fsave_vertices = [s["vertno"] for s in src_to]
    stc_morphed_all_epochs = []
    morph = mne.compute_source_morph(fwd['src'], subject_to='fsaverage', src_to = src_to, spacing=fsave_vertices,
                                    
                                        subjects_dir= subjects_dir_anat)
    for _, stc in enumerate(stcs):
        stc_morphed = morph.apply(stc)
        stc_morphed_all_epochs.append(stc_morphed)
    
    return stc_morphed_all_epochs
    
    
    


def run_source_localization(subjects_dir, subject, task, drug):
    """
    Run the complete source localization pipeline for a single subject.

    This function performs:
    1. Load epochs and transformation matrix
    2. Resample epochs to 250 Hz (if not already done)
    3. Create/load source space, BEM solution, and forward model
    4. Compute inverse operator with dSPM method
    5. Morph source estimates to fsaverage
    6. Parcellate using HCP-MMP1 atlas
    7. Save parcellated source estimates

    Parameters
    ----------
    subjects_dir : str
        Path to the derivatives directory.
    subject : str
        Subject identifier (without 'sub-' prefix).
    task : str
        Task name (e.g., 'Music', 'Video', 'WithoutMusic').
    drug : str
        Drug condition ('LSD' or 'PLA').

    Returns
    -------
    None
        Parcellated source estimates are saved to disk.
    """
    subject = f"sub-{subject}"
    trans_file = f"{subjects_dir}/anat/{drug}/{subject}/bem/{subject}_trans_{task}.fif"
    trans = mne.read_trans(trans_file)
    epochs_file = f"{subjects_dir}/func/{task}/{drug}/{subject}/meg/{subject}_cleaned_epochs_meg.fif"

    epochs = mne.read_epochs(epochs_file, preload=True)
    epochs = epochs.pick_types(meg=True, eeg=False, ref_meg=False)
    
    if not os.path.exists(f"{subjects_dir}/func/{task}/{drug}/{subject}/meg/{subject}_cleaned_epochs_resampled_meg.fif"):
        epochs = epochs.resample(250)
        epochs.save(f"{subjects_dir}/func/{task}/{drug}/{subject}/meg/{subject}_cleaned_epochs_resampled_meg.fif", overwrite=True)
    
    else:
        epochs = mne.read_epochs(f"{subjects_dir}/func/{task}/{drug}/{subject}/meg/{subject}_cleaned_epochs_resampled_meg.fif", preload=True)
        
    # Create the source space
    src = create_source_space(subjects_dir, subject, drug)

    # Create or load the BEM solution
    bem_sol = bem(subjects_dir, subject, drug)

    # Create or load the forward model
    fwd_model = forward_model(subjects_dir, subject, epochs, trans, src, bem_sol, drug)

    # Compute the noise covariance matrix (identity matrix)
    noise_cov_data = np.eye(epochs.info['nchan']) 
    noise_cov = mne.Covariance(data=noise_cov_data, names=epochs.info['ch_names'], bads=[], projs=[], nfree=1)    
    
    # Create the inverse operator
    inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, fwd_model, noise_cov, loose=0.2, depth=0.8)
    print(f"Inverse operator created for subject {subject}.")

    # Apply the inverse solution using dSPM
    method = "dSPM"
    snr = 3.0
    lambda2 = 1.0 / snr**2
    
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator, lambda2,
                                                method=method)
    
    morphed_stc = morph_subject_activity_to_fsaverage(stcs, fwd_model, subject, subjects_dir, task, drug)    

    stc_data_parcellated = []
    for stc in morphed_stc:
        stc_data_parcellated.append(parcellation(stc))
        
    if not os.path.exists(f"{subjects_dir}/func/{task}/{drug}/{subject}/meg/source_estimates"):
        os.makedirs(f"{subjects_dir}/func/{task}/{drug}/{subject}/meg/source_estimates")
            
    np.savez_compressed(f"{subjects_dir}/func/{task}/{drug}/{subject}/meg/source_estimates/{subject}_.npz", stc_data_parcellated = stc_data_parcellated)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform source localization for multiple subjects in parallel using joblib.')
    parser.add_argument('-s', '--subjects', type=str, required=True, nargs='+', help='List of subject IDs to preprocess')
    parser.add_argument('-t', '--task', type=str, required=True, help='Task name (e.g., music, video)')
    parser.add_argument('-d', '--drug', type=str, required=True, help='Drug condition (e.g., LSD, PLA)')
    
    
    subjects_dir = '/Brain/private/v20subra/LSD_project/src_data/derivatives/'
    args = parser.parse_args()

    # Run subjects in parallel using joblib
    Parallel(n_jobs=1)(delayed(run_source_localization)(subjects_dir, subject, args.task, args.drug) for subject in args.subjects)
