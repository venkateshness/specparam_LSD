"""
MEG Preprocessing Pipeline for LSD Study.

This module provides a preprocessing pipeline for MEG data from the LSD study,
including power line artifact removal, bandpass filtering, ICA-based artifact
rejection, and AutoReject-based epoch cleaning.

The pipeline follows these steps:
    1. Power line artifact removal (notch filter at 50Hz and harmonics)
    2. Bandpass filtering (1-125 Hz)
    3. Epoch creation with manual artifact annotations
    4. AutoReject for pre-ICA epoch cleaning
    5. ICA for ECG/EOG artifact removal
    6. AutoReject for post-ICA epoch cleaning

Usage:
    python preprocessing.py -s 001 002 003 -t Music -d LSD
"""

import os
import argparse
from re import sub

import tqdm
import mne
from autoreject import AutoReject
from mne.preprocessing import ICA
from mne.report import Report
from joblib import Parallel, delayed
import numpy as np


def preprocess_meg(subject_id, input_dir, task, drug, eog_components, DERIVATIVES_DIR, power_line_freq=50):
    """
    Preprocess MEG data for a single subject.

    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., '001', '002').
    input_dir : str
        Path to the BIDS-formatted input directory.
    task : str
        Task name (e.g., 'Music', 'Video', 'WithoutMusic').
    drug : str
        Drug condition ('LSD' or 'PLA').
    eog_components : dict
        Nested dictionary containing manually identified EOG/ECG components
        for each subject, task, and drug condition.
    DERIVATIVES_DIR : str
        Path to the derivatives output directory.
    power_line_freq : int, optional
        Power line frequency in Hz (default: 50).

    Returns
    -------
    None
        Preprocessed data is saved to the derivatives directory.
    """
    input_file = os.path.join(input_dir, f'sub-{subject_id}', f'ses-01', 'meg', f'sub-{subject_id}_ses-01_task-{task}_meg.fif')
    
    os.makedirs(os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}'), exist_ok=True)
    os.makedirs(os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', 'meg'), exist_ok=True)
    
    report = Report(title=f'MEG Report for Subject {subject_id}', verbose=False)
    
    # Step 1: Power line artifact removal using notch filter
    raw = mne.io.read_raw_fif(input_file, preload=True, verbose=False)
    report.add_raw(raw, title=f'Subject {subject_id} - Pre powerline artefact removal', psd=True)
    
    raw.notch_filter(freqs=[power_line_freq, 2*power_line_freq, 3*power_line_freq, 4*power_line_freq, 5*power_line_freq], verbose=False)
    report.add_raw(raw, title=f'Subject {subject_id} - Power Line Artifact Removal', psd=True)
    raw.save(os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', 'meg', f'sub-{subject_id}_raw_notch_meg.fif'), overwrite=True)

    # Step 2: Bandpass filtering (1-125 Hz)
    raw.filter(l_freq=1, h_freq=125, verbose=False)
    report.add_raw(raw, title=f'Subject {subject_id} - Bandpass Filtering', psd=True)
    raw.save(os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', 'meg', f'sub-{subject_id}_raw_filtered_meg.fif'), overwrite=True)

    # Step 3: Create epochs and apply AutoReject
    # ICA works best on data without bad epochs, so first apply AutoReject
    # Reference: https://autoreject.github.io/stable/auto_examples/plot_autoreject_workflow.html
    events = mne.make_fixed_length_events(raw, duration=2, overlap=0)
    epochs = mne.Epochs(raw, events, event_id=1, tmin=0, tmax=1.999, baseline=None, preload=True, picks='meg')
    
    # Load manually annotated good epochs through 
    manually_annotated_good_epochs = np.load(f'/Brain/private/v20subra/LSD_project/src_data/derivatives/func/{task}/{drug}/sub-{subject_id}/meg/sub-{subject_id}_good_epochs_upon_visual_inspection_of_raw_filtered_epochs.npz')['arr_0']
    epochs = epochs[manually_annotated_good_epochs]
    epochs.save(os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', 'meg', f'sub-{subject_id}_epochs_meg.fif'), overwrite=True)
    
    ar = AutoReject(picks="mag",n_jobs=-1, random_state=99, n_interpolate=[1, 4, 8, 16, 32])
    ar.fit(epochs)
    
    reject_log = ar.get_reject_log(epochs, picks='mag')
    report.add_figure(reject_log.plot('horizontal'), title=f'Subject {subject_id} - Autoreject Log')
    report.add_epochs(epochs, title=f'Subject {subject_id} - Autoreject Applied for Epochs')
    epochs.save(os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', 'meg', f'sub-{subject_id}_AR_PreICA_ft_epochs_meg.fif'), overwrite=True)
    
    output_file_reject_log = os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', 'meg', f'sub-{subject_id}_reject_log_AR_pre_meg.fif')
    reject_log.save(output_file_reject_log, overwrite=True)
    
    # Step 4: ICA for artifact removal
    print(f"Subject {subject_id},{reject_log.bad_epochs} ")
    print(f"Subject {subject_id},{len(reject_log.bad_epochs)} ")
    
    ica = ICA(n_components=20, random_state=97, method="picard")
    ica.fit(epochs[~reject_log.bad_epochs], picks="mag")
    ica.save(os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', 'meg', f'sub-{subject_id}_ica_meg.fif'), overwrite=True)
    
    # Step 4.1: ECG artifact detection
    if raw['EEG059'][0].std()==0:  # Handle flat ECG channel
        ecg_indices = []
        report.add_ica(ica, inst=None, title=f'Subject {subject_id}, IC; ECG match with ECG channels: {ecg_indices}, flat_line',   n_jobs=-1)
    else:
        ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, ch_name="EEG059")
        ecg_indices, ecg_scores = ica.find_bads_ecg(ecg_epochs, ch_name="EEG059")
    
        report.add_ica(ica, inst=None, title=f'Subject {subject_id}, IC; ECG match with ECG channels: {ecg_indices}',  ecg_scores=ecg_scores,  n_jobs=-1)

    # Step 4.2: EOG artifact detection
    eog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name=["EEG057", "EEG058"])
    eog_indices, eog_scores = ica.find_bads_eog(eog_epochs, ch_name=["EEG057", "EEG058"])
    report.add_ica(ica, inst=None, title=f'Subject {subject_id}, IC; EOG match with EOG channels: {eog_indices}',  eog_scores=eog_scores,  n_jobs=-1)    

    # Step 4.3: Apply ICA with manually identified components
    if subject_id == '009' and task == 'WithoutMusic' and drug == 'PLA':
        ecg_components = [0]  # Override for this specific subject
    else:
        ecg_components = ecg_indices
    eog_components = eog_components[task][drug][f"{subject_id}"]
    to_exclude = list(set(ecg_components + eog_components))
    
    ica.exclude = to_exclude
    
    epochs_ar_clean_ICA = ica.apply(epochs, exclude=to_exclude)
    
    report.add_ica(ica, inst=None, title=f'Subject {subject_id} - ICA Applied for Epochs',  n_jobs=-1)
    report.add_epochs(epochs_ar_clean_ICA, title=f'Subject {subject_id} - ICA Applied for Epochs')
    
    # Step 5: Post-ICA AutoReject
    ar = AutoReject(picks="mag",n_jobs=-1, random_state=99, n_interpolate=[1, 4, 8, 16, 32])
    ar.fit(epochs_ar_clean_ICA)
    epochs_ar_clean_ICA_ar, reject_log_post_ICA = ar.transform(epochs_ar_clean_ICA, return_log=True)
    report.add_figure(reject_log_post_ICA.plot('horizontal'), title=f'Subject {subject_id} - Autoreject Log post_ICA')

    # Save cleaned epochs
    output_file = os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', 'meg', f'sub-{subject_id}_cleaned_epochs_meg.fif')
    epochs_ar_clean_ICA_ar.save(output_file, overwrite=True)

    report.add_epochs(epochs_ar_clean_ICA_ar, title=f'Subject {subject_id} - Cleaned Epochs', psd=True)

    report.save(os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', f'sub-{subject_id}_report.html'), overwrite=True)
    report.save(os.path.join(f"/Brain/private/v20subra/LSD_project/Preprocessing_Reports/{task}/{drug}", f'sub-{subject_id}_report.html'), overwrite=True)

    output_file_reject_log = os.path.join(DERIVATIVES_DIR, f'sub-{subject_id}', 'meg', f'sub-{subject_id}_reject_log_AR_post_meg.fif')
    reject_log_post_ICA.save(output_file_reject_log, overwrite=True)


def main():
    """
    Main entry point for the MEG preprocessing pipeline.

    Parses command-line arguments and runs preprocessing in parallel for all
    specified subjects.
    """
    parser = argparse.ArgumentParser(description='MEG Preprocessing Script with stages')
    parser.add_argument('-s', '--subjects', type=str, required=True, nargs='+')
    parser.add_argument('-t', '--task', type=str, required=True)
    parser.add_argument('-d', '--drug', type=str, required=True)
    args = parser.parse_args()

    # Manually identified EOG/ECG components per subject, task, and drug condition.
    # These components were visually inspected and selected for removal during ICA.
    eog_components = {

    "Music": {

    "LSD": {
    "010": [8, 11],
    "015": [0, 1, 2, 4, 5, 12],
    "016": [7, 8, 3, 5, 13], 
    "013": [0, 1, 7],
    "006": [19],
    "005": [2],
    "011": [14, 19],
    "003": [0, 1, 2],
    "018": [0, 11],
    "017": [0, 18, 19, 4],
    "009": [0, 1, 3, 4, 5, 11],
    "001": [0, 2, 3],
    "002": [0, 1],
    "004": [4, 12],
    "012": [1, 2],
    "014": [0, 4, 5],
    "019": [1, 4],
    "020": [6, 10]
    },

    "PLA": {
    "011": [8],
    "010": [19],
    "005": [9],
    "017": [15],
    "018": [19],
    "003": [17],
    "006": [],
    "009": [8, 16],
    "013": [8, 14, 11],
    "016": [14],
    "015": [0, 1, 3, 6],
    "001": [5, 11],
    "002": [11],
    "004": [15],
    "012": [13],
    "014": [7, 17],
    "019": [3, 14, 18],
    "020": [3, 10, 18]
    }

    },

    "Video": {

    "LSD": {
    "011": [11, 17],
    "016": [0, 1, 2, 3, 4, 5, 6, 7, 10],
    "006": [0, 3, 6, 15],
    "015": [1, 2, 3, 4, 11],
    "003": [0, 2],
    "010": [2, 5],
    "013": [0, 1],
    "005": [8, 13],
    "018": [0, 1, 12],
    "017": [0, 1, 7, 8, 9, 19],
    "009": [0, 1, 10]
    },

    "PLA": {
    "011": [4],
    "018": [0, 2],
    "015": [0, 1, 2, 5],
    "010": [0, 13],
    "017": [0, 13, 19],
    "016": [0, 9, 19],
    "005": [15, 16],
    "006": [0, 18],
    "003": [1, 3],
    "009": [0, 1],
    "013": [0, 1]
    }

    },

    "WithoutMusic": {
    "LSD": {
        "001": [0, 2],
        "002": [1, 2],
        "003": [1, 11],
        "004": [1, 16],
        "005": [16],
        "006": [12, 17],
        "009": [0],
        "010": [8, 10],
        "011": [7, 12],
        "012": [2, 5],
        "013": [0, 1],
        "014": [0, 2],
        "015": [1, 13, 5, 4],
        "016": [7, 8],
        "017": [3, 5, 6, 7],
        "018": [0, 10],
        "019": [0, 2],
        "020": [7, 8]
        },
    "PLA": {
        "001": [2, 6],
        "002": [],
        "003": [16],
        "004": [14, 5],
        "005": [],
        "006": [],
        "009": [0, 1, 11, 19],
        "010": [9, 19],
        "011": [9],
        "012": [15, 10, 14],
        "013": [6, 8, 17],
        "014": [11, 14],
        "015": [1, 12],
        "016": [0, 19],
        "017": [2],
        "018": [0, 14],
        "019": [4, 3, 17, 16],
        "020": [16, 18]
        }
    }
    }
    
    BIDS_DIR = f'/Brain/private/v20subra/LSD_project/src_data/fif_data_BIDS/{args.task}/{args.drug}/'
    DERIVATIVES_DIR = f'/Brain/private/v20subra/LSD_project/src_data/derivatives/func/{args.task}/{args.drug}/'
    
    task = args.task
    drug = args.drug

    Parallel(n_jobs=9)(delayed(preprocess_meg)(subject_id, BIDS_DIR, task, drug, eog_components, DERIVATIVES_DIR) for subject_id in args.subjects)


if __name__ == '__main__':
    main()
