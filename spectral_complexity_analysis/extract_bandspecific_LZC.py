import importlib
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import compute_features

importlib.reload(compute_features)


CONTEXTS = ["Music", "WithoutMusic"]
CONDITIONS = ["LSD", "PLA"]
SUBJECTS = [
    "001",
    "002",
    "003",
    "004",
    "005",
    "006",
    "009",
    "010",
    "012",
    "013",
    "014",
    "015",
    "016",
    "017",
    "018",
    "019",
    "020",
]

# Existing bandlimited source estimates are stored under these keys.
BANDS_FOR_LZC = ["Alpha", "Beta", "LowGamma", "MedGamma", "HighGamma"]


def _load_broadband_source_signal(base_path, condition, subject, context):
    source_dir = f"{base_path}/{condition}/sub-{subject}/meg/source_estimates"
    if context == "Music":
        source_path = f"{source_dir}/sub-{subject}_.npz"
    else:
        source_path = f"{source_dir}/sub-{subject}_stc.npz"
    return np.load(source_path)["stc_data_parcellated"]


def _load_bandlimited_source_signals(base_path, condition, subject):
    source_dir = f"{base_path}/{condition}/sub-{subject}/meg/source_estimates"
    bandlimited_path = f"{source_dir}/sub-{subject}_bandlimited.npz"
    bandlimited_data = np.load(bandlimited_path)
    return {band: bandlimited_data[band] for band in BANDS_FOR_LZC}


def _summarize_lzc(source_signal):
    regional = compute_features.lzc(source_signal=source_signal)
    global_value = np.mean([value["lzc"] for value in regional.values()])
    return global_value, regional


print(len(SUBJECTS))
for subj in tqdm(SUBJECTS):
    for condition in CONDITIONS:
        for context in CONTEXTS:
            base_path = f"/Brain/private/v20subra/LSD_project/src_data/derivatives/func/{context}"

            broadband_signal = _load_broadband_source_signal(
                base_path=base_path,
                condition=condition,
                subject=subj,
                context=context,
            )
            bandlimited_signals = _load_bandlimited_source_signals(
                base_path=base_path,
                condition=condition,
                subject=subj,
            )

            lzc_feature_dict = {}

            broadband_global, broadband_regional = _summarize_lzc(broadband_signal)
            lzc_feature_dict["Broadband_lz_global"] = broadband_global
            lzc_feature_dict["Broadband_lz_regional"] = broadband_regional

            for band_name, band_signal in bandlimited_signals.items():
                band_global, band_regional = _summarize_lzc(band_signal)
                lzc_feature_dict[f"{band_name}_lz_global"] = band_global
                lzc_feature_dict[f"{band_name}_lz_regional"] = band_regional

            output_path = (
                f"{base_path}/{condition}/sub-{subj}/meg/"
                f"sub-{subj}_neural_features_source_lzc.npz"
            )
            np.savez_compressed(output_path, **lzc_feature_dict)
