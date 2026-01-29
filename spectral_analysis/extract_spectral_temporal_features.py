
import numpy as np
from tqdm import tqdm
import importlib
import sys
import os
import helper_spectral_temporal_features
importlib.reload(helper_spectral_temporal_features)


contexts = ["Music", "WithoutMusic"]
conditions = ["LSD", "PLA"]
subjects = ["001", "002", "003", "004", "005", "006", "009", "010", "012", "013", "014", "015", "016", "017", "018", "019", "020" ]
print(len(subjects))
for subj in tqdm(subjects):
    for condition in conditions:
        for context in contexts:
            base_path = f'/Brain/private/v20subra/LSD_project/src_data/derivatives/func/{context}'
            fmax = 45
            psd_path = f'{base_path}/{condition}/sub-{subj}/meg/source_estimates/sub-{subj}_source_psd_{fmax}hz.npz'
            psd_data = np.load(psd_path)['psd_parcellated']
            psd_data_time = np.mean(psd_data, axis=0)
            # print(psd_data_time.shape)
            
            freqs = np.linspace(1, fmax, psd_data_time.shape[1])
            subfooof_global = helper_spectral_temporal_features.fooof(psd = psd_data_time, freqs = freqs, fmin = 1, fmax = fmax, verbose=False, isRegional=False)
            subPeaksPowers_global = helper_spectral_temporal_features.get_psd_peak_freqs(param_bundle=subfooof_global, freqs=freqs, isRegional=False)


            subfooof_regional = helper_spectral_temporal_features.fooof(psd = psd_data_time, freqs = freqs, fmin = 1, fmax = fmax, verbose=False, isRegional=True)
            subPeaksPowers_regional = helper_spectral_temporal_features.get_psd_peak_freqs(param_bundle=subfooof_regional, freqs=freqs, isRegional=True)

            if context == 'Music':
                source_signal = np.load(f'{base_path}/{condition}/sub-{subj}/meg/source_estimates/sub-{subj}_.npz')['stc_data_parcellated']
            if context == 'WithoutMusic':
                source_signal = np.load(f'{base_path}/{condition}/sub-{subj}/meg/source_estimates/sub-{subj}_stc.npz')['stc_data_parcellated']
            subHFD = helper_spectral_temporal_features.higuchi_fd_kmax(source_signal=source_signal, kmax=18, hyperparameter_search=False) # based on the hyperparameter search, we identified kmax=18 as the knee. TO INCLUDE in the repo
            
            subLZ = helper_spectral_temporal_features.lzc(source_signal=source_signal)
            

            feature_dict = {
                "fooof_global": subfooof_global,
                "peaks_powers_global": subPeaksPowers_global,
                
                "fooof_regional": subfooof_regional,
                "peaks_powers_regional": subPeaksPowers_regional,

                "hfd_global":  np.mean([value['hfd'] for value in subHFD.values()]),
                "hfd_regional": subHFD,

                "lz_global": np.mean([value['lzc'] for value in subLZ.values()]),
                "lz_regional": subLZ,
            }

            np.savez_compressed(f'{base_path}/{condition}/sub-{subj}/meg/sub-{subj}_neural_features_source_{fmax}.npz', **feature_dict)
