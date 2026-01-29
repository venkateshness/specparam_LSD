"""
Spectral and Temporal Feature Computation.

This module provides functions to compute spectral and temporal features
from MEG source-space data, including:
- FOOOF (Fitting Oscillations & One-Over-F) analysis for aperiodic/periodic decomposition
- Band-specific peak frequencies and power extraction
- Higuchi Fractal Dimension (HFD) for signal complexity
- Lempel-Ziv Complexity (LZC) for signal regularity

These features are computed on parcellated source-space data (360 Glasser regions).
"""

from fooof import FOOOF
from tqdm import tqdm
import numpy as np
import antropy
from lempel_ziv_complexity import lempel_ziv_complexity
from scipy.interpolate import interp1d

n_regions = 360  # Number of regions in the Glasser atlas


def fooof(psd, freqs, fmin, fmax, verbose=False, isRegional=True):
    """
    Fit FOOOF model to power spectral density data.

    Decomposes the PSD into aperiodic (1/f) and periodic (oscillatory) components.
    Notch filter artifacts at 50 Hz and 100 Hz are interpolated before fitting.

    Parameters
    ----------
    psd : ndarray
        Power spectral density, shape (n_regions, n_freqs) if isRegional=True,
        or (n_regions, n_freqs) averaged across regions if isRegional=False.
    freqs : ndarray
        Frequency values corresponding to the PSD, shape (n_freqs,).
    fmin : float
        Minimum frequency for FOOOF fitting.
    fmax : float
        Maximum frequency for FOOOF fitting.
    verbose : bool, optional
        Whether to print verbose output (default: False).
    isRegional : bool, optional
        If True, fit FOOOF separately for each region. If False, average across
        regions first (default: True).

    Returns
    -------
    results : dict or list
        If isRegional=True, dict with region indices as keys containing:
            - exponent: Aperiodic exponent (slope)
            - offset: Aperiodic offset
            - peaks: Peak parameters (center freq, power, bandwidth)
            - r2: R-squared of fit
            - error: Fit error
            - flat_spectrum: Flattened (periodic) spectrum
            - total_power: Total power in frequency range
        If isRegional=False, list with single dict also containing 'ap_fit'.
    """
    def interpolate_notch_filter_freqs(psd, freqs):
        notch_freqs = [50, 100]
        notch_width = 3

        frequencies_clean = freqs.copy()
        psd_clean = psd.copy()

        for notch_freq in notch_freqs:
            notch_indices = np.where((frequencies_clean > notch_freq - notch_width) & (frequencies_clean < notch_freq + notch_width))[0]
            frequencies_clean = np.delete(frequencies_clean, notch_indices)
            psd_clean = np.delete(psd_clean, notch_indices)

        f = interp1d(frequencies_clean, psd_clean, kind='linear', fill_value="extrapolate")
        frequencies_interp = np.linspace(freqs[0], freqs[-1], len(freqs))
        psd_interp = f(frequencies_interp)
        return frequencies_interp, psd_interp

    if isRegional==False:
        results = []
        fm = FOOOF(peak_width_limits=[1, 10], max_n_peaks=3, min_peak_height=0.15, aperiodic_mode='fixed')
        assert np.shape(psd) == (n_regions, freqs.shape[0],), "psd should be of shape (freqs,)"
        psd_averaged = np.mean(psd, axis=0)
        _, psd_averaged = interpolate_notch_filter_freqs(psd_averaged, freqs)
        fm.fit(freqs, psd_averaged, freq_range=[fmin, fmax])
        peak_params = fm.get_params('peak_params')  
        flat_spectrum = fm.power_spectrum - fm._ap_fit
        ap_fit = fm._ap_fit

        results.append({
            'exponent': fm.get_params('aperiodic_params', 'exponent'), 
            'offset': fm.get_params('aperiodic_params', 'offset'),
            'peaks': peak_params,
            "r2": fm.get_params('r_squared'),
            "error": fm.get_params("error"),
            "flat_spectrum": flat_spectrum,
            "total_power": np.trapz(psd_averaged, freqs),
            "ap_fit": ap_fit
        })
        
    if isRegional:
        assert np.shape(psd) == (n_regions, freqs.shape[0]), "psd should be of shape (regions, freqs)"

        results = {}
       
        for region in range(n_regions):
                fm = FOOOF(peak_width_limits=[1, 10], max_n_peaks=5, min_peak_height=0.15, aperiodic_mode='fixed') 
                _, psd[region] = interpolate_notch_filter_freqs(psd[region], freqs)
                fm.fit(freqs, psd[region], freq_range=[fmin, fmax])
                peak_params = fm.get_params('peak_params')  
                flat_spectrum = fm.power_spectrum - fm._ap_fit

                results[region] = {
                    'exponent': fm.get_params('aperiodic_params', 'exponent'), 
                    'offset': fm.get_params('aperiodic_params', 'offset'),
                    'peaks': peak_params,
                    "r2": fm.get_params('r_squared'),
                    "error": fm.get_params("error"),
                    "flat_spectrum": flat_spectrum,
                    "total_power": np.trapz(psd[region], freqs),
                }
        
        
    return results


def get_psd_peak_freqs(param_bundle, freqs, isRegional=False):
    """
    Extract peak frequencies and band-specific power from FOOOF results.

    Computes peak frequency, peak power, and total power for standard
    frequency bands (delta, theta, alpha, beta) from the flattened spectrum.

    Parameters
    ----------
    param_bundle : dict or list
        FOOOF results from the fooof() function.
    freqs : ndarray
        Frequency values, shape (n_freqs,).
    isRegional : bool, optional
        If True, compute for each region separately (default: False).

    Returns
    -------
    feature_dict : dict
        If isRegional=False, dict with keys like 'alpha_peak', 'alpha_peak_power',
        'alpha_power' for each frequency band.
        If isRegional=True, nested dict with region indices as outer keys.
    """
    freqs_range = {"delta": [1, 4], "theta": [4, 8], "alpha": [8, 13], "beta": [15, 30]}

    
    if isRegional == False:
        feature_dict = {}
        for freq, (low_f, high_f)  in freqs_range.items():
            idx_band = np.where((freqs >= low_f) & (freqs <= high_f))[0]

            band_freqs = freqs[idx_band]
            psd = param_bundle[0]['flat_spectrum']
            band_psd = psd[idx_band]
            peak_idx = np.argmax(band_psd)
            feature_dict[f"{freq}_peak"] = band_freqs[peak_idx]
            feature_dict[f"{freq}_peak_power"] = np.max(band_psd)
            feature_dict[f"{freq}_power"] = np.trapz(psd[idx_band], freqs[idx_band])
    
    if isRegional:
        feature_dict = {region: {} for region in range(n_regions)}
        for region in range(n_regions):
            for freq, (low_f, high_f)  in freqs_range.items():
                
                idx_band = np.where((freqs >= low_f) & (freqs <= high_f))[0]
                
                band_freqs = freqs[idx_band]
                psd = param_bundle[region]['flat_spectrum']
                band_psd = psd[idx_band]
                peak_idx = np.argmax(band_psd)
                feature_dict[region][f"{freq}_peak"] = band_freqs[peak_idx]
                feature_dict[region][f"{freq}_peak_power"] = np.max(band_psd)
                feature_dict[region][f"{freq}_power"] = np.trapz(psd[idx_band], freqs[idx_band])

    return feature_dict


def higuchi_fd_kmax(source_signal, kmax, hyperparameter_search=False):
    """
    Compute Higuchi Fractal Dimension for source-space signals.

    Calculates the Higuchi Fractal Dimension (HFD) for each brain region,
    averaged across epochs.

    Parameters
    ----------
    source_signal : ndarray
        Source-space time series, shape (n_epochs, n_regions, n_samples).
    kmax : int
        Maximum lag parameter for HFD computation.
    hyperparameter_search : bool, optional
        Reserved for hyperparameter optimization (default: False).

    Returns
    -------
    higuchi_fd_results : dict
        Dict with region indices as keys, each containing 'hfd' (mean HFD).
    """
    higuchi_fd_results = {region: {} for region in range(360)} 

    n_epochs, n_regions, _ = source_signal.shape
    
    for region in range(n_regions):
        fd_epochs = np.zeros(n_epochs)
        for epoch in range(n_epochs):   
            signal = source_signal[epoch, region, :]
            
            fd_epochs[epoch] = antropy.higuchi_fd(signal, kmax=kmax)
        higuchi_fd_results[region]['hfd'] = np.mean(fd_epochs)

    return higuchi_fd_results


def lzc(source_signal):
    """
    Compute normalized Lempel-Ziv Complexity for source-space signals.

    Calculates LZC for each brain region by concatenating epochs,
    binarizing around the mean, and normalizing by signal length.

    Parameters
    ----------
    source_signal : ndarray
        Source-space time series, shape (n_epochs, n_regions, n_samples).

    Returns
    -------
    complexity : dict
        Dict with region indices as keys, each containing 'lzc' (normalized LZC).
    """
    n_epochs, n_regions, n_samples = source_signal.shape

    complexity = {region: {} for region in range(n_regions)}
    for region in range(n_regions):
        stacked = np.hstack(source_signal[:, region, :])
        stacked_binary = np.where(stacked > np.mean(stacked), 1, 0)
        binary_str = ''.join(map(str, stacked_binary))
        n = len(binary_str)
        lzc_raw = lempel_ziv_complexity(binary_str)
        lzc_norm = (np.log(n) / n) * lzc_raw 
        complexity[region]['lzc'] = lzc_norm
        
    return complexity