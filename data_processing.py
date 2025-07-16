"""
Data processing utilities for spectral analysis
"""
import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn.preprocessing import StandardScaler

def safe_interpolate_spectra(X, original_wavelengths, new_wavelengths):
    """
    Safely interpolate spectra with comprehensive error handling
    """
    try:
        # Validate inputs
        if len(original_wavelengths) != X.shape[1]:
            raise ValueError(f"Wavelength count ({len(original_wavelengths)}) doesn't match spectral data columns ({X.shape[1]})")
        
        # Check for valid wavelength ranges
        orig_min, orig_max = min(original_wavelengths), max(original_wavelengths)
        new_min, new_max = min(new_wavelengths), max(new_wavelengths)
        
        if new_min < orig_min or new_max > orig_max:
            raise ValueError(f"Interpolation range ({new_min:.1f}-{new_max:.1f} nm) exceeds original data range ({orig_min:.1f}-{orig_max:.1f} nm)")
        
        # Check for sufficient data points
        if len(original_wavelengths) < 3:
            raise ValueError("Need at least 3 wavelength points for interpolation")
        
        # Check for duplicate wavelengths
        if len(set(original_wavelengths)) != len(original_wavelengths):
            raise ValueError("Duplicate wavelengths found in original data")
        
        # Sort wavelengths if not already sorted
        sort_indices = np.argsort(original_wavelengths)
        sorted_wavelengths = np.array(original_wavelengths)[sort_indices]
        
        interpolated_X = np.zeros((X.shape[0], len(new_wavelengths)))
        
        for i in range(X.shape[0]):
            try:
                # Sort spectral data according to wavelength order
                sorted_spectrum = X[i, sort_indices]
                
                # Check for NaN or infinite values
                if np.any(~np.isfinite(sorted_spectrum)):
                    raise ValueError(f"Invalid values (NaN/Inf) found in spectrum {i}")
                
                # Perform interpolation
                f = interpolate.interp1d(sorted_wavelengths, sorted_spectrum, 
                                       kind='linear', bounds_error=True, fill_value='extrapolate')
                interpolated_X[i, :] = f(new_wavelengths)
                
            except Exception as e:
                raise ValueError(f"Interpolation failed for spectrum {i}: {str(e)}")
        
        return interpolated_X
        
    except Exception as e:
        error_msg = f"Spectral interpolation failed: {str(e)}\n\n"
        error_msg += "Possible solutions:\n"
        error_msg += "1. Reduce the wavelength range (Min/Max Wave)\n"
        error_msg += "2. Increase spacing value\n"
        error_msg += "3. Check for missing or invalid spectral data\n"
        error_msg += "4. Disable resampling if not needed"
        raise ValueError(error_msg)

def preprocess_spectra(spectra, method):
    """
    Apply preprocessing to spectral data
    """
    try:
        if method == "Continuum Removal":
            # Apply continuum removal
            processed = np.zeros_like(spectra)
            for i in range(spectra.shape[0]):
                spectrum = spectra[i, :]
                # Find convex hull (simple approach)
                hull = np.maximum.accumulate(spectrum)
                hull = np.maximum.accumulate(hull[::-1])[::-1]
                processed[i, :] = spectrum / (hull + 1e-10)  # Avoid division by zero
            return processed
            
        elif method == "First Derivative":
            return np.gradient(spectra, axis=1)
            
        elif method == "Second Derivative":
            return np.gradient(np.gradient(spectra, axis=1), axis=1)
            
        elif method == "Absorbance":
            # Ensure no zero or negative values
            spectra_safe = np.maximum(spectra, 1e-10)
            return -np.log10(spectra_safe)
            
        return spectra
        
    except Exception as e:
        raise Exception(f"Preprocessing failed: {str(e)}")

def calculate_statistics(data):
    """
    Calculate comprehensive statistics for data
    """
    try:
        stats = {
            'Count': len(data),
            'Mean': np.mean(data),
            'Std Dev': np.std(data),
            'Min': np.min(data),
            'Q1': np.percentile(data, 25),
            'Median': np.median(data),
            'Q3': np.percentile(data, 75),
            'Max': np.max(data),
            'Skewness': float(pd.Series(data).skew()),
            'Kurtosis': float(pd.Series(data).kurtosis())
        }
        return stats
        
    except Exception as e:
        raise Exception(f"Statistics calculation failed: {str(e)}")

def filter_wavelengths(wavelengths, min_wave, max_wave, resampling, spacing):
    """
    Filter and optionally resample wavelengths
    """
    try:
        if not wavelengths:
            return None, None
        
        wavelengths_array = np.array([float(w) for w in wavelengths])
        min_wave = float(min_wave or min(wavelengths_array))
        max_wave = float(max_wave or max(wavelengths_array))
        
        mask = (wavelengths_array >= min_wave) & (wavelengths_array <= max_wave)
        filtered_wavelengths = wavelengths_array[mask]
        
        if resampling == "Yes":
            spacing = float(spacing or 10)
            new_wavelengths = np.arange(min_wave, max_wave + spacing, spacing)
            return filtered_wavelengths, new_wavelengths
        
        return filtered_wavelengths, None
        
    except Exception as e:
        raise Exception(f"Wavelength filtering failed: {str(e)}")