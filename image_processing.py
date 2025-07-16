"""
Image processing utilities for spectral analysis
"""
import numpy as np
import rasterio
from data_processing import preprocess_spectra, safe_interpolate_spectra

def process_image_for_prediction(image_data, wavelengths, preprocessing_method, 
                                scaler_X, filtered_wavelengths, new_wavelengths=None):
    """
    Process image data for prediction with the same preprocessing as training data
    """
    try:
        # Reshape image for processing
        original_shape = image_data.shape
        reshaped_data = image_data.reshape(original_shape[0], -1).T
        
        # Apply wavelength filtering
        wavelength_indices = [i for i, w in enumerate(wavelengths) 
                            if float(w) >= min(filtered_wavelengths) 
                            and float(w) <= max(filtered_wavelengths)]
        reshaped_data = reshaped_data[:, wavelength_indices]
        
        # Apply resampling if needed
        if new_wavelengths is not None:
            reshaped_data = safe_interpolate_spectra(reshaped_data, filtered_wavelengths, new_wavelengths)
        
        # Apply the same preprocessing as training data
        reshaped_data = preprocess_spectra(reshaped_data, preprocessing_method)
        
        # Scale the data using the same scaler as training
        reshaped_data_scaled = scaler_X.transform(reshaped_data)
        
        return reshaped_data_scaled, original_shape
        
    except Exception as e:
        raise Exception(f"Image processing failed: {str(e)}")

def save_prediction_image(predictions, original_shape, image_meta, file_path):
    """
    Save prediction results as a GeoTIFF image
    """
    try:
        # Reshape back to image dimensions
        prediction_image = predictions.reshape(original_shape[1], original_shape[2])
        
        # Mask values below 0
        prediction_image = np.where(prediction_image > 0, prediction_image, np.nan)
        
        # Update metadata for single band output
        image_meta.update(
            count=1,
            dtype=prediction_image.dtype
        )
        
        with rasterio.open(file_path, 'w', **image_meta) as dst:
            dst.write(prediction_image, 1)
        
        return prediction_image
        
    except Exception as e:
        raise Exception(f"Failed to save prediction image: {str(e)}")