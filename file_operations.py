"""
File operations for spectral analysis
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os

def save_results_to_excel(output_filename, results_data):
    """
    Save all analysis results to Excel file with multiple sheets
    """
    try:
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            # Main results
            results_df = pd.DataFrame({
                'Actual': results_data['y_test'],
                'Predicted': results_data['y_pred'],
                'Absolute Error': np.abs(results_data['y_test'] - results_data['y_pred'])
            })
            results_df.to_excel(writer, sheet_name=f'Predictions_{results_data["model_type"]}', index=False)
            
            # Correlogram data
            correlogram_df = pd.DataFrame({
                'Wavelength': results_data['filtered_wavelengths'],
                'Correlation': results_data['correlations']
            })
            correlogram_df.to_excel(writer, sheet_name=f'Correlogram_{results_data["model_type"]}', index=False)
            
            # Performance metrics
            performance_data = {
                'Metric': ['Test R²', 'Test RMSE', 'Train R²', 'Train RMSE', 'Test Size', 
                          'Soil Property', 'Preprocessing', 'Number of Cores'],
                'Value': [results_data['test_r2'], results_data['test_rmse'], 
                         results_data['train_r2'], results_data['train_rmse'],
                         results_data['test_size'], results_data['selected_property'], 
                         results_data['preprocessing'], results_data['n_cores']]
            }
            
            # Add cross-validation results if available
            if results_data.get('cv_results'):
                cv_results = results_data['cv_results']
                performance_data['Metric'].extend([
                    'CV Strategy', 'CV R² Mean', 'CV R² Std', 'CV RMSE Mean', 'CV RMSE Std'
                ])
                performance_data['Value'].extend([
                    cv_results['strategy'], cv_results['r2_mean'], cv_results['r2_std'],
                    cv_results['rmse_mean'], cv_results['rmse_std']
                ])
                
                # Add K-fold specific information
                if cv_results['strategy'] == 'K-Fold' and 'parameters' in cv_results:
                    performance_data['Metric'].append('Number of Folds')
                    performance_data['Value'].append(cv_results['parameters']['k_folds'])
            
            stats_df = pd.DataFrame(performance_data)
            stats_df.to_excel(writer, sheet_name=f'Performance_{results_data["model_type"]}', index=False)
            
            # Model parameters
            params_df = pd.DataFrame({
                'Parameter': list(results_data['params'].keys()),
                'Value': [str(v) for v in results_data['params'].values()]
            })
            params_df.to_excel(writer, sheet_name=f'Parameters_{results_data["model_type"]}', index=False)
            
            # Component optimization results if available
            if results_data.get('component_optimization_results'):
                comp_opt = results_data['component_optimization_results']
                comp_opt_df = pd.DataFrame({
                    'Components': comp_opt['Components'],
                    'RMSE': comp_opt['RMSE'],
                    'R2_Score': comp_opt['R2_Score']
                })
                comp_opt_df.to_excel(writer, sheet_name=f'ComponentOpt_{results_data["model_type"]}', index=False)
            
            # Cross-validation detailed results if available
            if results_data.get('cv_results') and results_data['cv_results']['cv_rmse_scores']:
                cv_results = results_data['cv_results']
                cv_detailed_df = pd.DataFrame({
                    'Fold': range(1, len(cv_results['cv_rmse_scores']) + 1),
                    'RMSE': cv_results['cv_rmse_scores'],
                    'R2': cv_results['cv_r2_scores']
                })
                cv_detailed_df.to_excel(writer, sheet_name=f'CrossValidation_{results_data["model_type"]}', index=False)
            
            # Data statistics if requested
            if results_data.get('export_stats') and results_data.get('data_stats'):
                data_stats = results_data['data_stats']
                
                # Input statistics
                input_stats_df = pd.DataFrame.from_dict(
                    data_stats['Input Data Statistics'], orient='index', columns=['Value'])
                input_stats_df.to_excel(writer, sheet_name=f'Input Statistics_{results_data["model_type"]}')
                
                # Spectral statistics
                for stat_type, stat_data in data_stats['Spectral Statistics'].items():
                    stat_df = pd.DataFrame.from_dict(stat_data, orient='index', columns=['Value'])
                    sheet_name = f'Spectral {stat_type}_{results_data["model_type"]}'
                    stat_df.to_excel(writer, sheet_name=sheet_name)
        
        return True
        
    except Exception as e:
        raise Exception(f"Failed to save results to Excel: {str(e)}")

def generate_default_filename(input_filename, selected_property, model_type):
    """
    Generate default filename for results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{input_filename}_{selected_property}_{model_type}_results_{timestamp}.xlsx"

def generate_model_filename(input_filename, selected_property, model_type):
    """
    Generate default filename for model saving
    """
    return f"{input_filename}_{selected_property}_{model_type}_model.joblib"