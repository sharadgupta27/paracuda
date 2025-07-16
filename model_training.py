"""
Model training utilities for spectral analysis
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from joblib import Parallel, delayed
import xgboost as xgb

def create_model(model_type, params, n_cores=1):
    """
    Create a model instance based on type and parameters
    """
    try:
        if model_type == "PLS-R":
            n_components = min(int(params['n_components']), 50)  # Reasonable limit
            return PLSRegression(n_components=n_components)
        
        elif model_type == "PCA":
            n_components = min(int(params['n_components']), 50)
            return PCA(n_components=n_components, 
                      svd_solver=params['svd_solver'],
                      whiten=params['whiten'])
        
        elif model_type == "Ridge":
            return Ridge(
                alpha=params['alpha'],
                solver=params['solver'],
                max_iter=params['max_iter']
            )
        
        elif model_type == "Lasso":
            return Lasso(
                alpha=params['alpha'],
                max_iter=params['max_iter'],
                selection=params['selection'],
                tol=params['tol']
            )
        
        elif model_type == "Linear":
            return LinearRegression(fit_intercept=params['fit_intercept'])
        
        elif model_type == "Random Forest":
            max_depth = None if str(params['max_depth']).lower() == "none" else int(params['max_depth'])
            
            # Parse max_features safely
            max_feat_raw = str(params['max_features']).strip().lower()
            if max_feat_raw == 'none':
                max_features = None
            elif max_feat_raw in ['sqrt', 'log2']:
                max_features = max_feat_raw
            else:
                try:
                    max_features = int(max_feat_raw)
                except ValueError:
                    try:
                        max_features = float(max_feat_raw)
                    except ValueError:
                        max_features = 'sqrt'
            
            return RandomForestRegressor(
                n_estimators=params['n_estimators'],
                max_depth=max_depth,
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=max_features,
                bootstrap=params['bootstrap'],
                n_jobs=n_cores,
                random_state=42
            )
        
        elif model_type == "XGBoost":
            return xgb.XGBRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                reg_alpha=params['reg_alpha'],
                reg_lambda=params['reg_lambda'],
                n_jobs=n_cores,
                random_state=42
            )
        
        else:  # SVM
            gamma_val = params['gamma'] if params['gamma'] in ['scale', 'auto'] else float(params['gamma'])
            return SVR(
                kernel=params['kernel'],
                C=params['C'],
                degree=params['degree'],
                gamma=gamma_val,
                epsilon=params['epsilon']
            )
            
    except Exception as e:
        raise Exception(f"Model creation failed: {str(e)}")

def evaluate_component_count(n_comp, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, model_type, scaler_y):
    """Standalone function to evaluate a specific number of components"""
    try:
        if model_type == "PLS-R":
            model = PLSRegression(n_components=n_comp)
            model.fit(X_train_scaled, y_train_scaled)
            y_pred_scaled = model.predict(X_test_scaled)
            
        elif model_type == "PCA":
            pca = PCA(n_components=n_comp)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)
            model = LinearRegression()
            model.fit(X_train_pca, y_train_scaled)
            y_pred_scaled = model.predict(X_test_pca)
        else:
            return n_comp, float('inf'), 0.0
        
        # Convert back to original scale
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return n_comp, rmse, r2
    except Exception as e:
        return n_comp, float('inf'), 0.0

def optimize_components_parallel(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, 
                               model_type, max_components, scaler_y, n_jobs):
    """Optimize number of components using parallel processing"""
    try:
        # Create a copy of scaler_y that can be pickled
        scaler_y_copy = StandardScaler()
        scaler_y_copy.mean_ = scaler_y.mean_.copy()
        scaler_y_copy.scale_ = scaler_y.scale_.copy()
        scaler_y_copy.var_ = scaler_y.var_.copy()
        scaler_y_copy.n_features_in_ = scaler_y.n_features_in_
        scaler_y_copy.n_samples_seen_ = scaler_y.n_samples_seen_
        
        # Test different numbers of components
        component_range = range(1, min(max_components + 1, X_train_scaled.shape[1] + 1, X_train_scaled.shape[0]))
        
        # Use parallel processing to evaluate different component counts
        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_component_count)(
                n_comp, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, model_type, scaler_y_copy
            ) for n_comp in component_range
        )
        
        # Extract results
        components = [r[0] for r in results]
        rmse_values = [r[1] for r in results]
        r2_values = [r[2] for r in results]
        
        # Find optimal number of components (minimum RMSE)
        optimal_idx = np.argmin(rmse_values)
        optimal_components = components[optimal_idx]
        
        return optimal_components, components, rmse_values, r2_values
        
    except Exception as e:
        return None, [], [], []

def parse_parameter_value(param_name, param_value, param_type):
    """Parse parameter value based on its expected type"""
    try:
        if param_value.lower() == "none":
            return None
        elif param_value.lower() in ["true", "false"]:
            return param_value.lower() == "true"
        elif param_type in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", 
                           "n_components", "degree", "max_iter"]:
            return int(float(param_value)) if param_value.lower() != "none" else None
        elif param_type in ["alpha", "C", "epsilon", "tol", "learning_rate", "subsample", 
                           "colsample_bytree", "reg_alpha", "reg_lambda"]:
            return float(param_value)
        else:
            return param_value
    except:
        return param_value