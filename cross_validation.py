"""
Cross-validation utilities for spectral analysis
"""
import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, LeavePOut, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def rmse_original_scale_factory(scaler_y):
    """
    Returns a scorer function that computes RMSE in the original y scale.
    """
    def rmse_original(y_true_scaled, y_pred_scaled):
        # Inverse transform both y_true and y_pred
        y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).ravel()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        return np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse_original

def r2_original_scale_factory(scaler_y):
    """
    Returns a scorer function that computes R2 in the original y scale.
    """
    def r2_original(y_true_scaled, y_pred_scaled):
        # Inverse transform both y_true and y_pred
        y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).ravel()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        return r2_score(y_true, y_pred)
    return r2_original

def perform_cross_validation(X_scaled, y_scaled, model, cv_strategy, cv_params, scaler_y, n_jobs):
    """
    Perform cross-validation and compute metrics on the original scale of y using cross_val_score.
    """
    try:
        # Create cross-validation strategy
        if cv_strategy == "K-Fold":
            cv = KFold(n_splits=cv_params['k_folds'], shuffle=cv_params['shuffle'], random_state=42)
        elif cv_strategy == "Leave-One-Out":
            cv = LeaveOneOut()
        elif cv_strategy == "Leave-P-Out":
            cv = LeavePOut(p=cv_params['p_out'])
        else:
            return [], [], 0.0, 0.0, 0.0, 0.0

        # Create scorers for original scale
        rmse_scorer = make_scorer(
            rmse_original_scale_factory(scaler_y),
            greater_is_better=False
        )
        
        r2_scorer = make_scorer(
            r2_original_scale_factory(scaler_y),
            greater_is_better=True
        )

        # Perform cross-validation for RMSE
        cv_rmse_scores = cross_val_score(
            model, X_scaled, y_scaled, cv=cv, scoring=rmse_scorer, n_jobs=n_jobs
        )
        cv_rmse_scores = -cv_rmse_scores  # Make positive

        # Perform cross-validation for R2
        cv_r2_scores = cross_val_score(
            model, X_scaled, y_scaled, cv=cv, scoring=r2_scorer, n_jobs=n_jobs
        )

        # Calculate statistics
        cv_rmse_mean = np.mean(cv_rmse_scores)
        cv_rmse_std = np.std(cv_rmse_scores)
        cv_r2_mean = np.mean(cv_r2_scores)
        cv_r2_std = np.std(cv_r2_scores)

        return cv_rmse_scores.tolist(), cv_r2_scores.tolist(), cv_rmse_mean, cv_r2_mean, cv_rmse_std, cv_r2_std

    except Exception as e:
        print(f"Cross-validation error: {str(e)}")
        return [], [], 0.0, 0.0, 0.0, 0.0