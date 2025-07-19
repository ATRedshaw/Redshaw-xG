import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
import numpy as np

def run_linear_regressions(lagged_df):
    """
    Fit separate linear regression models for each predictor to predict future_goals.
    """
    predictors = [
        'past_xg_basic', 'past_xg_situation', 'past_xg_shottype',
        'past_xg_advanced', 'past_goals'
    ]
    results = []

    for predictor in predictors:
        if predictor in lagged_df.columns and 'future_goals' in lagged_df.columns:
            temp_df = lagged_df[[predictor, 'future_goals']].dropna()
            if not temp_df.empty:
                X = temp_df[[predictor]]
                y = temp_df['future_goals']

                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)

                r2 = r2_score(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                coefficient = model.coef_[0]
                intercept = model.intercept_

                # For p-value, use statsmodels
                X_sm = sm.add_constant(X)
                sm_model = sm.OLS(y, X_sm)
                sm_results = sm_model.fit()
                p_value = sm_results.pvalues[predictor]

                results.append({
                    'model_name': f'Linear Regression: {predictor} vs Future Goals',
                    'predictor': predictor,
                    'R-squared': r2,
                    'MAE': mae,
                    'RMSE': rmse,
                    'Coefficient': coefficient,
                    'Intercept': intercept,
                    'P-value (Coefficient)': p_value,
                    'data': temp_df,
                    'predictions': y_pred
                })
            else:
                results.append({
                    'model_name': f'Linear Regression: {predictor} vs Future Goals',
                    'predictor': predictor,
                    'note': 'Not enough data for regression'
                })
        else:
            results.append({
                'model_name': f'Linear Regression: {predictor} vs Future Goals',
                'predictor': predictor,
                'note': 'Predictor or future_goals column missing'
            })
    return results