from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

def create_lag_features(df, lags=[1, 2, 3]):
    """
    Add lag features to the dataframe.
    """
    df_copy = df.copy()
    for lag in lags:
        df_copy[f'load_kW_t-{lag}'] = df_copy['load_kW'].shift(lag)
    return df_copy

def evaluate_model(y_true, y_pred):
    """
    Compute common evaluation metrics: MAE, RMSE, MAPE
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mae, rmse, mape

def plot_forecast_comparison(timestamps, actual, baseline, advanced):
    """
    Plot actual vs. baseline and advanced forecasts
    """
    plt.figure(figsize=(14, 6))
    plt.plot(timestamps, actual, label="Actual Load", marker="o")
    plt.plot(timestamps, baseline, label="Baseline Forecast", marker="x")
    plt.plot(timestamps, advanced, label="Advanced Forecast", marker="s")
    plt.title("Actual vs Baseline vs Advanced Forecast (24-Hour Horizon)")
    plt.xlabel("Timestamp")
    plt.ylabel("Load (kW)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_residuals(timestamps, baseline_residuals, advanced_residuals):
    """
    Plot residuals for both models
    """
    plt.figure(figsize=(14, 5))
    plt.plot(timestamps, baseline_residuals, label="Baseline Residuals", marker="x")
    plt.plot(timestamps, advanced_residuals, label="Advanced Residuals", marker="s")
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Residuals: Baseline vs Advanced Model")
    plt.xlabel("Timestamp")
    plt.ylabel("Error (kW)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
