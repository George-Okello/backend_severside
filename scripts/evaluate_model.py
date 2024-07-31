import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow


def evaluate_model(test_data, forecast, column):
    mae_log = mean_absolute_error(test_data, forecast)
    mse_log = mean_squared_error(test_data, forecast)

    # Convert log-transformed errors to original scale
    mae_original = np.expm1(mae_log)
    mse_original = np.expm1(mse_log)

    return {'MAE': mae_original, 'MSE': mse_original}


def plot_forecast(test_data, forecast, forecast_ci, column):
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, test_data.values, label='Actual')
    plt.plot(test_data.index, forecast, label='Forecast', linestyle='--')
    plt.fill_between(test_data.index, forecast_ci[:, 0], forecast_ci[:, 1], color='k', alpha=0.1)
    plt.legend()
    plt.title(f'Forecasted vs. Actual {column}')
    plt.show()


def main():
    mlflow.start_run()

    # Load the processed data
    final_df = pd.read_csv('data/processed/final_df.csv', index_col=['Collection Date', 'Trap'])
    print("Loaded DataFrame columns:", final_df.columns)
    exog_columns = ['Temperature', 'Relative Humidity', 'Gust', 'UV Index', 'Solar radiation', 'Rainfall']
    target_columns = ['log_Anopheles_sp', 'log_Cx_nigripapus', 'log_Ae_inf_/_atl', 'log_Total_Mosquitoes']

    forecast_steps = 14  # Number of days to forecast

    models = {}
    results = {}
    metrics = []

    for column in target_columns:
        if column not in final_df.columns:
            print(f"Error: {column} not found in DataFrame columns")
            continue
        # Split the data into train and test sets
        train_data = final_df[column].iloc[:-forecast_steps]
        test_data = final_df[column].iloc[-forecast_steps:]
        train_exog = final_df[exog_columns].iloc[:-forecast_steps]
        test_exog = final_df[exog_columns].iloc[-forecast_steps:]

        # Load the trained model
        model_path = f"models/{column}_sarimax_model"
        result = mlflow.sklearn.load_model(model_path)

        # Generate the forecasts for the test data
        forecast = result.get_forecast(steps=forecast_steps, exog=test_exog)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()

        # Evaluate the model
        metric = evaluate_model(test_data, forecast_mean, column)
        metrics.append({'Target Column': column, 'MAE': metric['MAE'], 'MSE': metric['MSE']})

        # Log the metrics
        mlflow.log_metric(f"{column}_MAE", metric['MAE'])
        mlflow.log_metric(f"{column}_MSE", metric['MSE'])

        # Reset index for plotting
        test_data_reset = test_data.reset_index(drop=True)
        forecast_mean_reset = forecast_mean.reset_index(drop=True)
        forecast_ci_reset = forecast_ci.reset_index(drop=True).values

        # Plot forecast vs actual
        plot_forecast(test_data_reset, forecast_mean_reset, forecast_ci_reset, column)

    mlflow.end_run()


if __name__ == "__main__":
    main()
