import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import mlflow
import mlflow.sklearn
from scripts.data_loader import load_data
from scripts.preprocess import preprocess_data


def train_sarimax(train_data, train_exog, column, order=(1, 1, 1), seasonal_order=(0, 0, 0, 365)):
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order, exog=train_exog, enforce_stationarity=False,
                    enforce_invertibility=False)
    result = model.fit(maxiter=1000, disp=False)
    return model, result


def train_models(df, exog_columns, target_columns, forecast_steps=14):
    metrics = []
    models = {}
    results = {}
    for column in target_columns:
        if column not in df.columns:
            print(f"Error: {column} not found in DataFrame columns")
            continue
        train_data = df[column].iloc[:-forecast_steps]
        test_data = df[column].iloc[-forecast_steps:]
        train_exog = df[exog_columns].iloc[:-forecast_steps]
        test_exog = df[exog_columns].iloc[-forecast_steps:]

        model, result = train_sarimax(train_data, train_exog, column)
        forecast = result.get_forecast(steps=forecast_steps, exog=test_exog)
        mae = mean_absolute_error(test_data, forecast.predicted_mean)
        mse = mean_squared_error(test_data, forecast.predicted_mean)

        metrics.append({'Target Column': column, 'MAE': mae, 'MSE': mse})
        models[column] = model
        results[column] = result

        # Save the model
        mlflow.sklearn.save_model(result, f"models/{column}_sarimax_model")

    return models, results, metrics


def main():
    mlflow.start_run()

    base_path = 'data/raw'
    mosquito_df = load_data(base_path, 2020, 2024)
    weather_file_path = 'data/raw/Weather Stations.xlsx'
    weather_sheet_name = 'Sheet5'
    mosquito_species_columns = ['Ae aegypti', 'Ae albopictus', 'Ae inf / atl', 'Ae sollicitans', 'Ae sp',
                                'Ae teaniorhynchus', 'Ae vexans', 'Anopheles sp', 'Coquilltedia sp', 'Culiseta sp',
                                'Cx Coronator', 'Cx erraticus', 'Cx nigripapus', 'Cx quinq', 'Cx restuans',
                                'Cx salinarius', 'Cx sp', 'Deinocerites sp', 'Mansonia sp', 'Orthopodo',
                                'Ps ciliata', 'Ps columbiae', 'Ps ferox', 'Ps howardii', 'Ps sp',
                                'Uranotaen sp', 'Wyemoyia sp']
    final_df = preprocess_data(mosquito_df, weather_file_path, weather_sheet_name, mosquito_species_columns, 3)
    print("Final DataFrame columns:", final_df.columns)
    final_df.to_csv('data/processed/final_df.csv', index=True)

    exog_columns = ['Temperature', 'Relative Humidity', 'Gust', 'UV Index', 'Solar radiation', 'Rainfall']
    target_columns = ['log_Anopheles_sp', 'log_Cx_nigripapus', 'log_Ae_inf_/_atl', 'log_Total_Mosquitoes']

    models, results, metrics = train_models(final_df, exog_columns, target_columns)

    for metric in metrics:
        mlflow.log_metric(metric['Target Column'] + "_MAE", np.expm1(metric['MAE']))
        mlflow.log_metric(metric['Target Column'] + "_MSE", np.expm1(metric['MSE']))

    mlflow.end_run()


if __name__ == "__main__":
    main()
