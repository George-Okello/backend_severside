import pandas as pd
import numpy as np


def filter_significant_columns(df, mosquito_species_columns, top_n=3):
    mosquito_df = df[mosquito_species_columns]
    column_means = mosquito_df.mean()
    significant_columns = column_means.nlargest(top_n).index.tolist()
    significant_columns += ['Collection Date', 'Total Mosquitoes', 'Trap']
    return df[significant_columns]


def preprocess_weather_data(weather_file_path, sheet_name='Sheet5'):
    weather_df = pd.read_excel(weather_file_path, sheet_name=sheet_name)
    weather_df['Collection Date'] = pd.to_datetime(weather_df['Collection Date'])
    columns_to_convert = ["Temperature", "Rainfall", "Relative Humidity", "UV Index", "Solar radiation"]
    weather_df[columns_to_convert] = weather_df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    return weather_df


def merge_data(mosquito_df, weather_df):
    merged_df = pd.merge(mosquito_df, weather_df, on='Collection Date', how='inner')
    return merged_df


def convert_column_types(df):
    df['Collection Date'] = pd.to_datetime(df['Collection Date'])
    for col in df.columns:
        if col != 'Collection Date':
            df[col] = df[col].astype(float)
    return df


def set_index(df):
    df.set_index(['Collection Date', 'Trap'], inplace=True)
    return df


def interpolate_and_fill_na(df):
    df = df.sort_index()
    df = df.reset_index()
    df = df.interpolate(method='linear')
    df.set_index(['Collection Date', 'Trap'], inplace=True)
    df['Cx nigripapus'] = df['Cx nigripapus'].fillna(method='bfill')
    df['Ae inf / atl'] = df['Ae inf / atl'].fillna(method='bfill')
    return df


def log_transform(df, columns):
    for col in columns:
        df[f'log_{col.replace(" ", "_")}'] = np.log1p(df[col])
    return df


def preprocess_data(mosquito_df, weather_file_path, weather_sheet_name, mosquito_species_columns, top_n):
    df_significant = filter_significant_columns(mosquito_df, mosquito_species_columns, top_n)
    weather_df = preprocess_weather_data(weather_file_path, weather_sheet_name)
    weather_columns = ["Temperature", "Relative Humidity", "Gust", "UV Index", "Solar radiation", "Rainfall"]
    weather_df = weather_df.dropna(subset=weather_columns, how='all')
    weather_df = convert_column_types(weather_df)
    merged_df = merge_data(df_significant, weather_df)
    final_df = set_index(merged_df)
    final_df = interpolate_and_fill_na(final_df)
    final_df = log_transform(final_df, ['Anopheles sp', 'Cx nigripapus', 'Ae inf / atl', 'Total Mosquitoes'])
    print("Columns after log transformation:", final_df.columns)
    return final_df
