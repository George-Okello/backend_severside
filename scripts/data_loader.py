import pandas as pd
import os


def read_excel_files(base_path, start_year, end_year):
    data_frames = []
    for year in range(start_year, end_year + 1):
        file_path = os.path.join(base_path, f'Light trap collections {year}.xlsx')
        df = pd.read_excel(file_path)
        data_frames.append(df)
    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df


def load_data(base_path, start_year, end_year):
    recent_data_df = read_excel_files(base_path, start_year, end_year)
    historical_data_path = os.path.join(base_path, 'Light trap collections 18 through year end 2019.xlsx')
    historical_data_df = pd.read_excel(historical_data_path)
    df = pd.concat([recent_data_df, historical_data_df], ignore_index=True)
    df = df.dropna(subset=['Collection Date'])
    if 'Comment' in df.columns:
        df = df.drop(columns=['Comment'])
    return df
