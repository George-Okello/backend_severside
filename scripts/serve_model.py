from flask import Flask, request, jsonify
import pandas as pd
import mlflow
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load the models
models = {
    'log_Anopheles_sp': mlflow.sklearn.load_model("../models/log_Anopheles_sp_sarimax_model"),
    'log_Cx_nigripapus': mlflow.sklearn.load_model("../models/log_Cx_nigripapus_sarimax_model"),
    'log_Ae_inf_/_atl': mlflow.sklearn.load_model("../models/log_Ae_inf_/_atl_sarimax_model"),
    'log_Total_Mosquitoes': mlflow.sklearn.load_model("../models/log_Total_Mosquitoes_sarimax_model"),
}


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    logging.debug(f"Received data: {data}")

    if data is None:
        return jsonify({"error": "No data provided"}), 400

    input_df = pd.DataFrame(data)
    logging.debug(f"Input DataFrame: {input_df}")

    # Ensure the input DataFrame has a datetime index and 'Trap' column
    if 'date' in input_df.columns and 'Trap' in input_df.columns:
        input_df['date'] = pd.to_datetime(input_df['date'])
        input_df.rename(columns={'date': 'Collection Date'}, inplace=True)
        input_df.set_index(['Collection Date', 'Trap'], inplace=True)
    else:
        return jsonify({"error": "Input data must contain 'date' and 'Trap' columns."}), 400

    # Extract exogenous variables for the prediction
    exog_columns = ['Temperature', 'Relative Humidity', 'Gust', 'UV Index', 'Solar radiation', 'Rainfall']
    exog_data = input_df[exog_columns]

    predictions = {}
    for target, model in models.items():
        try:
            # Make forecast with exogenous variables
            forecast = model.get_forecast(steps=len(input_df), exog=exog_data).predicted_mean
            predictions[target] = forecast.tolist()
        except Exception as e:
            logging.error(f"Error in model prediction for {target}: {e}")
            predictions[target] = str(e)

    return jsonify(predictions)


if __name__ == '__main__':
    app.run(debug=True)
