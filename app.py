from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

app = Flask(__name__)


# Function to build and evaluate models with Cross-Validation
def build_and_evaluate_model_cv(data_scaled, model):
    x = data_scaled[:, 1].reshape(-1, 1)
    y = data_scaled[:, 0]
    model.fit(x, y)
    scores = cross_val_score(model, x, y, cv=5, scoring='neg_mean_squared_error')
    avg_mse = -scores.mean()
    return model, avg_mse


# Function to load data dynamically based on the filename
def load_data(filename):
    data = pd.read_csv(filename)
    data.dropna(inplace=True)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[['Close', 'Volume']])
    close_mean = data['Close'].mean()
    close_std = data['Close'].std()
    volume_mean = data['Volume'].mean()
    volume_std = data['Volume'].std()
    return data_scaled, close_mean, close_std, volume_mean, volume_std


# API endpoint for predictions
@app.route('/predict', methods=['POST'])
def api_predict():
    volume = request.json['volume']
    filename = request.json['filename']
    data_scaled, close_mean, close_std, volume_mean, volume_std = load_data(filename)
    linear_model, _ = build_and_evaluate_model_cv(data_scaled, LinearRegression())
    rf_model, _ = build_and_evaluate_model_cv(data_scaled, RandomForestRegressor())
    volume_scaled = (volume - volume_mean) / volume_std
    linear_prediction = linear_model.predict(np.array([[volume_scaled]]))[0]
    rf_prediction = rf_model.predict(np.array([[volume_scaled]]))[0]
    linear_prediction_o = (linear_prediction * close_std) + close_mean
    rf_prediction_o = (rf_prediction * close_std) + close_mean
    response = {
        'Linear Regression Prediction': linear_prediction_o,
        'Random Forest Prediction': rf_prediction_o
    }
    return jsonify(response)


# Route to serve the HTML form and handle form submissions
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        filename = request.form['filename']
        volume = float(request.form['volume'])
        prediction = predict_for_template(filename, volume)
        return render_template('index.html', prediction=prediction)
    return render_template('index.html')


# Function to predict for the template
def predict_for_template(filename, volume):
    data_scaled, close_mean, close_std, volume_mean, volume_std = load_data(filename)
    linear_model, _ = build_and_evaluate_model_cv(data_scaled, LinearRegression())
    rf_model, _ = build_and_evaluate_model_cv(data_scaled, RandomForestRegressor())
    volume_scaled = (volume - volume_mean) / volume_std
    linear_prediction = linear_model.predict(np.array([[volume_scaled]]))[0]
    rf_prediction = rf_model.predict(np.array([[volume_scaled]]))[0]
    linear_prediction_o = (linear_prediction * close_std) + close_mean
    rf_prediction_o = (rf_prediction * close_std) + close_mean
    return {
        'Linear Regression Prediction': linear_prediction_o,
        'Random Forest Prediction': rf_prediction_o
    }


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
