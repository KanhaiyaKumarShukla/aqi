from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load model and scaler with error handling
try:
    saved_data = pickle.load(open('model.pkl', 'rb'))
    model = saved_data['model']
    scaler = saved_data['scaler']
except Exception as e:
    raise RuntimeError(f"Failed to load model or scaler: {str(e)}")

@app.route('/')
def home():
    return "AQI Prediction API - Send POST request to /predict with co, temperature, humidity"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate input parameters
        required_params = ['co', 'temperature', 'humidity']
        if not all(param in request.form for param in required_params):
            return jsonify({'error': 'Missing parameters. Required: co, temperature, humidity'}), 400
        
        # Convert and validate input values
        try:
            co = float(request.form['co'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
        except ValueError:
            return jsonify({'error': 'All parameters must be numeric values'}), 400
        
        # Create input array and scale it
        input_data = np.array([[co, temperature, humidity]], dtype=np.float32)
        
        try:
            input_scaled = scaler.transform(input_data)
        except Exception as e:
            return jsonify({'error': f'Scaling failed: {str(e)}'}), 500
        
        # Reshape for LSTM (samples, timesteps, features)
        input_reshaped = input_scaled.reshape(1, 1, input_scaled.shape[1])
        
        # Make prediction
        try:
            prediction = model.predict(input_reshaped)
            prediction_actual = scaler.inverse_transform(prediction)
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
        return jsonify({
            'prediction': {
                'CO_AQI': float(prediction_actual[0][0]),
                'Temperature': float(prediction_actual[0][1]),
                'Humidity': float(prediction_actual[0][2])
            },
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}', 'status': 'error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)