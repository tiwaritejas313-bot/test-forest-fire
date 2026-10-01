from flask import Flask, request, render_template
import pickle
import numpy as np


application = Flask(__name__)

# Try to load models (assuming they are in the parent 'ml 1' folder)
# We use try/except so the app doesn't crash if the paths are slightly off
try:
    scaler = pickle.load(open('../scaler.pkl', 'rb'))
    model = pickle.load(open('../ridge.pkl', 'rb'))
except FileNotFoundError:
    scaler = None
    model = None

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Grab all the form data (These must match the HTML names!)
            day = float(request.form.get('day', 1))
            month = float(request.form.get('month', 6))
            year = float(request.form.get('year', 2012))
            temperature = float(request.form.get('Temperature', 0))
            rh = float(request.form.get('RH', 0))
            ws = float(request.form.get('Ws', 0))
            rain = float(request.form.get('Rain', 0))
            ffmc = float(request.form.get('FFMC', 0))
            dmc = float(request.form.get('DMC', 0))
            isi = float(request.form.get('ISI', 0))
            classes = float(request.form.get('Classes', 0))
            region = float(request.form.get('Region', 0))
            
            # Combine into list (must match the order you trained your model on!)
            features = [day, month, year, temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]
            features = np.array([features])
            
            # 2. Scale features and Predict
            if scaler and model:
                scaled_features = scaler.transform(features)
                prediction = model.predict(scaled_features)
                result = round(prediction[0], 2)
            else:
                result = "Error: Models (scaler.pkl or ridge.pkl) not found!"
            
            # 3. Return the prediction back to the webpage
            return render_template('index.html', result=result)
            
        except Exception as e:
            return render_template('index.html', result=f"Error processing input: {e}")

if __name__ == "__main__":
    application.run(debug=True,port=8000)
