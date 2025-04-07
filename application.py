import pickle
import numpy as np
import numpy as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application


## import ride regressor and standard scaler
ridge_model = pickle.load(open("models/model.pkl",'rb'))
standard_scaler = pickle.load(open("models/scaler.pkl",'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata",methods = ['GET','POST'])
def predict_datapoint():
    if request.method == "POST":
        # Get all form data
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))# Kept as string if it's categorical
        Region = float(request.form.get('Region'))   # Kept as string if it's categorical

        # Prepare input data for prediction
        input_data = [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        
        # Scale the data
        scaled_data = standard_scaler.transform(input_data)
        
        # Make prediction
        prediction = ridge_model.predict(scaled_data)
        
        # Process prediction result
        result = round(float(prediction[0]), 2) 

        return render_template('home.html', results = result)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')  # Run Flask in debug mode