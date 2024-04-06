# Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load the dataset and preprocess the labels
df = pd.read_csv("sonar data.csv", header=None)
x = df.drop(columns=60, axis=1)
y = df[60]
y = y.astype(str)

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model = LogisticRegression()
model.fit(x, y)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from form
        input_data = request.form['input_data']

        # Remove class label from input data
        input_data = input_data.split(',')
        input_data = input_data[:-1]  # Exclude the last element, which is the class label

        # Convert input data to floats
        try:
            input_data = [float(value) for value in input_data]
        except ValueError:
            return "Invalid input format. Please provide comma-separated numeric values."

        # Make prediction
        prediction = model.predict([input_data])[0]

        # Determine prediction label
        prediction_label = 'Rock' if prediction == 'R' else 'Mine'

        return render_template('result.html', prediction_label=prediction_label)

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
