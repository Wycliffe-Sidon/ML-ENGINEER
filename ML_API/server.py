from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# Load the trained model
model = joblib.load('app/model.joblib')

# Define a Pydantic model for the input data
class PredictionData(BaseModel):
    Age: int
    Number: int
    Start: int

# Create a FastAPI instance
app = FastAPI()

# Root endpoint
@app.get('/')
def read_root():
    return {'message': 'Kyphosis model API'}

# Prediction endpoint
@app.post('/predict')
def predict(data: PredictionData):
    # Prepare the input data for prediction
    input_data = np.array([[data.Age, data.Number, data.Start]])

    # Make a prediction
    prediction = model.predict(input_data)

    # Return the entire prediction result as a list for JSON serialization
    return {'prediction': prediction.tolist()}