import streamlit as st
import joblib
import pandas as pd
# Load the trained model
model = joblib.load('linear_model.pkl')
# Function to get user input
# Function to get user input
def user_input_features():
    area = st.number_input('Area (in square feet)', min_value=0)
    # Add other input fields based on your dataset
    # For example, if you have a column 'bedrooms':
    bedrooms = st.number_input('Number of Bedrooms', min_value=0)
    # Add more features as needed
    # Example: bathrooms = st.number_input('Number of Bathrooms', min_value=0)
    
    # Create a DataFrame with the user input
    data = {
        'area': area,
        'bedrooms': bedrooms,
        # Add other features here
    }
    features = pd.DataFrame(data, index=[0])
    return features
# Streamlit app layout
st.title('Housing Price Prediction')
st.write('Enter the details below to predict the house price.')
# Get user input
input_data = user_input_features()
# Make prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write(f'The predicted price is: ${prediction[0]:,.2f}')