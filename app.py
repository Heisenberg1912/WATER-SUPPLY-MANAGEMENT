import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from datetime import datetime, timedelta

# Function to load the model, using st.experimental_singleton to cache it
@st.experimental_singleton
def load_model():
    model = tf.keras.models.load_model('water_usage_model.h5')  # Replace 'path_to_model' with your model path
    return model

model = load_model()

# Example of adding a title and some Streamlit components
st.title("Water Supply Management")

# Add more Streamlit components here
st.write("This is an example of a Streamlit application for managing water supply data.")

# Example placeholder for user inputs
start_date = st.date_input("Start date", datetime.now() - timedelta(days=30))
end_date = st.date_input("End date", datetime.now())

# Example placeholder for data processing
@st.experimental_memo
def generate_data(start_date, end_date):
    # Example: Generate random data for demonstration
    date_range = pd.date_range(start_date, end_date)
    data = pd.DataFrame({
        'Date': date_range,
        'Water Usage': np.random.rand(len(date_range)) * 100
    })
    return data

data = generate_data(start_date, end_date)
st.write("Generated Data:", data)

# Example plot
fig, ax = plt.subplots()
sns.lineplot(data=data, x='Date', y='Water Usage', ax=ax)
st.pyplot(fig)

# Model prediction example
# Assuming you have a trained model that takes 'Water Usage' as input
def predict_usage(model, data):
    # Example: Using a simple mean as a placeholder for model prediction
    prediction = data['Water Usage'].mean() + np.random.randn(len(data)) * 10
    return prediction

prediction = predict_usage(model, data)
data['Predicted Usage'] = prediction

st.write("Predicted Data:", data)

# Example plot for predictions
fig2, ax2 = plt.subplots()
sns.lineplot(data=data, x='Date', y='Water Usage', label='Actual', ax=ax2)
sns.lineplot(data=data, x='Date', y='Predicted Usage', label='Predicted', ax=ax2)
st.pyplot(fig2)

# Saving data example
if st.button("Save Data"):
    data.to_csv('predicted_water_usage.csv')
    st.write("Data saved to predicted_water_usage.csv")
