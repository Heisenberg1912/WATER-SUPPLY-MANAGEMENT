import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from datetime import datetime, timedelta

# Paths to model, scaler, and feature names
model_path = 'water_usage_model.h5'
scaler_path = 'scaler.pkl'
feature_names_path = 'feature_names.pkl'

# Load the trained model, scaler, and feature names
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(feature_names_path)
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None, None

model, scaler, feature_names = load_model()

# Function to generate synthetic data
def generate_synthetic_data(num_records):
    data = {
        'Household ID': np.random.randint(1, 1000000, size=num_records),
        'Ward': np.random.randint(1, 50, size=num_records),
        'Area': np.random.randint(1, 100, size=num_records),
        'Leakage Detected (Yes/No)': np.random.choice([0, 1], size=num_records, p=[0.9, 0.1]),
        'Disparity in Supply (Yes/No)': np.random.choice([0, 1], size=num_records, p=[0.95, 0.05]),
        'Income Level': np.random.choice([0, 1, 2], size=num_records, p=[0.3, 0.5, 0.2]),
        'Household Size': np.random.randint(1, 10, size=num_records),
        'Monthly Water Usage (Liters)': np.random.randint(1000, 3000, size=num_records),
        'Date': [datetime.now() - timedelta(days=i) for i in range(num_records)]
    }
    return pd.DataFrame(data)

# Function to predict and analyze water usage
def predict_and_analyze(df):
    try:
        # Prepare the data
        X = df[feature_names]
        y = df['Monthly Water Usage (Liters)']

        # Scale the data
        X_scaled = scaler.transform(X)

        # Make predictions
        y_pred = model.predict(X_scaled)

        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"Mean Absolute Error: {mae}")
        st.write(f"R^2 Score: {r2}")

        # Plot true vs predicted values
        fig, ax = plt.subplots()
        ax.scatter(y, y_pred, alpha=0.3)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("True vs Predicted Water Usage")
        st.pyplot(fig)

        # Plot residuals
        residuals = y - y_pred.flatten()
        fig, ax = plt.subplots()
        sns.histplot(residuals, bins=50, kde=True, ax=ax)
        ax.set_xlabel("Residuals")
        ax.set_title("Distribution of Residuals")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Function to generate and save reports
def generate_reports(df):
    try:
        # Generate monthly and yearly reports
        df['Date'] = pd.to_datetime(df['Date'])
        monthly_report = df.groupby(['Ward', 'Area', df['Date'].dt.to_period('M')])['Monthly Water Usage (Liters)'].sum().unstack()
        yearly_report = df.groupby(['Ward', 'Area', df['Date'].dt.to_period('Y')])['Monthly Water Usage (Liters)'].sum().unstack()

        # Save the reports
        monthly_report.to_csv('monthly_water_usage_report.csv')
        yearly_report.to_csv('yearly_water_usage_report.csv')

        st.write("Monthly and yearly reports have been generated and saved.")
        
        # Plot water usage trends
        fig1 = px.line(monthly_report.T)
        fig1.update_layout(title="Monthly Water Usage by Ward and Area", xaxis_title="Month", yaxis_title="Water Usage (Liters)")
        st.plotly_chart(fig1)
        
        fig2 = px.line(yearly_report.T)
        fig2.update_layout(title="Yearly Water Usage by Ward and Area", xaxis_title="Year", yaxis_title="Water Usage (Liters)")
        st.plotly_chart(fig2)
    except Exception as e:
        st.error(f"Error: {str(e)}")

def retrain_model(new_data):
    global model, scaler, feature_names

    try:
        # Prepare the data
        new_data['Leakage Detected (Yes/No)'] = new_data['Leakage Detected (Yes/No)'].map({'Yes': 1, 'No': 0})
        new_data['Disparity in Supply (Yes/No)'] = new_data['Disparity in Supply (Yes/No)'].map({'Yes': 1, 'No': 0})
        new_data['Income Level'] = new_data['Income Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
        X = new_data[feature_names]
        y = new_data['Monthly Water Usage (Liters)']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Define the model
        model = Sequential([
            Dense(256, input_dim=X_train.shape[1], activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='linear')  # Regression output
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

        # Train the model
        model.fit(X_train, y_train, epochs=100, batch_size=512, validation_split=0.2, verbose=1)

        # Evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"Retrained Model - Mean Squared Error: {mse}")
        st.write(f"Retrained Model - Mean Absolute Error: {mae}")
        st.write(f"Retrained Model - R^2 Score: {r2}")

        # Save the updated model and scaler
        model.save(model_path)
        joblib.dump(scaler, scaler_path)

        st.success("Model has been retrained and saved successfully!")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Streamlit layout
st.title("Water Usage Monitoring and Prediction")

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Generate & Predict", "Load & Retrain", "Interactive Plot"])

with tab1:
    if st.button("Generate Data and Predict"):
        num_records = st.slider("Number of records to generate", 1000, 10000, 10000)
        df = generate_synthetic_data(num_records)
        predict_and_analyze(df)
        generate_reports(df)

with tab2:
    uploaded_file = st.file_uploader("Choose a CSV file to retrain the model", type="csv")
    if uploaded_file:
        new_data = pd.read_csv(uploaded_file)
        retrain_model(new_data)

with tab3:
    if st.button("Generate Interactive Plot"):
        df = generate_synthetic_data(100)
        interactive_plot(df)
