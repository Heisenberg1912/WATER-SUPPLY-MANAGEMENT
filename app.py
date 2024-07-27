import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu

# Function to create and train a new model
def create_and_train_model(data):
    features = data[['Household Size', 'Num Days No Water', 'Avg Temp']].values
    target = data['Water Usage'].values

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    return model, scaler

# Function to load or create the model
@st.experimental_singleton
def load_or_create_model(data):
    try:
        model = tf.keras.models.load_model('water_usage_model.h5')
        scaler = joblib.load('scaler.pkl')
    except:
        model, scaler = create_and_train_model(data)
        model.save('water_usage_model.h5')
        joblib.dump(scaler, 'scaler.pkl')
    return model, scaler

# Navbar setup
with st.sidebar:
    selected = option_menu("Main Menu", ["Home", "Data", "Model", "About"], 
        icons=['house', 'database', 'gear', 'info'], menu_icon="cast", default_index=0)

# Home page
if selected == "Home":
    st.title("Water Supply Management")
    st.write("Welcome to the Water Supply Management System. Use the sidebar to navigate to different sections.")

# Data page
elif selected == "Data":
    st.title("Data Overview")
    date_option = st.selectbox("Select date range", ["1 month", "6 months", "1 year"])

    if date_option == "1 month":
        start_date = datetime.now() - timedelta(days=30)
    elif date_option == "6 months":
        start_date = datetime.now() - timedelta(days=182)
    elif date_option == "1 year":
        start_date = datetime.now() - timedelta(days=365)

    end_date = datetime.now()

    # Generate example household data
    @st.experimental_memo
    def generate_household_data(start_date, end_date):
        np.random.seed(42)  # For reproducible results
        num_households = 100
        dates = pd.date_range(start=start_date, end=end_date)
        data = pd.DataFrame({
            'Date': np.random.choice(dates, size=num_households),
            'Household ID': np.arange(1, num_households + 1),
            'Received Water': np.random.choice([True, False], size=num_households, p=[0.8, 0.2]),
            'Water Usage': np.random.rand(num_households) * 150,
            'Water Limit': np.random.choice([100, 150, 200], size=num_households),
            'Household Size': np.random.randint(1, 6, size=num_households),
            'Num Days No Water': np.random.randint(0, 30, size=num_households),
            'Avg Temp': np.random.rand(num_households) * 10 + 15  # Average temperature
        })
        return data

    # Update data based on selected date range
    if st.button("Update Data"):
        household_data = generate_household_data(start_date, end_date)
        st.write("### Household Data", household_data)

        # Calculate statistics
        total_households = len(household_data)
        households_receiving_water = household_data['Received Water'].sum()
        households_not_receiving_water = total_households - households_receiving_water

        used_within_limit = (household_data['Water Usage'] <= household_data['Water Limit']).sum()
        wasted_beyond_limit = (household_data['Water Usage'] > household_data['Water Limit']).sum()

        total_usage = household_data['Water Usage'].sum()
        total_wasted = household_data.loc[household_data['Water Usage'] > household_data['Water Limit'], 'Water Usage'].sum() - household_data.loc[household_data['Water Usage'] > household_data['Water Limit'], 'Water Limit'].sum()

        mean_usage = household_data['Water Usage'].mean()
        median_usage = household_data['Water Usage'].median()
        std_usage = household_data['Water Usage'].std()

        st.write(f"**Total households**: {total_households}")
        st.write(f"**Households receiving water**: {households_receiving_water}")
        st.write(f"**Households not receiving water**: {households_not_receiving_water}")
        st.write(f"**Households using water within limit**: {used_within_limit}")
        st.write(f"**Households wasting water beyond limit**: {wasted_beyond_limit}")
        st.write(f"**Total water usage (liters)**: {total_usage:.2f}")
        st.write(f"**Total water wasted (liters)**: {total_wasted:.2f}")
        st.write(f"**Mean water usage (liters)**: {mean_usage:.2f}")
        st.write(f"**Median water usage (liters)**: {median_usage:.2f}")
        st.write(f"**Standard deviation of water usage (liters)**: {std_usage:.2f}")

        # Interactive bar plot
        fig = px.bar(
            x=['Receiving Water', 'Not Receiving Water'], 
            y=[households_receiving_water, households_not_receiving_water],
            labels={'x': 'Household Status', 'y': 'Number of Households'},
            title='Households Receiving vs. Not Receiving Water'
        )
        st.plotly_chart(fig)

        fig2 = px.bar(
            x=['Within Limit', 'Beyond Limit'], 
            y=[used_within_limit, wasted_beyond_limit],
            labels={'x': 'Usage Status', 'y': 'Number of Households'},
            title='Households Using Water Within Limit vs. Beyond Limit'
        )
        st.plotly_chart(fig2)

        # Heatmap for water usage
        heatmap_data = household_data.pivot_table(values='Water Usage', index='Household ID', columns='Date', fill_value=0)
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        sns.heatmap(heatmap_data, ax=ax3, cmap='viridis')
        st.pyplot(fig3)

# Model page
elif selected == "Model":
    st.title("Model Training and Prediction")
    # Load or create model
    model, scaler = load_or_create_model(generate_household_data(datetime.now() - timedelta(days=365), datetime.now()))

    # Example of model prediction
    def predict_usage(model, data):
        # Ensure the data has the correct shape
        features = data[['Household Size', 'Num Days No Water', 'Avg Temp']].values
        features = scaler.transform(features)
        prediction = model.predict(features)
        return prediction.flatten()

    if st.button("Predict Usage"):
        household_data = generate_household_data(datetime.now() - timedelta(days=365), datetime.now())
        try:
            prediction = predict_usage(model, household_data)
            household_data['Predicted Usage'] = prediction

            st.write("### Predicted Data", household_data)

            # Interactive plot for predictions
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=household_data['Household ID'], y=household_data['Water Usage'], mode='lines', name='Actual'))
            fig4.add_trace(go.Scatter(x=household_data['Household ID'], y=household_data['Predicted Usage'], mode='lines', name='Predicted'))
            fig4.update_layout(title='Actual vs. Predicted Water Usage', xaxis_title='Household ID', yaxis_title='Water Usage (liters)')
            st.plotly_chart(fig4)

            # Saving data example
            if st.button("Save Data"):
                household_data.to_csv('predicted_household_water_usage.csv')
                st.write("Data saved to `predicted_household_water_usage.csv`")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# About page
elif selected == "About":
    st.title("About")
    st.write("This application is designed to manage water supply for households. It provides data analysis and predictive modeling for water usage. The system can predict future water usage based on various factors such as household size, days without water, and average temperature. The data is visualized using interactive plots for better understanding and decision making.")
