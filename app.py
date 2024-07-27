import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Function to load the model, using st.experimental_singleton to cache it
@st.experimental_singleton
def load_model():
    model = tf.keras.models.load_model('water_usage_model.h5')  # Use the specified model file
    return model

model = load_model()

# Example of adding a title and some Streamlit components
st.title("Water Supply Management")

# Date range selection
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
        'Avg Temp': np.random.rand(num_households) * 10 + 15,  # Average temperature
        'Season': np.random.choice(['Winter', 'Spring', 'Summer', 'Autumn'], size=num_households)
    })
    data = pd.get_dummies(data, columns=['Season'])  # One-hot encoding for categorical data
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

    # Example of model prediction
    def predict_usage(model, data):
        # Ensure the data has the correct shape
        features = data[['Water Usage', 'Household Size', 'Num Days No Water', 'Avg Temp', 'Season_Autumn', 'Season_Spring', 'Season_Summer', 'Season_Winter']]
        prediction = model.predict(features)
        return prediction.flatten()

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
else:
    st.write("Click the 'Update Data' button to generate data based on the selected date range.")
