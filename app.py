import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Function to load the model, using st.experimental_singleton to cache it
@st.experimental_singleton
def load_model():
    model = tf.keras.models.load_model('water_usage_model.h5')  # Replace 'path_to_model' with your model path
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
        'Water Limit': np.random.choice([100, 150, 200], size=num_households)
    })
    return data

# Update data based on selected date range
if st.button("Update Data"):
    household_data = generate_household_data(start_date, end_date)
    st.write("Household Data:", household_data)

    # Calculate statistics
    total_households = len(household_data)
    households_receiving_water = household_data['Received Water'].sum()
    households_not_receiving_water = total_households - households_receiving_water

    used_within_limit = (household_data['Water Usage'] <= household_data['Water Limit']).sum()
    wasted_beyond_limit = (household_data['Water Usage'] > household_data['Water Limit']).sum()

    total_usage = household_data['Water Usage'].sum()
    total_wasted = household_data.loc[household_data['Water Usage'] > household_data['Water Limit'], 'Water Usage'].sum() - household_data.loc[household_data['Water Usage'] > household_data['Water Limit'], 'Water Limit'].sum()

    st.write(f"Total households: {total_households}")
    st.write(f"Households receiving water: {households_receiving_water}")
    st.write(f"Households not receiving water: {households_not_receiving_water}")
    st.write(f"Households using water within limit: {used_within_limit}")
    st.write(f"Households wasting water beyond limit: {wasted_beyond_limit}")
    st.write(f"Total water usage (liters): {total_usage}")
    st.write(f"Total water wasted (liters): {total_wasted}")

    # Example plot
    fig, ax = plt.subplots()
    sns.barplot(x=['Receiving Water', 'Not Receiving Water'], y=[households_receiving_water, households_not_receiving_water], ax=ax)
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    sns.barplot(x=['Within Limit', 'Beyond Limit'], y=[used_within_limit, wasted_beyond_limit], ax=ax2)
    st.pyplot(fig2)

    # Example of model prediction (using a placeholder function)
    def predict_usage(model, data):
        # Placeholder for actual model prediction
        prediction = data['Water Usage'].mean() + np.random.randn(len(data)) * 10
        return prediction

    prediction = predict_usage(model, household_data)
    household_data['Predicted Usage'] = prediction

    st.write("Predicted Data:", household_data)

    # Example plot for predictions
    fig3, ax3 = plt.subplots()
    sns.lineplot(data=household_data, x='Household ID', y='Water Usage', label='Actual', ax=ax3)
    sns.lineplot(data=household_data, x='Household ID', y='Predicted Usage', label='Predicted', ax=ax3)
    st.pyplot(fig3)

    # Saving data example
    if st.button("Save Data"):
        household_data.to_csv('predicted_household_water_usage.csv')
        st.write("Data saved to predicted_household_water_usage.csv")
else:
    st.write("Click the 'Update Data' button to generate data based on the selected date range.")
