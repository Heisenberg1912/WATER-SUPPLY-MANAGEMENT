import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to simulate data loading
@st.cache
def load_data():
    # Simulated data: replace with your actual data source
    data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=100),
        'Household ID': np.random.randint(1, 10, 100),
        'Water Usage (Liters)': np.random.randint(50, 200, 100)
    })
    return data

# Function to calculate water waste
def calculate_waste(data):
    # Simulated waste calculation: replace with your actual logic
    data['Waste (Liters)'] = data['Water Usage (Liters)'] * 0.1
    return data

# Load and preprocess data
data = load_data()
data = calculate_waste(data)

# Streamlit app
st.title('Water Waste Management System')

# Sidebar for user input
st.sidebar.header('User Input')
household_id = st.sidebar.selectbox('Select Household ID', data['Household ID'].unique())
date_range = st.sidebar.date_input('Select Date Range', [data['Date'].min(), data['Date'].max()])

# Filter data based on user input
filtered_data = data[(data['Household ID'] == household_id) & 
                     (data['Date'] >= pd.to_datetime(date_range[0])) & 
                     (data['Date'] <= pd.to_datetime(date_range[1]))]

# Display filtered data
st.subheader('Filtered Data')
st.write(filtered_data)

# Plot water usage
st.subheader('Water Usage Over Time')
fig, ax = plt.subplots()
ax.plot(filtered_data['Date'], filtered_data['Water Usage (Liters)'], label='Water Usage')
ax.plot(filtered_data['Date'], filtered_data['Waste (Liters)'], label='Water Waste', linestyle='--')
ax.set_xlabel('Date')
ax.set_ylabel('Liters')
ax.legend()
st.pyplot(fig)

# Display statistics
st.subheader('Statistics')
total_usage = filtered_data['Water Usage (Liters)'].sum()
total_waste = filtered_data['Waste (Liters)'].sum()
st.write(f'Total Water Usage: {total_usage} Liters')
st.write(f'Total Water Waste: {total_waste} Liters')

# Additional features
st.sidebar.subheader('Additional Features')
if st.sidebar.checkbox('Show Data Summary'):
    st.subheader('Data Summary')
    st.write(data.describe())

if st.sidebar.checkbox('Download Data'):
    st.subheader('Download Data')
    csv = data.to_csv(index=False)
    st.download_button('Download CSV', csv, 'water_waste_data.csv', 'text/csv')

# Run the app
if __name__ == '__main__':
    st.run()
