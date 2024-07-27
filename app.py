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

# List of wards with ward numbers
wards = [
    "Sirapur", "Chandan Nagar", "Kaalaani Nagar", "Sukhadev Nagar", "Raaj Nagar", "Malhaaraganj",
    "Janata Colony", "Joona Risaala", "Vrindaavan", "Baanaganga", "Bhaageerathpura", "Govind Colony",
    "Sangamanagar", "Ashok Nagar", "Bijaasan", "Nandabaag", "Kushavaah Nagar", "Sant Kabeer", 
    "Vishvakarma", "Gauree Nagar", "Shyaam Nagar", "P.Dee.D.Upaa. Nagar", "Sv. Raajesh Joshee", 
    "Sant Baaleejee Mahaaraaj", "Nanda Nagar", "Jeen Maata", "Pashupati Naath", "Maam Tulaja Bhavaanee", 
    "Dr Shyaamaaprasaad Mukharjee Nagar", "Sant Ravidaas", "Mahaaraaja Chhatrasaal", 
    "Atal Bihaaree Baajapeyee", "Sookhaliya", "Shaheed BhagatSinh", "Lasudiya Moree", "Nepaaniya", 
    "Saamee Kripa", "Haajee Colony", "Naaharasha Havelee", "Khajaraana", "Kailaashapuree", 
    "Swami Vivekaanand", "Shreenagar", "H.I.G.", "Dr. Bheemrao Ambedkar", "Somanaath", 
    "Saradaar Vallabh Bhai", "Geeta Bhavan", "Tilak Nagar", "Brajeshvaree", "Maam Bhagavatee Nagar", 
    "Musakkhedi", "Dr Maulaana Aajaad Nagar", "Residency", "Saaooth Tukoganj", "Snehalata Ganj", 
    "Devi Ahilyaabai", "Emli Bazaar", "Harasiddhee", "Ranipuraa", "Taatyaa Saravate", "Raavajee Baazaar", 
    "Navalakha", "Chitaavad", "Sant Kavar Raam", "Shaheed Hemu Kolonee", "Mahaaraaja Holakar", 
    "Bambaee Baazaar", "Jawaahar Maarg", "Lok Naayak Nagar", "Dravid Nagar", "Lok Maanya Nagar", 
    "Lakshman Sinh Chauhaan", "Vishnupuree", "Paalada", "Mundlaa Naayta", "Billavali", "Choithram", 
    "Sukhniwas", "Dr Rajendra Prasaad", "Annapurna", "Sudaama Nagar", "Gumaastaa Nagar", 
    "Dawaarkapuri", "Prajaapat Nagar"
]
ward_numbers = list(range(1, 86))

ward_mapping = dict(zip(wards, ward_numbers))

# Function to create and train a new model
def create_and_train_model(data):
    features = data[['Household Size', 'Num Days No Water', 'Avg Temp', 'Season_Spring', 'Season_Summer']].values
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

# Ward selection
selected_ward = st.selectbox("Select Ward", wards)
selected_ward_number = ward_mapping[selected_ward]

# File uploader for the dataset
uploaded_file = st.file_uploader("Upload Parquet File", type=["parquet"])

if uploaded_file is not None:
    data = pd.read_parquet(uploaded_file)

    # Display the columns of the dataframe to inspect available data
    st.write("### Data Columns", data.columns)

    # Filter data for the selected ward
    if 'Ward' in data.columns:
        data = data[data['Ward'] == selected_ward_number]
    else:
        st.error("The uploaded file does not contain the 'Ward' column.")

    # Ensure all expected columns are present in data
    for col in ['Season_Spring', 'Season_Summer']:
        if col not in data.columns:
            data[col] = 0

    # Update data based on selected date range
    if st.button("Update Data"):
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data_filtered = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
            st.write(f"### Household Data for {selected_ward} Ward (Ward Number {selected_ward_number})", data_filtered)

            # Calculate statistics
            if 'Received Water' in data_filtered.columns and 'Water Usage' in data_filtered.columns and 'Water Limit' in data_filtered.columns:
                total_households = len(data_filtered)
                households_receiving_water = data_filtered['Received Water'].sum()
                households_not_receiving_water = total_households - households_receiving_water

                used_within_limit = (data_filtered['Water Usage'] <= data_filtered['Water Limit']).sum()
                wasted_beyond_limit = (data_filtered['Water Usage'] > data_filtered['Water Limit']).sum()

                total_usage = data_filtered['Water Usage'].sum()
                total_wasted = data_filtered.loc[data_filtered['Water Usage'] > data_filtered['Water Limit'], 'Water Usage'].sum() - data_filtered.loc[data_filtered['Water Usage'] > data_filtered['Water Limit'], 'Water Limit'].sum()

                mean_usage = data_filtered['Water Usage'].mean()
                median_usage = data_filtered['Water Usage'].median()
                std_usage = data_filtered['Water Usage'].std()

                col1, col2 = st.columns(2)
                with col1:
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

                with col2:
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
                fig3, ax3 = plt.subplots(figsize=(10, 8))
                heatmap_data = data_filtered.pivot_table(values='Water Usage', index='Household ID', columns='Date', fill_value=0)
                sns.heatmap(heatmap_data, ax=ax3, cmap='viridis')
                st.pyplot(fig3)

                # Load or create model
                model, scaler = load_or_create_model(data_filtered)

                # Example of model prediction
                def predict_usage(model, data):
                    # Ensure the data has the correct shape
                    features = data[['Household Size', 'Num Days No Water', 'Avg Temp', 'Season_Spring', 'Season_Summer']].values
                    features = scaler.transform(features)
                    prediction = model.predict(features)
                    return prediction.flatten()

                try:
                    prediction = predict_usage(model, data_filtered)
                    data_filtered['Predicted Usage'] = prediction

                    st.write("### Predicted Data", data_filtered)

                    # Interactive plot for predictions
                    fig4 = go.Figure()
                    fig4.add_trace(go.Scatter(x=data_filtered['Household ID'], y=data_filtered['Water Usage'], mode='lines', name='Actual'))
                    fig4.add_trace(go.Scatter(x=data_filtered['Household ID'], y=data_filtered['Predicted Usage'], mode='lines', name='Predicted'))
                    fig4.update_layout(title='Actual vs. Predicted Water Usage', xaxis_title='Household ID', yaxis_title='Water Usage (liters)')
                    st.plotly_chart(fig4)

                    # Saving data example
                    if st.button("Save Data"):
                        data_filtered.to_csv('predicted_household_water_usage.csv')
                        st.write("Data saved to `predicted_household_water_usage.csv`")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
            else:
                st.error("The necessary columns ('Received Water', 'Water Usage', 'Water Limit') are not present in the data.")
        else:
            st.error("The uploaded file does not contain the 'Date' column.")
else:
    st.write("Please upload the Parquet file to proceed.")
