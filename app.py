import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# Load the dataset
file_path = 'indore_water_usage_data_difficult2.parquet'
household_data = pd.read_parquet(file_path)

# Mapping of ward numbers to names
ward_names = {
    1: 'Sirapur', 2: 'Chandan Nagar', 3: 'Kaalaani Nagar', 4: 'Sukhadev Nagar', 5: 'Raaj Nagar',
    6: 'Malhaaraganj', 7: 'Janata Colony', 8: 'Joona Risaala', 9: 'Vrindaavan', 10: 'Baanaganga',
    11: 'Bhaageerathpura', 12: 'Govind Colony', 13: 'Sangamanagar', 14: 'Ashok Nagar', 15: 'Bijaasan',
    16: 'Nandabaag', 17: 'Kushavaah Nagar', 18: 'Sant Kabeer', 19: 'Vishvakarma', 20: 'Gauree Nagar',
    21: 'Shyaam Nagar', 22: 'P.Dee.D.Upaa. Nagar', 23: 'Sv. Raajesh Joshee', 24: 'Sant Baaleejee Mahaaraaj',
    25: 'Nanda Nagar', 26: 'Jeen Maata', 27: 'Pashupati Naath', 28: 'Maam Tulaja Bhavaanee',
    29: 'Dr Shyaamaaprasaad Mukharjee Nagar', 30: 'Sant Ravidaas', 31: 'Mahaaraaja Chhatrasaal',
    32: 'Atal Bihaaree Baajapeyee', 33: 'Sookhaliya', 34: 'Shaheed BhagatSinh', 35: 'Lasudiya Moree',
    36: 'Nepaaniya', 37: 'Saamee Kripa', 38: 'Haajee Colony', 39: 'Naaharasha Havelee', 40: 'Khajaraana',
    41: 'Kailaashapuree', 42: 'Swami Vivekaanand', 43: 'Shreenagar', 44: 'H.I.G.', 45: 'Dr. Bheemrao Ambedkar',
    46: 'Somanaath', 47: 'Saradaar Vallabh Bhai', 48: 'Geeta Bhavan', 49: 'Tilak Nagar', 50: 'Brajeshvaree',
    51: 'Maam Bhagavatee Nagar', 52: 'Musakkhedi', 53: 'Dr Maulaana Aajaad Nagar', 54: 'Residency',
    55: 'Saaooth Tukoganj', 56: 'Snehalata Ganj', 57: 'Devi Ahilyaabai', 58: 'Emli Bazaar', 59: 'Harasiddhee',
    60: 'Ranipuraa', 61: 'Taatyaa Saravate', 62: 'Raavajee Baazaar', 63: 'Navalakha', 64: 'Chitaavad',
    65: 'Sant Kavar Raam', 66: 'Shaheed Hemu Kolonee', 67: 'Mahaaraaja Holakar', 68: 'Bambaee Baazaar',
    69: 'Jawaahar Maarg', 70: 'Lok Naayak Nagar', 71: 'Dravid Nagar', 72: 'Lok Maanya Nagar',
    73: 'Lakshman Sinh Chauhaan', 74: 'Vishnupuree', 75: 'Paalada', 76: 'Mundlaa Naayta', 77: 'Billavali',
    78: 'Choithram', 79: 'Sukhniwas', 80: 'Dr Rajendra Prasaad', 81: 'Annapurna', 82: 'Sudaama Nagar',
    83: 'Gumaastaa Nagar', 84: 'Dawaarkapuri', 85: 'Prajaapat Nagar'
}

# Add ward names to the dataframe
household_data['Ward Name'] = household_data['Ward'].map(ward_names)

# Add latitude and longitude for each ward (example coordinates, please complete with actual data)
ward_coords = {
 'Sirapur': (22.7196, 75.8577),
    'Chandan Nagar': (22.7242, 75.8648),
    'Kaalaani Nagar': (22.7324, 75.8765),
    'Sukhadev Nagar': (22.7292, 75.8744),
    'Raaj Nagar': (22.7258, 75.8721),
    'Malhaaraganj': (22.7187, 75.8544),
    'Janata Colony': (22.7220, 75.8600),
    'Joona Risaala': (22.7190, 75.8617),
    'Vrindaavan': (22.7235, 75.8622),
    'Baanaganga': (22.7348, 75.8810),
    'Bhaageerathpura': (22.7354, 75.8792),
    'Govind Colony': (22.7310, 75.8755),
    'Sangamanagar': (22.7156, 75.8253),
    'Ashok Nagar': (22.7316, 75.8796),
    'Bijaasan': (22.7444, 75.8917),
    'Nandabaag': (22.7378, 75.8772),
    'Kushavaah Nagar': (22.7383, 75.8780),
    'Sant Kabeer': (22.7337, 75.8700),
    'Vishvakarma': (22.7344, 75.8730),
    'Gauree Nagar': (22.7366, 75.8752),
    'Shyaam Nagar': (22.7372, 75.8760),
    'P.Dee.D.Upaa. Nagar': (22.7288, 75.8682),
    'Sv. Raajesh Joshee': (22.7292, 75.8690),
    'Sant Baaleejee Mahaaraaj': (22.7260, 75.8656),
    'Nanda Nagar': (22.7320, 75.8732),
    'Jeen Maata': (22.7330, 75.8740),
    'Pashupati Naath': (22.7350, 75.8765),
    'Maam Tulaja Bhavaanee': (22.7360, 75.8770),
    'Dr Shyaamaaprasaad Mukharjee Nagar': (22.7370, 75.8785),
    'Sant Ravidaas': (22.7380, 75.8790),
    'Mahaaraaja Chhatrasaal': (22.7390, 75.8800),
    'Atal Bihaaree Baajapeyee': (22.7400, 75.8810),
    'Sookhaliya': (22.7410, 75.8820),
    'Shaheed BhagatSinh': (22.7420, 75.8830),
    'Lasudiya Moree': (22.7430, 75.8840),
    'Nepaaniya': (22.7440, 75.8850),
    'Saamee Kripa': (22.7450, 75.8860),
    'Haajee Colony': (22.7460, 75.8870),
    'Naaharasha Havelee': (22.7470, 75.8880),
    'Khajaraana': (22.7480, 75.8890),
    'Kailaashapuree': (22.7490, 75.8900),
    'Swami Vivekaanand': (22.7500, 75.8910),
    'Shreenagar': (22.7510, 75.8920),
    'H.I.G.': (22.7520, 75.8930),
    'Dr. Bheemrao Ambedkar': (22.7530, 75.8940),
    'Somanaath': (22.7540, 75.8950),
    'Saradaar Vallabh Bhai': (22.7550, 75.8960),
    'Geeta Bhavan': (22.7560, 75.8970),
    'Tilak Nagar': (22.7570, 75.8980),
    'Brajeshvaree': (22.7580, 75.8990),
    'Maam Bhagavatee Nagar': (22.7590, 75.9000),
    'Musakkhedi': (22.7600, 75.9010),
    'Dr Maulaana Aajaad Nagar': (22.7610, 75.9020),
    'Residency': (22.7620, 75.9030),
    'Saaooth Tukoganj': (22.7630, 75.9040),
    'Snehalata Ganj': (22.7640, 75.9050),
    'Devi Ahilyaabai': (22.7650, 75.9060),
    'Emli Bazaar': (22.7660, 75.9070),
    'Harasiddhee': (22.7670, 75.9080),
    'Ranipuraa': (22.7680, 75.9090),
    'Taatyaa Saravate': (22.7690, 75.9100),
    'Raavajee Baazaar': (22.7700, 75.9110),
    'Navalakha': (22.7710, 75.9120),
    'Chitaavad': (22.7720, 75.9130),
    'Sant Kavar Raam': (22.7730, 75.9140),
    'Shaheed Hemu Kolonee': (22.7740, 75.9150),
    'Mahaaraaja Holakar': (22.7750, 75.9160),
    'Bambaee Baazaar': (22.7760, 75.9170),
    'Jawaahar Maarg': (22.7770, 75.9180),
    'Lok Naayak Nagar': (22.7780, 75.9190),
    'Dravid Nagar': (22.7790, 75.9200),
    'Lok Maanya Nagar': (22.7800, 75.9210),
    'Lakshman Sinh Chauhaan': (22.7810, 75.9220),
    'Vishnupuree': (22.7820, 75.9230),
    'Paalada': (22.7830, 75.9240),
    'Mundlaa Naayta': (22.7840, 75.9250),
    'Billavali': (22.7850, 75.9260),
    'Choithram': (22.7860, 75.9270),
    'Sukhniwas': (22.7870, 75.9280),
    'Dr Rajendra Prasaad': (22.7880, 75.9290),
    'Annapurna': (22.7890, 75.9300),
    'Sudaama Nagar': (22.7900, 75.9310),
    'Gumaastaa Nagar': (22.7910, 75.9320),
    'Dawaarkapuri': (22.7920, 75.9330),
    'Prajaapat Nagar': (22.7930, 75.9340)
}

# Add Latitude and Longitude columns
household_data['Latitude'] = household_data['Ward Name'].map(lambda x: ward_coords.get(x, (None, None))[0])
household_data['Longitude'] = household_data['Ward Name'].map(lambda x: ward_coords.get(x, (None, None))[1])

# Fill missing values for Latitude and Longitude if any
household_data['Latitude'].fillna(22.7196, inplace=True)
household_data['Longitude'].fillna(75.8577, inplace=True)

# Filter out rows with missing or zero coordinates
map_data = household_data.dropna(subset=['Latitude', 'Longitude'])
map_data = map_data[(map_data['Latitude'] != 0) & (map_data['Longitude'] != 0)]

# Load CSS (optional)
with open('carousel.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Sidebar Menu Setup - Ensure it's only called once
with st.sidebar:
    st.image('1.png', width=200)  # Use the uploaded logo path
    selected = option_menu("Main Menu", ["Home", "Data", "Map", "About"], 
        icons=['house', 'database', 'map', 'info'], menu_icon="cast", default_index=0)

# Home Page - Ensure correct placement and layout
if selected == "Home":
    st.title("Water Supply Management System")
    st.write("Welcome to the Water Supply Management System. Use the sidebar to navigate to different sections.")
    
    # Home page content
    st.write("This application provides data analysis and visualization of water usage in various wards of Indore.")
    st.write("Navigate to the Data or Map sections to explore the analysis in more detail.")

# Data Page
elif selected == "Data":
    st.title("Data Overview")
    date_option = st.selectbox("Select date range", ["1 month", "6 months", "1 year"])

    if date_option == "1 month":
        start_date = pd.to_datetime("today") - pd.DateOffset(months=1)
    elif date_option == "6 months":
        start_date = pd.to_datetime("today") - pd.DateOffset(months=6)
    elif date_option == "1 year":
        start_date = pd.to_datetime("today") - pd.DateOffset(years=1)

    end_date = pd.to_datetime("today")

    # Filter data based on date range
    filtered_data = household_data[(household_data['Date'] >= start_date) & (household_data['Date'] <= end_date)]

    wards = filtered_data['Ward Name'].unique()
    selected_ward = st.selectbox("Select Ward", sorted(wards))

    if selected_ward:
        # Filter data based on selected ward
        ward_data = filtered_data[filtered_data['Ward Name'] == selected_ward]

        st.write(f"### Household Data for Ward {selected_ward}", ward_data)

        # Show some calculated statistics
        total_households = len(ward_data)
        households_receiving_water = ward_data['Leakage Detected (Yes/No)'].value_counts().get('No', 0)
        households_not_receiving_water = total_households - households_receiving_water

        used_within_limit = (ward_data['Monthly Water Usage (Liters)'] <= 100).sum()
        wasted_beyond_limit = (ward_data['Monthly Water Usage (Liters)'] > 100).sum()

        total_usage = ward_data['Monthly Water Usage (Liters)'].sum()
        total_wasted = ward_data.loc[ward_data['Monthly Water Usage (Liters)'] > 100, 'Monthly Water Usage (Liters)'].sum() - 100 * wasted_beyond_limit

        mean_usage = ward_data['Monthly Water Usage (Liters)'].mean()
        median_usage = ward_data['Monthly Water Usage (Liters)'].median()
        std_usage = ward_data['Monthly Water Usage (Liters)'].std()

        st.write(f"**Total households in Ward {selected_ward}**: {total_households}")
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
            title=f'Households Receiving vs. Not Receiving Water in Ward {selected_ward}'
        )
        st.plotly_chart(fig)

        # Additional visualization (box plot, heatmap, etc.)
        fig4 = px.box(ward_data, x='Income Level', y='Monthly Water Usage (Liters)', title=f'Water Usage by Income Level in Ward {selected_ward}')
        st.plotly_chart(fig4)

import plotly.express as px

# Map page
elif selected == "Map":
    st.title("Ward Map Overview")
    st.write("This map highlights wards with water disparity and leakage detection issues.")

    # Ensure map data has valid ward and water usage information
    map_data = household_data[['Ward Name', 'Monthly Water Usage (Liters)', 'Leakage Detected (Yes/No)', 'Disparity in Supply (Yes/No)']].drop_duplicates()

    # Check if map_data has any valid rows
    if map_data.empty or map_data['Ward Name'].nunique() == 0:
        st.warning("No data available for mapping. Please check your dataset.")
    else:
        # Create a pivot table to summarize water usage per ward for the heatmap
        heatmap_data = map_data.pivot_table(values='Monthly Water Usage (Liters)', 
                                            index='Ward Name', 
                                            aggfunc='mean', 
                                            fill_value=0)

        # Sort the wards by water usage
        heatmap_data = heatmap_data.sort_values(by='Monthly Water Usage (Liters)', ascending=False)

        # Reset the index to get ward names in a column
        heatmap_data = heatmap_data.reset_index()

        # Create an interactive heatmap using Plotly Express
        fig = px.density_heatmap(heatmap_data, 
                                 x='Ward Name', 
                                 y='Monthly Water Usage (Liters)', 
                                 z='Monthly Water Usage (Liters)',
                                 color_continuous_scale='coolwarm',
                                 labels={'Monthly Water Usage (Liters)': 'Water Usage (Liters)'},
                                 title="Average Monthly Water Usage by Ward")

        # Customize layout and axes
        fig.update_layout(
            xaxis_title="Ward Name",
            yaxis_title="Average Water Usage (Liters)",
            coloraxis_colorbar=dict(title="Water Usage (Liters)"),
            height=600
        )

        # Display the interactive heatmap
        st.plotly_chart(fig)


# About Page
elif selected == "About":
    st.title("About")
    st.write("This application is designed to manage water supply for households. It provides data analysis and predictive modeling for water usage. The system can predict future water usage based on various factors such as household size, days without water, and average temperature.")
