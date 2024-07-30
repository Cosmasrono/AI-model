import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from geopy.geocoders import Nominatim
import requests

# Load datasets (replace with your actual dataset paths)
agricultural_dataset = pd.read_csv('agricultural_dataset.csv')
insect_dataset = pd.read_csv('period.csv')

# Create a column for soil type in the agricultural dataset
agricultural_dataset['soil_type'] = np.nan

# Encode categorical soil type column in the agricultural dataset
soil_types = ['Sandy', 'Loamy', 'Clay', 'Silty', 'Peaty', 'Chalky', 'Unknown']
soil_type_encoder = LabelEncoder()
soil_type_encoder.fit(soil_types)
agricultural_dataset['soil_type'] = agricultural_dataset['soil_type'].fillna('Unknown')
agricultural_dataset['soil_type_encoded'] = soil_type_encoder.transform(agricultural_dataset['soil_type'])

# Update features and labels to include the encoded soil type
X_agriculture = agricultural_dataset.drop(columns=['label', 'soil_type'])
y_agriculture = agricultural_dataset['label']

# Scale the agricultural features
scaler_agriculture = StandardScaler()
X_agriculture_scaled = scaler_agriculture.fit_transform(X_agriculture)

# Train the agricultural model
agricultural_model = RandomForestClassifier(n_estimators=100, random_state=42)
agricultural_model.fit(X_agriculture_scaled, y_agriculture)

# Preprocess insect dataset
severity_encoder = LabelEncoder()
insect_dataset['Severity_Encoded'] = severity_encoder.fit_transform(insect_dataset['Severity'])

# Calculate average severity by crop
crop_severity = insect_dataset.groupby('Crop Affected')['Severity_Encoded'].mean().sort_values()

# Categorize crops based on severity
def categorize_crop(severity):
    if severity < 0.4:
        return "Low Risk"
    elif severity < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"

crop_risk = crop_severity.apply(categorize_crop)

# Function to detect location
def get_location():
    try:
        ip_address = requests.get('https://api64.ipify.org?format=json').json()['ip']
        response = requests.get(f'https://ipinfo.io/{ip_address}/json').json()
        loc = response['loc'].split(',')
        latitude, longitude = loc[0], loc[1]
        return latitude, longitude
    except Exception as e:
        st.error(f"Error detecting location: {str(e)}")
        return None, None

# Function to get location details
def get_location_details(latitude, longitude):
    try:
        geolocator = Nominatim(user_agent="crop_recommendation")
        location = geolocator.reverse(f"{latitude}, {longitude}")
        return location.address
    except Exception as e:
        st.error(f"Error getting location details: {str(e)}")
        return None

# Function to recommend crops
def recommend_crops(N, P, K, temperature, humidity, ph, rainfall, soil_type):
    latitude, longitude = get_location()
    if latitude and longitude:
        location = get_location_details(latitude, longitude)
    else:
        location = st.text_input('Location', 'Location could not be detected, please enter manually')
    
    # Encode the soil type
    soil_type_encoded = soil_type_encoder.transform([soil_type])[0]
    
    # Prepare input features for agricultural model
    input_features_agriculture = np.array([[N, P, K, temperature, humidity, ph, rainfall, soil_type_encoded]])
    input_features_agriculture_scaled = scaler_agriculture.transform(input_features_agriculture)
    
    # Predict probabilities for all crops
    probabilities = agricultural_model.predict_proba(input_features_agriculture_scaled)[0]
    crop_probabilities = list(zip(agricultural_model.classes_, probabilities))
    crop_probabilities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top two crop recommendations
    top_two_crops = crop_probabilities[:2]
    
    recommendations = {}
    for crop, _ in top_two_crops:
        if crop in crop_risk.index:
            pests_info = insect_dataset[insect_dataset['Crop Affected'].str.contains(crop, na=False)]
            recommendations[crop] = pests_info
        else:
            recommendations[crop] = pd.DataFrame()
    
    return recommendations

# Streamlit app
st.title("AI Crop Recommendation System")

# Input sliders for environmental factors
N = st.slider('Nitrogen', min_value=0.0, max_value=150.0, value=20.0)
P = st.slider('Phosphorus', min_value=0.0, max_value=150.0, value=20.0)
K = st.slider('Potassium', min_value=0.0, max_value=200.0, value=40.0)
temperature = st.slider('Temperature (°C)', min_value=0.0, max_value=50.0, value=25.0)
humidity = st.slider('Humidity (%)', min_value=0.0, max_value=100.0, value=60.0)
ph = st.slider('pH', min_value=0.0, max_value=14.0, value=6.0)
rainfall = st.slider('Rainfall (mm)', min_value=10.0, max_value=300.0, value=100.0)
soil_type = st.selectbox('Soil Type', ['Sandy', 'Loamy', 'Clay', 'Silty', 'Peaty', 'Chalky'])

# Button to trigger recommendation
if st.button("Recommend Crops"):
    recommendations = recommend_crops(N, P, K, temperature, humidity, ph, rainfall, soil_type)
    
    if recommendations:
        for crop, pests_info in recommendations.items():
            st.write(f"Recommended Crop: {crop}")
            
            if not pests_info.empty:
                st.subheader(f"Pest Conditions for {crop}")
                st.write(pests_info[['Insect Name', 'Start Period', 'End Period', 'Region', 'Severity']])
            else:
                st.write(" ")
    else:
        st.write(" ")

# Display additional information
st.subheader("Additional Information")

# Most affected crops by insect severity
most_affected_crops = crop_severity.tail(5)  # Adjust number as needed
st.write("Top 5 Most Affected Crops by Insect Severity:")
st.write(most_affected_crops)

# Most affected regions
region_severity = insect_dataset.groupby('Region')['Severity_Encoded'].mean().sort_values(ascending=False)
st.write("Top 5 Most Affected Regions:")
st.write(region_severity.head())

# Most common insects
insect_counts = insect_dataset['Insect Name'].value_counts()
st.write("Top 5 Most Common Insects:")
st.write(insect_counts.head())
