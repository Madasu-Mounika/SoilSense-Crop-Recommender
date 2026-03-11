import streamlit as st
import numpy as np
import pickle
import requests
import tensorflow as tf
from PIL import Image
from geopy.geocoders import Nominatim

st.set_page_config(page_title="AgriVision: Precision Geo-ML", layout="wide")

WEATHER_API_KEY = "67aee13f4971f8a79fe9316af8aea8ce"

@st.cache_resource
def load_models():
    try:
        c_model = pickle.load(open('crop_model.pkl', 'rb'))
        s_model = tf.keras.models.load_model('soil_classifier.h5')
        return c_model, s_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

crop_model, soil_model = load_models()

geo_map = {'Coastal': 0, 'Plains': 1, 'Inland': 2, 'Semi-Arid': 3, 
           'Arid': 4, 'Plateau': 5, 'Tropical': 6, 'Highland': 7}

def get_geo_and_weather(city):
    geolocator = Nominatim(user_agent="agrivision_pro_2026")
    location = geolocator.geocode(city)
    if location:
        lat, lon = location.latitude, location.longitude
        alt_url = f"https://api.opentopodata.org/v1/aster30m?locations={lat},{lon}"
        alt_resp = requests.get(alt_url).json()
        altitude = alt_resp['results'][0]['elevation'] if 'results' in alt_resp else 250
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        w_data = requests.get(weather_url).json()
        temp = w_data['main']['temp'] if 'main' in w_data else 28.0
        humidity = w_data['main']['humidity'] if 'main' in w_data else 70.0
        rain = w_data.get('rain', {}).get('1h', 0) * 100 + 800 
        if altitude > 800: g_type = "Highland"
        elif altitude < 50 and humidity > 75: g_type = "Coastal"
        elif rain < 600: g_type = "Semi-Arid"
        elif lat < 15 and rain > 1500: g_type = "Tropical"
        elif 300 <= altitude <= 800: g_type = "Plateau"
        else: g_type = "Inland"
        return temp, humidity, rain, altitude, g_type
    return 28.0, 70.0, 1000, 200, "Inland"

st.title("🌾Hybrid Soil Analysis & Crop Recommendation System")
st.sidebar.header("📍 Geographic Details")
city = st.sidebar.text_input("Enter Farmer's City", "Warangal")
temp, hum, rain, alt, gtype = get_geo_and_weather(city)

st.sidebar.success(f"**Location Found!**\n\nZone: {gtype} | Alt: {int(alt)}m")
st.sidebar.info(f"**Live Weather:**\n\n🌡️ {temp}°C | 💧 {hum}% Humidity")

col1, col2 = st.columns(2)
with col1:
    st.subheader("📸 Step 1: Soil Texture Analysis")
    uploaded_file = st.file_uploader("Upload Soil Photo", type=["jpg", "png", "jpeg"])
    
    # Define your soil classes (Ensure these match the order of your training folders!)
    soil_classes = ['Alluvial Soil', 'Arid_Soil', 'Black_Soil', 'Laterite_Soil', 'Mountain_Soil','Red_Soil', 'Yellow_Soil']
    
    detected_soil = "Unknown" # Default before upload
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
        st.image(img, width=300, caption="Sampled Soil")
        
        # --- CNN PREDICTION LOGIC ---
        if soil_model is not None:
            # Prepare image for the model
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Get prediction
            prediction_probs = soil_model.predict(img_array)
            predicted_class_index = np.argmax(prediction_probs)
            detected_soil = soil_classes[predicted_class_index]
            
            st.info(f"🔍 **Detected Soil Type:** {detected_soil}")
        else:
            st.error("Soil model not loaded correctly.")
            

with col2:
    st.subheader("🧪 Step 2: Nutrient Input")
    n = st.slider("Nitrogen (N)", 0, 150, 70)
    p = st.slider("Phosphorus (P)", 0, 150, 45)
    k = st.slider("Potassium (K)", 0, 150, 40)
    ph = st.slider("Soil pH Level", 0.0, 14.0, 6.5)

if st.button("🚀Get Recommendation"):
    g_code = geo_map.get(gtype, 2)
    features = np.array([[n, p, k, temp, hum, ph, rain, alt, g_code]])
    if crop_model:
        prediction = crop_model.predict(features)[0].upper()
        st.balloons()
        st.markdown(f"""
            <div style="background-color:#1b5e20; padding:40px; border-radius:15px; text-align:center;">
                <h3 style="color:white;">Recommended Crop for {city.title()}</h3>
                <h1 style="color:white; font-size:70px;">{prediction}</h1>
                <p style="color:#c8e6c9;">Matched for <b>{gtype}</b> conditions at <b>{int(alt)}m</b> altitude.</p>
            </div>
        """, unsafe_allow_html=True)
