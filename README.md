#🌾 Hybrid Soil Analysis & Crop Recommendation System
An intelligent AI-powered agriculture support system that analyzes soil images and environmental conditions to recommend the most suitable crops for farmers.
The system combines Computer Vision, Machine Learning, and Geographic Data to make accurate crop recommendations.

# 🚀Features
 -Soil Image Analysis using a CNN deep learning model
 -Soil Nutrient Input (Nitrogen, Phosphorus, Potassium, pH)
 -Location-based Weather Data
 -Geographic Zone Detection
 -Crop Recommendation using Machine Learning
 -Interactive Streamlit Web Interface

 # 🧠 Technologies Used
 - Python
- Streamlit
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Geopy
- OpenWeatherMap API
- OpenTopoData API
- Pillow

  # 📂 Project Structure
SoilSense-Crop-Recommender/
│
├── app.py                  
├── crop_model.pkl         
├── soil_classifier.h5     
├── requirements.txt        
├── screenshots/            
└── README.md

⚙️ Installation

1️⃣ Clone the Repository
git clone https://github.com/yourusername/repositoryname.git
cd repositoryname
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Application
streamlit run app.py

#  How the System Works

1️⃣ User enters the **city name** and the system fetches weather, altitude, and geographic zone data.
2️⃣ User uploads **one soil image**, and a CNN model detects the soil type.
3️⃣ User inputs **soil nutrients (N, P, K) and pH values**.
4️⃣ The machine learning model analyzes all parameters and **recommends the most suitable crop**.

#  Machine Learning Models

### Soil Classification Model
- Model: Convolutional Neural Network (CNN)
- Framework: TensorFlow / Keras
- Input: Soil image (224 × 224)
- Output: Soil type category
### Crop Recommendation Model
- Model: Supervised Machine Learning Model
- Input Features:
  - Nitrogen
  - Phosphorus
  - Potassium
  - Temperature
  - Humidity
  - pH
  - Rainfall
  - Altitude
  - Geographic zone
- Output: Recommended crop

# 🌍 Applications
- Smart farming
- Precision agriculture
- Crop planning systems
- Farmer advisory platforms
# 🔮 Future Improvements
- Fertilizer recommendation system
- Crop yield prediction
- Soil nutrient detection using images
- Satellite-based soil analysis

