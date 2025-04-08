import streamlit as st
from streamlit_option_menu import option_menu
import base64
import numpy as np
import joblib
import time  # For delay
import streamlit.components.v1 as components  # For animation

st.set_page_config(page_title="Cardio Estimator", page_icon="🫀")

# Load Model and Scaler
model = joblib.load("cvd_model.pkl")
scaler = joblib.load("scaler.pkl")

def add_bg_image(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def home():
    add_bg_image("2.jpg")
    st.title("Welcome to Cardiovascular Disease Estimator")
    st.write("An AI-based tool to estimate the risk  cardiovascular diseases based on input parameters.")

def about():
    add_bg_image("2.jpg")
    st.title("About Us")
    developers = [
        {"name": "Ayush Gupta", "role": "Team Leader and SVM and UI Designer", "bio": "Trained Model on Support Vector Classifier Algorithm and designed UI of the website using Streamlit."},
        {"name": "Ayush Joshi", "role": "Dataset Explorer", "bio": "Explored various datasets and selected important features to be considered for estimation."},
        {"name": "Ayush Pandey", "role": "Naive Bayes Expert", "bio": "Trained Model on Naive Bayes Algorithm"},
        {"name": "Harsh Chaurasiya", "role": "Data Analyst", "bio": "Generated graphs and performed numerical data analysis for selected dataset and analysed various patterns among different features in the dataset."},
        {"name": "Keshav", "role": "Algorithm Master", "bio": "Trained Model on Random Forest and XG Boost Algorithm and selected best out of all algorithms to make the model as accurate as possible."}
    ]
    
    for dev in developers:
        st.markdown(f"""
            <div style="text-align: center; padding: 10px;">
                <h3>{dev['name']}</h3>
                <p style="font-size:24px; font-weight:bold; color:#ddd;">Role: {dev['role']}</p>
                <p style="font-size:18px; color:#ddd;">{dev['bio']}</p>
                <hr style="border: 1px solid #ddd;">
            </div>
        """, unsafe_allow_html=True)

def predictor():
    add_bg_image("2.jpg")
    st.title("Cardiovascular Disease Estimator")
    st.write("Enter patient details to predict cardiovascular disease risk.")

    # Input fields for all model features
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Gender", ["Male", "Female"])
    chestpain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    restingbp = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    serumcholestrol = st.number_input("Serum Cholesterol", min_value=100, max_value=400, value=200)
    fastingbloodsugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
    restingrelectro = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
    maxheartrate = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exerciseangia = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
    noofmajorvessels = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
    
    # Mappings for categorical variables
    sex_map = {"Male": 1, "Female": 0}
    chestpain_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }
    fbs_map = {"True": 1, "False": 0}
    restecg_map = {
        "Normal": 0,
        "ST-T Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    exang_map = {"Yes": 1, "No": 0}
    slope_map = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }

    # Apply mappings
    sex = sex_map[sex]
    chestpain = chestpain_map[chestpain]
    fastingbloodsugar = fbs_map[fastingbloodsugar]
    restingrelectro = restecg_map[restingrelectro]
    exerciseangia = exang_map[exerciseangia]
    slope = slope_map[slope]
    
    # Prepare input array
    input_data = np.array([[age, sex, chestpain, restingbp, serumcholestrol, fastingbloodsugar,
                            restingrelectro, maxheartrate, exerciseangia, oldpeak, slope,
                            noofmajorvessels]])

    # Apply standard scaler to numeric columns only
    input_data[:, [0, 3, 4, 7, 9]] = scaler.transform(input_data[:, [0, 3, 4, 7, 9]])

    if st.button("Estimate"):
        with st.spinner("🧠 AI is analyzing your data... Please wait..."):
            time.sleep(2)

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.error("⚠️ High Risk: Cardiovascular Disease Detected!")
            components.html(
                """<div style="text-align: center;">
                <lottie-player src="https://assets1.lottiefiles.com/packages/lf20_s6pfu4by.json" 
                background="transparent" speed="1" style="width: 200px; height: 200px;" loop autoplay></lottie-player>
                </div>""",
                height=220
            )
        else:
            st.success("✅ Low Risk: No Cardiovascular Disease Detected.")
            components.html(
                """<div style="text-align: center;">
                <lottie-player src="https://assets10.lottiefiles.com/packages/lf20_msdmfngy.json" 
                background="transparent" speed="1" style="width: 200px; height: 200px;" loop autoplay></lottie-player>
                </div>""",
                height=220
            )

st.sidebar.title("Menu")
page = st.sidebar.radio("", ["Home", "About", "Estimator"])

if page == "Home":
    home()
elif page == "About":
    about()
elif page == "Estimator":
    predictor()

