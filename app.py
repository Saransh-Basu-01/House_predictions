# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

# ---------------- STYLING ---------------- #
st.markdown("""
<style>
.main {
    background-color: #0f172a;
    color: white;
}

.stApp {
    background: linear-gradient(to right, #0f172a, #1e293b);
}

h1, h2, h3 {
    color: white;
}

.prediction-box {
    background-color: #1e293b;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #334155;
}

.metric-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #374151;
}

div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #
st.title("🏠 California House Price Predictor")
st.markdown(
    "Predict median house prices using Machine Learning with a clean Streamlit UI."
)

st.divider()

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("📊 About Model")

st.sidebar.info("""
Model Used:
- Random Forest Regressor

Preprocessing:
- Median Imputation
- Standard Scaling
- OneHot Encoding

Built using:
- Scikit-Learn
- Streamlit
""")

# ---------------- INPUT SECTION ---------------- #
st.subheader("📥 Enter House Details")

col1, col2, col3 = st.columns(3)

with col1:
    longitude = st.number_input(
        "Longitude",
        value=-122.23,
        format="%.5f"
    )

    latitude = st.number_input(
        "Latitude",
        value=37.88,
        format="%.5f"
    )

    housing_median_age = st.slider(
        "Housing Median Age",
        1, 100, 20
    )

with col2:
    total_rooms = st.number_input(
        "Total Rooms",
        min_value=1,
        value=880
    )

    total_bedrooms = st.number_input(
        "Total Bedrooms",
        min_value=1,
        value=129
    )

    population = st.number_input(
        "Population",
        min_value=1,
        value=322
    )

with col3:
    households = st.number_input(
        "Households",
        min_value=1,
        value=126
    )

    median_income = st.number_input(
        "Median Income",
        min_value=0.0,
        value=8.3252,
        format="%.4f"
    )

    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        [
            "<1H OCEAN",
            "INLAND",
            "ISLAND",
            "NEAR BAY",
            "NEAR OCEAN"
        ]
    )

st.divider()

# ---------------- PREDICTION ---------------- #
if st.button("🚀 Predict House Price"):

    input_data = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]
    })

    transformed_data = pipeline.transform(input_data)
    prediction = model.predict(transformed_data)[0]

    st.markdown("## 📈 Prediction Result")

    st.markdown(f"""
    <div class="prediction-box">
        <h1>💰 ${prediction:,.2f}</h1>
        <p>Estimated Median House Value</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Additional Insights
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏡 Rooms</h3>
            <h2>{total_rooms}</h2>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👨‍👩‍👦 Population</h3>
            <h2>{population}</h2>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🌊 Ocean Proximity</h3>
            <h2>{ocean_proximity}</h2>
        </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #
st.divider()

st.caption("Made with ❤️ using Streamlit and Machine Learning")