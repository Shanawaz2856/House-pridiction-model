import streamlit as st
import pickle
import json
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ── Load model & columns ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("banglore_home_prices_model.pickle", "rb") as f:
        model = pickle.load(f)
    with open("columns.json", "r") as f:
        data = json.load(f)
    columns = data["data_columns"]
    locations = [col for col in columns if col not in ("total_sqft", "bath", "bhk")]
    return model, columns, locations

model, data_columns, locations = load_artifacts()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏠 Bengaluru House Price Predictor")
st.markdown("Enter the property details below to get an estimated price.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("📍 Location", sorted(locations))
    sqft = st.number_input("📐 Total Square Feet", min_value=100, max_value=10000, value=1000, step=50)

with col2:
    bhk = st.number_input("🛏️ BHK (Bedrooms)", min_value=1, max_value=10, value=2)
    bath = st.number_input("🚿 Bathrooms", min_value=1, max_value=10, value=2)

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(np.array(data_columns) == location.lower())[0]
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if len(loc_index) > 0:
        x[loc_index[0]] = 1
    return round(model.predict([x])[0], 2)

if st.button("🔍 Predict Price", use_container_width=True, type="primary"):
    price = predict_price(location, sqft, bath, bhk)
    st.success(f"### Estimated Price: ₹ {price:.2f} Lakhs")
    st.caption(f"Property: {bhk} BHK | {sqft} sqft | {bath} Bath | {location}")

st.markdown("---")
st.caption("Model: Linear Regression | Dataset: Bengaluru House Data")
