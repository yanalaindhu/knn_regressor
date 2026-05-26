import streamlit as st
import numpy as np
import pickle

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
model = pickle.load(
    open("models/knn_model.pkl", "rb")
)

# ---------------------------------------------------
# LOAD SCALER
# ---------------------------------------------------
scaler = pickle.load(
    open("models/scaler.pkl", "rb")
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Insurance Prediction App"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("Insurance Charges Prediction")

st.write("Machine Learning Project using KNN Regressor")

# ---------------------------------------------------
# USER INPUTS
# ---------------------------------------------------

age = st.number_input(
    "Enter Age",
    min_value=1,
    max_value=100,
    value=25
)

sex = st.selectbox(
    "Select Gender",
    ["male", "female"]
)

bmi = st.number_input(
    "Enter BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=0
)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Select Region",
    [
        "southwest",
        "southeast",
        "northwest",
        "northeast"
    ]
)

# ---------------------------------------------------
# ENCODE INPUTS
# ---------------------------------------------------

sex_value = 1 if sex == "male" else 0

smoker_value = 1 if smoker == "yes" else 0

region_dict = {
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
}

region_value = region_dict[region]

# ---------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------

if st.button("Predict Insurance Charges"):

    input_data = np.array([[
        age,
        sex_value,
        bmi,
        children,
        smoker_value,
        region_value
    ]])

    # SCALE INPUT
    input_scaled = scaler.transform(input_data)

    # PREDICTION
    prediction = model.predict(input_scaled)

    # DISPLAY RESULT
    st.success(
        f"Predicted Insurance Charges = ${prediction[0]:.2f}"
    )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.write("--------------------------------")

st.write("KNN Regressor Insurance Project")