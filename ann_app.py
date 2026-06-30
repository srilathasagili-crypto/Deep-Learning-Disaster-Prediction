import streamlit as st
import numpy as np
import pickle

# Load model
model = tf.keras.models.load_model("model.keras", compile=False)

# Load preprocessor
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Feature names
feature_names = [
    "disaster_type",
    "latitude",
    "longitude",
    "severity_level",
    "affected_population",
    "estimated_economic_loss_usd",
    "response_time_hours",
    "aid_provided",
    "infrastructure_damage_index"
]

st.title("🌍 Disaster Prediction System")
st.write("Enter details below to predict whether it is a major disaster or not.")

# Input UI
inputs = []

for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    inputs.append(value)

# Predict button
if st.button("Predict"):
    try:
        data = np.array(inputs).reshape(1, -1)

        processed = preprocessor.transform(data)

        prediction = model.predict(processed)

        result = np.argmax(prediction)

        if result == 1:
            st.error("🚨 Major Disaster Predicted")
        else:
            st.success("✅ Not a Major Disaster")

    except Exception as e:
        st.error(f"Error: {str(e)}")
