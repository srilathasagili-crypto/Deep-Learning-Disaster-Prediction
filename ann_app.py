import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load model
model = tf.keras.models.load_model("model.keras")

# Load preprocessor
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

st.title("🌍 Disaster Prediction System")

st.write("Enter input values below:")

# Example: adjust based on your features
input_values = st.text_input("Enter comma-separated values")

if st.button("Predict"):
    try:
        # Convert input
        input_data = np.array([float(i) for i in input_values.split(",")]).reshape(1, -1)

        # Preprocess
        processed = preprocessor.transform(input_data)

        # Predict
        prediction = model.predict(processed)

        st.success(f"Prediction: {prediction.tolist()}")

    except Exception as e:
        st.error(str(e))
