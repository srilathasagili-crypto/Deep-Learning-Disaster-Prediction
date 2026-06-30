import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# Load model (safe loading)
model = tf.keras.models.load_model("model.keras", compile=False)

# Load preprocessor
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

st.title("🌍 Disaster Prediction System")

st.write("Enter comma-separated input values:")

input_values = st.text_input("Example: 1.2, 3.4, 5.6")

if st.button("Predict"):
    try:
        if not input_values:
            st.error("Please enter values")
        else:
            # Convert input safely
            input_data = np.array(
                [float(i.strip()) for i in input_values.split(",")]
            ).reshape(1, -1)

            # Preprocess
            processed = preprocessor.transform(input_data)

            # Predict
            prediction = model.predict(processed)

            # If classification → convert to class
            result = np.argmax(prediction)

            st.success(f"Prediction Class: {result}")
            st.write("Raw Output:", prediction)

    except Exception as e:
        st.error(f"Error: {str(e)}")
