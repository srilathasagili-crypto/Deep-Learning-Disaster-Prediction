import gradio as gr
import numpy as np
import pickle
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("model.keras", compile=False)

# Load preprocessor
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# IMPORTANT: match these with your dataset
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

# Prediction function
def predict(*inputs):
    try:
        # convert input to numpy array
        data = np.array(inputs).reshape(1, -1)

        # preprocessing
        processed = preprocessor.transform(data)

        # prediction
        prediction = model.predict(processed)

        return f"🌍 Prediction Class: {np.argmax(prediction)}"

    except Exception as e:
        return f"❌ Error: {str(e)}"


# UI
demo = gr.Interface(
    fn=predict,
    inputs=[gr.Number(label=name) for name in feature_names],
    outputs="text",
    title="🌍 Disaster Prediction System",
    description="Enter disaster details and click Predict"
)

demo.launch()
