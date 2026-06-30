import gradio as gr
import numpy as np
import pickle
import tensorflow as tf

model = tf.keras.models.load_model("model.keras", compile=False)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

def predict(input_text):
    data = np.array([float(i) for i in input_text.split(",")]).reshape(1, -1)
    processed = preprocessor.transform(data)
    prediction = model.predict(processed)
    return f"Prediction Class: {np.argmax(prediction)}"

demo = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="🌍 Disaster Prediction System"
)

demo.launch()
