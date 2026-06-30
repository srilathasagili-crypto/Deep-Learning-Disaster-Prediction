from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model.keras")

# Load preprocessor
with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

@app.route("/")
def home():
    return "ANN Model API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert input to array (assumes JSON list of features)
        input_data = np.array(data["input"]).reshape(1, -1)

        # Preprocess
        processed_data = preprocessor.transform(input_data)

        # Prediction
        prediction = model.predict(processed_data)

        # If classification (binary/multiclass)
        result = prediction.tolist()

        return jsonify({
            "prediction": result
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)