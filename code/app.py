from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import nbformat

app = Flask(__name__)
model = VGG16(weights='imagenet')  # Load pretrained VGG16 model
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the VGG16 model from Jupyter Notebook
def load_vgg16_from_notebook(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook_content = nbformat.read(f, as_version=4)
    # Extract relevant code (if needed, execute it dynamically)
    return model  # Placeholder, assuming we use a pre-trained model

@app.route('/')
def home():
    return render_template('index.html')  # Serve HTML frontend

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    image = load_img(file_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    
    preds = model.predict(image_array)
    decoded_preds = decode_predictions(preds, top=3)[0]
    
    results = [{'label': label, 'probability': float(prob)} for (_, label, prob) in decoded_preds]
    
    return jsonify({'predictions': results, 'image_url': file_path})

if __name__ == '__main__':
    app.run(debug=True)
