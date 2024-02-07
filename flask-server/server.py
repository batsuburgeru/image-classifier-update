from flask import Flask, request
from werkzeug.utils import secure_filename
import os
from flask import Flask
from classifier_model import predict_animal
from flask_cors import CORS
import shutil

app = Flask(__name__)
CORS(app, origins="*")

@app.route('/')
def index():
    return 'Server is running'

@app.route('/classifyImage', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return 'No image part in the request', 400
    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400
    
    filename = secure_filename(file.filename)
    image_path = os.path.join('./src/test-img', filename)
    file.save(image_path)
    # Now you can open this image and perform your classification on it
    # After classification, you can send back the results

    image_path = os.path.join('./src/test-img', filename)
    prediction = predict_animal(image_path)
    os.remove(image_path)

    return prediction, 200

if __name__ == "__main__":
    app.run(debug=True)