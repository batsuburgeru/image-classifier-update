from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask import Flask
from classifier_model import predict_animal, image_classify
from flask_cors import CORS

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
    
    model = image_classify()
    # Pass total_val as an argument to predict_animal function
    predicted_class, loss, accuracy, val_loss, val_accuracy, precision, recall, f1 = predict_animal(model, image_path)
    
    os.remove(image_path)

    response = {
        'Predicted_Class': predicted_class,
        'Accuracy_Result': float(accuracy),
        'Val_Accuracy': float(val_accuracy),
        'F1_Score': float(precision),
        'Precision_Result': float(recall),
        'Recall_Result': float(f1),
        'Loss': float(loss),
        'Val_Loss': float(val_loss),
        'Val_Accuracy': float(val_accuracy)
    }

    return jsonify(response), 200

@app.route('/correctClassification', methods=['POST'])
def correct_classification():
    if 'image' not in request.files:
        return 'No image part in the request', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    predicted_class = request.form.get('predictedClass').upper()

    directory = './src/Experiment1/train/' + predicted_class
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get a list of existing files in the directory
    existing_files = os.listdir(directory)

    # Extract the existing numbers from filenames
    existing_numbers = []
    for filename in existing_files:
        if filename.startswith(predicted_class):
            try:
                number = int(filename.split('_')[1].split('.')[0])
                existing_numbers.append(number)
            except ValueError:
                pass  # Ignore filenames that don't match the expected format

    # Find the next available number
    next_number = max(existing_numbers, default=0) + 1

    # Construct the new filename
    new_filename = f"{predicted_class}_{next_number}.jpg"  # Adjust the file extension as needed

    image_path = os.path.join(directory, new_filename)
    file.save(image_path)

    return 'Image saved successfully', 200

@app.route('/wrongClassification', methods=['POST'])
def wrong_classification():
    if 'image' not in request.files:
        return 'No image part in the request', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    correctedClass = request.form.get('correctedClass').upper()

    directory = './src/Experiment1/train/' + correctedClass
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get a list of existing files in the directory
    existing_files = os.listdir(directory)

    # Extract the existing numbers from filenames
    existing_numbers = []
    for filename in existing_files:
        if filename.startswith(correctedClass):
            try:
                number = int(filename.split('_')[1].split('.')[0])
                existing_numbers.append(number)
            except ValueError:
                pass  # Ignore filenames that don't match the expected format

    # Find the next available number
    next_number = max(existing_numbers, default=0) + 1

    # Construct the new filename
    new_filename = f"{correctedClass}_{next_number}.jpg"  # Adjust the file extension as needed

    image_path = os.path.join(directory, new_filename)
    file.save(image_path)

    return 'Image saved successfully', 200

if __name__ == "__main__":
    app.run(debug=True)