from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask import Flask
from classifier_model import predict_animal
from flask_cors import CORS
from tensorflow.keras.models import load_model


app = Flask(__name__)
CORS(app, origins="*")

# Load the trained model
model = load_model('./model_saved/model.h5')

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


if __name__ == "__main__":
    app.run(debug=True)