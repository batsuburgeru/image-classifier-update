# Importing the packages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

# Define constants
IMG_HEIGHT = 255
IMG_WIDTH = 255
tr_batch_size = 30
val_batch_size = 10
epochs = 25

# Check if the model file exists
model_file = './model_saved/model.h5'
if os.path.exists(model_file):
    # Load the trained model
    model = load_model(model_file)
    print('Loaded pre-trained model.')
    history_file = './model_saved/history.pkl'
    with open(history_file, 'rb') as f:
        history = pickle.load(f)
else:
    # Path to your local zip file
    database = './src/Experiment1'

    # Assigning variable names for the training and validation set
    train_dir = os.path.join(database, 'train')
    validation_dir = os.path.join(database, 'validation')

    # Directory with training pig pictures
    train_pig_dir = os.path.join(train_dir, 'pig')
    # Directory with training giraffe pictures
    train_giraffe_dir = os.path.join(train_dir, 'giraffe')
    # Directory with training moose pictures
    train_moose_dir = os.path.join(train_dir, 'moose')

    # Directory with validation pig pictures
    validation_pig_dir = os.path.join(validation_dir, 'pig')
    # Directory with validation giraffe pictures
    validation_giraffe_dir = os.path.join(validation_dir, 'giraffe')
    # Directory with validation moose pictures
    validation_moose_dir = os.path.join(validation_dir, 'moose')

    # Understanding the data
    num_pig_tr = len(os.listdir(train_pig_dir))
    num_giraffe_tr = len(os.listdir(train_giraffe_dir))
    num_moose_tr = len(os.listdir(train_moose_dir))
    num_pig_val = len(os.listdir(validation_pig_dir))
    num_giraffe_val = len(os.listdir(validation_giraffe_dir))
    num_moose_val = len(os.listdir(validation_moose_dir))

    # Total number of training and validation images
    total_train = num_pig_tr + num_giraffe_tr + num_moose_tr
    total_val = num_pig_val + num_giraffe_val + num_moose_val

    print('total training pig images:', num_pig_tr)
    print('total training giraffe images:', num_giraffe_tr)
    print('total training moose images:', num_moose_tr)
    print('total validation pig images:', num_pig_val)
    print('total validation giraffe images:', num_giraffe_val)
    print('total validation moose images:', num_moose_val)
    print("--")
    print("Total training images:", total_train)
    print("Total validation images:", total_val)

    # Setting variables for preprocessing and training the network
    tr_batch_size = 30
    val_batch_size = 10
    epochs = 25
    IMG_HEIGHT = 255
    IMG_WIDTH = 255

    # Prepping the data
    # Generator for our training data
    train_image_generator = ImageDataGenerator(rescale=1./255)
    # Generator for our validation data
    validation_image_generator = ImageDataGenerator(rescale=1./255)

    train_data_gen = train_image_generator.flow_from_directory(
        batch_size=tr_batch_size,
        directory=train_dir,
        shuffle=True,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='categorical',  # Use categorical for more than two classes
        classes=['pig', 'giraffe', 'moose']  # Specify class labels
    )

    val_data_gen = validation_image_generator.flow_from_directory(
        batch_size=val_batch_size,
        directory=validation_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='categorical',
        classes=['pig', 'giraffe', 'moose']
    )

    #Visualizing the training images
    sample_training_images, _ = next(train_data_gen)
    # This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
    def plotImages(images_arr):
        fig, axes = plt.subplots(1, 5, figsize=(20,20))
        axes = axes.flatten()
        for img, ax in zip( images_arr, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    #plotImages(sample_training_images[:5])

    
    #Creating the model
    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(3, activation='softmax')  # Change 1 to 3 for multi-class classification
    ])

    #Compiling the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    #Training the model - this takes quite a bit of time
    history = model.fit(
        train_data_gen,
        steps_per_epoch=max(1, total_train // tr_batch_size),
        epochs=epochs,
        validation_data=val_data_gen,
        validation_steps=max(1, total_val // val_batch_size)
    )

    # Save the trained model to disk
    model.save(model_file)
    print('Trained model saved.')

    history_file = './model_saved/history.pkl'
    with open(history_file, 'wb') as f:
        pickle.dump(history.history, f)
    print('Training history saved.')
    with open(history_file, 'rb') as f:
        history = pickle.load(f)


#Visualizing the training results
def plot_training_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

# Visualizing the new model
#plot_training_results(history)

# Function to preprocess the input image
def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Normalize the image
    img_array = img_array / 255.0

    return img_array

# Function to make predictions
def predict_animaltest(model, img_path, class_labels=["pig", "giraffe", "moose"]):
    processed_img = preprocess_image(img_path)
    predictions = model.predict(processed_img)
    class_labels = ['pig', 'giraffe', 'moose']
    predicted_class_index = tf.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_index]

    val_batch_size = 10
    IMG_HEIGHT = 255
    IMG_WIDTH = 255
    database = './src/Experiment1'

    validation_dir = os.path.join(database, 'validation')

    # Generator for validation data
    validation_image_generator = ImageDataGenerator()

    val_data_gen = validation_image_generator.flow_from_directory(
        batch_size=val_batch_size,
        directory=validation_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='categorical',
        classes=['pig', 'giraffe', 'moose']
    )
    val_data_gen.reset()
    # Get true labels
    y_true = val_data_gen.labels
    # Generate predictions on the validation data
    y_pred = model.predict(val_data_gen)

    # Convert predictions to class labels
    y_pred_classes = np.argmax(y_pred, axis=1)

    report = classification_report(y_true, y_pred_classes, output_dict=True)

    accuracy = report['accuracy']*100
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
        

    return predicted_class, accuracy, precision, recall, f1

def predict_animal(model, img_path, class_labels=["pig", "giraffe", "moose"]):
    processed_img = preprocess_image(img_path)
    predictions = model.predict(processed_img)

    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_index]

    # Get metrics at the final epoch
    final_epoch = epochs - 1
    val_batch_size = 10
    IMG_HEIGHT = 255
    IMG_WIDTH = 255
    database = './src/Experiment1'
    # Path to the history pickle file
    history_file = './model_saved/history.pkl'

    # Load the history object from the pickle file
    with open(history_file, 'rb') as f:
        history = pickle.load(f)

    validation_dir = os.path.join(database, 'validation')

    # Generator for validation data
    validation_image_generator = ImageDataGenerator()

    val_data_gen = validation_image_generator.flow_from_directory(
        batch_size=val_batch_size,
        directory=validation_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='categorical',
        classes=['pig', 'giraffe', 'moose']
    )
    # Evaluating the model
    val_data_gen.reset()  # Reset the validation generator to the beginning
    y_true = val_data_gen.classes
    y_pred = model.predict(val_data_gen)

    # Convert predictions to class labels
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate additional metrics
    loss = history['loss'][final_epoch]
    accuracy = history['accuracy'][final_epoch]
    val_loss = history['val_loss'][final_epoch]
    val_accuracy = history['val_accuracy'][final_epoch]

    # Calculate precision, recall, and f1-score
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')
    f1 = f1_score(y_true, y_pred_classes, average='weighted')

    # Return data as separate values
    return predicted_class, loss, accuracy, val_loss, val_accuracy, precision, recall, f1

#Sample Image Input Test
#test_image_path ="./src/test-img/37.jpg"
#prediction = predict_animal(model, test_image_path)
#print(f"The model predicts: {prediction}")