# Importing the packages
import zipfile
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt

# Path to your local zip file
zip_file_path = '/Users/Joshua/OneDrive/Documents/School Files/T.I.P. A.Y. 2023-2024/2nd Sem/Data Science 3/CNN/Experiment1.zip'

# Extract the contents of the zip file
extracted_folder_path = '/Users/Joshua/OneDrive/Documents/School Files/T.I.P. A.Y. 2023-2024/2nd Sem/Data Science 3/CNN/Experiment1'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)

# Assigning variable names for the training and validation set
train_dir = os.path.join(extracted_folder_path, 'train')
validation_dir = os.path.join(extracted_folder_path, 'validation')

# Directory with training bison pictures
train_bison_dir = os.path.join(train_dir, 'bison')
# Directory with training buffalo pictures
train_buffalo_dir = os.path.join(train_dir, 'buffalo')
# Directory with training moose pictures
train_moose_dir = os.path.join(train_dir, 'moose')

# Directory with validation bison pictures
validation_bison_dir = os.path.join(validation_dir, 'bison')
# Directory with validation buffalo pictures
validation_buffalo_dir = os.path.join(validation_dir, 'buffalo')
# Directory with validation moose pictures
validation_moose_dir = os.path.join(validation_dir, 'moose')

# Understanding the data
num_bison_tr = len(os.listdir(train_bison_dir))
num_buffalo_tr = len(os.listdir(train_buffalo_dir))
num_moose_tr = len(os.listdir(train_moose_dir))
num_bison_val = len(os.listdir(validation_bison_dir))
num_buffalo_val = len(os.listdir(validation_buffalo_dir))
num_moose_val = len(os.listdir(validation_moose_dir))

# Total number of training and validation images
total_train = num_bison_tr + num_buffalo_tr + num_moose_tr
total_val = num_bison_val + num_buffalo_val + num_moose_val

print('total training bison images:', num_bison_tr)
print('total training buffalo images:', num_buffalo_tr)
print('total training moose images:', num_moose_tr)
print('total validation bison images:', num_bison_val)
print('total validation buffalo images:', num_buffalo_val)
print('total validation moose images:', num_moose_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)

# Setting variables for preprocessing and training the network
batch_size = 128
epochs = 65
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Prepping the data
# Generator for our training data
train_image_generator = ImageDataGenerator(rescale=1./255)
# Generator for our validation data
validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical',  # Use categorical for more than two classes
    classes=['bison', 'buffalo', 'moose']  # Specify class labels
)

val_data_gen = validation_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical',
    classes=['bison', 'buffalo', 'moose']
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

plotImages(sample_training_images[:5])


#Creating the model
model = Sequential([Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),MaxPooling2D(), Conv2D(32, 3, padding='same', activation='relu'), MaxPooling2D(), Conv2D(64, 3, padding='same', activation='relu'), MaxPooling2D(), Flatten(), Dense(512, activation='relu'), Dense(1)])

#Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#Training the model - this takes quite a bit of time
history = model.fit(
    train_data_gen,
    steps_per_epoch=max(1, total_train // batch_size),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=max(1, total_val // batch_size)
)


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
plot_training_results(history)

#Applying Horizontal flip
image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,directory=train_dir,shuffle=True,target_size=(IMG_HEIGHT, IMG_WIDTH))
#Printing the results
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

#Randomly rotating the images
image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,directory=train_dir,shuffle=True,target_size=(IMG_HEIGHT, IMG_WIDTH))
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

#Applying Zoom Augmentation
# zoom_range from 0 - 1 where 1 = 100%.
image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5) #
train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,directory=train_dir,shuffle=True,target_size=(IMG_HEIGHT, IMG_WIDTH))
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

image_gen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True,
    zoom_range=0.5,
    fill_mode='nearest',  # Add this line
)
train_data_gen = image_gen_train.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical',  # Change to categorical
    classes=['bison', 'buffalo', 'moose']
)

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

#Creating the validation generator
image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(
    batch_size=batch_size,
    directory=validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical',
    classes=['bison', 'buffalo', 'moose']
)

#Applying Dropout
model_new = Sequential([
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
    Dense(3, activation='softmax')  # Change 1 to 3, and use softmax activation for multi-class classification
])


#Compiling the new model
model_new.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_new.summary()

# Training the model
history = model_new.fit_generator(train_data_gen,steps_per_epoch=max(1, total_train // batch_size),epochs=epochs,validation_data=val_data_gen, validation_steps=max(1, total_val // batch_size))

#Visualizing the new model
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

# Function to preprocess the input image
def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Normalize the image
    img_array = img_array / 255.0

    return img_array

# Function to make predictions
def predict_animal(model, img_path, class_labels=["bison", "buffalo", "moose"]):
    processed_img = preprocess_image(img_path)
    predictions = model.predict(processed_img)

    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_index]

    return predicted_class

test_image_path = "/Users/Joshua/OneDrive/Documents/School Files/T.I.P. A.Y. 2023-2024/2nd Sem/Data Science 3/CNN/test2/24.jpg"
prediction = predict_animal(model_new, test_image_path)
print(f"The model predicts: {prediction}")

