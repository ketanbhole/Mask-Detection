import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from PIL import Image
import numpy as np


# Paths to your dataset directories
train_dir = r'C:\Users\krisb\Downloads\New folder\archive (1)\Face Mask Dataset\Train'
test_dir = r'C:\Users\krisb\Downloads\New folder\archive (1)\Face Mask Dataset\Test'
val_dir = r'C:\Users\krisb\Downloads\New folder\archive (1)\Face Mask Dataset\Validation'

# Data augmentation and loading images using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2
)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(128, 128),
    class_mode='categorical',
    batch_size=32
)

# Validation and test generators (without data augmentation)
val_datagen = ImageDataGenerator(rescale=1.0/255)
val_generator = val_datagen.flow_from_directory(
    directory=val_dir,
    target_size=(128, 128),
    class_mode='categorical',
    batch_size=32
)

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(128, 128),
    class_mode='categorical',
    batch_size=32
)

# MobileNetV2 model initialization
mobilenet = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(128, 128, 3)
)

for layer in mobilenet.layers:
    layer.trainable = False

# Building the classification model
model = Sequential()
model.add(mobilenet)
model.add(Flatten())
model.add(Dense(2, activation='sigmoid'))
model.summary()

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Training the model
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator,
    steps_per_epoch=None,
    validation_steps=None
)

# Evaluating the model on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy}, Test loss: {test_loss}')

# Function for predicting mask or no mask for an image
def predict_mask(image_path):
    img = Image.open(image_path)
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    class_labels = ['With Mask', 'Without Mask']
    result = class_labels[predicted_class]

    return result

# Path to the image for prediction
image_path = r'C:\Users\kri\IMG_20085_1027.jpg'

# Make prediction
prediction_result = predict_mask(image_path)
print(f'Prediction: {prediction_result}')

# Save the trained model
model.save('masknet.h5')
