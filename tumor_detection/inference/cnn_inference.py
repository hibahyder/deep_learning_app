import os
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
# Define constants
image_width, image_height = 150, 150
num_channels = 3  # RGB images

def preprocess_image(img):
    img = img.resize((image_width, image_height))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalizing the pixel values

def predict(img):
    model = load_model('tumor_detection/models/cnn_model.h5')
    return model.predict(img)