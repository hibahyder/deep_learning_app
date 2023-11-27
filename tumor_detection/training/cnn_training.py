import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Define constants
image_width, image_height = 150, 150
num_channels = 3  # RGB images
batch_size = 32

# Define directory
data_dir = 'tumor_detection/data'

# Use ImageDataGenerator to load and preprocess data
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Create train and validation generators
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'  # Specify this as the training set
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # Specify this as the validation set
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_width, image_height, num_channels)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (tumor or non-tumor)
])


model_checkpoint_callback = ModelCheckpoint(
    filepath='tumor_detection/models/cnn_model.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy']
              )

# Train the model
history = model.fit(train_generator, epochs=10, validation_data=validation_generator, callbacks=[model_checkpoint_callback])