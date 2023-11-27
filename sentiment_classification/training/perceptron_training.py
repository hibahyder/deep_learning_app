import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load IMDb dataset
max_words = 10000
maxlen = 200
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_words)

# Preprocess the data
train_data = pad_sequences(train_data, maxlen=maxlen)
test_data = pad_sequences(test_data, maxlen=maxlen)

# Define the Perceptron model
model = Sequential()
model.add(Dense(1, activation='sigmoid', input_shape=(maxlen,)))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(test_data, test_labels)
print(f'Test accuracy: {accuracy * 100:.2f}%')

# Save the best model
model.save('perceptron_model.h5')