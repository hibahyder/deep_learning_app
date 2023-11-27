import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Attention
from tensorflow.keras.models import Sequential

# Load the IMDB dataset
max_words = 10000  # Consider the top 10,000 most common words
max_len = 200  # Limit the length of reviews to 200 words

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

# Build the model with an attention layer
model = Sequential()
model.add(Embedding(max_words, 128, input_length=max_len))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define ModelCheckpoint callback to save the best model
model_checkpoint_callback = ModelCheckpoint('sentiment_classification/models/lstm_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Train the model with ModelCheckpoint callback
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test), callbacks=[model_checkpoint_callback])