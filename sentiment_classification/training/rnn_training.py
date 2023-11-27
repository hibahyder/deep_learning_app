from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# Load the IMDB dataset
vocab_size = 10000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocab_size)

# Preprocess the sequences
max_len = 200
train_data = pad_sequences(train_data, maxlen=max_len)
test_data = pad_sequences(test_data, maxlen=max_len)

# Create the RNN model
embedding_dim = 128
rnn_units = 64

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(SimpleRNN(rnn_units))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define a callback to save the best model

model_checkpoint_callback = ModelCheckpoint(
    filepath='sentiment_classification/models/rnn_model.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

# Train the model with callbacks
batch_size = 128
epochs = 5

model.fit(
    train_data, train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(test_data, test_labels),
    callbacks=[model_checkpoint_callback]
)