from tensorflow.keras.datasets import imdb
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import os
parent_directory = os.path.dirname(os.getcwd())
sys.path.append(os.path.join(parent_directory, 'steamlit_app','sentiment_classification', 'training'))

word_to_index = imdb.get_word_index()
model_path = os.path.join(parent_directory, 'steamlit_app','sentiment_classification','models', 'Backpropagation_model.pkl')


with open(model_path, 'rb') as file:
    new_review_text = "That was an awesome moview, the story is wonderful"
    backprop = pickle.load(file)
    model = backprop
    max_review_length = 500
    new_review_tokens = [word_to_index.get(word, 0) for word in new_review_text.split()]
    new_review_tokens = pad_sequences([new_review_tokens], maxlen=max_review_length)
    prediction = model.predict(new_review_tokens)

    # Extract the first element if prediction is a list
    prediction = prediction[0] if isinstance(prediction, list) else prediction

    # Convert the prediction to a float (assuming it's a numeric value)
    prediction = float(prediction) if prediction is not None else None
    print(prediction)