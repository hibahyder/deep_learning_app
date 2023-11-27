from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb


class LSTMSentimentAnalyzer:
    def __init__(self):
        # Constants
        self.max_words = 10000
        self.max_len = 200
        # Load the saved best model
        self.loaded_model = load_model('sentiment_classification/models/lstm_model.h5')

    def predict_sentiment(self, review):
        word_index = imdb.get_word_index()
        review = review.lower().split()
        review = [word_index.get(word, 0) if word_index.get(word, 0) < self.max_words else self.max_words - 1 for word in review]
        review = pad_sequences([review], maxlen=self.max_len)
        prediction = self.loaded_model.predict(review)
        return prediction[0][0]
