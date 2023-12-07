from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import imdb
import joblib


# Load the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=None)

# Convert the sequence of word indices to a bag-of-words representation
vectorizer = CountVectorizer()
x_train_bow = vectorizer.fit_transform([' '.join(map(str, x)) for x in x_train])
x_test_bow = vectorizer.transform([' '.join(map(str, x)) for x in x_test])


# Split the data into training and validation sets
x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
    x_train_bow, y_train, test_size=0.2, random_state=42
)

# Create a Perceptron model
perceptron = Perceptron(random_state=42)

# Train the model
# perceptron.fit(x_train_split, y_train_split)

# # Make predictions on the validation set
# val_predictions = perceptron.predict(x_val_split)

# # Evaluate accuracy on the validation set
# accuracy = accuracy_score(y_val_split, val_predictions)
# print(f"Validation Accuracy: {accuracy:.4f}")

# joblib.dump(perceptron, r"D:\Users\hibah\Desktop\perceptron_imdb.joblib")
# joblib.dump(vectorizer, r"D:\Users\hibah\Desktop\vectorizer_imdb.joblib")