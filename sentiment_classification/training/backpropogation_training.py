from tensorflow.keras.datasets import imdb
# from back_prop import BackPropogation
import back_prop as bp
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
import pickle
# import joblib

top_words = 5000
(X_train, y_train), (X_test,y_test) = imdb.load_data(num_words=top_words)

max_review_length = 500

## tokenization
X_train = pad_sequences(X_train, maxlen=max_review_length)
X_test = pad_sequences(X_test, maxlen=max_review_length)


backprop = bp.BackPropogation(epochs=100,learning_rate=0.01,activation_function='sigmoid')
# training
backprop.fit(X_train, y_train)
# prediction 
pred = backprop.predict(X_test)
# find out accuracy of trained model
print(f"Accuracy : {accuracy_score(pred, y_test)}")
# save trained model
with open('c://Users/hibah/project/steamlit_app/steamlit_app/sentiment_classification/models/Backpropagation_model.pkl','wb') as file:
    pickle.dump(backprop, file)