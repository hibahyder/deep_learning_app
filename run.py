import streamlit as st
from sentiment_classification.inference.lstm_inference import LSTMSentimentAnalyzer
from sentiment_classification.inference.rnn_inference import RNNSentimentAnalyzer
from sentiment_classification.inference.dnn_inference import DNNSentimentAnalyzer
from tumor_detection.inference.cnn_inference import preprocess_image, predict
# from sentiment_classification.inference.perceptron import Perceptron
import joblib
from tensorflow.keras.datasets import imdb
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
from PIL import Image
import os
parent_directory = os.path.dirname(os.getcwd())
sys.path.append(os.path.join(parent_directory, 'steamlit_app','sentiment_classification', 'training'))

def main():
    st.set_page_config(page_title="DL Algorithm toolbox",
                       page_icon=":books:")
    st.header("Deep Learning App")

    add_selectbox = st.sidebar.selectbox(
        'What would you like to run?',
        ('Tumor Detection', 'Sentiment Classification'),
        index=None,
        placeholder="Select DL algorithm...",
    )

    st.subheader('{}'.format(add_selectbox), divider='rainbow')
    if add_selectbox == "Sentiment Classification":
        dataset_name = st.selectbox(
            'Which dataset?',
            ('IMDB movie review dataset', 'SMS Spam dataset'),
            index=None)
        if dataset_name == 'IMDB movie review dataset':
            method = st.radio(
                "Method",
                key="visibility",
                options=["LSTM", "RNN", "DNN", "Backpropogation", "Perceptron"],
            )
            st.write('Selected:', method)
            if method == "LSTM":
                placeholder="Select DL algorithm...",
                user_input1 = st.text_area('Enter your movie review here:', '')
                sentiment_analyzer = LSTMSentimentAnalyzer()
                if st.button('Predict'):
                    if user_input.strip() != '':
                        prediction = sentiment_analyzer.predict_sentiment(user_input)
                        if prediction >= 0.5:
                            st.success("The review is positive.")
                        else:
                            st.error("The review is negative.")
                    else:
                        st.warning("Please enter a review.")
            elif method == "RNN":
                user_input = st.text_area('Enter your movie review here:', '')
                sentiment_analyzer = RNNSentimentAnalyzer()
                if st.button('Predict'):
                    if user_input.strip() != '':
                        prediction = sentiment_analyzer.predict_sentiment(user_input)
                        if prediction >= 0.5:
                            st.success("The review is positive.")
                        else:
                            st.error("The review is negative.")
                    else:
                        st.warning("Please enter a review.")
            elif method == "DNN":
                user_input = st.text_area('Enter your movie review here:', '')
                sentiment_analyzer = DNNSentimentAnalyzer()
                if st.button('Predict'):
                    if user_input.strip() != '':
                        prediction = sentiment_analyzer.predict_sentiment(user_input)
                        if prediction >= 0.5:
                            st.success("The review is positive.")
                        else:
                            st.error("The review is negative.")
                    else:
                        st.warning("Please enter a review.")
            elif method == "Backpropogation":
                user_input = st.text_area('Enter your movie review here:', '')
                word_to_index = imdb.get_word_index()
                model_path = os.path.join(parent_directory, 'steamlit_app','sentiment_classification','models', 'Backpropagation_model.pkl')
                with open(model_path, 'rb') as file:
                    backprop = pickle.load(file)
                    max_review_length = 500
                    new_review_tokens = [word_to_index.get(word, 0) for word in user_input.split()]
                    new_review_tokens = pad_sequences([new_review_tokens], maxlen=max_review_length)
                if st.button('Predict'):
                    prediction = backprop.predict(new_review_tokens)
                    prediction = prediction[0] if isinstance(prediction, list) else prediction
                    prediction = float(prediction) if prediction is not None else None
                    if prediction > 0.5:
                        st.success("The review is positive")
                    else:
                        st.error("The review is negative")
            elif method =="Perceptron":
                imdb_perceptron = joblib.load("sentiment_classification/models/perceptron_imdb.joblib")

                def perceptron_predict_sentiment(review, vectorizer, model):
                    review_bow = vectorizer.transform([review])
                    prediction = imdb_perceptron.predict(review_bow)
                    if prediction > 0.3 :
                        st.success("The review is positive")
                    else:
                        st.error("The review is negative")
        
                review=st.text_input("Enter your review here")
                vectorizer = joblib.load("sentiment_classification/models/vectorizer_imdb.joblib")
                if st.button("Predict"):
                    perceptron_predict_sentiment(review,vectorizer,imdb_perceptron)
        elif dataset_name == 'SMS Spam dataset':
            method = st.radio(
                "Method",
                key="visibility",
                options=["LSTM", "RNN", "DNN", "Backpropogation", "Perceptron"],
            )
            st.write('Selected:', method)
            if method == "LSTM":
                placeholder="Select DL algorithm...",
                user_input1 = st.text_area('Enter your SMS here:', '')
                sentiment_analyzer = LSTMSentimentAnalyzer()
                if st.button('Predict'):
                    if user_input.strip() != '':
                        prediction = sentiment_analyzer.predict_sentiment(user_input)
                        if prediction >= 0.5:
                            st.success("spam.")
                        else:
                            st.error("non-spam")
                    else:
                        st.warning("Please enter a sms")
            elif method == "RNN":
                user_input = st.text_area('Enter your SMS here:', '')
                sentiment_analyzer = RNNSentimentAnalyzer()
                if st.button('Predict'):
                    if user_input.strip() != '':
                        prediction = sentiment_analyzer.predict_sentiment(user_input)
                        if prediction >= 0.5:
                            st.success("spam.")
                        else:
                            st.error("non-spam")
                    else:
                        st.warning("Please enter a sms")
            elif method == "DNN":
                user_input = st.text_area('Enter your SMS here:', '')
                sentiment_analyzer = DNNSentimentAnalyzer()
                if st.button('Predict'):
                    if user_input.strip() != '':
                        prediction = sentiment_analyzer.predict_sentiment(user_input)
                        if prediction >= 0.5:
                            st.success("spam.")
                        else:
                            st.error("non-spam")
                    else:
                        st.warning("Please enter a sms")
            elif method == "Backpropogation":
                user_input = st.text_area('Enter your SMS here:', '')
                backprop = BackPropogation(epochs=25,activation_function='sigmoid')
                if st.button('Predict'):
                    pass
            elif method =="Perceptron":
                imdb_perceptron = joblib.load("sentiment_classification/models/perceptron_imdb.joblib")

                def perceptron_predict_sentiment(review, vectorizer, model):
                    review_bow = vectorizer.transform([review])
                    prediction = imdb_perceptron.predict(review_bow)
                    if prediction > 0.3 :
                        st.success("spam")
                    else:
                        st.success("non-spam")
        
                review=st.text_input("Enter your sms here")
                vectorizer = joblib.load("sentiment_classification/models/vectorizer_imdb.joblib")
                if st.button("Predict"):
                    perceptron_predict_sentiment(review,vectorizer,imdb_perceptron)
    
    elif add_selectbox == "Tumor Detection":
        method = st.radio(
            "Method",
            options=["CNN"],
        )
        st.write('Selected:', method)

        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Perform inference if the user uploads an image
            if st.button('Predict'):
                processed_image = preprocess_image(image)
                prediction = predict(processed_image)
                
                # Assuming binary classification (tumor or non-tumor)
                if prediction[0] > 0.5:
                    st.warning('Prediction: Tumor')
                else:
                    st.success('Prediction: Non-Tumor')

if __name__ == '__main__':
    main()
