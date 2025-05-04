import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model('imdb_sentiment_model.h5')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_sentiment(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=500)
    
    prediction = model.predict(padded_sequences)
    
    sentiment = 'Positif' if prediction >= 0.5 else 'Negatif'
    return sentiment, prediction

st.title('Aplikasi Prediksi Sentimen IMDB')
st.write("Masukkan ulasan film dan kami akan menebak apakah sentimennya positif atau negatif.")

user_input = st.text_area("Masukkan Ulasan Dalam Bahasa Inggris")

if st.button('Prediksi'):
    if user_input:
        with st.spinner('Menganalisis sentimen...'):
            sentiment, prediction = predict_sentiment(user_input)
            st.write(f"Sentimen: {sentiment}")
            st.write(f"Confidence Score: {prediction[0][0]*100:.2f}%")
            if sentiment == 'Positif':
                st.success("Ulasan ini memiliki sentimen positif!")
            else:
                st.error("Ulasan ini memiliki sentimen negatif!")
    else:
        st.warning("Silakan masukkan ulasan terlebih dahulu!")
