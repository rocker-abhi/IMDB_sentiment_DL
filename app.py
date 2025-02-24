import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import nltk

# Download NLTK stopwords if not already downloaded
nltk.download("stopwords")
nltk.download("punkt")

# Load stopwords
stop_words = set(stopwords.words("english"))

# Function to remove stopwords
def remove_stop_words(text):
    tokens = word_tokenize(text)
    return " ".join([word for word in tokens if word.lower() not in stop_words])

# Function to clean text
def remove_unwanted_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only letters and spaces
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Load vectorizer and model with exception handling
vectorizer, model = None, None

try:
    with open("vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)
    st.success("Vectorizer loaded successfully!")
except FileNotFoundError:
    st.error("Error: 'vectorizer.pkl' file not found. Please check the file path.")
except Exception as e:
    st.error(f"An error occurred while loading the vectorizer: {e}")

try:
    model = tf.keras.models.load_model("model.h5")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Error: 'model.h5' file not found. Please check the file path.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")

# Streamlit UI
st.title("Sentiment Analysis with Streamlit")

# Text input box
user_input = st.text_input("Enter your text:", "")

# Submit button
if st.button("Submit"):
    if user_input.strip():  # Ensure input is not empty
        try:
            cleaned_text = remove_unwanted_text(user_input)
            cleaned_text = remove_stop_words(cleaned_text)

            # Ensure vectorizer and model are loaded
            if vectorizer is None or model is None:
                st.error("Error: Model or vectorizer not loaded. Please check the logs.")
            else:
                # Transform using vectorizer
                input_vector = vectorizer.transform([cleaned_text]).toarray()

                # Predict sentiment
                prediction = model.predict(input_vector)
                sentiment = "Positive" if prediction[0] > 0.5 else "Negative"

                st.write(f"Predicted Sentiment: **{sentiment}**")

        except ValueError as ve:
            st.error(f"Value Error: {ve}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter some text before submitting.")

