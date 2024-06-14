import streamlit as st
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the animal detection model
animal_detection_model = load_model("improved_animal_classification_model.h5")

# Load the animal emotion detection model
animal_emotion_detection_model = load_model("facial_expression_model.h5")

# Streamlit app title
st.title("Animal Detection and Emotion Recognition 🐦🐴 🐑 🐹 🐱 🐶 ")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Display the uploaded image
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

# Function to preprocess the uploaded image for animal detection
def preprocess_animal_detection_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    img = cv2.resize(img, (128, 128))  # Resize to 128x128
    img = img.astype('float32') / 255.0  # Normalize pixel values
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Function to preprocess the uploaded image for animal emotion detection
def preprocess_emotion_detection_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert RGB to grayscale
    img = cv2.resize(img, (48, 48))  # Resize to 48x48 (same as the input size of the emotion detection model)
    img = img.astype('float32') / 255.0  # Normalize pixel values
    return np.expand_dims(img, axis=0)  # Add batch dimension

# Function to get animal category name from label index
def get_animal_category_name(label_index):
    categories = ['cats', 'dog', 'parrot', 'horse', 'hamster', 'sheep']
    return categories[label_index]

# Function to get emotion category name from label index
def get_emotion_category_name(label_index):
    categories = ['Angry', 'Happy', 'Sad']
    return categories[label_index]

# Detect button
if uploaded_file is not None:
    if st.button("Detect"):
        # Read the uploaded image
        image = np.array(Image.open(uploaded_file))

        # Preprocess the image for animal detection
        animal_detection_image = preprocess_animal_detection_image(image)

        # Make predictions for animal detection
        animal_predictions = animal_detection_model.predict(animal_detection_image)
        animal_label_index = np.argmax(animal_predictions)
        animal_category = get_animal_category_name(animal_label_index)

        # Preprocess the image for animal emotion detection
        emotion_detection_image = preprocess_emotion_detection_image(image)

        # Make predictions for animal emotion detection
        emotion_predictions = animal_emotion_detection_model.predict(emotion_detection_image)
        emotion_label_index = np.argmax(emotion_predictions)
        emotion_category = get_emotion_category_name(emotion_label_index)

        # Combine predicted animal and emotion into a single line
        prediction_text = f"Predicted Animal: {animal_category}\n\nPredicted Emotion: {emotion_category}"

        # Add some styling to the textbox using HTML
        styled_text = f"<div style='background-color: #080808; padding: 10px; border-radius: 10px;'>{prediction_text}</div>"

        # Display the styled text using markdown
        st.markdown(styled_text, unsafe_allow_html=True)


# to run the model use this command:streamlit run c:\Users\Vaibhav\OneDrive\Desktop\Nullclass_AnimalEmotion\app.py [ARGUMENTS]