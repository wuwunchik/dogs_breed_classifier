import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Загрузка модели
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dogs_model.keras")

# Загрузка списка пород
@st.cache_data
def load_class_names():
    with open("class_names.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

# Предобработка изображения
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, axis=0)
    return tf.keras.applications.efficientnet.preprocess_input(img_array)

# Интерфейс
st.title("🐶 Определение породы собаки по изображению")

uploaded_file = st.file_uploader("Загрузите изображение собаки", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ваше изображение", use_column_width=True)

    model = load_model()
    class_names = load_class_names()

    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0]
    predicted_idx = np.argmax(prediction)
    confidence = prediction[predicted_idx]
    predicted_breed = class_names[predicted_idx]

    st.markdown(f"### 🐕 Порода: **{predicted_breed}**")
    st.markdown(f"Уверенность: **{confidence:.2%}**")
