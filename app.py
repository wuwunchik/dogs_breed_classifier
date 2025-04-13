import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import wikipedia
import re

# Установка языка Википедии
wikipedia.set_lang("ru")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dogs_model.keras")

@st.cache_data
def load_class_names():
    with open("class_names.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

# Функция получения информации из Википедии
def get_wikipedia_info(breed):
    try:
        search_result = wikipedia.search(breed.replace("_", " "))
        if not search_result:
            return "Описание не найдено.", "#"
        page = wikipedia.page(search_result[0])
        return wikipedia.summary(search_result[0], sentences=2), page.url
    except Exception:
        return "Описание недоступно.", "#"

# Форматирование имени породы
def format_class_name(class_name):
    return class_name.replace("_", " ")

# Загрузка модели и классов
model = load_model()
class_names = load_class_names()

st.title("🐶 Определение породы собаки")

uploaded_file = st.file_uploader("Загрузите изображение собаки", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_container_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_batch)
    predicted_class = class_names[np.argmax(prediction)]
    formatted_name = format_class_name(predicted_class)

    ru_title, summary, wiki_url = get_wikipedia_info(formatted_name)

    st.markdown(f"""
    <div style="text-align: center; padding-top: 20px;">
        <h3>🐾 Предсказанная порода: <b>{ru_title}</b></h3>
        <p style="font-size: 16px;">{summary}</p>
        <a href="{wiki_url}" target="_blank" style="font-size: 15px; color: #1f77b4;">Открыть в Википедии</a>
    </div>
    """, unsafe_allow_html=True)
