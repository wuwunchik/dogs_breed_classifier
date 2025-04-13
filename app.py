import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import wikipedia
import wikipediaapi

# Установка языка Википедии
wikipedia.set_lang("ru")

# Инициализация Wikipedia API для строгого поиска
wiki = wikipediaapi.Wikipedia('ru')

# Функция получения информации из Википедии
def get_wiki_page(title):
    page = wiki.page(title)
    if page.exists():
        return {
            "title": page.title,
            "summary": page.summary[:500] + "...",  # Обрезаем описание до 500 символов
            "url": page.fullurl
        }
    return {
        "title": title,
        "summary": "Информация не найдена 😔",
        "url": "#"
    }

# Форматирование имени породы (замена "_" на пробелы)
def format_class_name(class_name):
    return class_name.replace("_", " ")

# Загрузка модели и классов
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dogs_model.keras")

@st.cache_data
def load_class_names():
    with open("class_names.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

# Загружаем модель и имена классов
model = load_model()
class_names = load_class_names()

st.title("🐶 Определение породы собаки")

# Загружаем изображение собаки
uploaded_file = st.file_uploader("Загрузите изображение собаки", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Открытие изображения и его предварительный просмотр
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_container_width=True)

    # Подготовка изображения для предсказания
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Получение предсказания от модели
    prediction = model.predict(img_batch)
    predicted_class = class_names[np.argmax(prediction)]  # Класс с максимальной вероятностью

    # Форматирование имени породы
    formatted_name = format_class_name(predicted_class)

    # Получаем информацию из Википедии
    wiki_info = get_wiki_page(formatted_name)

    # Отображаем результаты
    st.markdown(f"""
    <div style="text-align: center; padding-top: 20px;">
        <h3>🐾 Предсказанная порода: <b>{wiki_info['title']}</b></h3>
        <p style="font-size: 16px;">{wiki_info['summary']}</p>
        <a href="{wiki_info['url']}" target="_blank" style="font-size: 15px; color: #1f77b4;">Открыть в Википедии</a>
    </div>
    """, unsafe_allow_html=True)
