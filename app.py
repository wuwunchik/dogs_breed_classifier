import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import wikipediaapi
from breed_translation import breed_translation

# Установка языка Википедии
wiki = wikipediaapi.Wikipedia(
    language='ru',
    user_agent="streamlit_dogs_breed_classifier/1.0 (https://example.com/contact; sashok.atochin@gmail.com)"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("dogs_model.keras")

@st.cache_data
def load_class_names():
    with open("class_names.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

# Устанавливаем wikipediaapi (если не установлена)
# Пример установки и теста поиска статьи по названию

wiki = wikipediaapi.Wikipedia('ru')

# Пример: ищем статью строго по названию
def get_wiki_page(title):
    page = wiki.page(title)
    if page.exists():
        return {
            "title": page.title,
            "summary": page.summary[:500] + "...",
            "url": page.fullurl
        }
    return None

# Загрузка модели
model = tf.keras.models.load_model("dogs_model.keras")

# Загрузка классов
class_names = load_class_names()

st.title("🐶 Определение породы собаки")

uploaded_file = st.file_uploader("Загрузите изображение собаки", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0  # Нормализация
    image_array = np.expand_dims(image_array, axis=0)

    # Предсказание породы
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    
    # Перевод породы на русский
    translated_breed = breed_translation.get(predicted_class_name, "Неизвестная порода")
    
    # Получаем информацию из Википедии
    wiki_info = get_wiki_page(translated_breed)
    
    # Отображение результатов
    st.image(uploaded_file, caption="Ваше изображение породы", use_column_width=True)
    st.write(f"🐾 Предсказанная порода: {translated_breed}")
    
    if wiki_info:
        st.write(f"Краткое описание: {wiki_info['summary']}")
        st.write(f"Подробнее: [Ссылка на Википедию]({wiki_info['url']})")
