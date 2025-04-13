import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import wikipedia
from breed_translation import breed_translation

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

# Получение информации из Википедии
def get_wiki_description(breed_name):
    try:
        # Используем перевод породы для поиска на Википедии
        breed_ru = breed_translation.get(breed_name, breed_name)  # Получаем русский перевод породы
        wiki_page = wikipedia.page(breed_ru, auto_suggest=False)
        return wiki_page.summary[:500] + "..."  # Краткое описание породы
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Есть несколько вариантов для {breed_name}. Например: {', '.join(e.options[:5])}."
    except wikipedia.exceptions.HTTPError as e:
        return "Не удалось получить информацию из Википедии."
    except Exception as e:
        return str(e)

# Интерфейс
st.title("🐶 Определение породы собаки по изображению")

uploaded_file = st.file_uploader("Загрузите изображение собаки", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_container_width=True)

    model = load_model()
    class_names = load_class_names()

    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0]
    predicted_idx = np.argmax(prediction)
    confidence = prediction[predicted_idx]
    predicted_breed = class_names[predicted_idx]

    # Получаем описание породы с Википедии
    breed_description = get_wiki_description(predicted_breed)

    # Отображение результатов
    st.markdown(f"### 🐕 Порода: **{breed_translation.get(predicted_breed, predicted_breed)}**")
    st.markdown(f"Уверенность: **{confidence:.2%}**")
    st.markdown(f"**Описание породы**: {breed_description}")
