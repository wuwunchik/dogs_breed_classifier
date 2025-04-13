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

def get_wiki_description_and_link(breed_name):
    try:
        breed_ru = breed_translation.get(breed_name, breed_name)
        wikipedia.set_lang('ru')  
        search_results = wikipedia.search(breed_ru, results=1)
        
        if search_results:
            page_title = search_results[0]
            wiki_page = wikipedia.page(page_title)
            return wiki_page.summary[:500] + "...", wiki_page.url  
        else:
            return "Информация о породе не найдена на Википедии.", None
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Есть несколько вариантов для {breed_name}. Например: {', '.join(e.options[:5])}.", None
    except wikipedia.exceptions.RedirectError as e:
        return f"Произошел редирект при поиске {breed_name}.", None
    except wikipedia.exceptions.HTTPTimeoutError as e:
        return "Ошибка подключения к Википедии. Попробуйте снова.", None
    except Exception as e:
        return str(e), None

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

    # Получаем описание породы и ссылку на Википедию
    breed_description, wiki_url = get_wiki_description_and_link(predicted_breed)

    # Отображение результатов
    st.markdown(f"### 🐕 Порода: **{breed_translation.get(predicted_breed, predicted_breed)}**")
    st.markdown(f"Уверенность: **{confidence:.2%}**")
    st.markdown(f"**Описание породы**: {breed_description}")

    if wiki_url:
        st.markdown(f"[Подробнее на Википедии]({wiki_url})")
