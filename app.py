import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load model
base = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

inputs = tf.keras.Input((224, 224, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.load_weights("buah_classifier.weights.h5")

# 2. Nama kelas
classes = ["APEL", "PISANG", "ANGGUR", "JERUK", "STROBERI"]

# 3. Tampilan aplikasi
st.title("🍎 Klasifikasi Buah")

file = st.file_uploader(
    "Upload gambar buah",
    type=["jpg", "jpeg", "png"]
)

# 4. Prediksi
if file:
    image = Image.open(file).convert("RGB")
    st.image(image, width=400)

    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)[0]
    index = np.argmax(prediction)

    st.success(f"Hasil: {classes[index]}")
    st.write(f"Confidence: {prediction[index] * 100:.2f}%")