from fastapi import FastAPI, UploadFile, File
from os import environ as env
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import uvicorn
import os
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
import shutil
import time
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

port = os.getenv("PORT")
if port is None:
  port = 8080

app = FastAPI()

t = time.time()
export_path = "saved_model_file".format(int(t))
model = tf.keras.models.load_model(
    export_path, custom_objects={"KerasLayer": hub.KerasLayer}
)

class_labels = ['healthy','nuklear']
# Fungsi untuk melakukan prediksi
def predict(image):
    image = image.resize((150, 150))  # Menyesuaikan ukuran gambar jika diperlukan
    image_array = np.array(image) / 255.0  # Mengonversi gambar menjadi array dan normalisasi
    image_array = np.expand_dims(image_array, axis=0)  # Menambahkan dimensi batch
    probabilities = model.predict(image_array)[0]
    class_idx = np.argmax(probabilities)

    return {class_labels[class_idx]: probabilities[class_idx]}

@app.get("/")
def index():
    return {"details":f"this is base url of {env['TITLE']}, ( {env['MY_VARIABLE']} page)"}

@app.post("/predict")
def prediction(file: UploadFile):
    #specify destination location uploaded file (file path)
    img_loc = "./images/" + file.filename
    #save uploaded file in specified path
    with open(img_loc, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    class_labels = ['healthy','nuklear']

    for image_path in img_loc:
        # Membaca gambar
        image = Image.open(image_path)
        # Melakukan prediksi
        prediction = predict(image)
        # Menampilkan hasil prediksi
        class_name = list(prediction.keys())[0]
        confidence = list(prediction.values())[0]
        print("Prediksi untuk %s: class: %s, confidence: %.2f%%" % (image_path, class_name, confidence * 100))


    return {"filename": img_loc}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=1200)