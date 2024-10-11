from fastapi import FastAPI
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from fastapi import FastAPI, File, UploadFile
from google.cloud import storage
import numpy as np
from io import BytesIO
from PIL import Image
import os
from datetime import datetime
import schedule
import boto3
import time
import threading
import requests
SUPPORTED_IMAGE_FORMATS = ["jpeg", "png"]
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


s3_client = boto3.client('s3')
BUCKET_NAME = "mybucketgithub" 
MODEL_FILE_KEY = "Image_resnet50.h5"

def download_model_from_s3(bucket_name, model_key, local_path):
    """Download the model from S3 to the local path."""
    s3_client.download_file(bucket_name, model_key, local_path)  # Download model from S3

# Download the model locally before loading it
local_model_path = "Image_resnet50.h5"  # Local path to save the model
download_model_from_s3(BUCKET_NAME, MODEL_FILE_KEY, local_model_path)
model = tf.keras.models.load_model(local_model_path)

classes = ['1', '10', '2', '3', '4', '5', '6', '7', '8', '9']

# Preprocess image
def preprocess_image(file):
    image = Image.open(BytesIO(file))
    image = image.resize((224, 224))  # Resize for your model's input
    image = np.array(image) / 255.0  # Normalize image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = preprocess_image(await file.read())
        prediction = model.predict(image)
        class_index = np.argmax(prediction[0])
        predicted_class = classes[class_index]
        return {"prediction": predicted_class}
    except Exception as e:
        return {"error": str(e)}
