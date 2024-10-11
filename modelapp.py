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

classes = ['1', '10', '2', '3', '4', '5', '6', '7', '8', '9']

def download_model_from_s3(bucket_name, model_key, local_path):
    """Download the model from S3 to the local path."""
    s3_client.download_file(bucket_name, model_key, local_path)

local_model_path = "Image_resnet50.h5"  # Local path to save the model
download_model_from_s3(BUCKET_NAME, MODEL_FILE_KEY, local_model_path)
model = tf.keras.models.load_model(local_model_path)

def preprocess_image(file):
    image = Image.open(BytesIO(file))
    image = image.resize((224, 224))  
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image


def upload_log_to_s3(content):
    """Upload log content to AWS S3."""
    try:
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=LOG_FILE_KEY)
        existing_log = obj['Body'].read().decode('utf-8')
    except s3_client.exceptions.NoSuchKey:
        existing_log = ""

    updated_log = existing_log + "\n" + content

    s3_client.put_object(Body=updated_log, Bucket=BUCKET_NAME, Key=LOG_FILE_KEY)
    
def download_image_from_s3(bucket_name, file_key):
    """Download an image from AWS S3 and return it as a numpy array."""
    obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    image_data = obj['Body'].read()

    # Open and preprocess the image
    image = Image.open(BytesIO(image_data))
    image = image.resize((224, 224))  # Resize to match the model input size
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_from_s3_folder(bucket_name, folder_name):
    """Perform batch prediction on images from the specified S3 folder."""
    results = []
    
    # List objects in the folder
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
    for obj in response.get('Contents', []):
        file_key = obj['Key']
        
        # Only process .jpg and .png files
        if file_key.endswith(".jpg") or file_key.endswith(".png"):
            image = download_image_from_s3(bucket_name, file_key)
            prediction = model.predict(image)
            class_index = np.argmax(prediction[0])
            predicted_class = classes[class_index]

            # Log the result and append it
            logging.info(f"Predicted {predicted_class} for {file_key}")
            upload_log_to_s3(f"Predicted {predicted_class} for {file_key}")
            results.append({"filename": file_key, "prediction": predicted_class})

    return results


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


@app.get("/predict-from-s3-folder")
async def predict_from_s3_folder_api():
    try:
        predictions = predict_from_s3_folder(BUCKET_NAME, PREDICT_FOLDER)
        return {"predictions": predictions}
    except Exception as e:
        return {"error": str(e)}
    




