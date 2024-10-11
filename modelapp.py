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
import logging
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
BUCKET_NAME = "mybucketgithub"  # Replace with your S3 bucket name
MODEL_FILE_KEY = "Image_resnet50.h5"
LOG_FILE_KEY = "mylog.txt"
PREDICT_FOLDER = "predict/"

log_file = "mylog.txt"  # Local log file
logging.basicConfig(
    filename=log_file,  # Log file to be stored temporarily
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def download_model_from_s3(bucket_name, model_key):
    """Download the model from S3 and load it."""
    local_model_path = "Image_resnet50.h5"
    s3_client.download_file(bucket_name, model_key, local_model_path)  
    logging.info(f"Model {model_key} downloaded from S3 bucket {bucket_name}")
    return tf.keras.models.load_model(local_model_path)
    
model = download_model_from_s3(BUCKET_NAME, MODEL_FILE_KEY)

classes = ['1', '10', '2', '3', '4', '5', '6', '7', '8', '9']

# Preprocess image
def preprocess_image(file):
    image = Image.open(BytesIO(file))
    image = image.resize((224, 224))  # Resize for your model's input
    image = np.array(image) / 255.0  # Normalize image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def upload_log_to_s3(content):
   """Upload the log file to S3."""
    try:
        # Ensure all log entries are written to the log file
        logging.shutdown()

        # Read the log file and print its content (for debugging)
        with open(log_file, "r") as log:
            log_content = log.read()
            print(f"Log content before uploading to S3:\n{log_content}")  # Debugging step
            
            # Upload log to S3
            s3_client.put_object(Bucket=BUCKET_NAME, Key=LOG_FILE_KEY, Body=log_content)
        
        print(f"Log file uploaded to S3 bucket {BUCKET_NAME}")
    except Exception as e:
        print(f"Failed to upload log to S3: {str(e)}")
   

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
    """Perform batch prediction on images from the S3 folder."""
    results = []
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)

    for obj in response.get('Contents', []):
        file_key = obj['Key']

        if file_key.endswith(".jpg") or file_key.endswith(".png"):
            image_data = s3_client.get_object(Bucket=bucket_name, Key=file_key)['Body'].read()
            image = preprocess_image(image_data)
            prediction = model.predict(image)
            class_index = np.argmax(prediction[0])
            predicted_class = classes[class_index]

            result = {"filename": file_key, "prediction": predicted_class}
            results.append(result)

            logging.info(f"Predicted {predicted_class} for {file_key}")

    return results

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = preprocess_image(await file.read())
        prediction = model.predict(image)
        class_index = np.argmax(prediction[0])
        predicted_class = classes[class_index]
        logging.info(f"Predicted class: {predicted_class} for uploaded image.")
        return {"prediction": predicted_class}
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return {"error": str(e)}

@app.get("/predict-from-s3-folder")
async def predict_from_s3_folder_api():
    try:
        predictions = predict_from_s3_folder(BUCKET_NAME, PREDICT_FOLDER)
        upload_log_to_s3()  # Upload log to S3 after batch prediction
        return {"predictions": predictions}
    except Exception as e:
        logging.error(f"Error during S3 folder prediction: {str(e)}")
        return {"error": str(e)}
