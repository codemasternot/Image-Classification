from fastapi import FastAPI
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

url = https://storage.googleapis.com/modelresnet/Image_resnet50.h5
response = requests.get(url)
model_path = "Image_resnet50.h5"
with open(model_path, "wb") as file:
    file.write(response.content)
model = tf.keras.models.load_model(model_path)
classes = ['1', '10', '2', '3', '4', '5', '6', '7', '8', '9']

BUCKET_NAME = "apitestdata"  
FOLDER_NAME = "sdffv"  # Folder containing the batch images

LOG_BUCKET_NAME = "mylogss"  # My log bucket name
LOG_FILE_NAME = "mylog.txt"
# Initialize Google Cloud Storage client
storage_client = storage.Client()

def upload_log_to_gcs(content):
    """Upload log content to Google Cloud Storage."""
    bucket = storage_client.bucket(LOG_BUCKET_NAME)
    blob = bucket.blob(LOG_FILE_NAME)
    
    # Get the existing log content if any
    if blob.exists():
        existing_log = blob.download_as_string().decode("utf-8")
    else:
        existing_log = ""

    # Append new log content and upload
    updated_log = existing_log + "\n" + content
    blob.upload_from_string(updated_log)

def download_image_from_gcs(bucket_name, file_name):
    """Download an image from GCS and return it as a numpy array."""
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    
    # Download image content into memory
    image_data = blob.download_as_bytes()
    
    # Open the image and preprocess it
    image = Image.open(BytesIO(image_data))
    image = image.resize((224, 224))  # Resize to match the model input size
    image = img_to_array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image
def predict_from_gcs_folder(bucket_name, folder_name):
    """Perform batch prediction on images from the specified GCS folder."""
    results = []
    
    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)
    
    # List all blobs (files) in the folder
    blobs = bucket.list_blobs(prefix=folder_name)
    
    for blob in blobs:
        # Only process .jpg and .png files
        if blob.name.endswith(".jpg") or blob.name.endswith(".png"):
            image = download_image_from_gcs(bucket_name, blob.name)
            prediction = model.predict(image)
            class_index = np.argmax(prediction[0])
            predicted_class = classes[class_index]
            
            # Log the result and append it
            logging.info(f"Predicted {predicted_class} for {blob.name}")
            results.append({"filename": blob.name, "prediction": predicted_class})
    
    return results

def read_imagefile(file: bytes) -> np.ndarray:
    image = Image.open(BytesIO(file))
    image = image.resize((224, 224))  
    image = img_to_array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_imagefile(await file.read())
        prediction = model.predict(image)
        class_index = np.argmax(prediction[0])
        predicted_class = classes[class_index]
        return {"prediction": predicted_class}
    except Exception as e:
        return {"error": str(e)}

def predict_from_folder(folder_path: str):
    results = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
           
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image) / 255.0
            image = np.expand_dims(image, axis=0)

            prediction = model.predict(image)
            class_index = np.argmax(prediction[0])
            predicted_class = classes[class_index]

            results.append({"filename": filename, "prediction": predicted_class})
    
    return results

@app.get("/predict-from-folder")
async def predict_from_folder_api():
    try:
        predictions = predict_from_folder(image_folder)
        return {"predictions": predictions}
    except Exception as e:
        return {"error": str(e)}

def schedule_prediction():
    print("Scheduling the daily predictions at 8 PM.")
    schedule.every().day.at("20:10").do(lambda: predict_from_folder(image_folder))

    while True:
        schedule.run_pending()
        time.sleep(60)
    
def run_scheduler():
    scheduler_thread = threading.Thread(target=schedule_prediction)
    scheduler_thread.daemon = True  
    scheduler_thread.start()
    
@app.on_event("startup")
async def startup_event():
    run_scheduler()





