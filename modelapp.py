from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import os
from datetime import datetime
import schedule
import time
import threading
SUPPORTED_IMAGE_FORMATS = ["jpeg", "png"]
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

model = load_model('C:\\Users\\Stephen\\Image_resnet50.h5')
classes = ['1', '10', '2', '3', '4', '5', '6', '7', '8', '9']

image_folder = "C:\\Users\\Stephen\\Desktop\\predict"  # My folder

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





