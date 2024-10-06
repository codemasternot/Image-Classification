from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image

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

def read_imagefile(file: bytes) -> np.ndarray:
    image = Image.open(BytesIO(file))
    image = image.resize((224, 224))  # Resize based on model input size
    image = img_to_array(image) / 255.0  # Normalize the pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
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





