import os
import uvicorn
import traceback
import numpy as np
from urllib.request import Request
from fastapi import FastAPI, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from utils import get_predicted_class, transform_image
from torchvision import models


load_options = tf.saved_model.LoadOptions(
    experimental_io_device='/job:localhost')

model = tf.saved_model.load("./model", options=load_options)


# Make sure to set `weights` as `'IMAGENET1K_V1'` to use the pretrained weights:
pt_model = models.densenet121(weights='IMAGENET1K_V1')
# Since we are using our model only for inference, switch to `eval` mode:
pt_model.eval()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    return "Hello world from ML endpoint!"


@app.post("/predict_image")
def predict_image(uploaded_file: UploadFile, response: Response):
    try:
        if uploaded_file.content_type not in ["image/jpeg", "image/png"]:
            response.status_code = 400
            return "File is Not an Image"

        img = uploaded_file.file.read()

        predicted = get_predicted_class(img, model)

        return predicted
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return f"Internal Server Error: {e}"


@app.post("/predict_image_2")
def predict_image(uploaded_file: UploadFile, response: Response):
    try:
        if uploaded_file.content_type not in ["image/jpeg", "image/png"]:
            response.status_code = 400
            return "File is Not an Image"

        img = uploaded_file.file.read()
        tensor = transform_image(image_bytes=img)

        outputs = model.forward(tensor)
        _, y_hat = outputs.max(1)
        return y_hat

    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return f"Internal Server Error: {e}"


port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0', port=port)
