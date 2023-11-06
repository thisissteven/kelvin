import os
import uvicorn
import traceback
from fastapi import FastAPI, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from deep_fake_detect_app import get_gan
from predict_utils import get_predicted_class
from PIL import Image

# curl command
# curl -X POST -F "uploaded_file=@./person.jpg" http://localhost:8080/predict_image

load_options = tf.saved_model.LoadOptions(
    experimental_io_device='/job:localhost')

model = tf.saved_model.load("./model", options=load_options)

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

        result, percentage, execution_time = get_predicted_class(img, model)

        img = Image.open(uploaded_file.file)

        directory = 'images/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the image to the specified file path
        file_path = os.path.join(directory, uploaded_file.filename)
        img.save(file_path)
            
        label, real_prob, execution_time_gan = get_gan()

        return [
            {
                "metode": "Auto Encoder",
                "result": result,
                "akurasi": percentage,
                "waktu": execution_time,
            },
            {
                "metode": "GAN",
                "result": label,
                "akurasi": f"{real_prob:.2f}%",
                "waktu": execution_time_gan,
            }
        ]
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return f"Internal Server Error: {e}"


port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0', port=port)
