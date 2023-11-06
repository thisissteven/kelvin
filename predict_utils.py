import cv2
import time
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import io


def load_image_into_numpy_array(data):
    nparr = np.frombuffer(data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_np is not None:
        return img_np
    else:
        raise ValueError("Could not decode the image")


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_predicted_class(image, model):
    # Load and preprocess the image using cv2
    start = time.time()
    img = load_image_into_numpy_array(image)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0

    # Rearrange channels if necessary (OpenCV uses BGR by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Expand dimensions to match the input shape expected by the model
    img = np.expand_dims(img, axis=0)

    # Make the prediction
    prediction = model(img)

    # Interpret the prediction (assuming binary classification)
    threshold = 0.5
    predicted_class = 1 if prediction > threshold else 0
    prediction_percentage = prediction[0][0] * 100

    result = ""
    percentage = ""

    if predicted_class == 1:
        result = "REAL"
        percentage = f"{prediction_percentage:.2f}%"

    else:
        result = "FAKE"
        percentage = f"{100 - prediction_percentage:.2f}%"
    end = time.time()

    execution_time = end - start
    execution_time = f"{execution_time:.2f} detik"

    return result, percentage, execution_time

def get_predicted_class_gan(image, model):
    # Load and preprocess the image using cv2
    start = time.time()
    img = load_image_into_numpy_array(image)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0

    # Rearrange channels if necessary (OpenCV uses BGR by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Expand dimensions to match the input shape expected by the model
    img = np.expand_dims(img, axis=0)

    # Make the prediction
    prediction = model(img)

    # Interpret the prediction (assuming binary classification)
    threshold = 0.5
    predicted_class = 1 if prediction > threshold else 0
    prediction_percentage = prediction[0][0] * 100

    result = ""
    percentage = ""

    if predicted_class == 1:
        result = "REAL"
        percentage = f"{prediction_percentage:.2f}%"

    else:
        result = "FAKE"
        percentage = f"{100 - prediction_percentage:.2f}%"
    end = time.time()

    execution_time = end - start
    execution_time = f"{execution_time:.2f} detik"

    return result, percentage, execution_time