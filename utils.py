import cv2
import time
import numpy as np


def load_image_into_numpy_array(data):
    nparr = np.frombuffer(data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_np is not None:
        return img_np
    else:
        raise ValueError("Could not decode the image")


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

    str = ""

    if predicted_class == 1:
        str = f"Predicted Class: REAL with {prediction_percentage:.2f}% confidence"
    else:
        str = f"Predicted Class: FAKE with {100 - prediction_percentage:.2f}% confidence"
    end = time.time()

    execution_time = end - start
    print(f"Execution time: {execution_time:.2f} seconds")

    return str
