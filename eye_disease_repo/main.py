import base64
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input
from eye_disease_repo.eye_disease_model import model
classes = ['normal', 'cataract', 'glaucoma', 'diabetic_retinopathy']


def predictEyeDisease(base64_data):
    if "," in base64_data:
        base64_data = base64_data.split(",")[1]

    image_data = base64.b64decode(base64_data)
    img = Image.open(BytesIO(image_data)).convert("RGB")
    img = img.resize((256, 256))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    pred = model.predict(img_array)
    predicted_class = np.argmax(pred, axis=1)[0]
    return classes[predicted_class]