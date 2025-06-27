import os 
from tensorflow.keras.models import load_model

model_filepath = os.path.join("eye_disease_repo","model_file","Cnn_model.h5")

model = load_model(model_filepath) 
