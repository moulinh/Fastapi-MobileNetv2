import os
import numpy as np
from tensorflow import keras
from app.preprocess.image import preprocess_image
from pathlib import Path

MODEL_NAME = "mobilenetv2_test1_finetuned.keras"
class_names = ["00-normal", "01-minor", "02-moderate", "03-severe"]

def load_image_model() -> keras.Model:
    project_root = Path.cwd()
    print("project_root : " + str(project_root))
    MODEL_PATH = "." + str(project_root) + "/" + MODEL_NAME
    print("MODEL_PATH : ", MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        print("*** Aucun modèle image trouvé !!!")
        model = None
    else: 
        model = keras.models.load_model(MODEL_PATH)
        print("Modèle image chargé.")
    return model

def predict_image(model: keras.Model, image_bytes: bytes) -> dict:
    array = preprocess_image(image_bytes)
    probas = model.predict(array)[0]
    digit = int(np.argmax(probas))
    class_name = class_names[digit] 
    confidence = round(float(np.max(probas)), 4)
    probabilities = {
        class_names[i]: round(float(probas[i]), 4) for i in range(len(class_names))
    }
    return {"class_name": class_name, "confidence": confidence, "probabilities": probabilities}