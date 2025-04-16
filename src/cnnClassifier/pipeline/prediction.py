import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
    
    def predict(self):
        # Cargar modelo
        #model = load_model(os.path.join("artifacts", "training", "model.h5"))
        model = load_model(os.path.join("model", "model.h5"))
        
        # Preparar imagen
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0  # Normalización importante
        test_image = np.expand_dims(test_image, axis=0)
        
        # Obtener predicción
        probabilities = model.predict(test_image)
        print("Probabilidades brutas:", probabilities)
        
        result = np.argmax(probabilities, axis=1)
        print("Clase predicha (índice):", result)
        
        # Verificar las clases
        class_mapping = {0: 'Adenocarcinoma Cancer', 1: 'Normal'}
        
        prediction = class_mapping[result[0]]
        return [{"image": prediction}]