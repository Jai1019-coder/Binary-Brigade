import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image
