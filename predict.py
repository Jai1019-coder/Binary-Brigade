import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import cv2

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights="imagenet", include_top=True, input_shape=(224, 224, 3))

def preprocess_image(image_path):
    """Loads and preprocesses an image for MobileNetV2"""
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def predict_disease(image_path):
    """Predicts a disease from a facial image using MobileNetV2"""
    img = preprocess_image(image_path)
    if img is None:
        return "Error: Unable to process image"

    prediction = model.predict(img)
    
    decoded_prediction = decode_predictions(prediction, top=1)[0]
    label = decoded_prediction[0][1]  # Extract class name (e.g., "acne", "eczema")
    confidence = decoded_prediction[0][2]  # Extract confidence score

    # Custom class mapping for specific diseases
    class_labels = {
        "acne": "Acne (Skin Disease)",
        "eczema": "Eczema (Skin Condition)",
        "allergy": "Allergic Reaction",
        "fever": "Possible Fever",
    }

    disease_prediction = class_labels.get(label.lower(), "Unknown Condition")

    return f"Prediction: {disease_prediction} (Confidence: {confidence:.2f})"
