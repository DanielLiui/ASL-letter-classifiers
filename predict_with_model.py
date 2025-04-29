# Pass in saved model to load_model(). To use the model to predict the letter of an 
# ASL hand sign image, run:
# >python predict_with_model.py <image path>

import numpy as np
import sys
from tensorflow.keras.models import load_model  # type: ignore
import cv2

model = load_model("asl_mobilenetv2_model.keras")

label_map = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(img_path):
    img = preprocess_image(img_path)
    pred = model.predict(img)
    predicted_class = np.argmax(pred)
    predicted_label = label_map[predicted_class]
    print(f"Predicted Letter: {predicted_label} (Confidence: {np.max(pred)*100:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py path_to_image.jpg")
    else:
        predict(sys.argv[1])
