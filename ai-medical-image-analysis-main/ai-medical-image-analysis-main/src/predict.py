import tensorflow as tf
import cv2
import numpy as np

print("Starting prediction...")

model = tf.keras.models.load_model("models/medical_model.h5")

print("Model loaded")

img = cv2.imread("data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg")

if img is None:
    print("Image not found!")
    exit()

img = cv2.resize(img, (224, 224)) / 255.0
img = np.reshape(img, (1, 224, 224, 3))

pred = model.predict(img)[0][0]

print("Raw prediction:", pred)
print("Final Prediction:", "PNEUMONIA" if pred > 0.5 else "NORMAL")