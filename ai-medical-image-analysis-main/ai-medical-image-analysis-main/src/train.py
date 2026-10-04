from preprocessing import load_data
from model import build_model
import matplotlib.pyplot as plt
import os

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

train_data, test_data = load_data()
model = build_model()

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=5
)

model.save("models/medical_model.h5")

# Plot accuracy
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.title("Accuracy")
plt.savefig("outputs/accuracy.png")
plt.show()