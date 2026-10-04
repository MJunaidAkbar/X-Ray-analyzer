import cv2
import matplotlib.pyplot as plt

img_path = "data/chest_xray/train/PNEUMONIA/person1_bacteria_1.jpeg"

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

plt.imshow(img, cmap='gray')
plt.title("Sample X-ray")
plt.axis('off')
plt.show()