import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt

model = load_model("dog_cat_model.keras")

test_img = cv2.imread(r'C:\6th sem\Deep Learning\PROJECT\unseen data\dog2.jpg')

test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

plt.imshow(test_img)

test_img = cv2.resize(test_img, (256,256))

test_img = test_img / 255.0

test_input = test_img.reshape(1,256,256,3)
pred = model.predict(test_input)
print(pred)
threshold = 0.5
if pred[0][0] < threshold:
    print("CAT")
else:    print("DOG")