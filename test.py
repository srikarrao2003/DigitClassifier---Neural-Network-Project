import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

model = tf.keras.models.load_model('D:\ml projects\Handwritten Digit Classifier\digitclassif.model')
ytest = [0,1,2,3,4,5,6,7,8,9]
xtest = []
for i in range(10):
    img = cv2.imread(f"testdata\digit{i}.png")[:,:,0]
    img = np.invert(np.array([img]))
    xtest.append(img[0])
    predict = model.predict(img)
    print(f"This model would probably be a number : {np.argmax(predict)}")
    plt.imshow(img[0],cmap = plt.cm.binary)
    plt.show()

