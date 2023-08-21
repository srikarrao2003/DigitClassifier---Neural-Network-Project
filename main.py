import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

data = tf.keras.datasets.mnist
(X_train,Y_train),(X_test,Y_test) = data.load_data()

X_train = tf.keras.utils.normalize(X_train,axis = 1)
X_test = tf.keras.utils.normalize(X_test,axis = 1)

model = tf.keras.models.Sequential(
    [
        Flatten(input_shape=(28,28)),
        Dense(128,activation='relu'),
        Dense(128,activation='relu'),
        Dense(10,activation='linear')
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)

model.fit(X_train,Y_train,epochs=40)

model.save('digitclassif.model')