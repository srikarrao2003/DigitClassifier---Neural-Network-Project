import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from keras import regularizers
data = tf.keras.datasets.mnist
(X_train,Y_train),(X_cv,Y_cv) = data.load_data()

X_train = tf.keras.utils.normalize(X_train,axis = 1)
X_cv = tf.keras.utils.normalize(X_cv,axis = 1)

model = tf.keras.models.Sequential(
    [
        Flatten(),
        Dense(128,activation='relu',kernel_regularizer = regularizers.l2(1e-4)),
        Dense(128,activation='relu',kernel_regularizer = regularizers.l2(1e-4)),
        Dense(10,activation='linear',kernel_regularizer = regularizers.l2(1e-4))
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',metrics=['accuracy']
)


model.fit(X_train,Y_train,epochs=3)

val_loss, val_acc = model.evaluate(X_train, Y_train)
print(val_loss)
print(val_acc)

val_loss, val_acc = model.evaluate(X_cv, Y_cv)
print(val_loss)
print(val_acc)

model.save('digitclassif.model')
