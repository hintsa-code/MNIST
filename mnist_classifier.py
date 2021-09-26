## Import Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

from mlxtend.data import loadlocal_mnist
import platform


from mlxtend.data import loadlocal_mnist
import platform

train_image='C:\\Users\\Hintsa\\Desktop\\Hintsa\\Course_Materials\\2021-2\\Neural_Network\\Project\\Data_Sets\\MNIST\\train-images.idx3-ubyte'
train_label='C:\\Users\\Hintsa\\Desktop\\Hintsa\\Course_Materials\\2021-2\\Neural_Network\\Project\\Data_Sets\\MNIST\\train-labels.idx1-ubyte'

X, y=loadlocal_mnist(
            images_path=train_image,
            labels_path=train_label)

model=2

if model == 1:
    model1=Sequential()
    
    model1.add(Dense(300, input_dim=784, activation='relu'))
    model1.add(Dense(500, activation='relu'))
    model1.add(Dense(1000, activation='relu'))
    model1.add(Dense(10, activation='softmax'))
    
    model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    hist1=model1.fit(X,y, epochs=100, batch_size=20)
elif model == 2:
    model2=Sequential()
    
    model2.add(Dense(units=1000, activation='sigmoid', input_shape=(784,)))
    model2.add(Dense(units=300, activation='sigmoid'))
    model2.add(Dense(units=10, activation='softmax'))
    
    model2.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
    history=model2.fit(X, y, batch_size=50, epochs=100)
##     loss, accuracy =model.evaluate(x_test, y_test, verbose=False)
    
