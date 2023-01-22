"""ColorClassifier.py

Objective: Classifies colors according to 11 distinct color categories: 
    red, green, blue,orange,yellow,purple,white,grey,brown,pink, and black
"""
#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

from keras import regularizers

from sklearn.metrics import confusion_matrix, classification_report
import seaborn

dataset = pd.read_csv('colors.csv', usecols = ['R','G','B','color_class'])
dataset

#one-hot encoding
dataset = pd.get_dummies(dataset, columns = ['color_class'])
dataset.head()

"""Reording columns

"""

dataset = dataset[['R', 'G', 'B', 'color_class_Red', 'color_class_Green', 'color_class_Blue', 'color_class_Yellow', 'color_class_Orange', 'color_class_Pink', 'color_class_Purple', 'color_class_Brown', 'color_class_Grey', 'color_class_Black', 'color_class_White']]
dataset

"""
**Split data into train and test set**

Train Dataset: Used to train the machine learning model.

Test Dataset: Used to evaluate the trained machine learning model i.e. it gives an unbiased estimate of the performance of the model as this data was not used to train the model."""

train_dataset = dataset.sample(frac=0.8, random_state=8) 
test_dataset = dataset.drop(train_dataset.index) #remove train_dataset from dataframe to get test_dataset
train_dataset
#test_dataset

"""**Split input features and output labels**

seperates "R,G,B, color name, and color_class" input colums from output color class column.
needed to be seperated for training

"""

train_labels = pd.DataFrame([train_dataset.pop(x) for x in ['color_class_Red', 'color_class_Green', 'color_class_Blue', 'color_class_Yellow', 'color_class_Orange', 'color_class_Pink', 'color_class_Purple', 'color_class_Brown', 'color_class_Grey', 'color_class_Black', 'color_class_White']]).T
train_labels

test_labels = pd.DataFrame([test_dataset.pop(x) for x in ['color_class_Red', 'color_class_Green', 'color_class_Blue', 'color_class_Yellow', 'color_class_Orange', 'color_class_Pink', 'color_class_Purple', 'color_class_Brown', 'color_class_Grey', 'color_class_Black', 'color_class_White']]).T
test_labels

"""**Building and compiling model**

using ANN model w/ keras & tensorflow library
"""

model = keras.Sequential([
    layers.Dense(3, kernel_regularizer=regularizers.l2(0.01), activation='relu', input_shape=[len(train_dataset.keys())]), #inputshape=[3]
    layers.Dense(24, kernel_regularizer=regularizers.l2(0.01), activation='relu'),
    layers.Dense(24, kernel_regularizer=regularizers.l2(0.01), activation='relu'),
    layers.Dense(16, kernel_regularizer=regularizers.l2(0.01), activation='relu'),
    layers.Dense(11)
  ])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(loss=loss_function,
                optimizer=optimizer,
                metrics=['accuracy'])

model.summary()

"""**Train model**"""

history = model.fit(x = train_dataset, y = train_labels, validation_split=0.2, epochs=5001, batch_size=2048, verbose=0, callbacks=[tfdocs.modeling.EpochDots()], shuffle=True)

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Basic': history}, metric = "accuracy")
plt.ylim([0, 1])
plt.ylabel('accuracy [Color]')

plotter.plot({'Basic': history}, metric = "loss")
plt.ylim([0, 1])
plt.ylabel('loss [Color]')

"""**Making Test Predictions for test dataset from train **

prediction by the ANN model returns a Numpy array of 11 floating-point numbers for every test example
"""

test_predictions = model.predict(test_dataset)
print("shape is {}".format(test_predictions.shape))  
test_predictions

test_predictions[0] 


#Selecting Class with highest confidence
predicted_encoded_test_labels = np.argmax(test_predictions, axis=1) #Returns the indices of the maximum values along each row(axis=1)
#Converting numpy array to pandas dataframe
predicted_encoded_test_labels = pd.DataFrame(predicted_encoded_test_labels, columns=['Predicted Labels'])
predicted_encoded_test_labels

#Converting One-Hot Encoded Actual Test set labels into Label Encoding format
actual_encoded_test_labels = np.argmax(test_labels.to_numpy(), axis=1) 
#Converting numpy array to pandas dataframe
actual_encoded_test_labels = pd.DataFrame(actual_encoded_test_labels, columns=['Actual Labels'])
actual_encoded_test_labels

"""**Evaluate model performance**"""

model.evaluate(x=test_dataset, y=test_labels)

"""accuracy is 89%

**Confusion Matrix**
"""


confusion_matrix_test = confusion_matrix(actual_encoded_test_labels, predicted_encoded_test_labels)
f,ax = plt.subplots(figsize=(16,12))
categories = ['Red', 'Green', 'Blue', 'Yellow', 'Orange', 'Pink', 'Purple', 'Brown', 'Grey', 'Black', 'White']
seaborn.heatmap(confusion_matrix_test, annot=True, cmap='Reds', fmt='d',
            xticklabels = categories,
            yticklabels = categories)
plt.show()

"""**Classification Report**"""

#Classification Report
target_names = ['Red', 'Green', 'Blue', 'Yellow', 'Orange', 'Pink', 'Purple', 'Brown', 'Grey', 'Black', 'White']
print(classification_report(actual_encoded_test_labels, predicted_encoded_test_labels, target_names=target_names))