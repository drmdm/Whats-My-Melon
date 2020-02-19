#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:42:32 2020

This script is used to train a CNN classifier on images of 3 types of melons.
Evaluation and Prediciton is done in a separate script: melon_CNN_eval.py

References:
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
IBM AI Engineering Lab DL0101EN-4-1

@author: mogmelon
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten, Dropout, MaxPooling2D
from keras import backend as K
import pickle
import datetime
plt.style.use('seaborn-dark')


#Split into train, validate and test data folders
datadir='/home/mogmelon/Python/Projects/MelonID/'
train_data_dir  = datadir+'data/train'
test_data_dir   = datadir+'data/test'

#define the classes
classes=['watermelon', 'canteloupe', 'honeydew']
n_classes=len(classes)
#2500 images required per class
n_train                 = 2220
n_test                  = 500
epochs                  = 20
batch_size              = 16
img_width, img_height   = 150, 150

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#Define the CNN model
def CNN():
    model = Sequential()
    
    #Convolutional Layer 1 w/ Pooling
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #Convolutional Layer 2 w/ Pooling
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #Convolutional Layer 3 w/ Pooling
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #Fully Connected Layer w/ Dropout
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    #Output Layer
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model=CNN()
print(keras.utils.print_summary(model))

#Create the Image Datasets using Keras ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

#Images must have suffix .jpg or.png
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    classes=classes,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    classes=classes,
    class_mode='categorical')

print('Train classes:\n %s' % train_generator.class_indices.keys())
print('Test classes:\n %s' % test_generator.class_indices.keys())

#Fit the Model and store the training output in history
history=model.fit_generator(
        train_generator,
        steps_per_epoch=n_train // batch_size,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=n_test // batch_size,
        shuffle=True)

#Save the model and history for subsequent evaluation and prediction
now=datetime.datetime.now()
now_str=now.strftime("%Y%m%dT%H%M")

model.save_weights(datadir+'weights_'+str(n_classes)+'_'+now_str+'.h5')
model.save('model_'+str(n_classes)+'_'+now_str+'.h5')

with open(datadir+'history_'+str(n_classes)+'_'+now_str+'.pickle', 'wb') as f:
    pickle.dump(history, f)



# Let's test one prediction!
img = load_img(datadir + 'data/predict/wmel.jpg', target_size=(img_width, img_height))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255

images = np.vstack([x])
prediction = model.predict_classes(images, batch_size=10)
predict_proba = model.predict_proba(images, batch_size=10)

#class_dict=train_generator.class_indices
text=('Prediction: %s, Probability: %s' % (classes[int(prediction)], np.max(predict_proba)))

plt.imshow(x[0])                           
plt.axis('off')
plt.text(0,0,text)
plt.show()



