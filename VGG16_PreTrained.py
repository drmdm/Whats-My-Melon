#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:24:52 2020

This Code runs a pretrained VGG16 model on all images in this line:
    for file in os.listdir(datadir+'data/predict/'):
        
Based on:
https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/

@author: drmdm
"""

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16, decode_predictions, preprocess_input
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import os

datadir='your_dir'

model = VGG16()
print(model.summary())
plot_model(model, to_file='vgg.png')

path=datadir+'/data/RCNN_test/predict/'
for file in os.listdir(path): 
    filepath=path+file

    # load an image from file
    image = load_img(filepath, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    print('%s (%.2f%%)' % (label[1], label[2]*100))
    
    #Reload for fullsize
    img = load_img(filepath)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255
    #text=("What's my Melon? %s\nHow sure are you? %3.0f%%" % (classes[int(prediction)].capitalize(), np.max(predict_proba)*100))
    text=('%s (%.2f%%)' % (label[1], label[2]*100))
    fig=plt.figure()
    plt.imshow(x[0])                           
    plt.axis('off')
    plt.title(text, loc='left')
    plt.show()
