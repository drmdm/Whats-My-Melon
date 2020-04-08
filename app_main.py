#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 22:44:58 2020

Ref:
https://github.com/Poseyy/StreamlitDemos/tree/master/Streamlit_Upload

@author: mogmelon
"""

import streamlit as st 
import skimage
import mrcnn
from PIL import Image
import glob
from mrcnn.config import Config
from mrcnn.model import MaskRCNN, log
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import warnings
warnings.simplefilter('ignore')

class PredictionConfig(Config):
    NAME = "MaskRCNN_cfg"
    NUM_CLASSES = 3 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.5
    MAX_GT_INSTANCES=50
    POST_NMS_ROIS_TRAINING = 3000  
    POST_NMS_ROIS_INFERENCE = 1000 
    RPN_NMS_THRESHOLD = 0.8
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

def prediction(file, confidence=0.95):
    success=True
    cfg = PredictionConfig()
    model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
    weights_fname ='./weights_all_melon_20200322T2009.h5'
    model.load_weights(weights_fname, by_name=True)   
    class_names=['BG', 'Watermelon', 'Canteloupe', 'Honeydew']
    
    image = skimage.io.imread(file)
    if len(image.shape) > 2 and image.shape[2] == 4:
        image = image[:,:,0:3]
    results = model.detect([image], verbose=0)[0] 

    fig=plt.figure()
    plt.imshow(image)   
    ax = plt.gca()
    keep=np.where(results['scores'] > confidence)
    keep=keep[0]
    cmap=['white', 'red', 'darkorange', 'gold']
    
    if len(keep) < 1:
        ax.axis('off')
        success=False
    else:
        for box in keep:
            y1, x1, y2, x2 = results['rois'][box]
            c_index=results['class_ids'][box]
            melon=class_names[c_index]
            score=(results['scores'][box])*100
            annotation = ('%s: %.1f%%' % (melon, score))
            color = cmap[c_index]
            width, height = x2 - x1, y2 - y1
            rect = Rectangle((x1, y1), width, height, fill=False, color=color)
            ax.add_patch(rect)
            ax.text(x1, y1, annotation, fontsize=10, color=color)
            ax.axis('off')
        
    plt.tight_layout()    
    fig.canvas.draw()

    return fig, success
    
st.title("What's My Melon?")
st.write("Upload a picture containing your unidentified melon(s) and What's My \
          Melon will detect and identify them for you.")

st.header("Upload your image for detection")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
 
if uploaded_file is not None:
    with st.spinner("Retrieving your melons..."):
        fig, success = prediction(uploaded_file, confidence=0.9)
        if success:
            st.success("Found the following melons:")
            st.balloons()
        else:
            st.error("No melons found. Please try another image")
    st.write(fig)