# imports 
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import keras
import cv2
import pickle
import sklearn
import pydicom
import random


def check_data_leakage(directory1, directory2):
    filenames_dir1 = directory1
    filenames_dir2 = directory2

    common_filenames = [f for f in filenames_dir2 if f in filenames_dir1]

    if len(common_filenames) > 0:
        print("Data leakage detected!")
        print("Common image filenames between the directories:", common_filenames)
    else:
        print("No data leakage detected.")

def load_and_preprocess_dicom(dcm_path):
    dcm_data = pydicom.dcmread(dcm_path)
    image_data = dcm_data.pixel_array
    # images have dimensions 512 x 512
    # INput size of model should be 299 x 299, [80:392 ,90:422] would be 312 x 332
    # Note: it's like this vertical:[top, bottom], [? ?]
    cropped_image_data = image_data[106:405 ,107:406] #This is 299 x299

    # Could still be improved? Sometimes the image shows the table, sometimes is shifted largely left or right..
    return cropped_image_data


# Grad-CAM

def get_img_array(img_path, size):
    img_resized = load_and_preprocess_dicom(fn)

    #img = io.imread(fn)
    #img_resized = resize(img, (image_height, image_width), anti_aliasing=True) 
    img_array = None

    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.repeat(img_array[...], 3, -1)
    #img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    #array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(img_array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()