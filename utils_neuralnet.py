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