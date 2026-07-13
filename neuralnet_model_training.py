# code modified from 
# Karimpour M, Taghinezhad N, Mehdizadeh A, Alavi M, Mahmoudi T. 
# A computer-aided diagnosis (CAD) system based on convolutional neural networks for lung cancer diagnosis from 2D [18F]- PET/CT images.
#  J Appl Clin Med Phys. 2025;26:e70285.
#  https://doi.org/10.1002/acm2.70285


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
from pathlib import Path
import pydicom_seg as dcmseg
import SimpleITK as sitk

from sklearn.metrics import auc, f1_score, roc_curve, recall_score, precision_score, accuracy_score, confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
#from google.colab import files
from keras.preprocessing import image
from keras.layers import Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import *
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.models import Model
from keras import backend as K
from tensorflow.keras import layers, Model
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img

from utils_pyradiomics import create_path_df, merge_and_clean
from utils_neuralnet import read_dicom_scans, find_neoplasm, extract_per_slice, get_cropped_arrays, pad_to_shape

#test train split
###### todo: once xgboost is finished, make sure to copy the exact test/train split from there #######
outcome = 'Overall.Stage'

general_dir = Path(os.path.expanduser('~/project/xAI-in-NSCLC/NSCLC-Radiomics'))
path_df = create_path_df(general_dir)
clinical_df = pd.read_csv(os.path.expanduser('~/project/xAI-in-NSCLC/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv'))
path_to_save = os.path.expanduser('~/project/xAI-in-NSCLC')

savingPath = os.path.expanduser('~/project/xAI-in-NSCLC/deep-learning results/Improved_temporaryWeights.weights.h5')
checkpoint_path = os.path.expanduser('~/project/xAI-in-NSCLC/deep-learning results/Improved_Checkpoint_temporaryWeights.weights.h5')
checkpoint_dir = os.path.dirname(checkpoint_path)
print(checkpoint_path)

mapping = {'I': 0, 'II': 0, 'IIIa':1, 'IIIb':1}
path_df = merge_and_clean(path_df, clinical_df, mapping, outcome)
path_df = path_df.dropna(subset=[outcome])
path_df[outcome] = path_df[outcome].map(mapping)
print(f'Outcomes after mapping: {path_df[outcome].unique()}')

# test train split by patient ID
temp_df = clinical_df.dropna(subset=[outcome])
X = temp_df['PatientID']
y = temp_df[outcome]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=240, stratify=y)


# save patient IDs for test train split later
train_labels = X_train.unique()
test_labels = X_test.unique()


# create test train split dfs for training
X_train = path_df.loc[path_df['PatientID'].isin(train_labels)]
y_train = X_train[outcome]
X_test = path_df.loc[path_df['PatientID'].isin(test_labels)]
y_test = X_test[outcome]

X_train = X_train.drop(columns='PatientID')#, 'slice_no'])
X_test = X_test.drop(columns='PatientID')#, 'slice_no'])

print(check_data_leakage(X_train, X_test))


#mapping: mapping = {'I': 0, 'II': 1, 'IIIa':2, 'IIIb':2}

print('TRAINING SET DISTRIBUTION: -------------------')
print(f'\nStage I: {len(y_train.loc[y_train == 0])} images ({(len(y_train.loc[y_train == 0])/len(y_train))*100:.1f}%)')
print(f'Stage II: {len(y_train.loc[y_train == 1])} images ({(len(y_train.loc[y_train == 1])/len(y_train))*100:.1f}%)')
print(f'Stage IIIa: {len(y_train.loc[y_train == 2])} images ({(len(y_train.loc[y_train == 2])/len(y_train))*100:.1f}%)')
print(f'Stage IIIb: {len(y_train.loc[y_train == 3])} images ({(len(y_train.loc[y_train == 3])/len(y_train))*100:.1f}%)')

print('\nTESTING SET DISTRIBUTION: --------------------')
print(f'\nStage I: {len(y_test.loc[y_test == 0])} images ({(len(y_test.loc[y_test == 0])/len(y_test))*100:.1f}%)')
print(f'Stage II: {len(y_test.loc[y_test == 1])} images ({(len(y_test.loc[y_test == 1])/len(y_test))*100:.1f}%)')
print(f'Stage IIIa: {len(y_test.loc[y_test == 2])} images ({(len(y_test.loc[y_test == 2])/len(y_test))*100:.1f}%)')
print(f'Stage IIIb: {len(y_test.loc[y_test == 3])} images ({(len(y_test.loc[y_test == 3])/len(y_test))*100:.1f}%)')

#Converting to onehot encoding, since differences between outcome levels are not linear

y_train = tf.keras.ops.one_hot(y_train, 2, axis=-1, dtype=None, sparse=False)
y_test = tf.keras.ops.one_hot(y_test, 2, axis=-1, dtype=None, sparse=False)


# Convert one-hot to class
y_train_labels = tf.argmax(y_train, axis=1).numpy()


# calculating class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weights = dict(enumerate(class_weights))
print(class_weights)

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)

# load image arrays
train_image_labels = X_train.loc[X_train['path'].str.endswith('.dcm'), 'path'].tolist()
train_images = np.array([load_and_preprocess_dicom(file)for file in train_image_labels])

test_image_labels = X_test.loc[X_test['path'].str.endswith('.dcm'), 'path'].tolist()
test_images = np.array([load_and_preprocess_dicom(file)for file in test_image_labels])

#Converting greyscale to rgb, InceptionV3 requires 3 channel input

print(f'Training set dimensions before: {train_images.shape}')
# img = img_array[0, :, :, 0]          # grayscale
# img_norm = (img - img.min()) / (img.max() - img.min())  # scale to 0–1
# img_rgb = np.stack([img_norm]*3, axis=-1)

# Min-max scaling
train_min = train_images.min()
train_max = train_images.max()

# Apply to all sets
X_train = (train_images - train_min) / (train_max - train_min)
X_test  = (test_images  - train_min) / (train_max - train_min)

train_images = np.repeat(train_images[..., np.newaxis], 3, -1)

print(f'Training set dimensions after: {train_images.shape}')

print(f'\nTesting set dimensions before: {test_images.shape}')
test_images = np.repeat(test_images[..., np.newaxis], 3, -1)
print(f'Testing set dimensions after: {test_images.shape}')

train_generator = train_datagen.flow(train_images, y_train,
        batch_size=80,
        shuffle = True)


pre_trained_model1 = InceptionV3(include_top=False,
                                        weights= 'imagenet',
                                        input_shape = (299, 299, 3)) #Could I specify input size here?

for layer in pre_trained_model1.layers:
    layer.trainable = False

last_layer1 = pre_trained_model1.get_layer('mixed10')
print(f'last layer output shape: {last_layer1.output.shape}')
last_output1 = last_layer1.output


# changed to reduce parameters
# x = layers.GlobalAveragePooling2D()(last_output1)
# x = layers.Dense(256, activation='relu')(x)
# x = layers.Dropout(0.3)(x)
# x = layers.Dense(4, activation='softmax')(x)

x = GlobalAveragePooling2D()(last_output1)
x = Flatten()(x)
x = Dense(2, activation='softmax')(x)


# x = layers.Flatten()(last_output1)
# x = layers.Dense(2048, activation='relu')(x)
# x = layers.Dense(1024, activation='relu')(x)
# x = layers.Dense(512, activation='relu')(x)
# x = layers.Dense(256, activation='relu')(x)
# x = layers.Dense(128, activation='relu')(x)
# x = layers.Dense(32, activation='relu')(x)
# x = layers.Dense(3, activation='softmax')(x) # originally: x = layers.Dense(1)(x)
# x = layers.Activation(tf.nn.sigmoid)(x)

model1 = Model(pre_trained_model1.input, x)


# saving checkpoints

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=True, verbose= 1)

model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= 0.001), loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
                        tf.keras.metrics.Precision(name='Precision'),
                        tf.keras.metrics.Recall(name='Recall'),
                        tf.keras.metrics.TruePositives(name='TP'),
                        tf.keras.metrics.TrueNegatives(name='TN'),
                        tf.keras.metrics.FalseNegatives(name='FN'),
                        tf.keras.metrics.FalsePositives(name='FP'),
                        tf.keras.metrics.AUC(name='AUC')])

start = time.time()
history = model1.fit(train_generator, epochs=20, validation_data=(test_images, y_test),
            verbose = 2,
            class_weight = class_weights,
            callbacks = [cp_callback])
T_single = time.time() - start

print('Finished training model.')
print(f'Model training took: {T_single / 60 /60:.2f} hours ({T_single / 60:.2f} minutes)')
print(f'Approx. {T_single/60 /20:2f} minutes per epoch, which is {T_single/20:2f} seconds')

predictions = model1.predict(test_images)
y_pred = tf.argmax(predictions, axis=1)
y_true = tf.argmax(y_test, axis=1)

from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, sample_weight=class_weights, digits=2))

# yeahhh I'm not sure whether this iwll imporve that much... Maybe subsample only a few patients per sample for the full dataset? Like 4 slices?
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure(figsize=(12,8))
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Log Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.xticks(epochs)  # force integer ticks
plt.legend()
plt.grid(True)
fig.savefig(path_to_save + 'neural_net_loss.png')