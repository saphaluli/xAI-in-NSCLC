# imports :)

import os
import time
import pickle
import pandas as pd
import argparse
import numpy as np
import xgboost as xgb
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from utils_pyradiomics import preprocessing_train, preprocessing_test, get_optimal_threshold
from utils_model_training import merge_and_clean


# datasets
clinical_df = pd.read_csv(os.path.expanduser('~/project/xAI-in-NSCLC/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv'))
features_df = pd.read_csv(os.path.expanduser('~/project/xAI-in-NSCLC/FULL_radiomics_features_per_slice.csv'))

# merging and cleaning datasets
merged_df = merge_and_clean(features_df, clinical_df)

# test train split by patient ID
temp_df = clinical_df.dropna(subset=['Histology'])
X = temp_df['PatientID']
y = temp_df['Histology']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=204, stratify=y)

# save patient IDs for deep learning test train split later
train_labels = X_train.unique()
test_labels = X_test.unique()


# create test train split dfs for training
X_train = merged_df.loc[merged_df['PatientID'].isin(train_labels)]
y_train = X_train['Histology']
X_test = merged_df.loc[merged_df['PatientID'].isin(test_labels)]
y_test = X_test['Histology']

X_train = X_train.drop(columns=['PatientID', 'Histology', 'slice_no'])
X_test = X_test.drop(columns=['PatientID', 'Histology', 'slice_no'])


# preprocess training dataset first 
mean_std, selector, to_drop, decor_dataset_train = preprocessing_train(X_train)
decor_dataset_test = preprocessing_test(X_test, mean_std, selector, to_drop)
print('features processed. New shape of training dataset:', decor_dataset_train.shape, 'before:', X_train.shape)

print(f'y_train: {y_train.value_counts()/len(y_train)} N = {len(y_train)}')
print(f'y_test: {y_test.value_counts()/len(y_test)} N = {len(y_test)}')

#model definition
model = xgb.XGBClassifier(enable_categorical=True, colsample_bytree=1, eta=0.01, max_depth=4,
                            objective='multi:softprob', eval_metric='logloss', nthread=8,
                            gamma=0.5, seed=204)

#measuring time to train one model
start_model1 = time.time()
model.fit(X_train, y_train)
T_single_model1 = time.time() - start_model1

# recursive feature elimination with cross validation:
min_features_to_select = 10

rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(10),
            scoring='roc_auc_ovr_weighted',
            min_features_to_select=min_features_to_select)

#measuring time for rfecv
start_rfecv = time.time()
rfecv.fit(decor_dataset_train, y_train)
T_single_rfecv = time.time() - start_rfecv
support = rfecv.support_




print(f'Training of singular xgboost model took {T_single_model1:.2f} seconds')
print(f'Recursive feature elimination took {T_single_rfecv / 60:.2f} minutes')