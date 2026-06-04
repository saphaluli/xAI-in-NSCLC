import os
import time
import pickle
import pandas as pd
import argparse
import numpy as np
import xgboost as xgb
import sklearn
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from utils_pyradiomics import preprocessing_train, preprocessing_test, get_optimal_threshold, merge_and_clean, get_multiclass_results

### SETTINGS
# is_by_patient -> splits dataset by patientID first if True
# is_single_slice -> takes only one slice per patient for training or testing if True
is_bypatient = bool(True)
is_single_slice = bool(False)

# True -> RFECV, False -> RFE
is_optimal_features = bool(True)

#datasets 
clinical_df = pd.read_csv(os.path.expanduser('~/project/xAI-in-NSCLC/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv'))
features_df = pd.read_csv(os.path.expanduser('~/project/xAI-in-NSCLC/FULL_radiomics_features_per_slice.csv'))
path_to_save = os.path.expanduser('~/project/xAI-in-NSCLC')

# merging and cleaning datasets
mapping = {'adenocarcinoma': 0, 'squamous cell carcinoma': 1, 'large cell': 2, 'nos':3 }
merged_df = merge_and_clean(features_df, clinical_df, mapping)

# only extract one slice per patient ID
if is_single_slice == True:
    merged_df = merged_df.groupby(by='PatientID').sample(n=1, random_state=310).reset_index(drop=True)


if is_bypatient == True:
    # test train split by patient ID
    temp_df = clinical_df.dropna(subset=['Histology'])
    X = temp_df['PatientID']
    y = temp_df['Histology']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=310, stratify=y)

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

else:
    #test train split NOT by patient ID, but by slice

    X = merged_df_clean.drop(columns=['PatientID', 'Histology'])
    y = merged_df_clean['Histology']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=310, stratify=y)



# preprocess training dataset first 
mean_std, selector, to_drop, decor_dataset_train = preprocessing_train(X_train)
decor_dataset_test = preprocessing_test(X_test, mean_std, selector, to_drop)
print('features processed. New shape of training dataset:', decor_dataset_train.shape, 'before:', X_train.shape)

mapping = {'adenocarcinoma': 0, 'squamous cell carcinoma': 1, 'large cell': 2, 'nos':3 }

#model definition
model = xgb.XGBClassifier(enable_categorical=True, colsample_bytree=1, eta=0.01, max_depth=4,
                            objective='multi:softprob', eval_metric='logloss', nthread=8,
                            gamma=0.5, seed=310)

#measuring time to train one model
#start_model1 = time.time()
#model.fit(X_train, y_train)
#T_single_model1 = time.time() - start_model1

# recursive feature elimination with cross validation:
min_features_to_select = 10

T_single_rfecv = None
T_single_rfe = None

if is_optimal_features == True:
    print('Performing RFECV')
    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(10),
                scoring='roc_auc_ovr_weighted',
                min_features_to_select=min_features_to_select)

    start_rfecv = time.time()
    rfecv.fit(decor_dataset_train, y_train)
    T_single_rfecv = time.time() - start_rfecv
    support = rfecv.support_

    fig, ax = plt.subplots()
    mean_scores = rfecv.cv_results_['mean_test_score']
    no_features = rfecv.cv_results_['n_features']
    ax.plot(no_features, mean_scores)
    ax.set(xlabel='no. of features', ylabel='roc_auc')
    ax.set_title('Results of RFECV cross validation')
    fig.savefig(fig.savefig(path_to_save + r'/RFECV_results.png'))

#todo: print out cross validation outcome

else:
    print('Performing RFE')
    rfe = RFE(estimator=model, n_features_to_select=10)
    start_rfe = time.time()
    rfe.fit(decor_dataset_train, y_train)
    T_single_rfe = time.time() - start_rfe
    support = rfe.support_


filtered_col = np.extract(support, np.array(decor_dataset_train.columns))
reduced_features_train_set = decor_dataset_train[filtered_col]
reduced_features_test_set = decor_dataset_test[filtered_col]


# parameters for XGBoost model

param_test_xgb = {
        'max_depth': range(2, 6, 1),
        'min_child_weight': [1], #range(1, 6, 2)
        'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5], #was originally [i * 0.1 for i in range(1, 10)]
        'n_estimators': [200, 300, 400, 500], #int(x) for x in np.linspace(start=200, stop=400, num=100)
    }

kfold = StratifiedKFold(n_splits=10, random_state=310, shuffle=True) #increase this back to 10 ?
gsearch = GridSearchCV(model, param_grid=param_test_xgb, scoring='roc_auc_ovr_weighted', n_jobs=4, cv=kfold, verbose=1)

#measure time for gsearch
start_gsearch = time.time()
gsearch.fit(reduced_features_train_set, y_train)
T_single_gsearch = time.time() - start_gsearch


# predictions
best_estimator = gsearch.best_estimator_
proba_train = best_estimator.predict_proba(reduced_features_train_set)
proba_test = best_estimator.predict_proba(reduced_features_test_set)

results_train = get_multiclass_results(y_train, proba_train, "train")
results_test = get_multiclass_results(y_test, proba_test, "test")

results_overall = pd.concat([results_train, results_test])


print('\nOUTCOME DISTRIBUTION & PERFORMANCE ACROSS TRAIN/TEST: --------------------------------')

print(f'\ny_train: {y_train.value_counts()/len(y_train)} N = {len(y_train)}')
print(f'\ny_test: {y_test.value_counts()/len(y_test)} N = {len(y_test)}')
print('\nModel results:')
print(results_overall)

print('\nSELECTED FEATURES: -----------------------------------------------------------------')

print('\nSelected features: ' + str(filtered_col))
print(f'\nSelected {len(filtered_col)} out of {decor_dataset_test.shape[1]} features')

print('\nFEATURE SELECTION AND TRAINING TIMES: ------------------------------------------------')

#print(f'Training of singular xgboost model took {T_single_model1:.2f} seconds')
if T_single_rfecv is not None:
    print(f'\nRecursive feature elimination with cross validation took {T_single_rfecv / 60:.2f} minutes')
if T_single_rfe is not None:
    print(f'\nRecursive feature elimination took {T_single_rfe/ 60:.2f} minutes')

print(f'Gridsearch took {T_single_gsearch/ 60:.2f} minutes')


if is_optimal_features == True:
    filename_rfecv = path_to_save +'rfecv_radiomics.pkl'
    pickle.dump(rfecv, open(filename_rfecv, 'wb'))
else:
    filename_rfe = path_to_save +'rfe_radiomics.pkl'
    pickle.dump(rfe, open(filename_rfe, 'wb'))

filename_gsearch = path_to_save +'gsearch_radiomics.pkl'
pickle.dump(gsearch, open(filename_gsearch, 'wb'))
filename_parameters =path_to_save + r"parameters_radiomics.pkl"
pickle.dump([mean_std,selector, to_drop,support],open(filename_parameters, 'wb'))
filename_proba_train =path_to_save + r"proba_train_radiomics.pkl"
pickle.dump(proba_train,open(filename_proba_train, 'wb'))
filename_proba_test =path_to_save + r"proba_test_radiomics.pkl"
pickle.dump(proba_test,open(filename_proba_test, 'wb'))