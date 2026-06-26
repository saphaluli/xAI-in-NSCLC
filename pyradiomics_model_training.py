# Code modified from 
#imports 
import os
import time
import random
import pickle
import pandas as pd
import argparse
import numpy as np
import xgboost as xgb
import sklearn
#import shap
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from utils_model_training import preprocessing_train, preprocessing_test, get_optimal_threshold, merge_and_clean, get_multiclass_results, get_stats_with_ci, generate_features_table

### SETTINGS
# is_by_patient -> splits dataset by patientID first if True
# is_single_slice -> takes only one slice per patient for training or testing if True
is_bypatient = bool(True)
is_single_slice = bool(False)

# True -> RFECV, False -> RFE
is_optimal_features = bool(True)

# datasets 
clinical_df = pd.read_csv(os.path.expanduser('~/project/xAI-in-NSCLC/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv'))
features_df = pd.read_csv(os.path.expanduser('~/project/xAI-in-NSCLC/full_radiomics_per_slice.csv'))
path_to_save = os.path.expanduser('~/project/xAI-in-NSCLC')

#outcome_stage = 'Histology'
outcome_stage = 'Overall.Stage'
# seed
seed = 204

#generating per-patient statistics

features_df = features_df.drop(columns=['slice_no'])
#features_df = generate_features_table(features_df)

# merging and cleaning datasets
mapping = {'I': 0, 'II': 0, 'IIIa':1, 'IIIb':1}
#mapping = {'adenocarcinoma': 0, 'squamous cell carcinoma': 1, 'large cell': 2, 'nos':3 }
merged_df = merge_and_clean(features_df, clinical_df, mapping, outcome_stage)


# only extract one slice per patient ID
if is_single_slice == True:
    merged_df = merged_df.groupby(by='PatientID').sample(n=1, random_state=seed).reset_index(drop=True)


if is_bypatient == True:
    # test train split by patient ID
    temp_df = clinical_df.dropna(subset=[outcome_stage])
    X = temp_df['PatientID']
    y = temp_df[outcome_stage]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)

    # save patient IDs for deep learning test train split later
    train_labels = X_train.unique()
    test_labels = X_test.unique()


    # create test train split dfs for training
    X_train = merged_df.loc[merged_df['PatientID'].isin(train_labels)]
    y_train = X_train[outcome_stage]
    X_test = merged_df.loc[merged_df['PatientID'].isin(test_labels)]
    y_test = X_test[outcome_stage]

    X_train = X_train.drop(columns=['PatientID', outcome_stage])#, 'slice_no'])
    X_test = X_test.drop(columns=['PatientID', outcome_stage])#, 'slice_no'])

else:
    #test train split NOT by patient ID, but by slice

    X = merged_df.drop(columns=['PatientID', outcome_stage])
    y = merged_df[outcome_stage]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)

print(f'Number of outcome classes: {len(y_train.unique())}')



# preprocess training dataset first 
mean_std, selector, to_drop, decor_dataset_train = preprocessing_train(X_train)
decor_dataset_test = preprocessing_test(X_test, mean_std, selector, to_drop)
print('features processed. New shape of training dataset:', decor_dataset_train.shape, 'before:', X_train.shape)

if len(y_train.unique()) > 2:
    objective = 'multi:softprob'
else:
    objective = 'binary:logistic'
    class_weight = len(y_train.loc[y_train == 0]) / len(y_train.loc[y_train == 1])
    print(f'class_weight: {class_weight}')

#model definition
model = xgb.XGBClassifier(enable_categorical=True, colsample_bytree=1, eta=0.01, max_depth=4,
                            objective=objective, scale_pos_weight=class_weight,  eval_metric='logloss', nthread=8, #'multi:softprob'
                            gamma=0.5, seed=seed)

#measuring time to train one model
#start_model1 = time.time()
#model.fit(X_train, y_train)
#T_single_model1 = time.time() - start_model1

# recursive feature elimination with cross validation:
min_features_to_select = 1

T_single_rfecv = None
T_single_rfe = None

scoring = 'roc_auc'#_ovr_weighted'

if is_optimal_features == True:

    print('Performing RFECV')
    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(10),
                scoring=scoring,
                min_features_to_select=min_features_to_select)

    start_rfecv = time.time()
    rfecv.fit(decor_dataset_train, y_train)
    T_single_rfecv = time.time() - start_rfecv
    support = rfecv.support_

    fig, ax = plt.subplots(figsize=(8, 6))
    no_features = rfecv.cv_results_['n_features']

    # plot line for each fold
    for k in range(10):
        scores = rfecv.cv_results_[f"split{k}_test_score"]
        ax.plot(no_features, scores, alpha=0.5, label=f"Fold {k}")

    # mean score
    mean_scores = rfecv.cv_results_['mean_test_score']
    ax.plot(no_features, mean_scores, color='black', linewidth=2, label='Mean')

    ax.set(xlabel='no. of features', ylabel='roc_auc')
    ax.set_title('Results of RFECV cross validation')
    ax.legend()
    fig.savefig(path_to_save + r'/RFECV_results.png')
    plt.close(fig)

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

kfold = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True) #increase this back to 10 ?
gsearch = GridSearchCV(model, param_grid=param_test_xgb, scoring=scoring, n_jobs=4, cv=kfold, verbose=1) #'roc_auc_ovr_weighted'

#measure time for gsearch
start_gsearch = time.time()
gsearch.fit(reduced_features_train_set, y_train)
T_single_gsearch = time.time() - start_gsearch


# predictions
best_estimator = gsearch.best_estimator_
proba_train = best_estimator.predict_proba(reduced_features_train_set)
proba_test = best_estimator.predict_proba(reduced_features_test_set)


if len(y_train.unique()) > 2:

    results_train = get_multiclass_results(y_train, proba_train, "train")
    results_test = get_multiclass_results(y_test, proba_test, "test")

    results_overall = pd.concat([results_train, results_test])

else:
    temp_proba_train = proba_train[:, 1]
    temp_proba_test = proba_test[:, 1]
    optimal_threshold = get_optimal_threshold(y_train, temp_proba_train, pos_label=1)

    df_distributions_train, df_results_train = get_stats_with_ci(y_train, temp_proba_train, 'train', optimal_threshold, nsamples=2000)
    df_distributions_test, df_results_test = get_stats_with_ci(y_test, temp_proba_test, 'test', optimal_threshold, nsamples=2000)

    results_overall = pd.concat([df_results_train, df_results_test])

#SHAP explanation 

explainer = shap.TreeExplainer(best_estimator)
shap_values = explainer(reduced_features_test_set)

if len(y_train.unique()) > 2:
    print('No multiclass SHAP implemented')

else:
    # beeswarm // whole model
    fig = shap.plots.beeswarm(shap_values, max_display=reduced_features_test_set.shape[1], plot_size=[10, 6], show=False)
    plt.savefig(str(path_to_save) + '/SHAP-figures/model_shap.png', bbox_inches='tight')

    # getting indices of positive and negative predictions

    neg = y_test.loc[y_test == 0]
    pos = y_test.loc[y_test == 1]

    neg_list = neg.index
    pos_list = pos.index

    neg_idx = random.choice(neg_list)
    pos_idx = random.choice(pos_list)

    neg_idx = y_test.index.get_loc(neg_idx)
    pos_idx = y_test.index.get_loc(pos_idx)

    # negative plot
    fig, ax = plt.subplots()

    shap.plots.waterfall(shap_values[neg_idx], max_display=10, show=False)
    fig.set_size_inches(10, 6)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(str(path_to_save) + '/SHAP-figures/Negative_patient_shap.png', bbox_inches='tight')

    plt.close(fig)

    # positive plot
    fig, ax = plt.subplots()

    shap.plots.waterfall(shap_values[pos_idx], max_display=10, show=False) #reduced_features_test_set.shape[1] for full
    fig.set_size_inches(10, 6)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(str(path_to_save) + '/SHAP-figures/Positive_patient_shap.png', bbox_inches='tight')

    plt.close(fig)

    print('Saved SHAP analysis plots')


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
    print(f'\nRecursive feature elimination with cross validation took {T_single_rfecv / 60:.2f} minutes or {T_single_rfecv:.2f} seconds')
if T_single_rfe is not None:
    print(f'\nRecursive feature elimination took {T_single_rfe/ 60:.2f} minutes or {T_single_rfe:.2f} seconds')

print(f'Gridsearch took {T_single_gsearch/ 60:.2f} minutes or {T_single_gsearch:.2f} seconds')



if is_optimal_features == True:
    filename_rfecv = path_to_save +'/pyradiomics_savedmodels/rfecv_radiomics.pkl'
    pickle.dump(rfecv, open(filename_rfecv, 'wb'))
else:
    filename_rfe = path_to_save +'/pyradiomics_savedmodels/rfe_radiomics.pkl'
    pickle.dump(rfe, open(filename_rfe, 'wb'))

filename_gsearch = path_to_save +'/pyradiomics_savedmodels/gsearch_radiomics.pkl'
pickle.dump(gsearch, open(filename_gsearch, 'wb'))
filename_parameters =path_to_save + r"/pyradiomics_savedmodels/parameters_radiomics.pkl"
pickle.dump([mean_std,selector, to_drop,support],open(filename_parameters, 'wb'))
filename_proba_train =path_to_save + r"/pyradiomics_savedmodels/proba_train_radiomics.pkl"
pickle.dump(proba_train,open(filename_proba_train, 'wb'))
filename_proba_test =path_to_save + r"/pyradiomics_savedmodels/proba_test_radiomics.pkl"
pickle.dump(proba_test,open(filename_proba_test, 'wb'))
filename_y_train =path_to_save + r"/pyradiomics_savedmodels/y_train_radiomics.pkl"
pickle.dump(y_train,open(filename_y_train, 'wb'))
filename_y_test =path_to_save + r"/pyradiomics_savedmodels/y_test_radiomics.pkl"
pickle.dump(y_test,open(filename_y_test, 'wb'))