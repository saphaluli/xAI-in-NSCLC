import os
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_selection import VarianceThreshold


### 1 - DIRECTORY DF SETUP

#turn directory into df with segmentation and dicom paths

def create_path_df(general_dir):
    
    path_records = []

    for patient_dir in general_dir.iterdir():
        if not patient_dir.is_dir():
            continue

        scan_id = patient_dir.name
        
        for study_dir in patient_dir.iterdir():
            if not study_dir.is_dir():
                continue

            for series_dir in study_dir.iterdir():
                if not series_dir.is_dir():
                    continue

                #select whether ct scan series or segmentation based on name/length
                if 'Segmentation' in series_dir.name and any(series_dir.glob('*.dcm')):
                    seg_series = series_dir
                    continue

                if any(series_dir.glob('*.dcm')) and len(list(series_dir.glob('*.dcm'))) >= 10:
                    ct_series = series_dir

            if ct_series is not None and seg_series is not None:
                path_records.append({
                    'scan_id': scan_id,
                    'path_ct': ct_series,
                    'path_mask':seg_series
                })
            else:
                print(f"No valid paths for {patient_dir.name}/{study_dir.name}: ct_series={ct_series is not None}, seg_series={seg_series is not None}")

    path_df = pd.DataFrame(path_records, columns=['scan_id', 'path_ct', 'path_mask'])

    return path_df

### 3 - FEATURE PREPROCESSING FROM: Partially from Radiomics_for_CEM https://github.com/precision-medicine-um/Radiomics_for_CEM

#Turning radiomics into per-patient statistics

def generate_features_table(df):
    means = df.groupby(by='PatientID').mean()
    means.columns = means.columns + '_mean'

    stdevs = df.groupby(by='PatientID').std()
    stdevs.columns = stdevs.columns +'_stdev'

    maxs = df.groupby(by='PatientID').max()
    maxs.columns = maxs.columns +'_max'

    mins = df.groupby(by='PatientID').min()
    mins.columns = mins.columns +'_min'

    full_df = pd.concat([means, stdevs, maxs, mins], axis=1, sort=True, ignore_index=False)
    full_df = full_df.reset_index()
    return full_df

############ pre-process feature table ######################

def get_correlated_features_to_drop(thres_dataset_train):
    cor = thres_dataset_train.corr('spearman').abs()
    upper_tri = cor.where(np.triu(np.ones(cor.shape), k=1).astype(bool))
    to_drop = []
    for column in upper_tri.columns:
        for row in upper_tri.columns:
            if upper_tri[column][row] > 0.85:
                if np.sum(upper_tri[column]) > np.sum(
                        upper_tri[row]):
                    to_drop.append(column)
                else:
                    to_drop.append(row)
    to_drop = np.unique(to_drop)
    return to_drop


def preprocessing_train(df_true_mask_train_features):  ##patient name needs to be removed
    ##normalize the features
    mean_std = {}
    print(f'Original size: {df_true_mask_train_features.shape[1]}')
    for var in df_true_mask_train_features.columns:
        temp_mean = df_true_mask_train_features[var].mean()
        temp_std = df_true_mask_train_features[var].std()
        mean_std[var] = (temp_mean, temp_std)
        df_true_mask_train_features[var] = (df_true_mask_train_features[var] - temp_mean) / temp_std
    ##remove low variance features
    selector = VarianceThreshold(threshold=0.01)
    variances = df_true_mask_train_features.var()
    selector.fit(df_true_mask_train_features)
    thres_dataset_train = df_true_mask_train_features.loc[:, selector.get_support()]
    print(f'Size after removing low-variance features: {thres_dataset_train.shape[1]}')
    print(f'Difference to previous: {df_true_mask_train_features.shape[1] - thres_dataset_train.shape[1]}')
    ## get_correlated_features_to_drop
    to_drop = get_correlated_features_to_drop(thres_dataset_train)
    decor_dataset = thres_dataset_train.drop(to_drop, axis=1)
    print(f'Size after removing highly correlated features: {decor_dataset.shape[1]}')
    print(f'Difference to previous: {thres_dataset_train.shape[1] - decor_dataset.shape[1]}')
    return mean_std, selector, to_drop, decor_dataset


def preprocessing_test(df_true_mask_test_features, mean_std, selector, to_drop):  ##apply parameters to test dataset
    for var in df_true_mask_test_features.columns:
        df_true_mask_test_features[var] = (df_true_mask_test_features[var] - mean_std[var][0]) / mean_std[var][1]
    thres_dataset_test = df_true_mask_test_features.loc[:, selector.get_support()]
    decor_dataset_test = thres_dataset_test.drop(to_drop, axis=1)
    return decor_dataset_test


### 4 - MODEL TRAINING AND FINETUNING (XGBoost)

def get_optimal_threshold(true_outcome, predictions, pos_label=1):
    ## to obtain a good threshold based on the train dataset for binary classification only
    y_type = sklearn.utils.multiclass.type_of_target(true_outcome)
    if y_type != 'binary':
        raise ValueError('get_optimal_threshold only supports binary classification. For multiclass, use argmax over class probabilities and multiclass metrics.')
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(true_outcome, predictions, pos_label=pos_label)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f'Youden\'s index: {optimal_threshold}')
    return optimal_threshold


def merge_and_clean(features_df, clinical_df, mapping, outcome):
    merged_df = pd.merge(features_df, clinical_df[['PatientID', outcome]], on='PatientID', how='left')
    merged_df = merged_df.sort_values(by=['PatientID'], ascending=True)
    merged_df_clean = merged_df.dropna(subset=[outcome]) # drop patients without histological assessment
    merged_df_clean[outcome] = merged_df_clean[outcome].map(mapping)
    merged_df_clean[outcome].unique()

    return merged_df_clean

def bootstrap(label, pred, f, nsamples=2000):

    label = np.asarray(label)
    pred = np.asarray(pred)

    stats = []
    for b in range(nsamples):
        random_list = np.random.randint(label.shape[0], size=label.shape[0])
        stats.append(f(label[random_list], pred[random_list]))
    return stats, np.percentile(stats, (2.5, 97.5))



def nom_den(label, pred, f):
    if f == sklearn.metrics.accuracy_score:
        n = np.sum(label == pred)
        d = len(pred)
    if f == sklearn.metrics.precision_score:
        n = np.sum(pred[label == 1])
        d = np.sum(pred)
    if f == sklearn.metrics.recall_score:
        n = np.sum(pred[label == 1])
        d = np.sum(label)
    if f == sklearn.metrics.f1_score:
        n = 0
        d = 0
    return n, d

def get_multiclass_results(y_true, proba, label, average='weighted'):
    ## multiclass classification: choose argmax class and evaluate multiclass metrics
    y_true_arr = np.asarray(y_true)
    y_pred = np.argmax(proba, axis=1)
    classes = np.unique(y_true)
    y_true_bin = sklearn.preprocessing.label_binarize(y_true, classes=classes)
    dict_results = {}
    dict_results["auc_ovr_weighted"] = sklearn.metrics.roc_auc_score(
        y_true_bin,
        np.asarray(proba),
        multi_class='ovr',
        average=average # maybe it's something to do with this?
    )
    dict_results["accuracy"] = sklearn.metrics.accuracy_score(y_true, y_pred)
    dict_results["precision"] = sklearn.metrics.precision_score(y_true, y_pred, average=average, zero_division=0)
    dict_results["recall"] = sklearn.metrics.recall_score(y_true, y_pred, average=average, zero_division=0)
    dict_results["f1 score"] = sklearn.metrics.f1_score(y_true, y_pred, average=average, zero_division=0)
    df_results = pd.DataFrame.from_dict([dict_results])
    df_results.index = [label]
    return df_results

def get_ci(label, pred, f):
    stats, ci = bootstrap(label, pred, f)
    n, d = nom_den(label, pred, f)
    return stats, ["%5d/%5d (%5d %% )  CI [%0.2f,%0.2f]" % (
    n, d, int(f(label, pred) * 100), ci[0], ci[1])]  # doesn't compute the mean of the score


def get_ci_for_auc(label, pred, nsamples=2000):
    auc_values = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    for b in range(nsamples):
        idx = np.random.randint(label.shape[0], size=label.shape[0])
        temp_pred = pred[idx]
        temp_fpr, temp_tpr, temp_thresholds = sklearn.metrics.roc_curve(label.iloc[idx], temp_pred)
        roc_auc = sklearn.metrics.auc(temp_fpr, temp_tpr)
        auc_values.append(roc_auc)
        interp_tpr = np.interp(mean_fpr, temp_fpr, temp_tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    ci_auc = np.percentile(auc_values, (2.5, 97.5))
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(label, pred)
    return auc_values, ["%0.2f CI [%0.2f,%0.2f]" % (sklearn.metrics.auc(fpr, tpr), ci_auc[0], ci_auc[1])]

def get_stats_with_ci(y_label, y_pred, label, optimal_threshold, nsamples=2000):
    ##optimal threshold: reuse the one computed on the train dataset
    ##label: index of the dataframe, can be "external radiomics results"
    ##returns a dataframe with auc accuracy precision recall f1-score
    dict_results = {}
    dict_distributions = {}
    dict_distributions["auc"], dict_results["auc"] = get_ci_for_auc(y_label, y_pred)
    y_pred_binary = (np.array(y_pred) > optimal_threshold).astype(int)
    dict_distributions["accuracy"], dict_results["accuracy"] = get_ci(y_label, y_pred_binary,
                                                                    sklearn.metrics.accuracy_score)
    dict_distributions["precision"], dict_results["precision"] = get_ci(y_label, y_pred_binary,
                                                                        sklearn.metrics.precision_score)
    dict_distributions["specificity"], dict_results["specificity"] = get_ci(np.ones(len(y_label)) - y_label,
                                                                            np.ones(len(y_pred_binary)) - y_pred_binary,
                                                                            sklearn.metrics.recall_score)
    dict_distributions["recall"], dict_results["recall"] = get_ci(y_label, y_pred_binary, sklearn.metrics.recall_score)
    dict_distributions["f1 score"], dict_results["f1 score"] = get_ci(y_label, y_pred_binary, sklearn.metrics.f1_score)
    df_results = pd.DataFrame.from_dict(dict_results)
    df_results = df_results.reset_index(drop=True)
    df_results.index = [label]
    df_distributions = pd.DataFrame.from_dict(dict_distributions)
    df_distributions = df_distributions.reset_index(drop=True)
    return df_distributions, df_results