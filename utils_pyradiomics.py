# IMPORTS

import os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pydicom
import pydicom_seg as dcmseg
from radiomics import featureextractor
import pandas as pd
import sklearn
from sklearn.feature_selection import VarianceThreshold


### 1 - DIRECTORY DF SETUP

#turn directory into df with segmentation and dicom paths

def create_path_df(general_dir):
    
    path_records = []

    for patient_dir in general_dir.iterdir():
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


### 2 - PYRADIOMICS SETUP AND EXTRACTION

# turn segmentation into slices instead of 3D object

def extract_slice(img, slice_no):
    size = list(img.GetSize())
    index = [0, 0, int(slice_no)]

    size[2] = 0  # extract 2D slice
    return sitk.Extract(img, size, index)


# set up feature extractor for pyradiomics

def initialize_feature_extractor():
    paramsFile = "CEM_extraction.yaml"
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile, shape2D=True, force2D=True,
                                                               force2Ddimension=True, resampledPixelSpacing=None)
    extractor.addProvenance(False)
    extractor.disableAllFeatures()
    extractor.enableImageTypes(Original={})

    extractor.enableFeatureClassByName('firstorder', enabled=True)
    extractor.enableFeatureClassByName('shape2D', enabled=True)
    extractor.enableFeatureClassByName('glcm', enabled=True)
    extractor.enableFeatureClassByName('glrlm', enabled=True)
    extractor.enableFeatureClassByName('glszm', enabled=True)
    extractor.enableFeatureClassByName('gldm', enabled=True)
    extractor.enableFeatureClassByName('ngtdm', enabled=True)
    return extractor

# fix alignment of the segmentation to the ct scan

def fix_seg(seg_img, ct_imgs):
    fixed_seg = sitk.Cast(seg_img, sitk.sitkUInt8)
    return fixed_seg

# per-slice extraction and record update
def extract_per_slice(extractor, fixed_seg, sitk_dcms, scan_id, records):
    for slice_no in range(sitk_dcms.GetSize()[2]):
        seg_slice = extract_slice(fixed_seg, slice_no)
        img_slice = extract_slice(sitk_dcms, slice_no)                # Check if segmentation contains label 1 -> some slices will have no segmentation
        
        if 1 not in sitk.GetArrayViewFromImage(seg_slice):
            continue

        features = extractor.execute(img_slice, seg_slice, label=1)
        record = {'PatientID': scan_id,
                  'slice_no': slice_no}
        
        record.update(features)
        records.append(record)
    return records


# extract radiomics features per slice

def extract_radiomics(path_df):

    # to create df from later
    records = []

    # this is to collect ct scans where there is mismatch between seg and ct slice counts
    mismatched_scans = []

    #some extractor and reader specifications
    seg_reader = dcmseg.SegmentReader()
    extractor = initialize_feature_extractor()
    ser_reader = sitk.ImageSeriesReader()

    for _, row in path_df.iterrows():
        scan_id = row['scan_id']
        ct_path = row['path_ct']
        mask_path = row['path_mask']

        # read segmentation file, read ct scan as series
        seg = pydicom.dcmread(list(mask_path.glob('*.dcm'))[0])
        result_seg = seg_reader.read(seg)
        dcm_paths = sorted(ct_path.glob('*.dcm'))
        dcm_files = ser_reader.GetGDCMSeriesFileNames(str(ct_path))
        ser_reader.SetFileNames(dcm_files)
        sitk_dcms = ser_reader.Execute()


        # find segmentation from neoplasm label
        seg_infos = result_seg.segment_infos
        for seg_num, info in seg_infos.items():

            #could make this more efficient by stopping the loop if the correct item was found?
            if 'Neoplasm' not in info.get('SegmentLabel', ''):
                continue
            neo_seg_num = seg_num
            neoplasm_segment_img = result_seg.segment_image(neo_seg_num)
        
            #need to cast the segmentation onto the same space as dicom image
            # otherwise radiomics will throw error because it thinks the segmentation is ever so slightly off due to data handling (by 0.0001 mm or so)
            fixed_seg = fix_seg(neoplasm_segment_img, sitk_dcms)

            # sanity check that they have the same dimensions, otherwise skip scan
            if fixed_seg.GetSize() != sitk_dcms.GetSize():
                print(f"Skipping {scan_id} due to size mismatch: seg={fixed_seg.GetSize()}, ct={sitk_dcms.GetSize()}")
                mismatched_scans.append(scan_id)
                continue

            fixed_seg.CopyInformation(sitk_dcms)

            #per-slice radiomics extraction
            records = extract_per_slice(extractor, fixed_seg, sitk_dcms, scan_id, records)

    return pd.DataFrame(records), mismatched_scans


### 3 - FEATURE PREPROCESSING FROM: Radiomics_for_CEM https://github.com/precision-medicine-um/Radiomics_for_CEM

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
    for var in df_true_mask_train_features.columns:
        temp_mean = df_true_mask_train_features[var].mean()
        temp_std = df_true_mask_train_features[var].std()
        mean_std[var] = (temp_mean, temp_std)
        df_true_mask_train_features[var] = (df_true_mask_train_features[var] - temp_mean) / temp_std
    ##remove low variance features
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(df_true_mask_train_features)
    thres_dataset_train = df_true_mask_train_features.loc[:, selector.get_support()]
    ## get_correlated_features_to_drop
    to_drop = get_correlated_features_to_drop(thres_dataset_train)
    decor_dataset = thres_dataset_train.drop(to_drop, axis=1)
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
    return optimal_threshold
