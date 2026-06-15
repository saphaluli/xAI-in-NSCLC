# IMPORTS

import os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pydicom
import pydicom_seg as dcmseg
from radiomics import featureextractor
import pandas as pd
from tqdm import tqdm
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


### 2 - PYRADIOMICS SETUP AND EXTRACTION

# set up feature extractor for pyradiomics

def initialize_feature_extractor():
    paramsFile = "CEM_extraction.yaml"
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile, shape2D=True, force2D=True,
                                                            force2Ddimension=0, resampledPixelSpacing=None) #originally: force2DDimension=True, now set to 0 for axial plane
    extractor.addProvenance(False) #It's not necessary to resample PixelSpacing since it is consistent across the dataset.
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

# turn segmentation into slices instead of 3D object

def extract_slice(img, slice_no):
    size = list(img.GetSize())
    index = [0, 0, int(slice_no)]

    size[2] = 0  # extract 2D slice
    return sitk.Extract(img, size, index)

# fix alignment of the segmentation to the ct scan

def fix_seg(seg_img, ct_imgs):
    fixed_seg = sitk.Cast(seg_img, sitk.sitkUInt8)
    return fixed_seg

# per-slice extraction and record update
def extract_per_slice(extractor, fixed_seg, sitk_dcms, scan_id, records):
    for slice_no in range(sitk_dcms.GetSize()[2]):
        seg_slice = extract_slice(fixed_seg, slice_no)
        img_slice = extract_slice(sitk_dcms, slice_no)
        
        # Check if segmentation contains label 1 -> some slices will have no segmentation (label 0)
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
    ser_reader = sitk.ImageSeriesReader()
    extractor = initialize_feature_extractor()

    for _, row in tqdm(path_df.iterrows()):
        scan_id = row['scan_id']
        ct_path = row['path_ct']
        mask_path = row['path_mask']

        print(f'started processing scan: {scan_id}')

        # read segmentation file, read ct scan as series
        seg = pydicom.dcmread(list(mask_path.glob('*.dcm'))[0])
        result_seg = seg_reader.read(seg)
        dcm_files = ser_reader.GetGDCMSeriesFileNames(str(ct_path))
        ser_reader.SetFileNames(dcm_files)
        sitk_dcms = ser_reader.Execute()


        # find segmentation from neoplasm label
        seg_infos = result_seg.segment_infos
        for seg_num, info in seg_infos.items():

            if 'Neoplasm' in info.get('SegmentLabel', ''): #neoplasm label is available in all patients
                break
            else:
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

        print(f'finished processing scan: {scan_id}')

    return pd.DataFrame(records), mismatched_scans