# imports
import os
import sys
import time
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pydicom
import pydicom_seg as dcmseg
from radiomics import featureextractor
import pandas as pd
import logging
from concurrent.futures import ProcessPoolExecutor
from test_utils_pyradiomics import extract_radiomics, create_path_df

# hiding pyradiomics info, clogs up terminal
logging.getLogger('radiomics').setLevel(logging.ERROR)
logging.getLogger('radiomics.featureextractor').setLevel(logging.ERROR)
logging.getLogger('pyradiomics').setLevel(logging.ERROR)


general_dir = Path(os.path.expanduser('~/project/xAI-in-NSCLC/NSCLC-Radiomics'))
#todo: Add option to provide csv with path?

path_df = create_path_df(general_dir)
path_df = path_df.sort_values(by='scan_id', ascending=True, ignore_index=True) #make sure to ignore index

print(f'Processing {path_df.shape[0]} scans. Should be: 422')
start = time.time()
features, mismatched_scans = extract_radiomics(path_df)
end = time.time()
records_df = pd.DataFrame(features)

print(f'time elapsed: {(end - start) / 60:.2f} minutes')
print('mismatched scan list: ', mismatched_scans)
records_df.to_csv('FULL_radiomics_features_per_slice.csv', index=False)