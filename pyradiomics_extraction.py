# imports
import os
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
from utils_pyradiomics import extract_radiomics, create_path_df

# hiding pyradiomics info, clogs up terminal
logging.getLogger('radiomics').setLevel(logging.ERROR)
logging.getLogger('radiomics.featureextractor').setLevel(logging.ERROR)
logging.getLogger('pyradiomics').setLevel(logging.ERROR)

general_dir = Path(os.path.expanduser('~/project/xAI-in-NSCLC/NSCLC-Radiomics'))
path_csv_dir = None

#below gives error -> str object has no attribute 'glob'
#if stored in df need to convert to path first                   
#if path_csv_dir.exists():
#    path_df = pd.read_csv(path_csv_dir)
#else:
path_df = create_path_df(general_dir)

start = time.time()
features, mismatched_scans = extract_radiomics(path_df)
end = time.time()

records_df = pd.DataFrame(features)

#records_df, mismatched_scans = extract_radiomics(path_df)

print(f'time elapsed: {(end - start) / 60:.2f} minutes')
print('mismatched scan list: ', mismatched_scans)
records_df.to_csv('FULL_radiomics_features_per_slice.csv', index=False)



