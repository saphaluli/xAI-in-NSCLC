# imports
import os
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pydicom
import pydicom_seg as dcmseg
from radiomics import featureextractor
import pandas as pd
import logging
from utils_pyradiomics import extract_radiomics, create_path_df

# hiding pyradiomics info, clogs up terminal
logging.getLogger('radiomics').setLevel(logging.ERROR)
logging.getLogger('radiomics.featureextractor').setLevel(logging.ERROR)
logging.getLogger('pyradiomics').setLevel(logging.ERROR)

general_dir = Path(os.path.expanduser('~/Documents/Test-NSCLC/'))
path_csv_dir = Path(os.path.expanduser('~/Documents/GitHub/xAI-in-NSCLC/path_records.csv'))

#below gives error -> str object has no attribute 'glob'
#if stored in df need to convert to path first                   
#if path_csv_dir.exists():
#    path_df = pd.read_csv(path_csv_dir)
#else:
path_df = create_path_df(general_dir)

records_df, mismatched_scans = extract_radiomics(path_df)

print(mismatched_scans)
records_df.to_csv('radiomics_features_per_slice.csv', index=False)



