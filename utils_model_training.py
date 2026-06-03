#imports

import numpy as np
import pandas as pd

def merge_and_clean(features_df, clinical_df):
    merged_df = pd.merge(features_df, clinical_df[['PatientID', 'Histology']], on='PatientID', how='left')
    merged_df = merged_df.sort_values(by=['PatientID'], ascending=True)
    merged_df_clean = merged_df.dropna(subset=['Histology'])

    mapping = {'adenocarcinoma': 0, 'squamous cell carcinoma': 1, 'large cell': 2, 'nos':3 }
    merged_df_clean['Histology'] = merged_df_clean['Histology'].map(mapping)
    merged_df_clean['Histology'].unique()

    return merged_df_clean
