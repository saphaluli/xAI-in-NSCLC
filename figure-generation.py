import os
import pandas as pd
import numpy as np
from descriptives_table import variable_selection, split_by_class, calc_metrics, metrics_to_df


dir_path = os.path.expanduser('~/Documents/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv')
save_path = os.path.expanduser('~//Documents/GitHub/xAI-in-NSCLC/Table_1.csv')
df = pd.read_csv(dir_path)
df = df.drop(labels='PatientID', axis=1)
#enter name of outcome column here
outcome = 'Histology'

#Determine all of the variables that exist
#Determine whether these variables are categorical or continuous
cat_var, cont_var, outcome_classes = variable_selection(df, outcome)

#split dataframes by class 
df_dict = split_by_class(df,outcome)

#Calculate the corresponding metric for each of the
metrics_dicts = {
    name: calc_metrics(df, cat_var, cont_var)
    for name, df in df_dict.items()
}

metric_frames = {
    name: metrics_to_df(mdict, column_name=name)
    for name, mdict in metrics_dicts.items()
}

combined = pd.concat(metric_frames.values(), axis=1)
combined = combined.fillna('-')

combined.to_csv(save_path, index=True)