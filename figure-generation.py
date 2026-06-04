import os
import pandas as pd
import numpy as np
from utils_descriptives import variable_selection, split_by_class, calc_metrics, metrics_to_df

#insert the tables and figures you want here:
#Note: the figures should correspond to the figure number in the paper.

desired = [
            #'table_1',
            'figure_1'
]

# insert directories here.
clinical_dir_path = os.path.expanduser('~/project/xAI-in-NSCLC/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv')
radiomics_dir_path = os.path.expanduser('~/project/xAI-in-NSCLC/FULL-radiomics_features_per_slice.csv')


### TABLE 1- DESCRIPTIVES FOR 

if 'table_1' in desired:
    df = pd.read_csv(clinical_dir_path)
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

    combined.to_csv(os.path.expanduser('~/project/xAI-in-NSCLC/Table_1.csv'), index=True)

if 'figure_1' is in desired:
    df = pd.read_csv(clinical_dir_path)
    df = df.drop(labels=['PatientID', 'slice_no'], axis=1)

    feature_names = sorted(filter ( lambda k: k.startswith("original_"), df.columns))

    #make dataframe with features
    d = df.iloc[:,feature_names]
    # Choose a subset of features for clustering
    dd = d.iloc[:,1:50]

    pp = sns.clustermap(dd.corr(), linewidths=.5, figsize=(13,13))
    _ = plt.setp(pp.ax_heatmap.get_yticklabels(), rotation=0)

    plt.show()