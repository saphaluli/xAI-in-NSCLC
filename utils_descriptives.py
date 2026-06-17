import os
import pandas as pd
import numpy as np
from scipy.stats import f_oneway


# select variables and determine type
def variable_selection(df, class_col):
    df_var = df.drop(labels = class_col, axis = 1)
    cols = df_var.columns
    cat_var = []
    cont_var = []


    for col in cols:
        if len(df[col].unique()) <= 10:
            cat_var.append(col)
        else:
            cont_var.append(col)
    
    outcome_classes = df[class_col].dropna().unique()
        
    return cat_var, cont_var, outcome_classes

# split dataset by class
def split_by_class(df, class_col):
    df_dict = {}
    outcome_classes = df[class_col].dropna().unique()
    
    df_dict['Total'] = df

    for cls in outcome_classes:
        df_subset = df.loc[df[class_col] == cls]
        df_dict[cls] = df_subset

    return df_dict

def calc_metrics(df, cat_var, cont_var):
    
    metrics_dict = {}

    df = df.dropna(subset=outcome)
    
    for var in cat_var:
        # calculate % and absolute counts for each metric (count / total)
        subclasses = df[var].dropna().unique().tolist()
        subclasses.sort()

        var_metrics = []


        for cls in subclasses:
            
            #for each calsl, count how many have cls true, and take it over th etotal for that var

            count = df.loc[df[var] == cls][var].count()
            total = df[var].count()
            pc = (count / total) * 100

            #Chi square?

            var_metrics.append((('{}').format(cls), ('{}% ({} / {})').format(round(pc, 2), count, total)))

        metrics_dict[var] = var_metrics

 
    for var in cont_var:

        mean = df[var].mean()
        std = df[var].std()

        var_metrics = [(None, ('{} ± {}'.format(round(mean, 2), round(std, 2))))]

        metrics_dict[var] = var_metrics

    return metrics_dict   

def metrics_to_df(metrics_dict, column_name):
    rows = []
    for variable, entries in metrics_dict.items():
        for level, metric in entries:
            rows.append((variable, level, metric))
    
    df = pd.DataFrame(rows, columns=["variable", "level", column_name])
    return df.set_index(["variable", "level"])   