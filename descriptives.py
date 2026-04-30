import os
import pandas as pd
import numpy as np


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
def split_by_class(df, class_col, outcome_classes):
    df_list = []
    for cls in outcome_classes:
        df_subset = df.loc[df[class_col] == cls]
        df_list.append(df_subset)

    return df_list

def calc_metrics(df, cat_var, cont_var):

    metrics_dict = {}
    
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

            var_metrics.append((('{}').format(cls), ('{}% ({} / {})').format(round(pc, 2), count, total)))

        metrics_dict[var] = var_metrics

 
    for var in cont_var:

        mean = df[var].mean()
        std = df[var].std()

        var_metrics = [(('{} ± {}'.format(round(mean, 2), round(std, 2))))]

        metrics_dict[var] = var_metrics

    return metrics_dict





    

        
        