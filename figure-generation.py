import os
import pandas as pd
import numpy as np
from descriptives.py import variable_selection, split_by_class, calc_metrics


dir_path = os.expanduser('~/Documents/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv')
df = pd.read_csv(dir_path)
#enter name of outcome column here
outcome = 'Histology'

#Determine all of the variables that exist
#Determine whether these variables are categorical or continuous
cat_var, cont_var, outcome_classes = variable_selection(df, outcome)

#split dataframes by class 
df_list = split_by_class(df, outcome, outcome_classes)

#Calculate the corresponding metric for each of the
for df in df_list:

    