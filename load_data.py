# -*- coding: utf-8 -*-
"""
Load data for complete-case analysis
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
import sklearn.metrics  as metrics
from sklearn.svm import l1_min_c
import matplotlib.pyplot as plt
import math as ma
from sklearn.neighbors import KNeighborsRegressor



np.random.seed(10)

main_df = pd.read_csv('4C_Data_merged.csv')

# Create test variable CAD50
main_df = main_df.replace('.', np.nan)

main_df['CAD50'] = np.nan

main_df['CAD50_positive'] = ((main_df['prev_pci'].convert_objects(convert_numeric=True)==1) 
| (main_df['prev_cabg'].convert_objects(convert_numeric=True)==1) 
| (main_df['final_number_dis_territories'].convert_objects(convert_numeric=True)==1) 
| (main_df['final_number_dis_territories'].convert_objects(convert_numeric=True)==2) 
| (main_df['final_number_dis_territories'].convert_objects(convert_numeric=True)==3) 
| (main_df['grafts_number_dis_territories'].convert_objects(convert_numeric=True)==1) 
| (main_df['grafts_number_dis_territories'].convert_objects(convert_numeric=True)==2) 
| (main_df['grafts_number_dis_territories'].convert_objects(convert_numeric=True)==3)).astype(int).replace(0,np.nan)

main_df.loc[main_df['CAD50_positive'] == 1, 'CAD50'] = 1

main_df.loc[main_df['final_number_dis_territories'].convert_objects(convert_numeric=True) == 0, 'CAD50'] = 0

# Create variable which includes "9" as moderate CAD, 0 no CAD, 1 CAD, 0.5 moderate CAD
main_df['CAD50_moderate'] = main_df['CAD50']
main_df.loc[main_df['final_number_dis_territories'].convert_objects(convert_numeric=True) == 9, 'CAD50_moderate'] = 0.5

# Create variable which includes "9" as CAD
main_df['CAD50_moderate_as_CAD'] = main_df['CAD50']
main_df.loc[main_df['final_number_dis_territories'].convert_objects(convert_numeric=True) == 9, 'CAD50_moderate_as_CAD'] = 1

# Create variable which includes "9" as not CAD
main_df['CAD50_moderate_as_noCAD'] = main_df['CAD50']
main_df.loc[main_df['final_number_dis_territories'].convert_objects(convert_numeric=True) == 9, 'CAD50_moderate_as_noCAD'] = 0


# Remove variables not of interest - leave only metabolities and output
metsout_df = pd.concat([main_df.ix[:,203:428], main_df['CAD50']], axis=1)

# How many rows with all metabolite measures nan? Is Gp a metabolite? Yes?
metsout_df.ix[:,0:225].isnull().all(axis=1)
sum(metsout_df.ix[:,0:225].isnull().all(axis=1))
# 1086


# Remove rows where ALL variables are nan
metsout_nan_df = metsout_df.dropna(axis = 0, how='all', subset = metsout_df.ix[:,0:225].columns)
# 2323 records remaining 

# Remove rows where output is missing
metsout_nan1_df = metsout_nan_df.dropna(subset = metsout_nan_df.ix[:,225:226].columns)
# 1474 records remaining (using CAD50)

# Replace TAG and NDEF with nan
metsout_nan2_df = metsout_nan1_df.replace('NDEF', np.nan)
metsout_nan3_df = metsout_nan2_df.replace('TAG', np.nan)

# Convert columns which had TAG, NDEF in to numeric
for i in range(0,225):
    metsout_nan3_df.ix[:,i] = metsout_nan3_df.ix[:,i].convert_objects(convert_numeric=True)
    
# Cleaned data: EITHER leave missing or drop all
data_withmissing_df = metsout_nan3_df
data_withoutmissing_df = metsout_nan3_df.dropna()
# 694 remaining (using CAD50)
    
#data_df = metsout_nan3_df
data_df = metsout_nan3_df.dropna()

# Print non-standardised dataset to csv
data_withmissing_df.to_csv('full_nonstandard_data_with_missing.csv')
#data_withoutmissing_df.to_csv('nonstandard_data_without_missing.csv')

##################################################################
### Summaries of data ###
##################################################################

# How many patients with any metabolomics data (from original)
sum(main_df.ix[:,203:428].notnull().any(axis=1))
# 2323 (1086 not)

# How many patients with full metabolomics data 
sum(main_df.ix[:,203:428].notnull().all(axis=1))
# 2060 (1349 not)

# How many patients with outcomes (from original) (either test var and calculated as Gaz said)
sum(main_df['CAD50'].notnull())
# 2029 for testvaribale
# 1817 for CAD50 

# How many patients with outcomes and some metabolomics
sum((main_df.ix[:,203:428].notnull().any(axis=1) & main_df['CAD50'].notnull()))
# 1661 testvariable
# 1474 CAD50

# How many patients with outcomes and all metabolomics
sum((main_df.ix[:,203:428].replace('TAG', np.nan).replace('NDEF', np.nan).notnull().all(axis=1) & main_df['testvariable'].notnull()))
# 794 testvariable
# 694 CAD50 


# Differences in outcome variable between complete/incomplete 
sum(main_df['CAD50'])



##################################################################
### Standardise data ###
##################################################################

data1 = data_df

# Standarize data to zero mean and unit variance
col_means = data1.ix[:,0:225].mean(axis=0)
data1_mean_adjusted = data1.ix[:,0:225] - col_means # effectively 0, but not quite due to numerical innacuracies
# TO DO: make this accurate?

col_variance = data1_mean_adjusted.var(axis=0)
col_sd = [0] * len(col_variance)
for x in list(range(len(col_variance))):
    col_sd[x] = ma.sqrt(col_variance[x])

data1_standardised = data1_mean_adjusted/col_sd

data1_full_sd = pd.concat([data1_standardised, data1.ix[:,225]], axis=1)



##################################################################
### Create training and test sets ###
##################################################################

# Create training set and test set, 3/4 vs 1/4. Randomize split across rows to
# avoid systematic factors in ordering to interfere.

#train1, test1 = train_test_split(data1_full_sd, test_size = 0.25, random_state=42)
train1, test1 = train_test_split(data1_full_sd, test_size = 0.25, random_state=42)

# Re-set indicies on train and test sets to be able to split them
train1 = train1.reset_index(drop=True)
test1 = test1.reset_index(drop=True)

