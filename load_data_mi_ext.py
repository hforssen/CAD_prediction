# -*- coding: utf-8 -*-
"""
Load data
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



main_df['diabetes'] = np.nan
# Set to 0 when CAD50 available, and diab_final not 1 (zero can be coded as null in dataset)
main_df['diabetes'] = (((main_df['CAD50'].isnull()==False) & (main_df['diab_final'] != 1))==False).astype(int)
main_df['diabetes'] = main_df['diabetes'].replace(1,np.nan)
#Set to 1 when diab_final = 1
main_df.loc[main_df['diab_final'] == 1, 'diabetes'] = 1

main_df['statin'] = np.nan
# Set to 0 when CAD50 available, and diab_final not 1 (zero can be coded as null in dataset)
main_df['statin'] = (((main_df['CAD50'].isnull()==False) & (main_df['statin_meds_ref'] != '1'))==False).astype(int)
main_df['statin'] = main_df['statin'].replace(1,np.nan)
#Set to 1 when diab_final = 1
main_df.loc[main_df['statin_meds_ref'] == '1', 'statin'] = 1

main_df['current_smoking'] = np.nan
# Set to zero if a past smoker or never smoked, 1 for current smoker
main_df.loc[main_df['rf_smoking'] == '2', 'current_smoking'] = 0
main_df.loc[main_df['rf_smoking'] == '3', 'current_smoking'] = 0
main_df.loc[main_df['rf_smoking'] == '1', 'current_smoking'] = 1

main_df['sex'] = np.nan
# Set to zero if a past smoker or never smoked, 1 for current smoker
main_df.loc[main_df['Sex'] == '2', 'sex'] = 0
main_df.loc[main_df['Sex'] == '1', 'sex'] = 1



# Remove variables not of interest - leave only metabolities and output
metsout_df = pd.concat([main_df.ix[:,203:428], main_df['Age'], main_df['sex'], main_df['hypertension_4c'],  main_df['obesity_bmi_v'], main_df['diabetes'],
                        main_df['statin'], main_df['current_smoking'], main_df['CAD50']], axis=1)



# How many rows with all metabolite measures nan? Is Gp a metabolite? Yes?
metsout_df.ix[:,0:225].isnull().all(axis=1)
sum(metsout_df.ix[:,0:225].isnull().all(axis=1))
# 1086


# Remove rows where ALL variables are nan
metsout_nan_df = metsout_df.dropna(axis = 0, how='all', subset = metsout_df.ix[:,0:225].columns)
# 2323 records remaining 

# Remove rows where output is missing
metsout_nan1_df = metsout_nan_df.dropna(subset = metsout_nan_df.ix[:,232:233].columns)
# 1474 records remaining (using CAD50)

# Replace TAG and NDEF with nan
metsout_nan2_df = metsout_nan1_df.replace('NDEF', np.nan)
metsout_nan3_df = metsout_nan2_df.replace('TAG', np.nan)

# Convert columns which had TAG, NDEF in to numeric
for i in range(0,227):
    metsout_nan3_df.ix[:,i] = metsout_nan3_df.ix[:,i].convert_objects(convert_numeric=True)
    
# Cleaned data: EITHER leave missing or drop all
data_withmissing_df = metsout_nan3_df
data_withoutmissing_df = metsout_nan3_df.dropna()
# 694 remaining (using CAD50)
    
data_df = metsout_nan3_df
#data_df = metsout_nan3_df.dropna()


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

data1 = data_df.dropna()

# Standarize data to zero mean and unit variance
col_means = data1.ix[:,0:225].mean(axis=0)
data1_mean_adjusted = data1.ix[:,0:225] - col_means # effectively 0, but not quite due to numerical innacuracies
# TO DO: make this accurate?

col_variance = data1_mean_adjusted.var(axis=0)
col_sd = [0] * len(col_variance)
for x in list(range(len(col_variance))):
    col_sd[x] = ma.sqrt(col_variance[x])

data1_standardised = data1_mean_adjusted/col_sd

data1_full_sd = pd.concat([data1_standardised, data1.ix[:,225:233]], axis=1)

#data1_full_sd.to_csv('full_data_with_missing_ext.csv')

################################
### Load non-standardised multiple imputation (MI) from R  ###
data1_full_imp = pd.read_csv('imp_data1_NEW_fullextended.csv').ix[:,1:234]

data1_full_imp1 = pd.read_csv('imp_data1_NEW_fullextended.csv').ix[:,1:234]
data1_full_imp2 = pd.read_csv('imp_data2_NEW_fullextended.csv').ix[:,1:234]
data1_full_imp3 = pd.read_csv('imp_data3_NEW_fullextended.csv').ix[:,1:234]
data1_full_imp4 = pd.read_csv('imp_data4_NEW_fullextended.csv').ix[:,1:234]
data1_full_imp5 = pd.read_csv('imp_data5_NEW_fullextended.csv').ix[:,1:234]


# No negatives when non-standardised imputation done. Better!
# No negatives on any of the imputed datasets, use this!

##################################################################
### Check that imputed values are realistic ###
##################################################################

# Find range of values of orginial data
data_max = data_df.max()
data_min = data_df.min()


# Unstandardise imputed dataset and check that no negatives
#data1_full_imp = (data1_full_sd_imp.ix[:,0:225]*col_sd) + col_means.values

# Negatives?
(data1_full_imp <0).sum(axis=1).sum(axis=0)

(data1_full_imp1 <0).sum(axis=1).sum(axis=0)
(data1_full_imp2 <0).sum(axis=1).sum(axis=0)
(data1_full_imp3 <0).sum(axis=1).sum(axis=0)
(data1_full_imp4 <0).sum(axis=1).sum(axis=0)
(data1_full_imp5 <0).sum(axis=1).sum(axis=0)


# Columns containing imputed negatives
data1_full_imp[(data1_full_imp <0).sum(axis=1) > 0 ]

data1_full_imp1[(data1_full_imp1 <0).sum(axis=1) > 0 ]
data1_full_imp2[(data1_full_imp2 <0).sum(axis=1) > 0 ]
data1_full_imp3[(data1_full_imp3 <0).sum(axis=1) > 0 ]
data1_full_imp4[(data1_full_imp4 <0).sum(axis=1) > 0 ]
data1_full_imp5[(data1_full_imp5 <0).sum(axis=1) > 0 ]

(data1.isnull()).sum(axis=1)

# There should be no negatives!
(data1 <0).sum(axis=1)
(data1 <0).sum()


##################################################################
### Standardise imputed dataset ###
##################################################################

# Rename colums to undo R formatting
data1_full_imp = data1_full_imp.rename(columns=lambda x: x.replace('_.', '_%'))
data1_full_imp = data1_full_imp.rename(columns=lambda x: x.replace('.FA', '/FA'))
data1_full_imp = data1_full_imp.rename(columns=lambda x: x.replace('.Ap', '/Ap'))
data1_full_imp = data1_full_imp.rename(columns=lambda x: x.replace('.PG', '/PG'))
data1_full_imp = data1_full_imp.rename(columns=lambda x: x.replace('.', '-'))



data1_full_imp1 = data1_full_imp1.rename(columns=lambda x: x.replace('_.', '_%'))
data1_full_imp1 = data1_full_imp1.rename(columns=lambda x: x.replace('.FA', '/FA'))
data1_full_imp1 = data1_full_imp1.rename(columns=lambda x: x.replace('.Ap', '/Ap'))
data1_full_imp1 = data1_full_imp1.rename(columns=lambda x: x.replace('.PG', '/PG'))
data1_full_imp1 = data1_full_imp1.rename(columns=lambda x: x.replace('.', '-'))

data1_full_imp2 = data1_full_imp2.rename(columns=lambda x: x.replace('_.', '_%'))
data1_full_imp2 = data1_full_imp2.rename(columns=lambda x: x.replace('.FA', '/FA'))
data1_full_imp2 = data1_full_imp2.rename(columns=lambda x: x.replace('.Ap', '/Ap'))
data1_full_imp2 = data1_full_imp2.rename(columns=lambda x: x.replace('.PG', '/PG'))
data1_full_imp2 = data1_full_imp2.rename(columns=lambda x: x.replace('.', '-'))

data1_full_imp3 = data1_full_imp3.rename(columns=lambda x: x.replace('_.', '_%'))
data1_full_imp3 = data1_full_imp3.rename(columns=lambda x: x.replace('.FA', '/FA'))
data1_full_imp3 = data1_full_imp3.rename(columns=lambda x: x.replace('.Ap', '/Ap'))
data1_full_imp3 = data1_full_imp3.rename(columns=lambda x: x.replace('.PG', '/PG'))
data1_full_imp3 = data1_full_imp3.rename(columns=lambda x: x.replace('.', '-'))

data1_full_imp4 = data1_full_imp4.rename(columns=lambda x: x.replace('_.', '_%'))
data1_full_imp4 = data1_full_imp4.rename(columns=lambda x: x.replace('.FA', '/FA'))
data1_full_imp4 = data1_full_imp4.rename(columns=lambda x: x.replace('.Ap', '/Ap'))
data1_full_imp4 = data1_full_imp4.rename(columns=lambda x: x.replace('.PG', '/PG'))
data1_full_imp4 = data1_full_imp4.rename(columns=lambda x: x.replace('.', '-'))

data1_full_imp5 = data1_full_imp5.rename(columns=lambda x: x.replace('_.', '_%'))
data1_full_imp5 = data1_full_imp5.rename(columns=lambda x: x.replace('.FA', '/FA'))
data1_full_imp5 = data1_full_imp5.rename(columns=lambda x: x.replace('.Ap', '/Ap'))
data1_full_imp5 = data1_full_imp5.rename(columns=lambda x: x.replace('.PG', '/PG'))
data1_full_imp5 = data1_full_imp5.rename(columns=lambda x: x.replace('.', '-'))



data1_imp_mean_adjusted = data1_full_imp.ix[:,0:225] - col_means

data1_imp_mean_adjusted1 = data1_full_imp1.ix[:,0:225] - col_means
data1_imp_mean_adjusted2 = data1_full_imp2.ix[:,0:225] - col_means
data1_imp_mean_adjusted3 = data1_full_imp3.ix[:,0:225] - col_means
data1_imp_mean_adjusted4 = data1_full_imp4.ix[:,0:225] - col_means
data1_imp_mean_adjusted5 = data1_full_imp5.ix[:,0:225] - col_means



data1_standardised = data1_imp_mean_adjusted/col_sd

data1_standardised1 = data1_imp_mean_adjusted1/col_sd
data1_standardised2 = data1_imp_mean_adjusted2/col_sd
data1_standardised3 = data1_imp_mean_adjusted3/col_sd
data1_standardised4 = data1_imp_mean_adjusted4/col_sd
data1_standardised5 = data1_imp_mean_adjusted5/col_sd


data1_full_sd_imp = pd.concat([data1_standardised, data1_full_imp.ix[:,225]], axis=1)

data1_full_sd_imp1 = pd.concat([data1_standardised1, data1_full_imp1.ix[:,225:233]], axis=1)
data1_full_sd_imp2 = pd.concat([data1_standardised2, data1_full_imp2.ix[:,225:233]], axis=1)
data1_full_sd_imp3 = pd.concat([data1_standardised3, data1_full_imp3.ix[:,225:233]], axis=1)
data1_full_sd_imp4 = pd.concat([data1_standardised4, data1_full_imp4.ix[:,225:233]], axis=1)
data1_full_sd_imp5 = pd.concat([data1_standardised5, data1_full_imp5.ix[:,225:233]], axis=1)

##################################################################
### Create training and test sets ###
##################################################################

# Create training set and test set, 3/4 vs 1/4. Randomize split across rows to
# avoid systematic factors in ordering to interfere.

#train1, test1 = train_test_split(data1_full_sd, test_size = 0.25, random_state=42)
train1, test1 = train_test_split(data1_full_sd_imp, test_size = 0.25, random_state=42)

train11, test11 = train_test_split(data1_full_sd_imp1, test_size = 0.25, random_state=42)
train12, test12 = train_test_split(data1_full_sd_imp2, test_size = 0.25, random_state=42)
train13, test13 = train_test_split(data1_full_sd_imp3, test_size = 0.25, random_state=42)
train14, test14 = train_test_split(data1_full_sd_imp4, test_size = 0.25, random_state=42)
train15, test15 = train_test_split(data1_full_sd_imp5, test_size = 0.25, random_state=42)

# Re-set indicies on train and test sets to be able to split them
train1 = train1.reset_index(drop=True)

train11 = train11.reset_index(drop=True)
train12 = train12.reset_index(drop=True)
train13 = train13.reset_index(drop=True)
train14 = train14.reset_index(drop=True)
train15 = train15.reset_index(drop=True)


test1 = test1.reset_index(drop=True)

test11 = test11.reset_index(drop=True)
test12 = test12.reset_index(drop=True)
test13 = test13.reset_index(drop=True)
test14 = test14.reset_index(drop=True)
test15 = test15.reset_index(drop=True)

 


#### MDS plot of standardised data
#from sklearn import manifold
#from sklearn.metrics import euclidean_distances
#
#similarities = euclidean_distances(data1_full_sd_imp)
#mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
#                   dissimilarity="precomputed", n_jobs=1)
#pos = mds.fit(similarities).embedding_

