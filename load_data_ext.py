# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 16:26:32 2016

@author: ucabhmf
"""

# -*- coding: utf-8 -*-
"""
Load data for complete-case analysis including age, (smoking?), hypertension
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
main_df.loc[main_df['Sex'] == '2', 'sex'] = 0 # female
main_df.loc[main_df['Sex'] == '1', 'sex'] = 1



# Remove variables not of interest - leave only metabolities and output
metsout_df = pd.concat([main_df.ix[:,203:428], main_df['Age'], main_df['sex'], main_df['hypertension_4c'], 
                        main_df['statin'],  main_df['CAD50']], axis=1)




# added 4 variables

#gen diab_ghaz=0 if CAD50!=. & diab_final!=1
#replace diab_ghaz=1 if diab_final==1
#
#gen statin_ghaz=0 if CAD50!=. & statin_meds_ref!=1
#replace statin_ghaz=1 if CAD50!=. & statin_meds_ref==1


# How many rows with all metabolite measures nan? Is Gp a metabolite? Yes?
metsout_df.ix[:,0:225].isnull().all(axis=1)
sum(metsout_df.ix[:,0:225].isnull().all(axis=1))
# 1086


# Remove rows where ALL variables are nan
metsout_nan_df = metsout_df.dropna(axis = 0, how='all', subset = metsout_df.ix[:,0:225].columns)
# 2323 records remaining 

# Remove rows where output is missing
metsout_nan1_df = metsout_nan_df.dropna(subset = metsout_nan_df.ix[:,229:230].columns)
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

# Print non-standardised dataset to csv
#data_withmissing_df.to_csv('full_nonstandard_data_with_missing.csv')

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

age_mean = data_df['Age'].mean()
mean_variance =  data_df['Age'].var()
age_standardised = (data1['Age']-age_mean)/(ma.sqrt(mean_variance))



data1_full_sd = pd.concat([data1_standardised, age_standardised, data1.ix[:,226:230]], axis=1)

#data1_full_sd.to_csv('full_data_with_missing.csv')

##################################################################
### Imputation ###
##################################################################

################################
### Which columns to include in imputation ###

# Find columns from whole dataset with no missing data when CAD50 not null.


################################
### Column means ###

#col_means = data_df.mean()
#data_imp_df = data_df
#
#for i in range(0,224):
#    data_imp_df.ix[:,i] = data_df.ix[:,i].replace(np.nan, col_means.ix[i])

################################
### K-NN imputation ###
#
## Need data to be standarised before applying, otherwise different scales will bias the distance calculations
## High dimensional, could calculate PCA then use this the calculate K-NN distances and neighbours
##       - only "large scale variations" would be included
#
## Method: 
##   - For each row with missing data
##       - find k nearest neighbours based on the complete data using available columns
##       - for each columns with missing data
##           - predict average from the neighbours
#
## Split into datasets with no missing values and missing values (excluding output var)
#nomissing_df = data1_full_sd.ix[:,0:225].dropna()
#missing_df = data1_full_sd.ix[:,0:225][data1_full_sd.ix[:,0:225].isnull().any(axis=1)]
#
#imputed_df = missing_df
#
#imputer_knn = KNeighborsRegressor(n_neighbors=5, algorithm='auto')
#
#miss = len(missing_df)
#
## For each row with missing data
#for i in range (0,miss):
#    # Find "training matrix" of available columns
#    print i
#    train_vals = nomissing_df.loc[:, (missing_df.iloc[[i]].notnull()).any(axis=0)] 
#    # only select columns where data exists in column of interest. 
#    # need to calculate distances so can only use available data
#    
#    pred_vals = missing_df.iloc[[i]].loc[:, (missing_df.iloc[[i]].notnull()).any(axis=0)]
#    # data point to use in distance calculations
#
#    is_there = (missing_df.iloc[[i]].notnull()).any(axis=0)
#    missing = is_there[is_there == False]
#    # Find the columns that are missing in the ith row
#    
#    for j in missing.index:
#        print j
#        train_target = nomissing_df[j] # training data outome
#        
#        imputer_knn.fit(train_vals,train_target)
#        imputed_df.iloc[i][j] = imputer_knn.predict(pred_vals)
#        
##check no more missings
#sum(imputed_df.isnull().any(axis=1))
#
## Combine to full dataset again
#data1_full_sd_imp = pd.concat([(pd.concat([nomissing_df, imputed_df], axis=0)), data1_full_sd.ix[:,225]],axis =1)        

################################
### Load multiple imputation (MI) from R  ###
#data1_full_sd_imp = pd.read_csv('imp_data1.csv').ix[:,1:227]

################################
### Load non-standardised multiple imputation (MI) from R  ###
#data1_full_imp = pd.read_csv('imp_data5_nonstand.csv').ix[:,1:227]
#
## No negatives when non-standardised imputation done. Better!
## No negatives on any of the imputed datasets, use this!
#
###################################################################
#### Check that imputed values are realistic ###
###################################################################
#
## Find range of values of orginial data
#data_max = data_df.max()
#data_min = data_df.min()
#
#
## Unstandardise imputed dataset and check that no negatives
##data1_full_imp = (data1_full_sd_imp.ix[:,0:225]*col_sd) + col_means.values
#
## Negatives?
#(data1_full_imp <0).sum(axis=1).sum(axis=0)
#
## Columns containing imputed negatives
#data1_full_imp[(data1_full_imp <0).sum(axis=1) > 0 ]
#
#(data1.isnull()).sum(axis=1)
#
## There should be no negatives!
#(data1 <0).sum(axis=1)
#(data1 <0).sum()
#
#
###################################################################
#### Standardise imputed dataset ###
###################################################################
#
## Rename colums to undo R formatting
#data1_full_imp = data1_full_imp.rename(columns=lambda x: x.replace('_.', '_%'))
#data1_full_imp = data1_full_imp.rename(columns=lambda x: x.replace('.FA', '/FA'))
#data1_full_imp = data1_full_imp.rename(columns=lambda x: x.replace('.Ap', '/Ap'))
#data1_full_imp = data1_full_imp.rename(columns=lambda x: x.replace('.PG', '/PG'))
#data1_full_imp = data1_full_imp.rename(columns=lambda x: x.replace('.', '-'))
#
#
#data1_imp_mean_adjusted = data1_full_imp.ix[:,0:225] - col_means
#
#data1_standardised = data1_imp_mean_adjusted/col_sd
#
#data1_full_sd_imp = pd.concat([data1_standardised, data1_full_imp.ix[:,225]], axis=1)
#
#

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

 


#### MDS plot of standardised data
#from sklearn import manifold
#from sklearn.metrics import euclidean_distances
#
#similarities = euclidean_distances(data1_full_sd_imp)
#mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
#                   dissimilarity="precomputed", n_jobs=1)
#pos = mds.fit(similarities).embedding_

