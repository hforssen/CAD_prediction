# -*- coding: utf-8 -*-
"""
Initial analysis
"""

import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
#from sklearn.linear_model import metrics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib  import cm
import numpy as np
import math as ma
import statsmodels.graphics.correlation as stat
#import seaborn as sns

import load_data as ld
import load_data_ext as ld_ext
import load_data_mi as ldmi
import load_data_mi_ext_new as ldmi_ext

##############################################################################
# Variable summaries: standardised and non-standardised means, variance, range etc
full_data = ld.data_withmissing_df 

col_means = full_data.ix[:,0:225].mean(axis=0)
col_variance = full_data.ix[:,0:225].var(axis=0)
col_median = full_data.ix[:,0:225].median(axis=0)

q100 = full_data.ix[:,0:225].max(axis=0)
q0 = full_data.ix[:,0:225].min(axis=0)
col_range = q100 - q0

q75, q25 = np.percentile(full_data.ix[:,0:225], [75 ,25],axis=0)
col_iqr = q75 - q25

missings_per_column = full_data.isnull().sum()

# Combine all into a table, remove CAD50 
metabolite_summary = pd.concat([col_means, col_variance,col_median,col_range,missings_per_column], axis=1).drop(['CAD50'])
metabolite_summary.columns = ['Mean', 'Variance','Median','Range','Number of Missing Values']

#metabolite_summary.to_csv('metabolite_summary_table.csv')

############################################################################################
# Table 1 calculations

full_data = ld_ext.data_withmissing_df.copy(deep=True)

sex_mean = full_data['sex'].mean(axis=0) # male = 1, female = 0
sex_male = sum(full_data['sex'])
print 'sex mean '+ str(sex_mean) +' sex male '+ str(sex_male)

age_mean = full_data['Age'].mean(axis=0) # male = 1, female = 0
age_sd = ma.sqrt(full_data['Age'].var(axis=0))
print 'age mean '+ str(age_mean) +' age sd '+ str(age_sd)

bmi_mean = full_data['obesity_bmi_v'].mean(axis=0) # male = 1, female = 0
bmi_sd = ma.sqrt(full_data['obesity_bmi_v'].var(axis=0))
print 'bmi mean '+ str(bmi_mean) +' bmi sd '+ str(bmi_sd)

hypertension_mean = full_data['hypertension_4c'].mean(axis=0) # male = 1, female = 0
hypertension_yes = sum(full_data['hypertension_4c'])
print 'hypertension mean '+ str(hypertension_mean) +' current hypertension '+ str(hypertension_yes)

statin_use_mean = full_data['statin'].mean(axis=0) # male = 1, female = 0
statin_yes = sum(full_data['statin'])
print 'statin mean '+ str(statin_use_mean) +' statin yes '+ str(statin_yes)

diabetes_mean = full_data['diabetes'].mean(axis=0) # male = 1, female = 0
diabetes_yes = sum(full_data['diabetes'])
print 'diabetes mean '+ str(diabetes_mean) +' diabetes yes '+ str(diabetes_yes)

smoker_mean = full_data['current_smoking'].mean(axis=0) # male = 1, female = 0
smoker_yes =  np.nansum(full_data['current_smoking'])
print 'smoker mean '+ str(smoker_mean) +' smoker yes '+ str(smoker_yes)

CAD_mean = full_data['CAD50'].mean(axis=0) # male = 1, female = 0
CAD_yes =  np.nansum(full_data['CAD50'])
print 'CAD mean '+ str(CAD_mean) +' CAD yes '+ str(CAD_yes)



############################################################################################

#Summary of complete case
full_complete_data = ld.data_withoutmissing_df 

col_means_comp = full_complete_data.ix[:,0:225].mean(axis=0)
col_variance_comp = full_complete_data.ix[:,0:225].var(axis=0)
col_median_comp = full_complete_data.ix[:,0:225].median(axis=0)

q100_comp = full_complete_data.ix[:,0:225].max(axis=0)
q0_comp = full_complete_data.ix[:,0:225].min(axis=0)
col_range_comp = q100 - q0

q75_comp, q25_comp = np.percentile(full_complete_data.ix[:,0:225], [75 ,25],axis=0)
col_iqr = q75 - q25

# Combine all into a table, remove CAD50 
metabolite_summary_comp = pd.concat([col_means_comp, col_variance_comp,col_median_comp,col_range_comp], axis=1)
metabolite_summary_comp.columns = ['Mean', 'Variance','Median','Range']

metabolite_summary_comp.to_csv('metabolite_completecase_summary_table.csv')

# Comparison of complete and missing case: difference in means and varainces?
means_diff = col_means - col_means_comp
variance_diff = col_variance - col_variance_comp
metabolite_difference = pd.concat([col_means, col_means_comp, means_diff, col_variance, col_variance_comp, variance_diff], axis=1)
metabolite_difference.columns = ['Missing Case Mean','Complete Case Mean','Difference in Mean','Missing Case Variance','Complete Case Variance','Difference in Variance']

metabolite_difference.to_csv('metabolite_difference_table.csv')
 
# https://stanford.edu/~mwaskom/software/seaborn/examples/many_pairwise_correlations.html
# Correlation between variables, plot heatmap
corr = ld.data_df.corr()

N, M = 12, 12
fig, ax = plt.subplots(figsize=(N, M))
stat.plot_corr(corr, ax=ax)




# CAD/noCAD split

CAD_missing_case = sum(ld.data_withmissing_df.ix[:,225])
total_missing_case = len(ld.data_withmissing_df.ix[:,225])
noCAD_missing_case = total_missing_case-CAD_missing_case
propCAD_missing_case = CAD_missing_case/total_missing_case

CAD_complete_case = sum(ld.data_withoutmissing_df.ix[:,225])
total_complete_case = len(ld.data_withoutmissing_df.ix[:,225])
noCAD_complete_case = total_complete_case - CAD_complete_case
propCAD_complete_case = CAD_complete_case/total_complete_case




# CAD/noCAD split in training and test sets of missing data case
CAD_missing_train = sum(ldmi.train1.ix[:,225])
total_missing_train = len(ldmi.train1.ix[:,225])
noCAD_missing_train = total_missing_train-CAD_missing_train
propCAD_missing_train = (CAD_missing_train*1.0)/total_missing_train

CAD_missing_test = sum(ldmi.test1.ix[:,225])
total_missing_test = len(ldmi.test1.ix[:,225])
noCAD_missing_test = total_missing_test-CAD_missing_test
propCAD_missing_test = (CAD_missing_test*1.0)/total_missing_test


# CAD/noCAD split in training and test sets of complete data case
CAD_complete_train = sum(ld.train1.ix[:,225])
total_complete_train = len(ld.train1.ix[:,225])
noCAD_complete_train = total_complete_train-CAD_complete_train
propCAD_complete_train = (CAD_complete_train*1.0)/total_complete_train

CAD_complete_test = sum(ld.test1.ix[:,225])
total_complete_test = len(ld.test1.ix[:,225])
noCAD_complete_test = total_complete_test-CAD_complete_test
propCAD_complete_test = (CAD_complete_test*1.0)/total_complete_test



##############################################################################
####### additional missing data analysis - in addition to load_data ##########################

# number of missing per column
missings_column_20 = sum(i < 50 for i in missings_per_column)
missings_column_40 = sum((i > 49) & (i < 100) for i in missings_per_column)
missings_column_60 = sum((i > 99) & (i < 150) for i in missings_per_column)
missings_column_80 = sum((i > 149) & (i < 200) for i in missings_per_column)
missings_column_100 = sum((i > 199) & (i < 250) for i in missings_per_column)
#missings_column_120 = sum((i > 99) & (i < 120) for i in missings_per_column)
missings_column_upper = sum(i > 250  for i in missings_per_column)

missings_colgroups = [missings_column_20,missings_column_40,missings_column_60,missings_column_80,missings_column_100,missings_column_upper]

plt.bar([0,10,20,30,40,60], missings_colgroups, 9, color="blue")




missings_per_row = full_data.isnull().sum(axis=1).order()

missings_row_10 = sum(i < 10 for i in missings_per_row)
missings_row_20 = sum((i > 9) & (i < 20) for i in missings_per_row)
missings_row_30 = sum((i > 19) & (i < 30) for i in missings_per_row)
missings_row_40 = sum((i > 29) & (i < 40) for i in missings_per_row)
missings_row_50 = sum((i > 39) & (i < 50) for i in missings_per_row)
missings_row_60 = sum((i > 49) & (i < 60) for i in missings_per_row)
missings_row_upper = sum(i > 59  for i in missings_per_row)

missings_rowgroups = [missings_row_10,missings_row_20,missings_row_30,missings_row_40,missings_row_50,missings_row_60,missings_row_upper]

plt.bar([0,10,20,30,40,50,60], missings_rowgroups, 9, color="blue")



##############################################################################
# TSNE plot of imputed metabolites
metabolites_imp = ldmi.data1_full_sd_imp1.ix[:,0:225]

model = TSNE(n_components=2, random_state=0)
tsne_vals = model.fit_transform(metabolites_imp)
tsne_vals1 = tsne_vals[:,0]
tsne_vals2 = tsne_vals[:,1]
tsne_col = ldmi.data1_full_sd_imp1.ix[:,225]
my_colours = ['coral' if i == 0 else 'c' for i in tsne_col]

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.set_title("t-SNE visulatisation of metabolite variables",fontsize=14)
ax.set_xlabel("t-SNE calculated factor 1",fontsize=12)
ax.set_ylabel("t-SNE calculated factor 2",fontsize=12)
ax.grid(False,linestyle='',color='0.75')
# scatter with colormap mapping to z value
ax.scatter(tsne_vals1,tsne_vals2,s=25,c=my_colours, marker = 'o', cmap = cm.brg, edgecolors='none')

plt.show()


##############################################################################
###################### Correlation heatmap  ######################
##############################################################################

metabilites = ld.data_withoutmissing_df.ix[:,0:225]

#https://github.com/drquant/Python_Finance/blob/master/Correlation_Matrix_Example.py
corr = metabilites.corr()
fig = plt.figure(figsize=(20,20))
plt.imshow(corr, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns);



#########################################################################################################################
############################## Plots based on "significant" variable profiles ##############################
#########################################################################################################################

# L1 penalised regression

# With confounders
dat = ldmi_ext.data1_full_sd_imp1
metabolites_plot = pd.concat([dat['sex'],dat['Age'],dat['hypertension_4c'],dat['MUFA/FA'],dat['IDL-TG_%'],dat['Gln'],dat['AcAce'],dat['IDL-PL_%'],
                              dat['M-HDL-FC'],dat['TotCho'],dat['M-HDL-FC_%'],dat['Crea']],axis=1)

model = TSNE(n_components=2, random_state=0)
tsne_vals = model.fit_transform(metabolites_plot)
tsne_vals1 = tsne_vals[:,0]
tsne_vals2 = tsne_vals[:,1]
tsne_col = ldmi.data1_full_sd_imp1.ix[:,225]
my_colours = ['coral' if i == 0 else 'c' for i in tsne_col]

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.set_title("t-SNE visulatisation of most important metabolites and \n confounders from L1 penalised regression analysis",fontsize=14)
ax.set_xlabel("t-SNE calculated factor 1",fontsize=12)
ax.set_ylabel("t-SNE calculated factor 2",fontsize=12)
ax.grid(False,linestyle='',color='0.75')
# scatter with colormap mapping to z value
ax.scatter(tsne_vals1,tsne_vals2,s=25,c=my_colours, marker = 'o', cmap = cm.brg, edgecolors='none');
plt.show()




# Without confounders
dat = ldmi_ext.data1_full_sd_imp1
metabolites_plot = pd.concat([dat['LA/FA']
,dat['Phe']
,dat['Gp']
,dat['XL-VLDL-PL_%']
,dat['Alb']
,dat['XL-VLDL-FC_%']
,dat['S-VLDL-PL_%']
,dat['Gln']
,dat['XS-VLDL-C_%']
,dat['Cit']
,dat['Glc']
,dat['Val']
,dat['Tyr']
,dat['XXL-VLDL-PL_%']
,dat['His']
,dat['XXL-VLDL-FC_%']
,dat['XL-HDL-FC_%']
,dat['IDL-FC_%']
,dat['XL-HDL-FC']
,dat['IDL-CE']
,dat['M-HDL-PL_%']
,dat['SM']
,dat['Leu']
,dat['L-HDL-PL_%']
,dat['L-HDL-CE_%']
,dat['XL-HDL-TG_%']
,dat['HDL3-C']
,dat['L-HDL-TG']
,dat['Ile']
,dat['M-HDL-FC_%']
,dat['L-HDL-TG_%']
,dat['Ace']
],axis=1)

model = TSNE(n_components=2, random_state=0)
tsne_vals = model.fit_transform(metabolites_plot)
tsne_vals1 = tsne_vals[:,0]
tsne_vals2 = tsne_vals[:,1]
tsne_col = ldmi.data1_full_sd_imp1.ix[:,225]
my_colours = ['coral' if i == 0 else 'c' for i in tsne_col]

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.set_title("t-SNE visulatisation of most important metabolites \n from L1 penalised regression analysis",fontsize=14)
ax.set_xlabel("t-SNE calculated factor 1",fontsize=12)
ax.set_ylabel("t-SNE calculated factor 2",fontsize=12)
ax.grid(False,linestyle='',color='0.75')
# scatter with colormap mapping to z value
ax.scatter(tsne_vals1,tsne_vals2,s=25,c=my_colours, marker = 'o', cmap = cm.brg, edgecolors='none');

plt.show()




# Random forest

# With confounders
dat = ldmi_ext.data1_full_sd_imp1
metabolites_plot = pd.concat([dat['sex'],dat['Age'],dat['IDL-TG_%'],dat['Gln'],dat['Phe'],dat['IDL-C_%'],
                              dat['Alb'],dat['Lac'],dat['Crea']],axis=1)

model = TSNE(n_components=2, random_state=0)
tsne_vals = model.fit_transform(metabolites_plot)
tsne_vals1 = tsne_vals[:,0]
tsne_vals2 = tsne_vals[:,1]
tsne_col = ldmi.data1_full_sd_imp1.ix[:,225]
my_colours = ['coral' if i == 0 else 'c' for i in tsne_col]

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.set_title("t-SNE visulatisation of the most important metabolites and \n confounders from random forest analysis",fontsize=14)
ax.set_xlabel("t-SNE calculated factor 1",fontsize=12)
ax.set_ylabel("t-SNE calculated factor 2",fontsize=12)
ax.grid(False,linestyle='',color='0.75')
# scatter with colormap mapping to z value
ax.scatter(tsne_vals1,tsne_vals2,s=25,c=my_colours, marker = 'o', cmap = cm.brg, edgecolors='none');

plt.show()



# Without confounders
dat = ldmi_ext.data1_full_sd_imp1
metabolites_plot = pd.concat([dat['IDL-TG_%'],dat['Gln'],dat['Phe'],dat['IDL-C_%'],
                              dat['Alb'],dat['Lac'],dat['Crea']],axis=1)

model = TSNE(n_components=2, random_state=0)
tsne_vals = model.fit_transform(metabolites_plot)
tsne_vals1 = tsne_vals[:,0]
tsne_vals2 = tsne_vals[:,1]
tsne_col = ldmi.data1_full_sd_imp1.ix[:,225]
my_colours = ['coral' if i == 0 else 'c' for i in tsne_col]

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.set_title("t-SNE visulatisation of the most important metabolites \n from random forest analysis",fontsize=14)
ax.set_xlabel("t-SNE calculated factor 1",fontsize=12)
ax.set_ylabel("t-SNE calculated factor 2",fontsize=12)
ax.grid(False,linestyle='',color='0.75')
# scatter with colormap mapping to z value
ax.scatter(tsne_vals1,tsne_vals2,s=25,c=my_colours, marker = 'o', cmap = cm.brg, edgecolors='none');

plt.show()




# Statistical associations BF
dat = ldmi_ext.data1_full_sd_imp1
metabolites_plot = pd.concat([dat['ApoA1']
,dat['FAw6/FA']
,dat['HDL-D']
,dat['HDL2-C']
,dat['IDL-TG_%']
,dat['L-HDL-C']
,dat['L-HDL-CE']
,dat['L-HDL-FC']
,dat['L-HDL-L']
,dat['L-HDL-P']
,dat['L-HDL-PL']
,dat['L-HDL-TG']
,dat['L-LDL-TG_%']
,dat['Lac']
,dat['M-HDL-C']
,dat['M-HDL-CE']
,dat['M-HDL-FC']
,dat['M-HDL-P']
,dat['M-VLDL-PL_%']
,dat['MUFA/FA']
,dat['PUFA/FA']
,dat['S-HDL-TG_%']
,dat['S-VLDL-CE_%']
,dat['S-VLDL-C_%']
,dat['S-VLDL-TG_%']
,dat['TG/PG']
,dat['XS-VLDL-C_%']
,dat['XS-VLDL-TG_%']
],axis=1)

model = TSNE(n_components=2, random_state=0)
tsne_vals = model.fit_transform(metabolites_plot)
tsne_vals1 = tsne_vals[:,0]
tsne_vals2 = tsne_vals[:,1]
tsne_col = ldmi.data1_full_sd_imp1.ix[:,225]
my_colours = ['coral' if i == 0 else 'c' for i in tsne_col]

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.set_title("t-SNE visulatisation of statistically \n significant metabolities after applying Bonferroni Correction",fontsize=14)
ax.set_xlabel("t-SNE calculated factor 1",fontsize=12)
ax.set_ylabel("t-SNE calculated factor 2",fontsize=12)
ax.grid(False,linestyle='',color='0.75')
# scatter with colormap mapping to z value
ax.scatter(tsne_vals1,tsne_vals2,s=25,c=my_colours, marker = 'o', cmap = cm.brg, edgecolors='none');

plt.show()





















