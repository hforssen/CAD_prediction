# -*- coding: utf-8 -*-
"""
ROC curve confounders
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import sklearn.metrics  as metrics
from sklearn.svm import l1_min_c
import matplotlib.pyplot as plt
import math as ma
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_fscore_support 
from sklearn import grid_search
from sklearn.manifold import TSNE
from matplotlib  import cm
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
import statsmodels.api as sm
from sklearn.manifold import TSNE
from matplotlib  import cm
from sklearn.decomposition import PCA

import load_data_ext as ld
import load_data_ext as ld_ext
import load_data_mi_ext_new as ldmi
import load_data_mi_ext_new as ldmi_ext

np.random.seed(10)

################################################################################################################################

################################################################################################################################

################################################################################################################################

################################################################################################################################

################################################################################################################################

################################################################################################################################

################################################################################################################################

# Random forest
np.random.seed(10)


train_data = ldmi.train11
test_data = ldmi.test11
truevals = np.array(test_data.ix[:,229]) 
sum(truevals)
sum(truevals-1)

# MI set 1
rf1 = RandomForestClassifier(n_estimators=5000, max_features = 0.325)
forest_fit1 = rf1.fit(ldmi.train11.ix[:,0:229],ldmi.train11.ix[:,229])

importances1 = forest_fit1.feature_importances_ 
forest_preds1= forest_fit1.predict(test_data.ix[:,0:229])
forest_acc1 = (forest_preds1 == truevals).sum()/(len(test_data)*1.0)
print forest_acc1

preds_proba1 = forest_fit1.predict_proba(test_data.ix[:,0:229])[:,1]



importances1_list = pd.TimeSeries(importances1)
metabolites = pd.TimeSeries(test_data.columns.values[0:229])
importances1_mat = pd.concat([metabolites, importances1_list], axis=1)
importances1_mat.sort(columns=1, ascending=False) # could plot?



# MI set 2
train_data = ldmi.train12
test_data = ldmi.test12
rf2 = RandomForestClassifier(n_estimators=5000, max_features = 0.325) 
forest_fit2 = rf2.fit(ldmi.train12.ix[:,0:229],ldmi.train12.ix[:,229])

importances2 = forest_fit2.feature_importances_ 
forest_preds2 = forest_fit2.predict(test_data.ix[:,0:229])
forest_acc2 = (forest_preds2 == truevals).sum()/(len(test_data)*1.0)
print forest_acc2 

preds_proba2 = forest_fit2.predict_proba(test_data.ix[:,0:229])[:,1]

importances2_list = pd.TimeSeries(importances2)
metabolites = pd.TimeSeries(test_data.columns.values[0:229])
importances2_mat = pd.concat([metabolites, importances2_list], axis=1)
importances2_mat.sort(columns=1, ascending=False)





# MI set 3
train_data = ldmi.train13
test_data = ldmi.test13
rf3 = RandomForestClassifier(n_estimators=5000, max_features = 0.325)
forest_fit3 = rf3.fit(ldmi.train13.ix[:,0:229],ldmi.train13.ix[:,229])

importances3 = forest_fit3.feature_importances_ 
forest_preds3 = forest_fit3.predict(test_data.ix[:,0:229])
forest_acc3 = (forest_preds3 == truevals).sum()/(len(test_data)*1.0)
print forest_acc3 

preds_proba3 = forest_fit3.predict_proba(test_data.ix[:,0:229])[:,1]

importances3_list = pd.TimeSeries(importances3)
metabolites = pd.TimeSeries(test_data.columns.values[0:229])
importances3_mat = pd.concat([metabolites, importances3_list], axis=1)
importances3_mat.sort(columns=1, ascending=False)




# MI set 4
train_data = ldmi.train14
test_data = ldmi.test14
rf4 = RandomForestClassifier(n_estimators=5000, max_features = 0.325)
forest_fit4 = rf4.fit(ldmi.train14.ix[:,0:229],ldmi.train14.ix[:,229])

importances4 = forest_fit4.feature_importances_ 
forest_preds4 = forest_fit4.predict(test_data.ix[:,0:229])
forest_acc4 = (forest_preds4 == truevals).sum()/(len(test_data)*1.0)
print forest_acc4 

preds_proba4 = forest_fit4.predict_proba(test_data.ix[:,0:229])[:,1]

importances4_list = pd.TimeSeries(importances4)
metabolites = pd.TimeSeries(test_data.columns.values[0:229])
importances4_mat = pd.concat([metabolites, importances4_list], axis=1)
importances4_mat.sort(columns=1, ascending=False)





# MI set 5
train_data = ldmi.train15
test_data = ldmi.test15
rf5 = RandomForestClassifier(n_estimators=5000, max_features = 0.325)
forest_fit5 = rf5.fit(ldmi.train15.ix[:,0:229],ldmi.train15.ix[:,229])

importances5 = forest_fit5.feature_importances_ 
forest_preds5 = forest_fit5.predict(ldmi.test15.ix[:,0:229])
forest_acc5 = (forest_preds5 == truevals).sum()/(len(ldmi.test15)*1.0)
print forest_acc5 

preds_proba5 = forest_fit5.predict_proba(ldmi.test1.ix[:,0:229])[:,1]

importances5_list = pd.TimeSeries(importances5)
metabolites = pd.TimeSeries(test_data.columns.values[0:229])
importances5_mat = pd.concat([metabolites, importances5_list], axis=1)
importances5_mat.sort(columns=1, ascending=False)







preds_proba_average_rf = (preds_proba1 + preds_proba2 + preds_proba3 + preds_proba4 + preds_proba5)/5
preds_proba_average_bin_rf = (preds_proba_average_rf >0.5 ).astype(int)

confusion_matrix(truevals, preds_proba_average_bin_rf, labels=[1, 0])
forest_av_acc_rf = (preds_proba_average_bin_rf == truevals).sum()/(len(ldmi.test14)*1.0)
print forest_av_acc_rf 



fpr_rf, tpr_rf, _ = roc_curve(truevals, preds_proba_average_rf)
# Calculate the AUC
roc_auc_rf = auc(fpr_rf, tpr_rf)
print 'ROC AUC: %0.3f' % roc_auc_rf




importances_average = pd.TimeSeries((importances1 + importances2 + importances5 + importances5 + importances5)/5)
importances_average_list = pd.TimeSeries(importances_average)
importances_average_mat = pd.concat([metabolites, importances_average], axis=1)
importances_average_mat_sorted = importances_average_mat.sort(columns=1, ascending=False)
importances_average_mat_sorted 
importances_average_mat_sorted.to_csv('importances_random_forest_ordered.csv')




################################################################################################################################

################################################################################################################################

################################################################################################################################

################################################################################################################################

################################################################################################################################

################################################################################################################################

################################################################################################################################

# L1 Regression

##########################################################################
############ Imputed data analysis 1 ############
##########################################################################

np.random.seed(10)

# Initial random model
mod1 = LogisticRegression(C=0.5, penalty='l1')

# Initialise full train and test sets
train_data = ldmi_ext.train11
train_dataX = ldmi_ext.train11.ix[:,0:229]
train_dataY = ldmi_ext.train11.ix[:,229]
test_data = ldmi_ext.test11
test_dataX = ldmi_ext.test11.ix[:,0:229]
test_dataY = ldmi_ext.test11.ix[:,229]


# Smallest value of C before all coefficients set to zero
min_l1_C = l1_min_c(train_dataX,train_dataY) 
'%f' % min_l1_C # 0.000028 ~= 0.00003

#create candidate values of C

c_vals = min_l1_C * np.logspace(0, 4, 17) 


cdict1 = {}
for c in c_vals:
    cdict1[c] = []



# Genaerate indicies to split data into 50 chunks
cv_index = [ma.ceil((len(train_data)/50)*x) for x in range(51)]

for i in range(50):
    # Split the data
    print(i)
    
    test_cv = train_data.ix[cv_index[i]:cv_index[i+1],:]
    train_cv = train_data.drop(train_data.index[cv_index[i]:cv_index[i+1]])

    trainy, testy = train_cv.ix[:,229], test_cv.ix[:,229]
    trainX, testX = train_cv.ix[:,0:229], test_cv.ix[:,0:229]
    

    for c in cdict1:
        mod1.set_params(C=c)
        logit_fit = mod1.fit(trainX.values, trainy.values)
        predy = mod1.predict(testX.values)
        error_rate = np.mean(predy != testy)
        cdict1[c].append(error_rate)

error_path1 = pd.DataFrame(cdict1).mean()
error_path1.plot(style = 'o-k', label = 'Error rate')
error_path1.cummin().plot(style = 'r-', label = 'Lower envelope')
plt.xlabel('C (regularization parameter)')
plt.ylabel('Prediction error rate')
plt.legend(loc = 'upper right')
plt.axis([-0.005, 1, 0.2, 0.35])


min_error_c1 = 0.032165  #minimum value from error path
logit_model_best1 = LogisticRegression(C = min_error_c1, penalty = 'l1')

logit_fit_best1 = logit_model_best1.fit(train_dataX, train_dataY)

# Get list of parameter estimates
a1 = pd.Series(np.asarray(logit_fit_best1.coef_)[0])
metabolite_names = pd.Series(np.asarray(list(train_dataX.columns.values)))

parameter_estimates_all1 = pd.DataFrame({'metabolite_names': metabolite_names,'values':a1})
parameter_estimates1 = parameter_estimates_all1.loc[parameter_estimates_all1['values'] != 0]

parameter_estimates_ordered1 = parameter_estimates_all1.loc[parameter_estimates_all1['values'] != 0].sort(columns='values', ascending=True)
parameter_estimates_ordered1['abs_values'] = abs(parameter_estimates_ordered1['values'])
parameter_estimates_ordered1 = parameter_estimates_ordered1.sort(columns='abs_values', ascending=False).ix[:,0:2]

parameter_estimates_ordered1



# Predict on test data

preds1 = logit_model_best1.predict(test_dataX)
truevals = np.array(test_dataY)

accuracy1 = ((preds1 == truevals).sum())/(len(test_dataX)*1.0)
print accuracy1
0.761517


# Confusion matrix
confusion_matrix1 = confusion_matrix(truevals, preds1, labels=[1, 0])

# total pos: 254
(sum(truevals)*1.0)/len(test_dataY)

# total neg: 115
sum(truevals-1)

preds_proba1 = logit_model_best1.predict_proba(test_dataX)[:,1]

# Determine the false positive and true positive rates
fpr1, tpr1, _ = roc_curve(truevals, preds_proba1)
 
# Calculate the AUC
roc_auc1 = auc(fpr1, tpr1)
print 'ROC AUC: %0.2f' % roc_auc1
 
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr1, tpr1, label='ROC curve (AUC = %0.2f)' % roc_auc1)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for L1 penalised logitic regression on Imputation dataset 1')
plt.legend(loc="lower right")
plt.show()


##########################################################################
############ Imputed data analysis 2 ############
##########################################################################


# Initial random model
mod1 = LogisticRegression(C=0.5, penalty='l1')

# Initialise full train and test sets
train_data = ldmi_ext.train12
train_dataX = ldmi_ext.train12.ix[:,0:229]
train_dataY = ldmi_ext.train12.ix[:,229]
test_data = ldmi_ext.test12
test_dataX = ldmi_ext.test12.ix[:,0:229]
test_dataY = ldmi_ext.test12.ix[:,229]


cdict2 = {}
for c in c_vals:
    cdict2[c] = []

# Cross validation to choose c. train1 and test1 already have randomized rows from train_test_split

# Genaerate indicies to split data into 50 chunks
cv_index = [ma.ceil((len(train_data)/50)*x) for x in range(51)]

for i in range(50):
    # Split the data
    print(i)
    
    test_cv = train_data.ix[cv_index[i]:cv_index[i+1],:]
    train_cv = train_data.drop(train_data.index[cv_index[i]:cv_index[i+1]])

    trainy, testy = train_cv.ix[:,229], test_cv.ix[:,229]
    trainX, testX = train_cv.ix[:,0:229], test_cv.ix[:,0:229]

    for c in cdict2:
        mod1.set_params(C=c)
        logit_fit = mod1.fit(trainX.values, trainy.values)
        predy = mod1.predict(testX.values)
        error_rate = np.mean(predy != testy)
        cdict2[c].append(error_rate)

error_path2 = pd.DataFrame(cdict2).mean()
error_path2.plot(style = 'o-k', label = 'Error rate')
error_path2.cummin().plot(style = 'r-', label = 'Lower envelope')
plt.xlabel('C (regularization parameter)')
plt.ylabel('Prediction error rate')
plt.legend(loc = 'upper right')
plt.axis([-0.005, 1, 0.2, 0.35])


min_error_c2 = 0.057198  # minimum value from error path
logit_model_best2 = LogisticRegression(C = min_error_c2, penalty = 'l1')

logit_fit_best2 = logit_model_best2.fit(train_dataX, train_dataY)


# Get list of parameter estimates
a2 = pd.Series(np.asarray(logit_fit_best2.coef_)[0])
metabolite_names = pd.Series(np.asarray(list(train_dataX.columns.values)))

parameter_estimates_all2 = pd.DataFrame({'metabolite_names': metabolite_names,'values':a2})
parameter_estimates2 = parameter_estimates_all2.loc[parameter_estimates_all2['values'] != 0]

parameter_estimates_ordered2 = parameter_estimates_all2.loc[parameter_estimates_all2['values'] != 0].sort(columns='values', ascending=True)
parameter_estimates_ordered2['abs_values'] = abs(parameter_estimates_ordered2['values'])
parameter_estimates_ordered2 = parameter_estimates_ordered2.sort(columns='abs_values', ascending=False).ix[:,0:2]

parameter_estimates_ordered2



# Predict on test data

preds2 = logit_model_best2.predict(test_dataX)
truevals = np.array(test_dataY)

accuracy2 = ((preds2 == truevals).sum())/(len(test_dataX)*1.0)
print accuracy2
0.7669376


# Confusion matrix
confusion_matrix2 = confusion_matrix(truevals, preds2, labels=[1, 0])


# total pos: 254
sum(truevals)

# total neg: 115
sum(truevals-1)

preds_proba2 = logit_model_best2.predict_proba(test_dataX)[:,1]

# Determine the false positive and true positive rates
fpr2, tpr2, _ = roc_curve(truevals, preds_proba2)
 
# Calculate the AUC
roc_auc2 = auc(fpr2, tpr2)
print 'ROC AUC: %0.2f' % roc_auc2

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr2, tpr2, label='ROC curve (AUC = %0.2f)' % roc_auc2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for L1 penalised logitic regression on Imputation dataset 1')
plt.legend(loc="lower right")
plt.show()


##########################################################################
############ Imputed data analysis 3 ############
##########################################################################


# Initial random model
mod1 = LogisticRegression(C=0.5, penalty='l1')

# Initialise full train and test sets
train_data = ldmi_ext.train13
train_dataX = ldmi_ext.train13.ix[:,0:229]
train_dataY = ldmi_ext.train13.ix[:,229]
test_data = ldmi_ext.test13
test_dataX = ldmi_ext.test13.ix[:,0:229]
test_dataY = ldmi_ext.test13.ix[:,229]


cdict3 = {}
for c in c_vals:
    cdict3[c] = []

# Cross validation to choose c. train1 and test1 already have randomized rows from train_test_split

# Genaerate indicies to split data into 50 chunks
cv_index = [ma.ceil((len(train_data)/50)*x) for x in range(51)]

for i in range(50):
    # Split the data
    print(i)
    
    test_cv = train_data.ix[cv_index[i]:cv_index[i+1],:]
    train_cv = train_data.drop(train_data.index[cv_index[i]:cv_index[i+1]])

    trainy, testy = train_cv.ix[:,229], test_cv.ix[:,229]
    trainX, testX = train_cv.ix[:,0:229], test_cv.ix[:,0:229]
    
    for c in cdict3:
        mod1.set_params(C=c)
        logit_fit = mod1.fit(trainX.values, trainy.values)
        predy = mod1.predict(testX.values)
        error_rate = np.mean(predy != testy)
        cdict3[c].append(error_rate)

error_path3 = pd.DataFrame(cdict3).mean()
error_path3.plot(style = 'o-k', label = 'Error rate')
error_path3.cummin().plot(style = 'r-', label = 'Lower envelope')
plt.xlabel('C (regularization parameter)')
plt.ylabel('Prediction error rate')
plt.legend(loc = 'upper right')
plt.axis([-0.005, 1, 0.2, 0.35])


min_error_c3 = 0.032165  # lowest value from error path
logit_model_best3 = LogisticRegression(C = min_error_c3, penalty = 'l1')

logit_fit_best3 = logit_model_best3.fit(train_dataX, train_dataY)

# Get list of parameter estimates
a3 = pd.Series(np.asarray(logit_fit_best3.coef_)[0])
metabolite_names = pd.Series(np.asarray(list(train_dataX.columns.values)))

parameter_estimates_all3 = pd.DataFrame({'metabolite_names': metabolite_names,'values':a3})
parameter_estimates3 = parameter_estimates_all3.loc[parameter_estimates_all3['values'] != 0]

parameter_estimates_ordered3 = parameter_estimates_all3.loc[parameter_estimates_all3['values'] != 0].sort(columns='values', ascending=True)
parameter_estimates_ordered3['abs_values'] = abs(parameter_estimates_ordered3['values'])
parameter_estimates_ordered3 = parameter_estimates_ordered3.sort(columns='abs_values', ascending=False).ix[:,0:2]

parameter_estimates_ordered3



# Predict on test data

preds3 = logit_model_best3.predict(test_dataX)
truevals = np.array(test_dataY)

accuracy3 = ((preds3 == truevals).sum())/(len(test_dataX)*1.0)
print accuracy3
0.7588075


# Confusion matrix
confusion_matrix3 = confusion_matrix(truevals, preds3, labels=[1, 0])

# total pos: 254
sum(truevals)

# total neg: 115
sum(truevals-1)

preds_proba3 = logit_model_best3.predict_proba(test_dataX)[:,1]

# Determine the false positive and true positive rates
fpr3, tpr3, _ = roc_curve(truevals, preds_proba3)
 
# Calculate the AUC
roc_auc3 = auc(fpr3, tpr3)
print 'ROC AUC: %0.2f' % roc_auc3

 
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr3, tpr3, label='ROC curve (AUC = %0.2f)' % roc_auc3)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for L1 penalised logitic regression on Imputation dataset 1')
plt.legend(loc="lower right")
plt.show()


##########################################################################
############ Imputed data analysis 4 ############
##########################################################################


# Initial random model
mod1 = LogisticRegression(C=0.5, penalty='l1')

# Initialise full train and test sets
train_data = ldmi_ext.train14
train_dataX = ldmi_ext.train14.ix[:,0:229]
train_dataY = ldmi_ext.train14.ix[:,229]
test_data = ldmi_ext.test14
test_dataX = ldmi_ext.test14.ix[:,0:229]
test_dataY = ldmi_ext.test14.ix[:,229]



cdict4 = {}
for c in c_vals:
    cdict4[c] = []

# Cross validation to choose c. train1 and test1 already have randomized rows from train_test_split

# Genaerate indicies to split data into 50 chunks
cv_index = [ma.ceil((len(train_data)/50)*x) for x in range(51)]

for i in range(50):
    # Split the data
    print(i)
    
    test_cv = train_data.ix[cv_index[i]:cv_index[i+1],:]
    train_cv = train_data.drop(train_data.index[cv_index[i]:cv_index[i+1]])

    trainy, testy = train_cv.ix[:,229], test_cv.ix[:,229]
    trainX, testX = train_cv.ix[:,0:229], test_cv.ix[:,0:229]
    

    for c in cdict4:
        mod1.set_params(C=c)
        logit_fit = mod1.fit(trainX.values, trainy.values)
        predy = mod1.predict(testX.values)
        error_rate = np.mean(predy != testy)
        cdict4[c].append(error_rate)

error_path4 = pd.DataFrame(cdict4).mean()
error_path4.plot(style = 'o-k', label = 'Error rate')
error_path4.cummin().plot(style = 'r-', label = 'Lower envelope')
plt.xlabel('C (regularization parameter)')
plt.ylabel('Prediction error rate')
plt.legend(loc = 'upper right')
plt.axis([-0.005, 1, 0.2, 0.35])


min_error_c4 = 0.101714  # minimum from error path
logit_model_best4 = LogisticRegression(C = min_error_c4, penalty = 'l1')

logit_fit_best4 = logit_model_best4.fit(train_dataX, train_dataY)

# Get list of parameter estimates
a4 = pd.Series(np.asarray(logit_fit_best4.coef_)[0])
metabolite_names = pd.Series(np.asarray(list(train_dataX.columns.values)))

parameter_estimates_all4 = pd.DataFrame({'metabolite_names': metabolite_names,'values':a4})
parameter_estimates4 = parameter_estimates_all4.loc[parameter_estimates_all4['values'] != 0]

parameter_estimates_ordered4 = parameter_estimates_all4.loc[parameter_estimates_all4['values'] != 0].sort(columns='values', ascending=True)
parameter_estimates_ordered4['abs_values'] = abs(parameter_estimates_ordered4['values'])
parameter_estimates_ordered4 = parameter_estimates_ordered4.sort(columns='abs_values', ascending=False).ix[:,0:2]

parameter_estimates_ordered4



# Predict on test data

preds4 = logit_model_best4.predict(test_dataX)
truevals = np.array(test_dataY)

accuracy4 = ((preds4 == truevals).sum())/(len(test_dataX)*1.0)
print accuracy4
0.691056910569


# Confusion matrix
confusion_matrix4 = confusion_matrix(truevals, preds4, labels=[1, 0])

# total pos: 254
sum(truevals)

# total neg: 115
sum(truevals-1)

preds_proba4 = logit_model_best4.predict_proba(test_dataX)[:,1]

# Determine the false positive and true positive rates
fpr4, tpr4, _ = roc_curve(truevals, preds_proba4)
 
# Calculate the AUC
roc_auc4 = auc(fpr4, tpr4)
print 'ROC AUC: %0.2f' % roc_auc4

 
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr4, tpr4, label='ROC curve (AUC = %0.2f)' % roc_auc4)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for L1 penalised logitic regression on Imputation dataset 1')
plt.legend(loc="lower right")
plt.show()


##########################################################################
############ Imputed data analysis 5 ############
##########################################################################


# Initial random model
mod1 = LogisticRegression(C=0.5, penalty='l1')

# Initialise full train and test sets
train_data = ldmi_ext.train15
train_dataX = ldmi_ext.train15.ix[:,0:229]
train_dataY = ldmi_ext.train15.ix[:,229]
test_data = ldmi_ext.test15
test_dataX = ldmi_ext.test15.ix[:,0:229]
test_dataY = ldmi_ext.test15.ix[:,229]


cdict5 = {}
for c in c_vals:
    cdict5[c] = []

# Cross validation to choose c. train1 and test1 already have randomized rows from train_test_split

# Genaerate indicies to split data into 50 chunks
cv_index = [ma.ceil((len(train_data)/50)*x) for x in range(51)]

for i in range(50):
    # Split the data
    print(i)
    
    test_cv = train_data.ix[cv_index[i]:cv_index[i+1],:]
    train_cv = train_data.drop(train_data.index[cv_index[i]:cv_index[i+1]])

    trainy, testy = train_cv.ix[:,229], test_cv.ix[:,229]
    trainX, testX = train_cv.ix[:,0:229], test_cv.ix[:,0:229]
    
    for c in cdict5:
        mod1.set_params(C=c)
        logit_fit = mod1.fit(trainX.values, trainy.values)
        predy = mod1.predict(testX.values)
        error_rate = np.mean(predy != testy)
        cdict5[c].append(error_rate)
        
error_path5 = pd.DataFrame(cdict5).mean()
error_path5.plot(style = 'o-k', label = 'Error rate')
error_path5.cummin().plot(style = 'r-', label = 'Lower envelope')
plt.xlabel('C (regularization parameter)')
plt.ylabel('Prediction error rate')
plt.legend(loc = 'upper right')
plt.axis([-0.005, 1, 0.2, 0.35])


min_error_c5 = 0.057198  # lowest value from error path
logit_model_best5 = LogisticRegression(C = min_error_c5, penalty = 'l1')

logit_fit_best5 = logit_model_best5.fit(train_dataX, train_dataY)


# Get list of parameter estimates
a5 = pd.Series(np.asarray(logit_fit_best5.coef_)[0])
metabolite_names = pd.Series(np.asarray(list(train_dataX.columns.values)))

parameter_estimates_all5 = pd.DataFrame({'metabolite_names': metabolite_names,'values':a5})
parameter_estimates5 = parameter_estimates_all5.loc[parameter_estimates_all5['values'] != 0]

parameter_estimates_ordered5 = parameter_estimates_all5.loc[parameter_estimates_all5['values'] != 0].sort(columns='values', ascending=True)
parameter_estimates_ordered5['abs_values'] = abs(parameter_estimates_ordered5['values'])
parameter_estimates_ordered5 = parameter_estimates_ordered5.sort(columns='abs_values', ascending=False).ix[:,0:2]

parameter_estimates_ordered5



# Predict on test data

preds5 = logit_model_best5.predict(test_dataX)
truevals = np.array(test_dataY)

accuracy5 = ((preds5 == truevals).sum())/(len(test_dataX)*1.0)
print accuracy5
0.750677506775


# Confusion matrix
confusion_matrix5 = confusion_matrix(truevals, preds5, labels=[1, 0])


# total pos: 254
sum(truevals)

# total neg: 115
sum(truevals-1)

preds_proba5 = logit_model_best5.predict_proba(test_dataX)[:,1]

# Determine the false positive and true positive rates
fpr5, tpr5, _ = roc_curve(truevals, preds_proba5)
 
# Calculate the AUC
roc_auc5 = auc(fpr5, tpr5)
print 'ROC AUC: %0.2f' % roc_auc5

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr5, tpr5, label='ROC curve (AUC = %0.2f)' % roc_auc5)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for L1 penalised logitic regression on Imputation dataset 1')
plt.legend(loc="lower right")
plt.show()




#################################################################################################
######################### Combine imputation results - average ##################################
#################################################################################################

preds_proba_average_L1 = (preds_proba1+preds_proba2+preds_proba3+preds_proba4+preds_proba5)/5
preds_proba_average_bin_L1 =  (preds_proba_average_L1 > 0.5).astype(int)

accuracy_proba_average_L1 = ((preds_proba_average_bin_L1== truevals).sum())/(len(test_dataX)*1.0)
print accuracy_proba_average_L1

confusion_matrix_average_L1 = confusion_matrix(truevals, preds_proba_average_bin_L1, labels=[1, 0])

fpr_L1, tpr_L1, _ = roc_curve(truevals, preds_proba_average_L1)
 
# Calculate the AUC
roc_auc_L1 = auc(fpr_L1, tpr_L1)
print 'ROC AUC: %0.3f' % roc_auc_L1

################################################################################################################################

################################################################################################################################

################################################################################################################################

################################################################################################################################

################################################################################################################################

################################################################################################################################

################################################################################################################################

# PCA regression

data = ld_ext.data_df

age_miss = data['Age'].isnull().sum() #0
sex_miss = data['sex'].isnull().sum() #0
statin_miss = data['statin'].isnull().sum() #0
hypertension_miss = data['hypertension_4c'].isnull().sum() #0


logged_data1 = np.log(data.ix[:,0:225]+1)

# Standarize data to zero mean and unit variance
col_means = logged_data1.mean(axis=0)
logged_data1_mean_adjusted = logged_data1 - col_means 
# TO DO: make this accurate?

col_variance = logged_data1_mean_adjusted.var(axis=0)
col_sd = [0] * len(col_variance)
for x in list(range(len(col_variance))):
    col_sd[x] = ma.sqrt(col_variance[x])

logged_data1_standardised = logged_data1_mean_adjusted/col_sd
logged_data = pd.concat([logged_data1_standardised, data.ix[:,225:234]], axis=1)
logged_data['intercept'] = 1



# Create matrix of confounders
regression_mat_initial = pd.concat([logged_data['intercept'],logged_data['Age'],logged_data['sex'],logged_data['statin'],
                                   logged_data['hypertension_4c']],axis=1)


###############################################################################
############## PCA decomposition of metabolites analysis ##############

data_imp = ldmi_ext.data1_full_sd_imp

metabolities_mat_imp = ldmi_ext.data1_full_sd_imp.ix[:,0:225]

pca = PCA(n_components=15, copy=True)
pca.fit(metabolities_mat_imp)
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Principle Components', size = 13)
plt.ylabel('Proportion of variance Captured', size = 13)
plt.title('Variance explained by \n successive Principle Components', size = 14) # use top 6

pca = PCA(n_components=6, copy=True)
pca.fit(metabolities_mat_imp)
pca_components = pca.components_

pd.DataFrame(pca_components)

# Calculate PC values of data
PC_data = np.zeros(shape=(1474,6))

for i in range(0,6):
    for j in range(0,1474):
        PC_data[j,i]= np.dot(pca_components[i,:],metabolities_mat_imp.ix[j,:].values)
        
PC_data_mat = pd.DataFrame(PC_data)
PC_data_mat['age'] = ldmi_ext.data1_full_sd_imp['Age']
PC_data_mat['sex'] = ldmi_ext.data1_full_sd_imp['sex']
PC_data_mat['statin'] = ldmi_ext.data1_full_sd_imp['statin']
PC_data_mat['hypertension'] = ldmi_ext.data1_full_sd_imp['hypertension_4c']
PC_data_mat['intercept'] = 1
PC_data_mat['CAD50'] = ldmi_ext.data1_full_sd_imp['CAD50'] 
PC_data_mat.columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','age','sex','statin','hypertension','intercept','CAD50']



###############################################################
# Calculate PCA values of train and test data to allow prediction
data_train = ldmi_ext.train1 #1105 rows
data_test = ldmi_ext.test1 #369 rows

metabolities_train = data_train.ix[:,0:225]
metabolities_test = data_test.ix[:,0:225]

PC_data_train_mat = np.zeros(shape=(1105,6))
PC_data_test_mat = np.zeros(shape=(369,6))


for i in range(0,6):
    for j in range(0,1105):
        PC_data_train_mat[j,i]= np.dot(pca_components[i,:],metabolities_train.ix[j,:].values)
        
PC_data_train = pd.DataFrame(PC_data_train_mat)
PC_data_train['age'] = data_train['Age']
PC_data_train['sex'] = data_train['sex']
PC_data_train['statin'] = data_train['statin']
PC_data_train['hypertension'] = data_train['hypertension_4c']
PC_data_train['intercept'] = 1
PC_data_train['CAD50'] = data_train['CAD50'] 
PC_data_train.columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','age','sex','statin','hypertension','intercept','CAD50']


for i in range(0,6):
    for j in range(0,369):
        PC_data_test_mat[j,i]= np.dot(pca_components[i,:],metabolities_test.ix[j,:].values)
        
PC_data_test = pd.DataFrame(PC_data_test_mat)
PC_data_test['age'] = data_test['Age']
PC_data_test['sex'] = data_test['sex']
PC_data_test['statin'] = data_test['statin']
PC_data_test['hypertension'] = data_test['hypertension_4c']
PC_data_test['intercept'] = 1
PC_data_test['CAD50'] = data_test['CAD50'] 
PC_data_test.columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','age','sex','statin','hypertension','intercept','CAD50']


# Run regressions

logit_mod2 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train.ix[:,0:10]],axis=1), missing = 'drop')
result_mod2 = logit_mod2.fit()
result_mod2.summary()

#drop PC3
logit_mod3 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train['PC1'],PC_data_train['PC2']
                                                ,PC_data_train.ix[:,3:10]],axis=1), missing = 'drop')
result_mod3 = logit_mod3.fit()
result_mod3.summary()

#drop PC4
logit_mod4 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train['PC1'],PC_data_train['PC2'],
                                                 PC_data_train['PC5'],PC_data_train['PC6'],PC_data_train.ix[:,6:10]],axis=1)
                                                 , missing = 'drop')
result_mod4 = logit_mod4.fit()
result_mod4.summary()

#drop PC5
logit_mod5 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train['PC1'],PC_data_train['PC2'],
                                                 PC_data_train['PC6'],PC_data_train.ix[:,6:10]],axis=1), missing = 'drop')
result_mod5 = logit_mod5.fit()
result_mod5.summary()


logit_mod5 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train['PC1'],PC_data_train['PC2'],
                                                 PC_data_train['PC6'],PC_data_train.ix[:,6:10]],axis=1), missing = 'drop')
result_mod5 = logit_mod5.fit()
result_mod5.summary()

preds1 = result_mod5.predict(pd.concat([PC_data_test['intercept'],PC_data_test['PC1'],PC_data_test['PC2'],
                                                 PC_data_test['PC6'],PC_data_test.ix[:,6:10]],axis=1))



preds_bin1 = (preds1 > 0.5).astype(int)
truevals = np.array(PC_data_test.ix[:,11])
accuracy1 = ((preds_bin1== truevals).sum())/(len(PC_data_test)*1.0)
print accuracy1
0.756097560976


###############################################################################
############## PCA decomposition of metabolites analysis Impuation set 2 ##############

###### Impuatation set 2 ######

data_imp = ldmi_ext.data1_full_sd_imp2

metabolities_mat_imp = ldmi_ext.data1_full_sd_imp2.ix[:,0:225]

pca = PCA(n_components=15, copy=True)
pca.fit(metabolities_mat_imp)
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Principle Components', size = 13)
plt.ylabel('Proportion of variance Captured', size = 13)
plt.title('Variance explained by \n successive Principle Components', size = 14) # use top 6

pca = PCA(n_components=6, copy=True)
pca.fit(metabolities_mat_imp)
pca_components = pca.components_

pd.DataFrame(pca_components)

# Calculate PC values of data
PC_data = np.zeros(shape=(1474,6))

for i in range(0,6):
    for j in range(0,1474):
        PC_data[j,i]= np.dot(pca_components[i,:],metabolities_mat_imp.ix[j,:].values)
        
PC_data_mat = pd.DataFrame(PC_data)
PC_data_mat['age'] = ldmi_ext.data1_full_sd_imp2['Age']
PC_data_mat['sex'] = ldmi_ext.data1_full_sd_imp2['sex']
PC_data_mat['statin'] = ldmi_ext.data1_full_sd_imp2['statin']
PC_data_mat['hypertension'] = ldmi_ext.data1_full_sd_imp2['hypertension_4c']
PC_data_mat['intercept'] = 1
PC_data_mat['CAD50'] = ldmi_ext.data1_full_sd_imp2['CAD50'] 
PC_data_mat.columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','age','sex','statin','hypertension','intercept','CAD50']



###############################################################
# Calculate PCA values of train and test data to allow prediction
data_train = ldmi_ext.train12 #1105 rows
data_test = ldmi_ext.test12 #369 rows

metabolities_train = data_train.ix[:,0:225]
metabolities_test = data_test.ix[:,0:225]

PC_data_train_mat = np.zeros(shape=(1105,6))
PC_data_test_mat = np.zeros(shape=(369,6))


for i in range(0,6):
    for j in range(0,1105):
        PC_data_train_mat[j,i]= np.dot(pca_components[i,:],metabolities_train.ix[j,:].values)
        
PC_data_train = pd.DataFrame(PC_data_train_mat)
PC_data_train['age'] = data_train['Age']
PC_data_train['sex'] = data_train['sex']
PC_data_train['statin'] = data_train['statin']
PC_data_train['hypertension'] = data_train['hypertension_4c']
PC_data_train['intercept'] = 1
PC_data_train['CAD50'] = data_train['CAD50'] 
PC_data_train.columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','age','sex','statin','hypertension','intercept','CAD50']


for i in range(0,6):
    for j in range(0,369):
        PC_data_test_mat[j,i]= np.dot(pca_components[i,:],metabolities_test.ix[j,:].values)
        
PC_data_test = pd.DataFrame(PC_data_test_mat)
PC_data_test['age'] = data_test['Age']
PC_data_test['sex'] = data_test['sex']
PC_data_test['statin'] = data_test['statin']
PC_data_test['hypertension'] = data_test['hypertension_4c']
PC_data_test['intercept'] = 1
PC_data_test['CAD50'] = data_test['CAD50'] 
PC_data_test.columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','age','sex','statin','hypertension','intercept','CAD50']


# Run regressions

logit_mod2 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train.ix[:,0:10]],axis=1), missing = 'drop')
result_mod2 = logit_mod2.fit()
result_mod2.summary()

#drop PC3
logit_mod3 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train['PC1'],PC_data_train['PC2']
                                                ,PC_data_train.ix[:,3:10]],axis=1), missing = 'drop')
result_mod3 = logit_mod3.fit()
result_mod3.summary()

#drop PC4
logit_mod4 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train['PC1'],PC_data_train['PC2'],
                                                 PC_data_train['PC5'],PC_data_train['PC6'],PC_data_train.ix[:,6:10]],axis=1)
                                                 , missing = 'drop')
result_mod4 = logit_mod4.fit()
result_mod4.summary()

#drop PC6
logit_mod5 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train['PC1'],PC_data_train['PC2'],
                                                 PC_data_train['PC5'],PC_data_train.ix[:,6:10]],axis=1), missing = 'drop')
result_mod5 = logit_mod5.fit()
result_mod5.summary()


logit_mod5 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train['PC1'],PC_data_train['PC2'],
                                                 PC_data_train['PC5'],PC_data_train.ix[:,6:10]],axis=1), missing = 'drop')
result_mod5 = logit_mod5.fit()
result_mod5.summary()

preds2 = result_mod5.predict(pd.concat([PC_data_test['intercept'],PC_data_test['PC1'],PC_data_test['PC2'],
                                                 PC_data_test['PC5'],PC_data_test.ix[:,6:10]],axis=1))



preds_bin2 = (preds2 > 0.5).astype(int)
truevals = np.array(PC_data_test.ix[:,11])
accuracy2 = ((preds_bin2== truevals).sum())/(len(PC_data_test)*1.0)
print accuracy2
0.764227642276



###############################################################################
############## PCA decomposition of metabolites analysis Impuation set 3 ##############

###### Impuatation set 3 ######

data_imp = ldmi_ext.data1_full_sd_imp3

metabolities_mat_imp = ldmi_ext.data1_full_sd_imp3.ix[:,0:225]

pca = PCA(n_components=15, copy=True)
pca.fit(metabolities_mat_imp)
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Principle Components', size = 13)
plt.ylabel('Proportion of variance Captured', size = 13)
plt.title('Variance explained by \n successive Principle Components', size = 14) # use top 6

pca = PCA(n_components=6, copy=True)
pca.fit(metabolities_mat_imp)
pca_components = pca.components_

pd.DataFrame(pca_components)

# Calculate PC values of data
PC_data = np.zeros(shape=(1474,6))

for i in range(0,6):
    for j in range(0,1474):
        PC_data[j,i]= np.dot(pca_components[i,:],metabolities_mat_imp.ix[j,:].values)
        
PC_data_mat = pd.DataFrame(PC_data)
PC_data_mat['age'] = ldmi_ext.data1_full_sd_imp3['Age']
PC_data_mat['sex'] = ldmi_ext.data1_full_sd_imp3['sex']
PC_data_mat['statin'] = ldmi_ext.data1_full_sd_imp3['statin']
PC_data_mat['hypertension'] = ldmi_ext.data1_full_sd_imp3['hypertension_4c']
PC_data_mat['intercept'] = 1
PC_data_mat['CAD50'] = ldmi_ext.data1_full_sd_imp3['CAD50'] 
PC_data_mat.columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','age','sex','statin','hypertension','intercept','CAD50']



###############################################################
# Calculate PCA values of train and test data to allow prediction
data_train = ldmi_ext.train13 #1105 rows
data_test = ldmi_ext.test13 #369 rows

metabolities_train = data_train.ix[:,0:225]
metabolities_test = data_test.ix[:,0:225]

PC_data_train_mat = np.zeros(shape=(1105,6))
PC_data_test_mat = np.zeros(shape=(369,6))


for i in range(0,6):
    for j in range(0,1105):
        PC_data_train_mat[j,i]= np.dot(pca_components[i,:],metabolities_train.ix[j,:].values)
        
PC_data_train = pd.DataFrame(PC_data_train_mat)
PC_data_train['age'] = data_train['Age']
PC_data_train['sex'] = data_train['sex']
PC_data_train['statin'] = data_train['statin']
PC_data_train['hypertension'] = data_train['hypertension_4c']
PC_data_train['intercept'] = 1
PC_data_train['CAD50'] = data_train['CAD50'] 
PC_data_train.columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','age','sex','statin','hypertension','intercept','CAD50']


for i in range(0,6):
    for j in range(0,369):
        PC_data_test_mat[j,i]= np.dot(pca_components[i,:],metabolities_test.ix[j,:].values)
        
PC_data_test = pd.DataFrame(PC_data_test_mat)
PC_data_test['age'] = data_test['Age']
PC_data_test['sex'] = data_test['sex']
PC_data_test['statin'] = data_test['statin']
PC_data_test['hypertension'] = data_test['hypertension_4c']
PC_data_test['intercept'] = 1
PC_data_test['CAD50'] = data_test['CAD50'] 
PC_data_test.columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','age','sex','statin','hypertension','intercept','CAD50']


# Run regressions

logit_mod2 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train.ix[:,0:10]],axis=1), missing = 'drop')
result_mod2 = logit_mod2.fit()
result_mod2.summary()

#drop PC3
logit_mod3 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train['PC1'],PC_data_train['PC2']
                                                ,PC_data_train.ix[:,3:10]],axis=1), missing = 'drop')
result_mod3 = logit_mod3.fit()
result_mod3.summary()

#drop PC4
logit_mod4 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train['PC1'],PC_data_train['PC2'],
                                                 PC_data_train['PC5'],PC_data_train['PC6'],PC_data_train.ix[:,6:10]],axis=1)
                                                 , missing = 'drop')
result_mod4 = logit_mod4.fit()
result_mod4.summary()

#drop PC6
logit_mod5 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train['PC1'],PC_data_train['PC2'],
                                                 PC_data_train['PC5'],PC_data_train.ix[:,6:10]],axis=1), missing = 'drop')
result_mod5 = logit_mod5.fit()
result_mod5.summary()


logit_mod5 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train['PC1'],PC_data_train['PC2'],
                                                 PC_data_train['PC5'],PC_data_train.ix[:,6:10]],axis=1), missing = 'drop')
result_mod5 = logit_mod5.fit()
result_mod5.summary()

preds3 = result_mod5.predict(pd.concat([PC_data_test['intercept'],PC_data_test['PC1'],PC_data_test['PC2'],
                                                 PC_data_test['PC5'],PC_data_test.ix[:,6:10]],axis=1))



preds_bin3 = (preds3 > 0.5).astype(int)
truevals = np.array(PC_data_test.ix[:,11])
accuracy3 = ((preds_bin3== truevals).sum())/(len(PC_data_test)*1.0)
print accuracy3
0.764227642276


###############################################################################
############## PCA decomposition of metabolites analysis Impuation set 4 ##############

###### Impuatation set 4 ######

data_imp = ldmi_ext.data1_full_sd_imp4

metabolities_mat_imp = ldmi_ext.data1_full_sd_imp4.ix[:,0:225]

pca = PCA(n_components=15, copy=True)
pca.fit(metabolities_mat_imp)
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Principle Components', size = 13)
plt.ylabel('Proportion of variance Captured', size = 13)
plt.title('Variance explained by \n successive Principle Components', size = 14) # use top 6

pca = PCA(n_components=6, copy=True)
pca.fit(metabolities_mat_imp)
pca_components = pca.components_

pd.DataFrame(pca_components)

# Calculate PC values of data
PC_data = np.zeros(shape=(1474,6))

for i in range(0,6):
    for j in range(0,1474):
        PC_data[j,i]= np.dot(pca_components[i,:],metabolities_mat_imp.ix[j,:].values)
        
PC_data_mat = pd.DataFrame(PC_data)
PC_data_mat['age'] = ldmi_ext.data1_full_sd_imp4['Age']
PC_data_mat['sex'] = ldmi_ext.data1_full_sd_imp4['sex']
PC_data_mat['statin'] = ldmi_ext.data1_full_sd_imp4['statin']
PC_data_mat['hypertension'] = ldmi_ext.data1_full_sd_imp4['hypertension_4c']
PC_data_mat['intercept'] = 1
PC_data_mat['CAD50'] = ldmi_ext.data1_full_sd_imp4['CAD50'] 
PC_data_mat.columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','age','sex','statin','hypertension','intercept','CAD50']



###############################################################
# Calculate PCA values of train and test data to allow prediction
data_train = ldmi_ext.train14 #1105 rows
data_test = ldmi_ext.test14 #369 rows

metabolities_train = data_train.ix[:,0:225]
metabolities_test = data_test.ix[:,0:225]

PC_data_train_mat = np.zeros(shape=(1105,6))
PC_data_test_mat = np.zeros(shape=(369,6))


for i in range(0,6):
    for j in range(0,1105):
        PC_data_train_mat[j,i]= np.dot(pca_components[i,:],metabolities_train.ix[j,:].values)
        
PC_data_train = pd.DataFrame(PC_data_train_mat)
PC_data_train['age'] = data_train['Age']
PC_data_train['sex'] = data_train['sex']
PC_data_train['statin'] = data_train['statin']
PC_data_train['hypertension'] = data_train['hypertension_4c']
PC_data_train['intercept'] = 1
PC_data_train['CAD50'] = data_train['CAD50'] 
PC_data_train.columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','age','sex','statin','hypertension','intercept','CAD50']


for i in range(0,6):
    for j in range(0,369):
        PC_data_test_mat[j,i]= np.dot(pca_components[i,:],metabolities_test.ix[j,:].values)
        
PC_data_test = pd.DataFrame(PC_data_test_mat)
PC_data_test['age'] = data_test['Age']
PC_data_test['sex'] = data_test['sex']
PC_data_test['statin'] = data_test['statin']
PC_data_test['hypertension'] = data_test['hypertension_4c']
PC_data_test['intercept'] = 1
PC_data_test['CAD50'] = data_test['CAD50'] 
PC_data_test.columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','age','sex','statin','hypertension','intercept','CAD50']


# Run regressions

logit_mod2 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train.ix[:,0:10]],axis=1), missing = 'drop')
result_mod2 = logit_mod2.fit()
result_mod2.summary()

#drop PC3
logit_mod3 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train['PC1'],PC_data_train['PC2']
                                                ,PC_data_train.ix[:,3:10]],axis=1), missing = 'drop')
result_mod3 = logit_mod3.fit()
result_mod3.summary()

#drop PC4
logit_mod4 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train['PC1'],PC_data_train['PC2'],
                                                 PC_data_train['PC5'],PC_data_train['PC6'],PC_data_train.ix[:,6:10]],axis=1)
                                                 , missing = 'drop')
result_mod4 = logit_mod4.fit()
result_mod4.summary()

#drop PC6
logit_mod5 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train['PC1'],PC_data_train['PC2'],
                                                 PC_data_train['PC5'],PC_data_train.ix[:,6:10]],axis=1), missing = 'drop')
result_mod5 = logit_mod5.fit()
result_mod5.summary()


logit_mod5 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train['PC1'],PC_data_train['PC2'],
                                                 PC_data_train['PC5'],PC_data_train.ix[:,6:10]],axis=1), missing = 'drop')
result_mod5 = logit_mod5.fit()
result_mod5.summary()

preds4 = result_mod5.predict(pd.concat([PC_data_test['intercept'],PC_data_test['PC1'],PC_data_test['PC2'],
                                                 PC_data_test['PC5'],PC_data_test.ix[:,6:10]],axis=1))


preds_bin4 = (preds4 > 0.5).astype(int)
truevals = np.array(PC_data_test.ix[:,11])
accuracy4 = ((preds_bin4== truevals).sum())/(len(PC_data_test)*1.0)
print accuracy4
0.761517615176


###############################################################################
############## PCA decomposition of metabolites analysis Impuation set 5 ##############

###### Impuatation set 5 ######

data_imp = ldmi_ext.data1_full_sd_imp5

metabolities_mat_imp = ldmi_ext.data1_full_sd_imp5.ix[:,0:225]

pca = PCA(n_components=15, copy=True)
pca.fit(metabolities_mat_imp)
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Principle Components', size = 13)
plt.ylabel('Proportion of variance Captured', size = 13)
plt.title('Variance explained by \n successive Principle Components', size = 14) # use top 6

pca = PCA(n_components=6, copy=True)
pca.fit(metabolities_mat_imp)
pca_components = pca.components_

pd.DataFrame(pca_components)

# Calculate PC values of data
PC_data = np.zeros(shape=(1474,6))

for i in range(0,6):
    for j in range(0,1474):
        PC_data[j,i]= np.dot(pca_components[i,:],metabolities_mat_imp.ix[j,:].values)
        
PC_data_mat = pd.DataFrame(PC_data)
PC_data_mat['age'] = ldmi_ext.data1_full_sd_imp5['Age']
PC_data_mat['sex'] = ldmi_ext.data1_full_sd_imp5['sex']
PC_data_mat['statin'] = ldmi_ext.data1_full_sd_imp5['statin']
PC_data_mat['hypertension'] = ldmi_ext.data1_full_sd_imp5['hypertension_4c']
PC_data_mat['intercept'] = 1
PC_data_mat['CAD50'] = ldmi_ext.data1_full_sd_imp5['CAD50'] 
PC_data_mat.columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','age','sex','statin','hypertension','intercept','CAD50']




###############################################################
# Calculate PCA values of train and test data to allow prediction
data_train = ldmi_ext.train15 #1105 rows
data_test = ldmi_ext.test15 #369 rows

metabolities_train = data_train.ix[:,0:225]
metabolities_test = data_test.ix[:,0:225]

PC_data_train_mat = np.zeros(shape=(1105,6))
PC_data_test_mat = np.zeros(shape=(369,6))


for i in range(0,6):
    for j in range(0,1105):
        PC_data_train_mat[j,i]= np.dot(pca_components[i,:],metabolities_train.ix[j,:].values)
        
PC_data_train = pd.DataFrame(PC_data_train_mat)
PC_data_train['age'] = data_train['Age']
PC_data_train['sex'] = data_train['sex']
PC_data_train['statin'] = data_train['statin']
PC_data_train['hypertension'] = data_train['hypertension_4c']
PC_data_train['intercept'] = 1
PC_data_train['CAD50'] = data_train['CAD50'] 
PC_data_train.columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','age','sex','statin','hypertension','intercept','CAD50']


for i in range(0,6):
    for j in range(0,369):
        PC_data_test_mat[j,i]= np.dot(pca_components[i,:],metabolities_test.ix[j,:].values)
        
PC_data_test = pd.DataFrame(PC_data_test_mat)
PC_data_test['age'] = data_test['Age']
PC_data_test['sex'] = data_test['sex']
PC_data_test['statin'] = data_test['statin']
PC_data_test['hypertension'] = data_test['hypertension_4c']
PC_data_test['intercept'] = 1
PC_data_test['CAD50'] = data_test['CAD50'] 
PC_data_test.columns = ['PC1', 'PC2','PC3','PC4','PC5','PC6','age','sex','statin','hypertension','intercept','CAD50']


# Run regressions

logit_mod2 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train.ix[:,0:10]],axis=1), missing = 'drop')
result_mod2 = logit_mod2.fit()
result_mod2.summary()

#drop PC3
logit_mod3 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train['PC1'],PC_data_train['PC2']
                                                ,PC_data_train.ix[:,3:10]],axis=1), missing = 'drop')
result_mod3 = logit_mod3.fit()
result_mod3.summary()

#drop PC4
logit_mod4 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train['PC1'],PC_data_train['PC2'],
                                                 PC_data_train['PC5'],PC_data_train['PC6'],PC_data_train.ix[:,6:10]],axis=1)
                                                 , missing = 'drop')
result_mod4 = logit_mod4.fit()
result_mod4.summary()

#drop PC5
logit_mod5 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train['PC1'],PC_data_train['PC2'],
                                                 PC_data_train['PC6'],PC_data_train.ix[:,6:10]],axis=1), missing = 'drop')
result_mod5 = logit_mod5.fit()
result_mod5.summary()


logit_mod5 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train['PC1'],PC_data_train['PC2'],
                                                 PC_data_train['PC6'],PC_data_train.ix[:,6:10]],axis=1), missing = 'drop')
result_mod5 = logit_mod5.fit()
result_mod5.summary()

preds5 = result_mod5.predict(pd.concat([PC_data_test['intercept'],PC_data_test['PC1'],PC_data_test['PC2'],
                                                 PC_data_test['PC6'],PC_data_test.ix[:,6:10]],axis=1))


preds_bin5 = (preds5 > 0.5).astype(int)
truevals = np.array(PC_data_test.ix[:,11])
accuracy5 = ((preds_bin5== truevals).sum())/(len(PC_data_test)*1.0)
print accuracy5
0.756097560976


###################################################################################################################
##################### aveage predictions with confounders #####################
###################################################################################################################



preds_proba_average = (preds1 + preds2 + preds3 + preds4 + preds5)/5
preds_proba_average_bin = (preds_proba_average >0.5 ).astype(int)

confusion_matrix(truevals, preds_proba_average_bin, labels=[1, 0])
PCA_av_acc = (preds_proba_average_bin == truevals).sum()/(len(ldmi.test14)*1.0)
print PCA_av_acc 


fprPCA, tprPCA, _ = roc_curve(truevals, preds_proba_average)
# Calculate the AUC
roc_aucPCA = auc(fprPCA, tprPCA)
print 'ROC AUC: %0.3f' % roc_aucPCA
 
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fprPCA, tprPCA, label='ROC curve (area = %0.3f)' % roc_aucPCA)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', size = 12)
plt.ylabel('True Positive Rate', size = 12)
plt.title('ROC Curve for PCA regression averaged across the imputation datasets')
plt.legend(loc="lower right")
plt.show()




######################################################################################################################




plt.figure(figsize=(8, 6))
plt.plot(fprPCA, tprPCA, label='ROC curve of PCA regression (area = %0.3f)' % roc_aucPCA, color = 'firebrick')
plt.plot(fpr_L1, tpr_L1, label='ROC curve of L1 regression (area = %0.3f)' % roc_auc_L1, color = 'green')
plt.plot(fpr_rf, tpr_rf, label='ROC curve of random forest (area = %0.3f)' % roc_auc_rf, color = 'mediumblue')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity', size = 12)
plt.ylabel('Sensitivity', size = 12)
plt.title('ROC Curve for predictions using PCA regression, penalised \n logistic regression and random forest, including confounders ')
plt.legend(loc="lower right")
plt.show()
