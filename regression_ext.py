# -*- coding: utf-8 -*-
"""
Regression

@author: ucabhmf
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
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import statsmodels.api as sm
from sklearn.manifold import TSNE
from matplotlib  import cm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support 


import load_data_ext as ld_ext
import load_data_mi_ext_new as ldmi_ext

np.random.seed(10)


############################################################################

# Quick logistic regression with lasso penalty, chosen with cross validation
# website used for much code:


# Initial random model
mod1 = LogisticRegression(C=0.5, penalty='l1')

# Smallest value of C before all coefficients set to zero
min_l1_C = l1_min_c(ld_ext.train1.ix[:,0:229], ld_ext.train1.ix[:,229]) 
'%f' % min_l1_C # 0.000028 ~= 0.00003

#create candidate values of C
c_vals = min_l1_C * np.logspace(0, 4, 15)

# Create a dictionary whose keys are the candidate values of C. 
# The dictionary will hold the error rates in each CV trial for that
# value of C.
cdict = {}
for c in c_vals:
    cdict[c] = []

# Cross validation to choose c. train1 and test1 already have randomized rows from train_test_split

# Genaerate indicies to split data into 50 chunks
cv_index = [ma.ceil((len(ld_ext.train1)/50)*x) for x in range(51)]

for i in range(50):
    # Split the data
    print(i)
    
    test_cv = ld_ext.train1.ix[cv_index[i]:cv_index[i+1],:]
    train_cv = ld_ext.train1.drop(ld_ext.train1.index[cv_index[i]:cv_index[i+1]])

    trainy, testy = train_cv.ix[:,229], test_cv.ix[:,229]
    trainX, testX = train_cv.ix[:,0:229], test_cv.ix[:,0:229]
    
    for c in cdict:
        mod1.set_params(C=c)
        logit_fit = mod1.fit(trainX.values, trainy.values)
        predy = mod1.predict(testX.values)
        error_rate = np.mean(predy != testy)
        cdict[c].append(error_rate)

#### Plot ####
error_path = pd.DataFrame(cdict).mean()
error_path.plot(style = 'o-k', label = 'Error rate')
error_path.cummin().plot(style = 'r-', label = 'Lowest error')
plt.xlabel('Regularization parameter')
plt.ylabel('Prediction error rate')
plt.legend(loc = 'upper right')
plt.title('Plot of error rate with increasing regularisation \n  parameter values based on cross-validation results' )
plt.axis([-0.01, 0.9, 0.2, 0.4])
####     ####

#best using regularisation #0.226784
min_error_c = 0.122346
logit_model_best = LogisticRegression(C = min_error_c, penalty = 'l1')

logit_fit_best = logit_model_best.fit(ld_ext.train1.ix[:,0:229], ld_ext.train1.ix[:,229])
keep_terms = ld_ext.train1.ix[:,0:229].columns[np.where(logit_fit_best.coef_ > 0)[1]]
keep_terms2 = ld_ext.train1.ix[:,0:229].columns[np.where(logit_fit_best.coef_ < 0)[1]]


# Get list of parameter estimates
a = pd.Series(np.asarray(logit_fit_best.coef_)[0])
metabolite_names = pd.Series(np.asarray(list(ld_ext.train1.ix[:,0:229].columns.values)))

parameter_estimates_all = pd.DataFrame({'metabolite_names': metabolite_names,'values':a})
parameter_estimates = parameter_estimates_all.loc[parameter_estimates_all['values'] != 0]

parameter_estimates_ordered = parameter_estimates_all.loc[parameter_estimates_all['values'] != 0].sort(columns='values', ascending=True)
parameter_estimates_ordered['abs_values'] = abs(parameter_estimates_ordered['values'])
parameter_estimates_ordered = parameter_estimates_ordered.sort(columns='abs_values', ascending=False).ix[:,0:2]

parameter_estimates_ordered

# Predict on test data

preds = logit_model_best.predict(ld_ext.test1.ix[:,0:229])
truevals = np.array(ld_ext.test1.ix[:,229])



accuracy = ((preds == truevals).sum())/(len(ld_ext.test1.ix[:,0:229])*1.0)
print accuracy


# Confusion matrix
confusion_matrix(truevals, preds, labels=[1, 0])


# total pos: 117
sum(truevals)

# total neg: 57
sum(truevals-1)


# Plot AUC curve
preds_proba = logit_model_best.predict_proba(ld_ext.test1.ix[:,0:229])[:,1]

# Determine the false positive and true positive rates
fpr, tpr, _ = roc_curve(truevals, preds_proba)
 
# Calculate the AUC
roc_auc = auc(fpr, tpr)
print 'ROC AUC: %0.2f' % roc_auc
#0.68
 
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for L1 penalised logitic regression')
plt.legend(loc="lower right")
plt.show()



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
'%f' % min_l1_C 


c_vals = min_l1_C * np.logspace(0, 4, 17) 


cdict1 = {}
for c in c_vals:
    cdict1[c] = []

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
print 'ROC AUC: %0.3f' % roc_auc1

# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr1, tpr1, label='ROC curve (AUC = %0.3f)' % roc_auc1)
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
##################################  Plot cross validation graphs  ####################################
#################################################################################################

plt.figure(figsize=(10,12))
plt.subplot(3, 2, 1)
error_path1.plot(style = 'o-k', label = 'Error rate')
error_path1.cummin().plot(style = 'r-', label = 'Lower envelope')
plt.xlabel('regularization parameter - dataset 1')
plt.ylabel('Prediction error rate')
plt.legend(loc = 'upper right')
plt.axis([-0.005, 1, 0.2, 0.35])

plt.subplot(3, 2, 2)
error_path2.plot(style = 'o-k', label = 'Error rate')
error_path2.cummin().plot(style = 'r-', label = 'Lower envelope')
plt.xlabel('regularization parameter - dataset 2')
plt.ylabel('Prediction error rate')
plt.legend(loc = 'upper right')
plt.axis([-0.005, 1, 0.2, 0.35])

plt.subplot(3, 2, 3)
error_path3.plot(style = 'o-k', label = 'Error rate')
error_path3.cummin().plot(style = 'r-', label = 'Lower envelope')
plt.xlabel('regularization parameter - dataset 3')
plt.ylabel('Prediction error rate')
plt.legend(loc = 'upper right')
plt.axis([-0.005, 1, 0.2, 0.35])

plt.subplot(3, 2, 4)
error_path4.plot(style = 'o-k', label = 'Error rate')
error_path4.cummin().plot(style = 'r-', label = 'Lower envelope')
plt.xlabel('regularization parameter - dataset 4')
plt.ylabel('Prediction error rate')
plt.legend(loc = 'upper right')
plt.axis([-0.005, 1, 0.2, 0.35])

plt.subplot(3, 2, 5)
error_path5.plot(style = 'o-k', label = 'Error rate')
error_path5.cummin().plot(style = 'r-', label = 'Lower envelope')
plt.xlabel('regularization parameter - dataset 5')
plt.ylabel('Prediction error rate')
plt.legend(loc = 'upper right')
plt.axis([-0.005, 1, 0.2, 0.35])

plt.show()





#################################################################################################
######################### Combine imputation results - average ##################################
#################################################################################################


# Better way to calculate average probability
preds_proba_average = (preds_proba1+preds_proba2+preds_proba3+preds_proba4+preds_proba5)/5
preds_proba_average_bin =  (preds_proba_average > 0.5).astype(int)

accuracy_proba_average = ((preds_proba_average_bin== truevals).sum())/(len(test_dataX)*1.0)
print accuracy_proba_average

confusion_matrix_average = confusion_matrix(truevals, preds_proba_average_bin, labels=[1, 0])



fpr_av_logistic, tpr_av_logistic, _ = roc_curve(truevals, preds_proba_average)
 
# Calculate the AUC
roc_auc_av = auc(fpr_av_logistic, tpr_av_logistic)
print 'ROC AUC: %0.3f' % roc_auc_av

 
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr_av_logistic, tpr_av_logistic, label='ROC curve (AUC = %0.3f)' % roc_auc_av)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', size = 14)
plt.ylabel('True Positive Rate', size = 14)
plt.title('ROC Curve for L1 penalised logitic regression averaged across the \n imputation datasets with both metabolites and clinical confounders', size = 14)
plt.legend(loc="lower right")
plt.show()


# Average parameter estimates

parameter_estimates_all_average = parameter_estimates_all1.copy(deep=True)
parameter_estimates_all_average['values'] = (parameter_estimates_all1['values'] + parameter_estimates_all2['values'] 
                                            + parameter_estimates_all3 ['values'] + parameter_estimates_all4 ['values']
                                            + parameter_estimates_all5 ['values'])/5

parameter_estimates_average = parameter_estimates_all_average.loc[parameter_estimates_all_average['values'] != 0]

parameter_estimates_ordered_average = parameter_estimates_all_average.loc[parameter_estimates_all_average['values'] != 0].sort(columns='values', ascending=True)
parameter_estimates_ordered_average['abs_values'] = abs(parameter_estimates_ordered_average['values'])
parameter_estimates_ordered_average = parameter_estimates_ordered_average.sort(columns='abs_values', ascending=False).ix[:,0:2]

parameter_estimates_ordered_average




# Parameters consistently included in model
parameter_estimates_all_count1 = (parameter_estimates_all1['values'] != 0).astype(int)
parameter_estimates_all_count2 = (parameter_estimates_all2['values'] != 0).astype(int)
parameter_estimates_all_count3 = (parameter_estimates_all3['values'] != 0).astype(int)
parameter_estimates_all_count4 = (parameter_estimates_all4['values'] != 0).astype(int)
parameter_estimates_all_count5 = (parameter_estimates_all5['values'] != 0).astype(int)

parameter_estimates_all_count = parameter_estimates_all1.copy(deep=True)
parameter_estimates_all_count['values'] = parameter_estimates_all_count1 + parameter_estimates_all_count2 + parameter_estimates_all_count3 + parameter_estimates_all_count4 + parameter_estimates_all_count5
# Parameter counts which are not zero
parameter_estimates_all_count_ordered = parameter_estimates_all_count.loc[parameter_estimates_all_count['values'] != 0].sort(columns='values', ascending=False)


# Parameters included summary
parameter_estimates_summary = parameter_estimates_all_average.copy(deep=True)
parameter_estimates_summary['average value'] = parameter_estimates_all_average['values']
parameter_estimates_summary['count'] = parameter_estimates_all_count['values']
parameter_estimates_summary['value_1'] = parameter_estimates_all1['values']
parameter_estimates_summary['value_2'] = parameter_estimates_all2['values']
parameter_estimates_summary['value_3'] = parameter_estimates_all3['values']
parameter_estimates_summary['value_4'] = parameter_estimates_all4['values']
parameter_estimates_summary['value_5'] = parameter_estimates_all5['values']
parameter_estimates_summary['abs_value'] = abs(parameter_estimates_summary['average value'])
parameter_estimates_summary_ordered = parameter_estimates_summary.loc[parameter_estimates_summary['count'] != 0].sort(columns='abs_value', ascending=False)
del parameter_estimates_summary_ordered['values']
del parameter_estimates_summary_ordered['abs_value']
print parameter_estimates_summary_ordered

parameter_estimates_summary_ordered
parameter_estimates_summary_ordered.to_csv('parameter_estimates_summary_ordered_EXT.csv')





##### INCLUDE ANALYSIS OF which patients wrongly classified, eg high probability = correct, medium always wrong?

preds_proba_average_confident80 = preds_proba_average[(preds_proba_average > 0.8) | (preds_proba_average < 0.2)]
preds_proba_average_confident80_bin = (preds_proba_average_confident80 > 0.5).astype(int)
truevals_confident80 = truevals[(preds_proba_average > 0.8) | (preds_proba_average < 0.2)]

accuracy_proba_average_confident80 = ((preds_proba_average_confident80_bin== truevals_confident80).sum())/(len(truevals_confident80)*1.0)
print accuracy_proba_average_confident80



preds_proba_average_confident60 = preds_proba_average[(preds_proba_average > 0.6) | (preds_proba_average < 0.4)]
preds_proba_average_confident60_bin = (preds_proba_average_confident60 > 0.5).astype(int)
truevals_confident60 = truevals[(preds_proba_average > 0.6) | (preds_proba_average < 0.4)]

accuracy_proba_average_confident60 = ((preds_proba_average_confident60_bin== truevals_confident60).sum())/(len(truevals_confident60)*1.0)
print accuracy_proba_average_confident60



preds_proba_average_confident90 = preds_proba_average[(preds_proba_average > 0.9) | (preds_proba_average < 0.1)]
preds_proba_average_confident90_bin = (preds_proba_average_confident90 > 0.5).astype(int)
truevals_confident90 = truevals[(preds_proba_average > 0.9) | (preds_proba_average < 0.1)]

accuracy_proba_average_confident90 = ((preds_proba_average_confident90_bin== truevals_confident90).sum())/(len(truevals_confident90)*1.0)
print accuracy_proba_average_confident90

fpr_av, tpr_av, _ = roc_curve(truevals_confident80, preds_proba_average_confident80)
 
# Calculate the AUC
roc_auc_av = auc(fpr_av, tpr_av)
print 'ROC AUC: %0.2f' % roc_auc_av
#0.65 for 5.4 regularisation
#0.66 for 1.6 regularsation
 
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr_av, tpr_av, label='ROC curve (AUC = %0.2f)' % roc_auc_av)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for L1 penalised logitic regression averaged across the imputation datasets')
plt.legend(loc="lower right")
plt.show()


