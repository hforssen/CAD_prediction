# -*- coding: utf-8 -*-
"""
Combined ROC curves no conf
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
from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

import load_data as ld
import load_data_mi_ext_new as ldmi_ext
import load_data_ext as ld_ext
import load_data as ld
import load_data_mi as ldmi


################################################################################################################################

################################################################################################################################

################################################################################################################################

################################################################################################################################

################################################################################################################################

################################################################################################################################

################################################################################################################################

# L1 regression



np.random.seed(10)

# Initial random model
mod1 = LogisticRegression(C=0.5, penalty='l1')

# Initialise full train and test sets
train_data = ldmi.train11
train_dataX = ldmi.train11.ix[:,0:225]
train_dataY = ldmi.train11.ix[:,225]
test_data = ldmi.test11
test_dataX = ldmi.test11.ix[:,0:225]
test_dataY = ldmi.test11.ix[:,225]


# Smallest value of C before all coefficients set to zero
min_l1_C = l1_min_c(train_dataX,train_dataY) 
'%f' % min_l1_C 

c_vals = min_l1_C * np.logspace(0, 3.7, 15) 

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

    trainy, testy = train_cv.ix[:,225], test_cv.ix[:,225]
    trainX, testX = train_cv.ix[:,0:225], test_cv.ix[:,0:225]
    
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
plt.axis([-0.005, 3, 0.25, 0.35])


min_error_c1 = 0.714310  #minimum value from error path
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
0.69918699187


# Confusion matrix
confusion_matrix1 = confusion_matrix(truevals, preds1, labels=[1, 0])


# total pos: 254
sum(truevals)

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
train_data = ldmi.train12
train_dataX = ldmi.train12.ix[:,0:225]
train_dataY = ldmi.train12.ix[:,225]
test_data = ldmi.test12
test_dataX = ldmi.test12.ix[:,0:225]
test_dataY = ldmi.test12.ix[:,225]


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

    trainy, testy = train_cv.ix[:,225], test_cv.ix[:,225]
    trainX, testX = train_cv.ix[:,0:225], test_cv.ix[:,0:225]
    
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
plt.axis([-0.005, 3, 0.25, 0.35])


min_error_c2 = 1.312722  # minimum value from error path
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
0.70460704607


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
train_data = ldmi.train13
train_dataX = ldmi.train13.ix[:,0:225]
train_dataY = ldmi.train13.ix[:,225]
test_data = ldmi.test13
test_dataX = ldmi.test13.ix[:,0:225]
test_dataY = ldmi.test13.ix[:,225]


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

    trainy, testy = train_cv.ix[:,225], test_cv.ix[:,225]
    trainX, testX = train_cv.ix[:,0:225], test_cv.ix[:,0:225]
    
    # Fit the model for each candidate penalty parameter, C.
    # then compute the error rate in the test data and add
    # it to the dictionary entry for that candidate.
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
plt.axis([-0.005, 3, 0.25, 0.35])


min_error_c3 = 0.714310  # lowest value from error path
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
0.693766937669


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
train_data = ldmi.train14
train_dataX = ldmi.train14.ix[:,0:225]
train_dataY = ldmi.train14.ix[:,225]
test_data = ldmi.test14
test_dataX = ldmi.test14.ix[:,0:225]
test_dataY = ldmi.test14.ix[:,225]


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

    trainy, testy = train_cv.ix[:,225], test_cv.ix[:,225]
    trainX, testX = train_cv.ix[:,0:225], test_cv.ix[:,0:225]
    
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
plt.axis([-0.005, 3, 0.25, 0.35])
# suggests no regularisation is best, but then end up overfitting since we have so many variables!



min_error_c4 = 1.312722  # minimum from error path
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
train_data = ldmi.train15
train_dataX = ldmi.train15.ix[:,0:225]
train_dataY = ldmi.train15.ix[:,225]
test_data = ldmi.test15
test_dataX = ldmi.test15.ix[:,0:225]
test_dataY = ldmi.test15.ix[:,225]


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

    trainy, testy = train_cv.ix[:,225], test_cv.ix[:,225]
    trainX, testX = train_cv.ix[:,0:225], test_cv.ix[:,0:225]

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
plt.axis([-0.005, 3, 0.25, 0.35])
# suggests no regularisation is best, but then end up overfitting since we have so many variables!



min_error_c5 = 0.714310  # lowest value from error path
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
0.688346883469


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

# Better way to calculate average probability
preds_proba_averageL1 = (preds_proba1+preds_proba2+preds_proba3+preds_proba4+preds_proba5)/5
preds_proba_averageL1_bin =  (preds_proba_averageL1 > 0.5).astype(int)
truevals = np.array(test_dataY)

accuracy_proba_average_regressionL1 = ((preds_proba_averageL1_bin== truevals).sum())/(len(test_dataX)*1.0)
print accuracy_proba_average_regressionL1

fpr_av_L1, tpr_av_L1, _ = roc_curve(truevals, preds_proba_averageL1)
 
# Calculate the AUC
roc_auc_av_L1 = auc(fpr_av_L1, tpr_av_L1)
print 'ROC AUC: %0.3f' % roc_auc_av_L1



################################################################################################################################

################################################################################################################################

################################################################################################################################

################################################################################################################################

################################################################################################################################

################################################################################################################################

################################################################################################################################


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



logit_mod_no_conf = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train.ix[:,0:6]],axis=1), missing = 'drop')
result_mod_no_conf = logit_mod_no_conf.fit()
result_mod_no_conf.summary()

# Drop PC3
logit_mod_no_conf2 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train.ix[:,0:2],PC_data_train.ix[:,3:6]],axis=1), missing = 'drop')
result_mod_no_conf2 = logit_mod_no_conf2.fit()
result_mod_no_conf2.summary()

# Drop PC5
logit_mod_no_conf3 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train.ix[:,0:2],
                                                                PC_data_train.ix[:,3:4],PC_data_train.ix[:,5:6]],axis=1), missing = 'drop')
result_mod_no_conf3 = logit_mod_no_conf3.fit()
result_mod_no_conf3.summary()

preds1c = result_mod_no_conf3.predict(pd.concat([PC_data_test['intercept'],PC_data_test.ix[:,0:2],
                                                                PC_data_test.ix[:,3:4],PC_data_test.ix[:,5:6]],axis=1))


preds_bin1c = (preds1c > 0.5).astype(int)
truevals = np.array(PC_data_test.ix[:,11])
accuracy1c = ((preds_bin1c== truevals).sum())/(len(PC_data_test)*1.0)
print accuracy1c
0.691056910569




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



logit_mod_no_conf = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train.ix[:,0:6]],axis=1), missing = 'drop')
result_mod_no_conf = logit_mod_no_conf.fit()
result_mod_no_conf.summary()

# Drop PC3
logit_mod_no_conf2 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train.ix[:,0:2],PC_data_train.ix[:,3:6]],axis=1), missing = 'drop')
result_mod_no_conf2 = logit_mod_no_conf2.fit()
result_mod_no_conf2.summary()

# Drop PC4
logit_mod_no_conf3 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train.ix[:,0:2],
                                                                PC_data_train.ix[:,4:6]],axis=1), missing = 'drop')
result_mod_no_conf3 = logit_mod_no_conf3.fit()
result_mod_no_conf3.summary()

preds2c = result_mod_no_conf3.predict(pd.concat([PC_data_test['intercept'],PC_data_test.ix[:,0:2],
                                                                PC_data_test.ix[:,4:6]],axis=1))

preds_bin2c = (preds2c > 0.5).astype(int)
truevals = np.array(PC_data_test.ix[:,11])
accuracy2c = ((preds_bin2c== truevals).sum())/(len(PC_data_test)*1.0)
print accuracy2c
0.685636856369




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



logit_mod_no_conf = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train.ix[:,0:6]],axis=1), missing = 'drop')
result_mod_no_conf = logit_mod_no_conf.fit()
result_mod_no_conf.summary()

# Drop PC3
logit_mod_no_conf2 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train.ix[:,0:2],PC_data_train.ix[:,3:6]],axis=1), missing = 'drop')
result_mod_no_conf2 = logit_mod_no_conf2.fit()
result_mod_no_conf2.summary()

# Drop PC5
logit_mod_no_conf3 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train.ix[:,0:2],
                                                                PC_data_train.ix[:,3:4],PC_data_train.ix[:,5:6]],axis=1), missing = 'drop')
result_mod_no_conf3 = logit_mod_no_conf3.fit()
result_mod_no_conf3.summary()

preds3c = result_mod_no_conf3.predict(pd.concat([PC_data_test['intercept'],PC_data_test.ix[:,0:2],
                                                                PC_data_test.ix[:,3:4],PC_data_test.ix[:,5:6]],axis=1))


preds_bin3c = (preds3c > 0.5).astype(int)
truevals = np.array(PC_data_test.ix[:,11])
accuracy3c = ((preds_bin3c== truevals).sum())/(len(PC_data_test)*1.0)
print accuracy3c
0.685636856369




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



logit_mod_no_conf = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train.ix[:,0:6]],axis=1), missing = 'drop')
result_mod_no_conf = logit_mod_no_conf.fit()
result_mod_no_conf.summary()

# Drop PC3
logit_mod_no_conf2 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train.ix[:,0:2],PC_data_train.ix[:,3:6]],axis=1), missing = 'drop')
result_mod_no_conf2 = logit_mod_no_conf2.fit()
result_mod_no_conf2.summary()

# Drop PC4
logit_mod_no_conf3 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train.ix[:,0:2],
                                                                PC_data_train.ix[:,4:6]],axis=1), missing = 'drop')
result_mod_no_conf3 = logit_mod_no_conf3.fit()
result_mod_no_conf3.summary()

preds4c = result_mod_no_conf3.predict(pd.concat([PC_data_test['intercept'],PC_data_test.ix[:,0:2],
                                                                PC_data_test.ix[:,4:6]],axis=1))



preds_bin4c = (preds4c > 0.5).astype(int)
truevals = np.array(PC_data_test.ix[:,11])
accuracy4c = ((preds_bin4c== truevals).sum())/(len(PC_data_test)*1.0)
print accuracy4c
0.685636856369


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


##### Analysis without confounders ######

logit_mod_no_conf = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train.ix[:,0:6]],axis=1), missing = 'drop')
result_mod_no_conf = logit_mod_no_conf.fit()
result_mod_no_conf.summary()

# Drop PC3
logit_mod_no_conf2 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train.ix[:,0:2],PC_data_train.ix[:,3:6]],axis=1), missing = 'drop')
result_mod_no_conf2 = logit_mod_no_conf2.fit()
result_mod_no_conf2.summary()

# Drop PC5
logit_mod_no_conf3 = sm.Logit(PC_data_train['CAD50'], pd.concat([PC_data_train['intercept'],PC_data_train.ix[:,0:2],
                                                                PC_data_train.ix[:,3:4],PC_data_train.ix[:,5:6]],axis=1), missing = 'drop')
result_mod_no_conf3 = logit_mod_no_conf3.fit()
result_mod_no_conf3.summary()

preds5c = result_mod_no_conf3.predict(pd.concat([PC_data_test['intercept'],PC_data_test.ix[:,0:2],
                                                                PC_data_test.ix[:,3:4],PC_data_test.ix[:,5:6]],axis=1))

#preds = result_mod_no_conf.predict(pd.concat([PC_data_test['intercept'],PC_data_test.ix[:,0:6]],axis=1))

preds_bin5c = (preds5c > 0.5).astype(int)
truevals = np.array(PC_data_test.ix[:,11])
accuracy5c = ((preds_bin5c== truevals).sum())/(len(PC_data_test)*1.0)
print accuracy5c
0.691056910569


###################################################################################################################
##################### aveage predictions - without confounders #####################
###################################################################################################################



preds_proba_averagec = (preds1c + preds2c + preds3c + preds4c + preds5c)/5
preds_proba_average_binc = (preds_proba_averagec >0.5 ).astype(int)

confusion_matrix(truevals, preds_proba_average_binc, labels=[1, 0])
PCA_av_accc = (preds_proba_average_binc == truevals).sum()/(len(ldmi.test14)*1.0)
print PCA_av_accc 



fprPCAc, tprPCAc, _ = roc_curve(truevals, preds_proba_averagec)
# Calculate the AUC
roc_aucPCAc = auc(fprPCAc, tprPCAc)
print 'ROC AUC: %0.3f' % roc_aucPCAc
 
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fprPCAc, tprPCAc, label='ROC curve (area = %0.3f)' % roc_aucPCAc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', size = 12)
plt.ylabel('True Positive Rate', size = 12)
plt.title('ROC Curve for PCA regression averaged across the imputation datasets')
plt.legend(loc="lower right")
plt.show()










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
truevals = np.array(test_data.ix[:,225]) # these the same for all imp sets
sum(truevals)
sum(truevals-1)

# MI set 1
rf1 = RandomForestClassifier(n_estimators=5000, max_features = 0.2)#, max_depth = 15)
forest_fit1 = rf1.fit(ldmi.train11.ix[:,0:225],ldmi.train11.ix[:,225])

importances1 = forest_fit1.feature_importances_ #but doesn't tell you which feature is which!
forest_preds1= forest_fit1.predict(test_data.ix[:,0:225])
forest_acc1 = (forest_preds1 == truevals).sum()/(len(test_data)*1.0)
print forest_acc1

preds_proba1 = forest_fit1.predict_proba(test_data.ix[:,0:225])[:,1]



importances1_list = pd.TimeSeries(importances1)
metabolites = pd.TimeSeries(test_data.columns.values[0:225])
importances1_mat = pd.concat([metabolites, importances1_list], axis=1)
importances1_mat.sort(columns=1, ascending=False) # could plot?



# MI set 2
train_data = ldmi.train12
test_data = ldmi.test12
rf2 = RandomForestClassifier(n_estimators=5000, max_features = 0.2) #, max_depth = 15)
forest_fit2 = rf2.fit(ldmi.train12.ix[:,0:225],ldmi.train12.ix[:,225])

importances2 = forest_fit2.feature_importances_ #but doesn't tell you which feature is which!
forest_preds2 = forest_fit2.predict(test_data.ix[:,0:225])
forest_acc2 = (forest_preds2 == truevals).sum()/(len(test_data)*1.0)
print forest_acc2 

preds_proba2 = forest_fit2.predict_proba(test_data.ix[:,0:225])[:,1]

importances2_list = pd.TimeSeries(importances2)
metabolites = pd.TimeSeries(test_data.columns.values[0:225])
importances2_mat = pd.concat([metabolites, importances2_list], axis=1)
importances2_mat.sort(columns=1, ascending=False)









# MI set 3
train_data = ldmi.train13
test_data = ldmi.test13
rf3 = RandomForestClassifier(n_estimators=5000, max_features = 0.2)
forest_fit3 = rf3.fit(ldmi.train13.ix[:,0:225],ldmi.train13.ix[:,225])

importances3 = forest_fit3.feature_importances_ 
forest_preds3 = forest_fit3.predict(test_data.ix[:,0:225])
forest_acc3 = (forest_preds3 == truevals).sum()/(len(test_data)*1.0)
print forest_acc3 

preds_proba3 = forest_fit3.predict_proba(test_data.ix[:,0:225])[:,1]

importances3_list = pd.TimeSeries(importances3)
metabolites = pd.TimeSeries(test_data.columns.values[0:225])
importances3_mat = pd.concat([metabolites, importances3_list], axis=1)
importances3_mat.sort(columns=1, ascending=False)




# MI set 4
train_data = ldmi.train14
test_data = ldmi.test14
rf4 = RandomForestClassifier(n_estimators=5000, max_features = 0.2)
forest_fit4 = rf4.fit(ldmi.train14.ix[:,0:225],ldmi.train14.ix[:,225])

importances4 = forest_fit4.feature_importances_ 
forest_preds4 = forest_fit4.predict(test_data.ix[:,0:225])
forest_acc4 = (forest_preds4 == truevals).sum()/(len(test_data)*1.0)
print forest_acc4 

preds_proba4 = forest_fit4.predict_proba(test_data.ix[:,0:225])[:,1]

importances4_list = pd.TimeSeries(importances4)
metabolites = pd.TimeSeries(test_data.columns.values[0:225])
importances4_mat = pd.concat([metabolites, importances4_list], axis=1)
importances4_mat.sort(columns=1, ascending=False)





# MI set 5
train_data = ldmi.train15
test_data = ldmi.test15
rf5 = RandomForestClassifier(n_estimators=5000, max_features = 0.2)
forest_fit5 = rf5.fit(ldmi.train15.ix[:,0:225],ldmi.train15.ix[:,225])

importances5 = forest_fit5.feature_importances_ 
forest_preds5 = forest_fit5.predict(ldmi.test15.ix[:,0:225])
forest_acc5 = (forest_preds5 == truevals).sum()/(len(ldmi.test15)*1.0)
print forest_acc5 

preds_proba5 = forest_fit5.predict_proba(ldmi.test1.ix[:,0:225])[:,1]

importances5_list = pd.TimeSeries(importances5)
metabolites = pd.TimeSeries(test_data.columns.values[0:225])
importances5_mat = pd.concat([metabolites, importances5_list], axis=1)
importances5_mat.sort(columns=1, ascending=False)




importances_average = pd.TimeSeries((importances1 + importances2 + importances5 + importances5 + importances5)/5)
importances_average_list = pd.TimeSeries(importances_average)
importances_average_mat = pd.concat([metabolites, importances_average], axis=1)
importances_average_mat_sorted = importances_average_mat.sort(columns=1, ascending=False)
importances_average_mat_sorted 
importances_average_mat_sorted.to_csv('importances_random_forest_ordered.csv')



importances_average = pd.TimeSeries((importances1 + importances2 + importances5 + importances5 + importances5)/5)
importances_average_list = pd.TimeSeries(importances_average)
importances_average_mat = pd.concat([metabolites, importances_average], axis=1)
importances_average_mat_sorted = importances_average_mat.sort(columns=1, ascending=False)
importances_average_mat_sorted 
importances_average_mat_sorted.to_csv('importances_random_forest_ordered.csv')





# Plot average importances
importances_re_ordered = importances_average_mat_sorted.reset_index(drop=True).copy(deep=True)
importances_re_ordered_no_top = importances_re_ordered.ix[1:40,:].reset_index(drop=True).copy(deep=True)


plt.figure(figsize=(12,8))
plt.title("Feature importances excluding confounders", size=13)
plt.bar(range(importances_re_ordered.ix[0:40,:].shape[0]), importances_re_ordered.ix[0:40,:][1],
       color="gray", align="center", width =0.75)
plt.xticks(range(importances_re_ordered.ix[0:40,:].shape[0]), importances_re_ordered.ix[0:40,:][0],rotation=90, fontsize=12)
plt.yticks(fontsize = 12)
plt.xlim([-1, importances_re_ordered.ix[0:40,:].shape[0]])
plt.show()


plt.figure(figsize=(12,8))
plt.title("Feature importances excluding confounders, truncated to exclude creatine", size=13)
plt.bar(range(importances_re_ordered_no_top.ix[0:39,:].shape[0]), importances_re_ordered_no_top.ix[0:39,:][1],
       color="gray", align="center", width =0.75)
plt.xticks(range(importances_re_ordered_no_top.ix[0:39,:].shape[0]), importances_re_ordered_no_top.ix[0:39,:][0],rotation=90, fontsize=12)
plt.yticks(fontsize = 12)
plt.xlim([-1, importances_re_ordered_no_top.ix[0:39,:].shape[0]])
plt.show()



preds_proba_average_rf = (preds_proba1 + preds_proba2 + preds_proba3 + preds_proba4 + preds_proba5)/5
preds_proba_average_bin_rf = (preds_proba_average_rf >0.5 ).astype(int)

confusion_matrix(truevals, preds_proba_average_bin_rf, labels=[1, 0])
forest_av_acc_rf = (preds_proba_average_bin_rf == truevals).sum()/(len(ldmi.test14)*1.0)
print forest_av_acc_rf 




fpr_av_rf, tpr_av_rf, _ = roc_curve(truevals, preds_proba_average_rf)
# Calculate the AUC
roc_auc_av_rf = auc(fpr_av_rf, tpr_av_rf)
print 'ROC AUC: %0.3f' % roc_auc_av_rf

plt.figure(figsize=(8, 6))
plt.plot(fprPCAc, tprPCAc, label='ROC curve of PCA regression (area = %0.3f)' % roc_aucPCAc, color = 'firebrick')
plt.plot(fpr_av_L1, tpr_av_L1, label='ROC curve of L1 regression (area = %0.3f)' % roc_auc_av_L1, color = 'green')
plt.plot(fpr_av_rf, tpr_av_rf, label='ROC curve of random forest (area = %0.3f)' % roc_auc_av_rf, color = 'mediumblue')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity', size = 12)
plt.ylabel('Sensitivity', size = 12)
plt.title('ROC Curve for predictions using PCA regression, \n penalised regression and random forest, not including confounding variables')
plt.legend(loc="lower right")
plt.show()
