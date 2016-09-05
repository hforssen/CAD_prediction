# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Random forest with smaller cross validation
@author: ucabhmf
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

import load_data_ext as ld
import load_data_mi_ext_new as ldmi


np.random.seed(10)

##########################################################################
############ Complete case analysis ############
##########################################################################

# Create random forest classifier, gini index
rf = RandomForestClassifier(n_estimators=1000)

forest_fit = rf.fit(ld.train1.ix[:,0:229],ld.train1.ix[:,229])

#work out how to match importances to features
importances = np.sort(forest_fit.feature_importances_) #but doesn't tell you which feature is which!


forest_preds_proba = forest_fit.predict_proba(ld.test1.ix[:,0:229])[:,1]
forest_preds = forest_fit.predict(ld.test1.ix[:,0:229])

truevals = np.array(ld.test1.ix[:,229])

forest_acc = (forest_preds == truevals).sum()/(len(ld.test1)*1.0)# why is it different each time I run it? So few trees!
print forest_acc #sometimes as high as 0.718592964824, sometimes 0.65


# Create random forest classifier, cross-entropy
sum(ld.train1.isnull().any(axis=0))

sum(truevals ==1)/(len(ld.test1)*1.0)

# Calculate confusion matrix
confusion_matrix(truevals, forest_preds, labels=[1, 0])
#array([[237,  17],
#       [ 89,  26]])
#varies depending on forest generated

roc_auc_score(truevals, forest_preds)
fpr, tpr, thresholds = roc_curve(truevals, forest_preds)
auc(fpr, tpr)

# total pos: 117
sum(truevals)

# total neg: 57 => 0.6724 positive examples in test set
sum(truevals-1)

# Generally much worse performance than with imputation, nearly twice as much data used!

ld.test1.ix[:,0:229].mean(axis=1)

fpr, tpr, _ = roc_curve(truevals, forest_preds_proba)
 
# Calculate the AUC
roc_auc = auc(fpr, tpr)



       
#########################################################################################    
# Cross validation on number of features
#########################################################################################

pred_acc = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

auc_vals = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

f1_vals = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}
     
pre_vals = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}
          
rec_vals = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}
     

pred_acc_average = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

auc_vals_average = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}
     
f1_vals_average = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}    

pre_vals_average = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

rec_vals_average ={0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}


max_feature_range = np.arange(0.05, 0.625, 0.025)



# Split training set into validation folds
cv_index = [ma.ceil((len(ld.train1)/5)*x) for x in range(6)]

for i in range(5):
    # Split the data
    
    test_cv = ld.train1.ix[cv_index[i]:cv_index[i+1],:]
    train_cv = ld.train1.drop(ld.train1.index[cv_index[i]:cv_index[i+1]])

    trainy, testy = train_cv.ix[:,229], test_cv.ix[:,229]
    trainX, testX = train_cv.ix[:,0:229], test_cv.ix[:,0:229]
    
    # Train forest for all CV values and predict on cv_test, save results
    for feat_range in pred_acc:
        print i, feat_range
        rf = RandomForestClassifier(n_estimators=5000, max_features = feat_range, oob_score = 'false')
        forest_fit = rf.fit(trainX.values,trainy.values)            
        forest_preds = forest_fit.predict(testX)
        forest_preds_proba = forest_fit.predict_proba(testX)[:,1]
        accuracy = np.mean(forest_preds != testy)
        fpr, tpr, _ = roc_curve(testy, forest_preds_proba)
        auc_pt = auc(fpr, tpr)  
        pre, rec, f1, _ = precision_recall_fscore_support(testy, forest_preds, average = 'binary')
        
        pred_acc[feat_range].append(accuracy)
        auc_vals[feat_range].append(auc_pt)
        f1_vals[feat_range].append(f1)
        pre_vals[feat_range].append(pre)
        rec_vals[feat_range].append(rec)
            
  

for feat_range in pred_acc:
    pred_acc_average[feat_range].append(sum(pred_acc[feat_range])/5)
    pred_acc_average[feat_range] = pred_acc_average[feat_range][0]

    auc_vals_average[feat_range].append(sum(auc_vals[feat_range])/5)
    auc_vals_average[feat_range] = auc_vals_average[feat_range][0]
    
    f1_vals_average[feat_range].append(sum(f1_vals[feat_range])/5)
    f1_vals_average[feat_range] = f1_vals_average[feat_range][0]
    
    pre_vals_average[feat_range].append(sum(pre_vals[feat_range])/5)
    pre_vals_average[feat_range] = pre_vals_average[feat_range][0]
    
    rec_vals_average[feat_range].append(sum(rec_vals[feat_range])/5) 
    rec_vals_average[feat_range] = rec_vals_average[feat_range][0]




pred_acc_average_ts = pd.Series(pred_acc_average)
auc_vals_average_ts = pd.Series(auc_vals_average)
f1_vals_average_ts = pd.Series(f1_vals_average)
pre_vals_average_ts = pd.Series(pre_vals_average)
rec_vals_average_ts = pd.Series(rec_vals_average)

    
plt.plot(pred_acc_average_ts)
plt.plot(auc_vals_average_ts)
plt.plot(f1_vals_average_ts)
plt.plot(pre_vals_average_ts)
plt.plot(rec_vals_average_ts)



####### BEST MODEL and diagnostics based on plots showing best parameters ####### extend with best parameters

rf = RandomForestClassifier(n_estimators=1000)
forest_fit = rf.fit(ld.train1.ix[:,0:229],ld.train1.ix[:,229])

# http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
importances = np.sort(forest_fit.feature_importances_) 


# predictions on test set
forest_preds_proba = forest_fit.predict_proba(ld.test1.ix[:,0:229])[:,1]
forest_preds = forest_fit.predict(ld.test1.ix[:,0:229])

truevals = np.array(ld.test1.ix[:,229])

forest_acc = (forest_preds == truevals).sum()/(len(ld.test1)*1.0)# why is it different each time I run it? So few trees!
print forest_acc #sometimes as high as 0.718592964824, sometimes 0.65


sum(ld.train1.isnull().any(axis=0))

# percentage of 1s in dataset
sum(truevals ==1)/(len(ld.test1)*1.0) 

# Calculate confusion matrix
confusion_matrix(truevals, forest_preds, labels=[1, 0])


# Calculate roc score and curve
roc_auc_score(truevals, forest_preds_proba)
fpr, tpr, thresholds = roc_curve(truevals, forest_preds_proba)
auc(fpr, tpr)

# total pos: 117
sum(truevals)

# total neg: 57 => 0.6724 positive examples in test set
sum(truevals-1)

# Generally much worse performance than with imputation, nearly twice as much data used!

ld.test1.ix[:,0:229].mean(axis=1)



# Calculate ROC curve
fpr, tpr, _ = roc_curve(truevals, forest_preds_proba)
 
# Calculate the AUC
roc_auc = auc(fpr, tpr)









# plot roc curve using cross-validation results




# plot # trees vs missclassification error for best parameter settings
trees_acc_mat = [0 for x in range(420)] 
trees_oob_mat = [0 for x in range(420)]
trees_auc_mat = [0 for x in range(420)]


n_features_range = np.arange(20, 8000, 20)
k=0

for i in n_features_range:
    print i, k
    rf = RandomForestClassifier(n_estimators=i, max_features = 50, max_depth = 6, oob_score = 'true') # to edit based on cross val results
    forest_fit = rf.fit(ld.train1.ix[:,0:229],ld.train1.ix[:,229])
    forest_preds = forest_fit.predict(ld.test1.ix[:,0:229])
    forest_preds_proba = forest_fit.predict_proba(ld.test1.ix[:,0:229])[:,1]

    trees_oob_mat[k] = rf.oob_score_
    trees_acc_mat[k] = (forest_preds == truevals).sum()/(len(ld.test1)*1.0)
    
    fpr, tpr, _ = roc_curve(truevals, forest_preds_proba)
    trees_auc_mat[k] = auc(fpr, tpr)        

    k=k+1


plt.figure(figsize=(10, 8))
#my_xticks = ['5%', '10%', '15%', '20%','25%', '30%', '35%', '40%','45%', '50%', '55%', '60%', '65%']
#plt.xticks(range(0,len(max_feature_range)), my_xticks)
#plt.yticks(np.arange(0.60, 0.72, 0.01))
plt.plot(range(0,len(n_features_range))[0:300], trees_acc_mat[0:300], color = 'green')
plt.plot(range(0,len(n_features_range))[0:300], trees_oob_mat[0:300], color = 'blue')
#plt.xlim([0.0, 12])
#plt.ylim([0.6, 0.72])
plt.grid(b=True, which='major', color='lightgrey', linestyle='-')
plt.grid(b=True, which='minor', color='lightgrey', linestyle='-')
plt.xlabel('Number of trees in forest ensemble')
plt.ylabel('Accuracy')
plt.title('Prediction accuracy and Out-of-Bag score over varying number of trees in the forest ensemble')
plt.legend(loc="lower right")
plt.show()




# http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#example-model-selection-plot-roc-py


# http://www.ultravioletanalytics.com/2014/12/16/kaggle-titanic-competition-part-x-roc-curves-and-auc/
preds_proba = forest_fit.predict_proba(ld.test1.ix[:,0:229])[:,1]

 
# Determine the false positive and true positive rates
fpr, tpr, _ = roc_curve(truevals, preds_proba)
 
# Calculate the AUC
roc_auc = auc(fpr, tpr)
print 'ROC AUC: %0.2f' % roc_auc
 
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()



##########################################################################
############ Imputed data analysis ############
##########################################################################

############################ 1 ############################

pred_acc1 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}
     
auc_vals1 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}
     
f1_vals1 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}
     
pre_vals1 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

rec_vals1 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

pred_acc_average1 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

auc_vals_average1 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

f1_vals_average1 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

pre_vals_average1 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

rec_vals_average1 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}     


train_data = ldmi.train11
train_dataX = ldmi.train11.ix[:,0:229]
train_dataY = ldmi.train11.ix[:,229]
test_data = ldmi.test11
test_dataX = ldmi.test11.ix[:,0:229]
test_dataY = ldmi.test11.ix[:,229]
               

max_feature_range = np.arange(0.05, 0.625, 0.025)


# Split training set into validation folds
cv_index = [ma.ceil((len(ldmi.train11)/5)*x) for x in range(6)]

for i in range(5):
    # Split the data
    
    test_cv = train_data.ix[cv_index[i]:cv_index[i+1],:]
    train_cv = train_data.drop(ldmi.train11.index[cv_index[i]:cv_index[i+1]])

    trainy, testy = train_cv.ix[:,229], test_cv.ix[:,229]
    trainX, testX = train_cv.ix[:,0:229], test_cv.ix[:,0:229]
    
    # Train forest for all CV values and predict on cv_test, save results
    for feat_range in pred_acc1:
        print i, feat_range
        print 1
        rf = RandomForestClassifier(n_estimators=5000, max_features = feat_range, oob_score = 'false')
        forest_fit = rf.fit(trainX.values,trainy.values)            
        forest_preds = forest_fit.predict(testX)
        forest_preds_proba = forest_fit.predict_proba(testX)[:,1]
        accuracy = np.mean(forest_preds != testy)
        fpr, tpr, _ = roc_curve(testy, forest_preds_proba)
        auc_pt = auc(fpr, tpr)  
        pre, rec, f1, _ = precision_recall_fscore_support(testy, forest_preds, average = 'binary')
        
        pred_acc1[feat_range].append(accuracy)
        auc_vals1[feat_range].append(auc_pt)
        f1_vals1[feat_range].append(f1)
        pre_vals1[feat_range].append(pre)
        rec_vals1[feat_range].append(rec)
        


for feat_range in pred_acc1:
    pred_acc_average1[feat_range].append(sum(pred_acc1[feat_range])/5)
    #pred_acc_average1[feat_range] = pred_acc_average1[feat_range][0]

    auc_vals_average1[feat_range].append(sum(auc_vals1[feat_range])/5)
    #auc_vals_average1[feat_range] = auc_vals_average1[feat_range][0]
    
    f1_vals_average1[feat_range].append(sum(f1_vals1[feat_range])/5)
    #f1_vals_average1[feat_range] = f1_vals_average1[feat_range][0]
    
    pre_vals_average1[feat_range].append(sum(pre_vals1[feat_range])/5)
    #pre_vals_average1[feat_range] = pre_vals_average1[feat_range][0]
    
    rec_vals_average1[feat_range].append(sum(rec_vals1[feat_range])/5) 
    #rec_vals_average1[feat_range] = rec_vals_average1[feat_range][0]


pred_acc_average_df1 = pd.DataFrame.from_dict(pred_acc_average1, orient='index')
auc_vals_average_df1 = pd.DataFrame.from_dict(auc_vals_average1, orient='index')
f1_vals_average_df1 = pd.DataFrame.from_dict(f1_vals_average1, orient='index')
pre_vals_average_df1 = pd.DataFrame.from_dict(pre_vals_average1, orient='index')
rec_vals_average_df1 = pd.DataFrame.from_dict(rec_vals_average1, orient='index')



############################ 2 ############################

pred_acc2 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}
     
auc_vals2 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}
     
f1_vals2 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}
     
pre_vals2 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

rec_vals2 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

pred_acc_average2 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

auc_vals_average2 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

f1_vals_average2 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

pre_vals_average2 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

rec_vals_average2 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []} 

train_data = ldmi.train12
train_dataX = ldmi.train12.ix[:,0:229]
train_dataY = ldmi.train12.ix[:,229]
test_data = ldmi.test12
test_dataX = ldmi.test12.ix[:,0:229]
test_dataY = ldmi.test12.ix[:,229]


max_feature_range = np.arange(0.05, 0.625, 0.025)


# Split training set into validation folds
cv_index = [ma.ceil((len(ldmi.train12)/5)*x) for x in range(6)]

for i in range(5):
    # Split the data
    
    test_cv = train_data.ix[cv_index[i]:cv_index[i+1],:]
    train_cv = train_data.drop(ldmi.train12.index[cv_index[i]:cv_index[i+1]])

    trainy, testy = train_cv.ix[:,229], test_cv.ix[:,229]
    trainX, testX = train_cv.ix[:,0:229], test_cv.ix[:,0:229]
    
    # Train forest for all CV values and predict on cv_test, save results
    for feat_range in pred_acc2:
        print i, feat_range
        print 2
        rf = RandomForestClassifier(n_estimators=5000, max_features = feat_range, oob_score = 'false')
        forest_fit = rf.fit(trainX.values,trainy.values)            
        forest_preds = forest_fit.predict(testX)
        forest_preds_proba = forest_fit.predict_proba(testX)[:,1]
        accuracy = np.mean(forest_preds != testy)
        fpr, tpr, _ = roc_curve(testy, forest_preds_proba)
        auc_pt = auc(fpr, tpr)  
        pre, rec, f1, _ = precision_recall_fscore_support(testy, forest_preds, average = 'binary')
        
        pred_acc2[feat_range].append(accuracy)
        auc_vals2[feat_range].append(auc_pt)
        f1_vals2[feat_range].append(f1)
        pre_vals2[feat_range].append(pre)
        rec_vals2[feat_range].append(rec)
        


for feat_range in pred_acc2:
    pred_acc_average2[feat_range].append(sum(pred_acc2[feat_range])/5)
    #pred_acc_average1[feat_range] = pred_acc_average1[feat_range][0]

    auc_vals_average2[feat_range].append(sum(auc_vals2[feat_range])/5)
    #auc_vals_average1[feat_range] = auc_vals_average1[feat_range][0]
    
    f1_vals_average2[feat_range].append(sum(f1_vals2[feat_range])/5)
    #f1_vals_average1[feat_range] = f1_vals_average1[feat_range][0]
    
    pre_vals_average2[feat_range].append(sum(pre_vals2[feat_range])/5)
    #pre_vals_average1[feat_range] = pre_vals_average1[feat_range][0]
    
    rec_vals_average2[feat_range].append(sum(rec_vals2[feat_range])/5) 
    #rec_vals_average1[feat_range] = rec_vals_average1[feat_range][0]


pred_acc_average_df2 = pd.DataFrame.from_dict(pred_acc_average2, orient='index')
auc_vals_average_df2 = pd.DataFrame.from_dict(auc_vals_average2, orient='index')
f1_vals_average_df2 = pd.DataFrame.from_dict(f1_vals_average2, orient='index')
pre_vals_average_df2 = pd.DataFrame.from_dict(pre_vals_average2, orient='index')
rec_vals_average_df2 = pd.DataFrame.from_dict(rec_vals_average2, orient='index')


############################ 3 ############################

pred_acc3 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}
     
auc_vals3 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}
     
f1_vals3 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}
     
pre_vals3 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

rec_vals3 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

pred_acc_average3 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

auc_vals_average3 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

f1_vals_average3 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

pre_vals_average3 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

rec_vals_average3 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []} 

train_data = ldmi.train13
train_dataX = ldmi.train13.ix[:,0:229]
train_dataY = ldmi.train13.ix[:,229]
test_data = ldmi.test13
test_dataX = ldmi.test13.ix[:,0:229]
test_dataY = ldmi.test13.ix[:,229]


max_feature_range = np.arange(0.05, 0.625, 0.025)


# Split training set into validation folds
cv_index = [ma.ceil((len(ldmi.train13)/5)*x) for x in range(6)]

for i in range(5):
    # Split the data
    
    test_cv = train_data.ix[cv_index[i]:cv_index[i+1],:]
    train_cv = train_data.drop(ldmi.train13.index[cv_index[i]:cv_index[i+1]])

    trainy, testy = train_cv.ix[:,229], test_cv.ix[:,229]
    trainX, testX = train_cv.ix[:,0:229], test_cv.ix[:,0:229]
    
    # Train forest for all CV values and predict on cv_test, save results
    for feat_range in pred_acc3:
        print i, feat_range
        print 3
        rf = RandomForestClassifier(n_estimators=5000, max_features = feat_range, oob_score = 'false')
        forest_fit = rf.fit(trainX.values,trainy.values)            
        forest_preds = forest_fit.predict(testX)
        forest_preds_proba = forest_fit.predict_proba(testX)[:,1]
        accuracy = np.mean(forest_preds != testy)
        fpr, tpr, _ = roc_curve(testy, forest_preds_proba)
        auc_pt = auc(fpr, tpr)  
        pre, rec, f1, _ = precision_recall_fscore_support(testy, forest_preds, average = 'binary')
        
        pred_acc3[feat_range].append(accuracy)
        auc_vals3[feat_range].append(auc_pt)
        f1_vals3[feat_range].append(f1)
        pre_vals3[feat_range].append(pre)
        rec_vals3[feat_range].append(rec)
        


for feat_range in pred_acc3:
    pred_acc_average3[feat_range].append(sum(pred_acc3[feat_range])/5)
    #pred_acc_average1[feat_range] = pred_acc_average1[feat_range][0]

    auc_vals_average3[feat_range].append(sum(auc_vals3[feat_range])/5)
    #auc_vals_average1[feat_range] = auc_vals_average1[feat_range][0]
    
    f1_vals_average3[feat_range].append(sum(f1_vals3[feat_range])/5)
    #f1_vals_average1[feat_range] = f1_vals_average1[feat_range][0]
    
    pre_vals_average3[feat_range].append(sum(pre_vals3[feat_range])/5)
    #pre_vals_average1[feat_range] = pre_vals_average1[feat_range][0]
    
    rec_vals_average3[feat_range].append(sum(rec_vals3[feat_range])/5) 
    #rec_vals_average1[feat_range] = rec_vals_average1[feat_range][0]


pred_acc_average_df3 = pd.DataFrame.from_dict(pred_acc_average3, orient='index')
auc_vals_average_df3 = pd.DataFrame.from_dict(auc_vals_average3, orient='index')
f1_vals_average_df3 = pd.DataFrame.from_dict(f1_vals_average3, orient='index')
pre_vals_average_df3 = pd.DataFrame.from_dict(pre_vals_average3, orient='index')
rec_vals_average_df3 = pd.DataFrame.from_dict(rec_vals_average3, orient='index')


############################ 4 ############################

pred_acc4 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}
     
auc_vals4 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}
     
f1_vals4 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}
     
pre_vals4 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

rec_vals4 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

pred_acc_average4 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

auc_vals_average4 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

f1_vals_average4 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

pre_vals_average4 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

rec_vals_average4 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []} 

train_data = ldmi.train14
train_dataX = ldmi.train14.ix[:,0:229]
train_dataY = ldmi.train14.ix[:,229]
test_data = ldmi.test14
test_dataX = ldmi.test14.ix[:,0:229]
test_dataY = ldmi.test14.ix[:,229]


max_feature_range = np.arange(0.05, 0.625, 0.025)


# Split training set into validation folds
cv_index = [ma.ceil((len(ldmi.train14)/5)*x) for x in range(6)]

for i in range(5):
    # Split the data
    
    test_cv = train_data.ix[cv_index[i]:cv_index[i+1],:]
    train_cv = train_data.drop(ldmi.train14.index[cv_index[i]:cv_index[i+1]])

    trainy, testy = train_cv.ix[:,229], test_cv.ix[:,229]
    trainX, testX = train_cv.ix[:,0:229], test_cv.ix[:,0:229]
    
    # Train forest for all CV values and predict on cv_test, save results
    for feat_range in pred_acc4:
        print i, feat_range
        print 1
        rf = RandomForestClassifier(n_estimators=5000, max_features = feat_range, oob_score = 'false')
        forest_fit = rf.fit(trainX.values,trainy.values)            
        forest_preds = forest_fit.predict(testX)
        forest_preds_proba = forest_fit.predict_proba(testX)[:,1]
        accuracy = np.mean(forest_preds != testy)
        fpr, tpr, _ = roc_curve(testy, forest_preds_proba)
        auc_pt = auc(fpr, tpr)  
        pre, rec, f1, _ = precision_recall_fscore_support(testy, forest_preds, average = 'binary')
        
        pred_acc4[feat_range].append(accuracy)
        auc_vals4[feat_range].append(auc_pt)
        f1_vals4[feat_range].append(f1)
        pre_vals4[feat_range].append(pre)
        rec_vals4[feat_range].append(rec)
        


for feat_range in pred_acc4:
    pred_acc_average4[feat_range].append(sum(pred_acc4[feat_range])/5)
    #pred_acc_average1[feat_range] = pred_acc_average1[feat_range][0]

    auc_vals_average4[feat_range].append(sum(auc_vals4[feat_range])/5)
    #auc_vals_average1[feat_range] = auc_vals_average1[feat_range][0]
    
    f1_vals_average4[feat_range].append(sum(f1_vals4[feat_range])/5)
    #f1_vals_average1[feat_range] = f1_vals_average1[feat_range][0]
    
    pre_vals_average4[feat_range].append(sum(pre_vals4[feat_range])/5)
    #pre_vals_average1[feat_range] = pre_vals_average1[feat_range][0]
    
    rec_vals_average4[feat_range].append(sum(rec_vals4[feat_range])/5) 
    #rec_vals_average1[feat_range] = rec_vals_average1[feat_range][0]


pred_acc_average_df4 = pd.DataFrame.from_dict(pred_acc_average4, orient='index')
auc_vals_average_df4 = pd.DataFrame.from_dict(auc_vals_average4, orient='index')
f1_vals_average_df4 = pd.DataFrame.from_dict(f1_vals_average4, orient='index')
pre_vals_average_df4 = pd.DataFrame.from_dict(pre_vals_average4, orient='index')
rec_vals_average_df4 = pd.DataFrame.from_dict(rec_vals_average4, orient='index')


############################ 5 ############################

pred_acc5 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}
     
auc_vals5 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}
     
f1_vals5 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}
     
pre_vals5 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

rec_vals5 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

pred_acc_average5 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

auc_vals_average5 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

f1_vals_average5 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

pre_vals_average5 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []}

rec_vals_average5 = {0.05:[], 0.075: [],
     0.1: [], 0.125: [],
     0.15: [], 0.175: [],
     0.2: [], 0.225: [],
     0.25: [], 0.275: [],
     0.3: [], 0.325: [],
     0.35: [], 0.375: [],
     0.4: [], 0.425: [],
     0.45: [], 0.475: [],
     0.5: [], 0.525: [],
     0.55: [], 0.575: [],
     0.6: []} 

train_data = ldmi.train15
train_dataX = ldmi.train15.ix[:,0:229]
train_dataY = ldmi.train15.ix[:,229]
test_data = ldmi.test15
test_dataX = ldmi.test15.ix[:,0:229]
test_dataY = ldmi.test15.ix[:,229]


max_feature_range = np.arange(0.05, 0.625, 0.025)

# Split training set into validation folds
cv_index = [ma.ceil((len(ldmi.train15)/5)*x) for x in range(6)]

for i in range(5):
    # Split the data
    
    test_cv = train_data.ix[cv_index[i]:cv_index[i+1],:]
    train_cv = train_data.drop(ldmi.train15.index[cv_index[i]:cv_index[i+1]])

    trainy, testy = train_cv.ix[:,229], test_cv.ix[:,229]
    trainX, testX = train_cv.ix[:,0:229], test_cv.ix[:,0:229]
    
    # Train forest for all CV values and predict on cv_test, save results
    for feat_range in pred_acc5:
        print i, feat_range
        print 5
        rf = RandomForestClassifier(n_estimators=5000, max_features = feat_range, oob_score = 'false')
        forest_fit = rf.fit(trainX.values,trainy.values)            
        forest_preds = forest_fit.predict(testX)
        forest_preds_proba = forest_fit.predict_proba(testX)[:,1]
        accuracy = np.mean(forest_preds != testy)
        fpr, tpr, _ = roc_curve(testy, forest_preds_proba)
        auc_pt = auc(fpr, tpr)  
        pre, rec, f1, _ = precision_recall_fscore_support(testy, forest_preds, average = 'binary')
        
        pred_acc5[feat_range].append(accuracy)
        auc_vals5[feat_range].append(auc_pt)
        f1_vals5[feat_range].append(f1)
        pre_vals5[feat_range].append(pre)
        rec_vals5[feat_range].append(rec)
        


for feat_range in pred_acc5:
    pred_acc_average5[feat_range].append(sum(pred_acc5[feat_range])/5)
    #pred_acc_average1[feat_range] = pred_acc_average1[feat_range][0]

    auc_vals_average5[feat_range].append(sum(auc_vals5[feat_range])/5)
    #auc_vals_average1[feat_range] = auc_vals_average1[feat_range][0]
    
    f1_vals_average5[feat_range].append(sum(f1_vals5[feat_range])/5)
    #f1_vals_average1[feat_range] = f1_vals_average1[feat_range][0]
    
    pre_vals_average5[feat_range].append(sum(pre_vals5[feat_range])/5)
    #pre_vals_average1[feat_range] = pre_vals_average1[feat_range][0]
    
    rec_vals_average5[feat_range].append(sum(rec_vals5[feat_range])/5) 
    #rec_vals_average1[feat_range] = rec_vals_average1[feat_range][0]


pred_acc_average_df5 = pd.DataFrame.from_dict(pred_acc_average5, orient='index')
auc_vals_average_df5 = pd.DataFrame.from_dict(auc_vals_average5, orient='index')
f1_vals_average_df5 = pd.DataFrame.from_dict(f1_vals_average5, orient='index')
pre_vals_average_df5 = pd.DataFrame.from_dict(pre_vals_average5, orient='index')
rec_vals_average_df5 = pd.DataFrame.from_dict(rec_vals_average5, orient='index')


##########################################################################################################
################################ Plots and analysis for each imp set ################################

pred_acc_average_average = (pred_acc_average_df1 + pred_acc_average_df2 + pred_acc_average_df3 + pred_acc_average_df4 + pred_acc_average_df5)/5
auc_vals_average_average = (auc_vals_average_df1 + auc_vals_average_df2 + auc_vals_average_df3 + auc_vals_average_df4 + auc_vals_average_df5)/5
f1_vals_average_average = (f1_vals_average_df1 + f1_vals_average_df2 + f1_vals_average_df3 + f1_vals_average_df4 + f1_vals_average_df5)/5
pre_vals_average_average = (pre_vals_average_df1 + pre_vals_average_df2 + pre_vals_average_df3 + pre_vals_average_df4 + pre_vals_average_df5)/5
rec_vals_average_average = (rec_vals_average_df1 + rec_vals_average_df2 + rec_vals_average_df3 + rec_vals_average_df4 + rec_vals_average_df5)/5


pred_acc_average_average_temp = pred_acc_average_average.reset_index()
auc_vals_average_average_temp = auc_vals_average_average.reset_index()
f1_vals_average_average_temp = f1_vals_average_average.reset_index()
pre_vals_average_average_temp = pre_vals_average_average.reset_index()
rec_vals_average_average_temp = rec_vals_average_average.reset_index()

pred_acc_average_average_sort = pred_acc_average_average_temp.sort(columns='index', ascending = True)
auc_vals_average_average_sort = auc_vals_average_average_temp.sort(columns='index', ascending = True)
f1_vals_average_average_sort = f1_vals_average_average_temp.sort(columns='index', ascending = True)
pre_vals_average_average_sort = pre_vals_average_average_temp.sort(columns='index', ascending = True)
rec_vals_average_average_sort = rec_vals_average_average_temp.sort(columns='index', ascending = True)


plt.plot(pred_acc_average_average_sort.ix[:,0],pred_acc_average_average_sort.ix[:,1])
plt.plot(auc_vals_average_average_sort.ix[:,0],auc_vals_average_average_sort.ix[:,1])
plt.plot(f1_vals_average_average_sort.ix[:,0],f1_vals_average_average_sort.ix[:,1])
plt.plot(pre_vals_average_average_sort.ix[:,0],pre_vals_average_average_sort.ix[:,1])
plt.plot(rec_vals_average_average_sort.ix[:,0],rec_vals_average_average_sort.ix[:,1])

plt.figure(figsize=(6,5))
plt.xlim([0.05, 0.6])
plt.ylim([0.25, 0.3])
plt.xlabel('Proportion of features used',size = '14')
plt.ylabel('Average prediction error',size = '14')
plt.plot(pred_acc_average_average_sort.ix[:,0],pred_acc_average_average_sort.ix[:,1])
plt.title('Prediction error',size='14')
plt.show()


plt.figure(figsize=(6,5))
plt.plot(auc_vals_average_average_sort.ix[:,0],auc_vals_average_average_sort.ix[:,1])
plt.xlim([0.05, 0.6])
plt.ylim([0.69, 0.72])
plt.xlabel('Proportion of features used',size = '14')
plt.ylabel('Average AUC',size = '14')
plt.title('AUC',size = '14')



pred_acc_average_average_sort.to_csv('pred_acc_average_average_sort_ext.csv')
auc_vals_average_average_sort.to_csv('auc_vals_average_average_sort_ext.csv')
f1_vals_average_average_sort.to_csv('f1_vals_average_average_sort_ext.csv')
pre_vals_average_average_sort.to_csv('pre_vals_average_average_sort_ext.csv')
rec_vals_average_average_sort.to_csv('rec_vals_average_average_sort_ext.csv')

pred_acc_average_df1.to_csv('pred_acc_average_df1_ext.csv')
pred_acc_average_df2.to_csv('pred_acc_average_df2_ext.csv')
pred_acc_average_df3.to_csv('pred_acc_average_df3_ext.csv')
pred_acc_average_df4.to_csv('pred_acc_average_df4_ext.csv')
pred_acc_average_df5.to_csv('pred_acc_average_df5_ext.csv')

auc_vals_average_df1.to_csv('auc_vals_average_df1_ext.csv')
auc_vals_average_df2.to_csv('auc_vals_average_df2_ext.csv')
auc_vals_average_df3.to_csv('auc_vals_average_df3_ext.csv')
auc_vals_average_df4.to_csv('auc_vals_average_df4_ext.csv')
auc_vals_average_df5.to_csv('auc_vals_average_df5_ext.csv')

f1_vals_average_df1.to_csv('f1_vals_average_df1_ext.csv')
f1_vals_average_df2.to_csv('f1_vals_average_df2_ext.csv')
f1_vals_average_df3.to_csv('f1_vals_average_df3_ext.csv')
f1_vals_average_df4.to_csv('f1_vals_average_df4_ext.csv')
f1_vals_average_df5.to_csv('f1_vals_average_df5_ext.csv')

pre_vals_average_df1.to_csv('pre_vals_average_df1_ext.csv')
pre_vals_average_df2.to_csv('pre_vals_average_df2_ext.csv')
pre_vals_average_df3.to_csv('pre_vals_average_df3_ext.csv')
pre_vals_average_df4.to_csv('pre_vals_average_df4_ext.csv')
pre_vals_average_df5.to_csv('pre_vals_average_df5_ext.csv')

rec_vals_average_df1.to_csv('rec_vals_average_df1_ext.csv')
rec_vals_average_df2.to_csv('rec_vals_average_df2_ext.csv')
rec_vals_average_df3.to_csv('rec_vals_average_df3_ext.csv')
rec_vals_average_df4.to_csv('rec_vals_average_df4_ext.csv')
rec_vals_average_df5.to_csv('rec_vals_average_df5_ext.csv')




################## Average predictions ##################
np.random.seed(10)
# 0.325 seems ok. No significant improvemnt after, and want to avoid overfitting!

train_data = ldmi.train11
test_data = ldmi.test11
truevals = np.array(test_data.ix[:,229]) # these the same for all imp sets
sum(truevals)
sum(truevals-1)

# MI set 1
rf1 = RandomForestClassifier(n_estimators=5000, max_features = 0.325)#, max_depth = 15)
forest_fit1 = rf1.fit(ldmi.train11.ix[:,0:229],ldmi.train11.ix[:,229])

importances1 = forest_fit1.feature_importances_ #but doesn't tell you which feature is which!
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
rf2 = RandomForestClassifier(n_estimators=5000, max_features = 0.325) #, max_depth = 15)
forest_fit2 = rf2.fit(ldmi.train12.ix[:,0:229],ldmi.train12.ix[:,229])

importances2 = forest_fit2.feature_importances_ #but doesn't tell you which feature is which!
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
rf3 = RandomForestClassifier(n_estimators=5000, max_features = 0.325)#, max_depth = 15)
forest_fit3 = rf3.fit(ldmi.train13.ix[:,0:229],ldmi.train13.ix[:,229])

importances3 = forest_fit3.feature_importances_ #but doesn't tell you which feature is which!
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
rf4 = RandomForestClassifier(n_estimators=5000, max_features = 0.325)#, max_depth = 15)
forest_fit4 = rf4.fit(ldmi.train14.ix[:,0:229],ldmi.train14.ix[:,229])

importances4 = forest_fit4.feature_importances_ #but doesn't tell you which feature is which!
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
rf5 = RandomForestClassifier(n_estimators=5000, max_features = 0.325)#, max_depth = 15)
forest_fit5 = rf5.fit(ldmi.train15.ix[:,0:229],ldmi.train15.ix[:,229])

importances5 = forest_fit5.feature_importances_ #but doesn't tell you which feature is which!
forest_preds5 = forest_fit5.predict(ldmi.test15.ix[:,0:229])
forest_acc5 = (forest_preds5 == truevals).sum()/(len(ldmi.test15)*1.0)
print forest_acc5 

preds_proba5 = forest_fit5.predict_proba(ldmi.test1.ix[:,0:229])[:,1]

importances5_list = pd.TimeSeries(importances5)
metabolites = pd.TimeSeries(test_data.columns.values[0:229])
importances5_mat = pd.concat([metabolites, importances5_list], axis=1)
importances5_mat.sort(columns=1, ascending=False)




importances_average = pd.TimeSeries((importances1 + importances2 + importances5 + importances5 + importances5)/5)
importances_average_list = pd.TimeSeries(importances_average)
importances_average_mat = pd.concat([metabolites, importances_average], axis=1)
importances_average_mat_sorted = importances_average_mat.sort(columns=1, ascending=False)
importances_average_mat_sorted 
importances_average_mat_sorted.to_csv('importances_random_forest_ordered_EXT.csv')



# Plot average importances
importances_re_ordered = importances_average_mat_sorted.reset_index(drop=True).copy(deep=True)
importances_re_ordered_no_top = importances_re_ordered.ix[1:40,:].reset_index(drop=True).copy(deep=True)
importances_re_ordered_no_top2 = importances_re_ordered.ix[2:40,:].reset_index(drop=True).copy(deep=True)

plt.figure(figsize=(12,8))
plt.title("Feature importances", size=13)
plt.bar(range(importances_re_ordered.ix[0:40,:].shape[0]), importances_re_ordered.ix[0:40,:][1],
       color="gray", align="center", width =0.8)
plt.xticks(range(importances_re_ordered.ix[0:40,:].shape[0]), importances_re_ordered.ix[0:40,:][0],rotation=90, fontsize=12)
plt.yticks(fontsize = 12)
plt.xlim([-1, importances_re_ordered.ix[0:40,:].shape[0]])
plt.show()


plt.figure(figsize=(12,8))
plt.title("Feature importances", size=13)
plt.bar(range(importances_re_ordered_no_top.ix[0:39,:].shape[0]), importances_re_ordered_no_top.ix[0:39,:][1],
       color="gray", align="center", width =0.8)
plt.xticks(range(importances_re_ordered_no_top.ix[0:39,:].shape[0]), importances_re_ordered_no_top.ix[0:39,:][0],rotation=90, fontsize=12)
plt.yticks(fontsize = 12)
plt.xlim([-1, importances_re_ordered_no_top.ix[0:39,:].shape[0]])
plt.show()


plt.figure(figsize=(12,8))
plt.title("Feature importances", size=13)
plt.bar(range(importances_re_ordered_no_top2.ix[0:39,:].shape[0]), importances_re_ordered_no_top2.ix[0:39,:][1],
       color="gray", align="center", width =0.8)
plt.xticks(range(importances_re_ordered_no_top2.ix[0:39,:].shape[0]), importances_re_ordered_no_top2.ix[0:39,:][0],rotation=90, fontsize=12)
plt.yticks(fontsize = 12)
plt.xlim([-1, importances_re_ordered_no_top2.ix[0:39,:].shape[0]])
plt.show()




preds_proba_average = (preds_proba1 + preds_proba2 + preds_proba3 + preds_proba4 + preds_proba5)/5
preds_proba_average_bin = (preds_proba_average >0.5 ).astype(int)

confusion_matrix(truevals, preds_proba_average_bin, labels=[1, 0])
forest_av_acc = (preds_proba_average_bin == truevals).sum()/(len(ldmi.test14)*1.0)
print forest_av_acc 
print roc_auc_score(truevals, forest_preds5)



fpr, tpr, _ = roc_curve(truevals, preds_proba_average)
# Calculate the AUC
roc_auc = auc(fpr, tpr)
print 'ROC AUC: %0.2f' % roc_auc
 
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', size = 13)
plt.ylabel('True Positive Rate', size = 13)
plt.title('ROC Curve for random forest predictions averaged \n across the imputation datasets, including confounders', size = 13)
plt.legend(loc="lower right")
plt.show()



# Alterned bounds
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



preds_proba_average_confident50 = preds_proba_average[(preds_proba_average > 0.4) & (preds_proba_average < 0.6)]
preds_proba_average_confident50_bin = (preds_proba_average_confident50 > 0.5).astype(int)
truevals_confident50 = truevals[(preds_proba_average > 0.4) & (preds_proba_average < 0.6)]

accuracy_proba_average_confident50 = ((preds_proba_average_confident50_bin== truevals_confident50).sum())/(len(truevals_confident50)*1.0)
print accuracy_proba_average_confident50