# -*- coding: utf-8 -*-
"""
Individual metabolite ananlysis, kind of replicate 3 pop cohort study

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
import load_data_mi_ext_new as ldmi
import load_data_ext as ld_ext

np.random.seed(10)


###############################################################################

# PREDICTIVE
# Phenylalanie (Phe), MUFA% (MUFA/FA), omega-6 fatty acid (FAw6), DHA  (DHA), PUFA

#Significant
# Phenlylalaline Phe , Tyrosene Tyr, Lactate Lac , Pyruvate NOT AVAILABLE???, Beta-hydroxbutyrate bOHBut,
#MUFA, Omega3-fatty acisds%, DHA%, PUFA%, MUFA%, Unsaturation degree UnSat,
# (inflamation) glycoprotein acetyls GP, ApoA, ApoB, Triglycerides (Serum-TG),
# XL-VLDL, VL-VLDV, L-VLDV, M,S, XS, IDL, L-LDL, M-LDL, M-LDL,
#S-LDL , L-HDL, M-HDL, HDL particle size, (cholesterol) VLDL C, 
# IDL C, LDL C, HDL C, HDL2 C

# They used Cox proportional hazards regression models, we just use logitic regression on logged values

#Missing case
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

# Create dictionary to store results summaries
metabolites = list(data.columns.values)[0:225]


# Regression on Gaz's smaller set of confounders
summary_dict_gaz = {}
for m in metabolites:
    summary_dict_gaz[m] = []

for m in metabolites:
    regression_mat = pd.concat([logged_data[m], regression_mat_initial], axis=1)
    logit_mod = sm.Logit(logged_data['CAD50'], regression_mat, missing = 'drop')
    result_mod = logit_mod.fit()
    summary_dict_gaz[m] = result_mod.summary()

summary_dict_gaz

# Significant variables:
#Metabolite          coef        std err    z          P>|z|         95% conf int
#Ala                 0.1551      0.065      2.371      0.018         0.027     0.283
#Alb                -0.1945      0.065     -2.994      0.003        -0.322    -0.067
#*ApoA1              -0.2773      0.068     -4.097      0.000        -0.410    -0.145
#ApoB/ApoA1          0.1808      0.067      2.682      0.007         0.049     0.313
#EstC               -0.1335      0.067     -2.005      0.045        -0.264    -0.003
#FAw6               -0.1437      0.065     -2.211      0.027        -0.271    -0.016
#*FAw6/FA            -0.2604      0.068     -3.840      0.000        -0.393    -0.128
#FreeC              -0.1335      0.063     -2.114      0.035        -0.257    -0.010
#Glc                 0.1806      0.067      2.693      0.007         0.049     0.312
#Gln                -0.1996      0.070     -2.858      0.004        -0.337    -0.063
#Gp                  0.1536      0.066      2.344      0.019         0.025     0.282
#HDL-C              -0.3130      0.070     -4.442      0.000        -0.451    -0.175
#*HDL-D              -0.2889      0.070     -4.134      0.000        -0.426    -0.152
#*HDL2-C             -0.3241      0.070     -4.608      0.000        -0.462    -0.186
#IDL-C              -0.1495      0.065     -2.312      0.021        -0.276    -0.023
#IDL-CE             -0.1386      0.064     -2.154      0.031        -0.265    -0.012
#IDL-C_%            -0.2120      0.075     -2.843      0.004        -0.358    -0.066
#IDL-FC             -0.1652      0.064     -2.575      0.010        -0.291    -0.039
#IDL-FC_%           -0.1598      0.076     -2.099      0.036        -0.309    -0.011
#IDL-PL             -0.1560      0.064     -2.422      0.015        -0.282    -0.030
#IDL-PL_%           -0.1867      0.066     -2.826      0.005        -0.316    -0.057
#IDL-TG              0.1683      0.064      2.640      0.008         0.043     0.293
#*IDL-TG_%            0.3433      0.068      5.032      0.000         0.210     0.477
#Ile                 0.2027      0.068      2.965      0.003         0.069     0.337
#*L-HDL-C            -0.2916      0.070     -4.193      0.000        -0.428    -0.155
#*L-HDL-CE           -0.2939      0.070     -4.220      0.000        -0.430    -0.157
#*L-HDL-FC           -0.2702      0.068     -3.959      0.000        -0.404    -0.136
#*L-HDL-L            -0.3142      0.071     -4.437      0.000        -0.453    -0.175
#*L-HDL-P            -0.3092      0.070     -4.423      0.000        -0.446    -0.172
#*L-HDL-PL           -0.3255      0.071     -4.606      0.000        -0.464    -0.187
#*L-HDL-TG           -0.2459      0.069     -3.566      0.000        -0.381    -0.111
#L-LDL-C            -0.1269      0.064     -1.968      0.049        -0.253    -0.001
#L-LDL-FC           -0.1595      0.064     -2.486      0.013        -0.285    -0.034
#L-LDL-FC_%         -0.1685      0.075     -2.238      0.025        -0.316    -0.021
#L-LDL-PL           -0.1388      0.064     -2.164      0.030        -0.265    -0.013
#*L-LDL-TG_%          0.2611      0.068      3.815      0.000         0.127     0.395
#L-VLDL-C            0.1580      0.067      2.352      0.019         0.026     0.290
#L-VLDL-CE           0.1373      0.067      2.061      0.039         0.007     0.268
#L-VLDL-FC           0.1682      0.068      2.458      0.014         0.034     0.302
#L-VLDL-L            0.1915      0.067      2.866      0.004         0.061     0.322
#L-VLDL-P            0.1767      0.070      2.533      0.011         0.040     0.313
#L-VLDL-PL           0.1772      0.068      2.596      0.009         0.043     0.311
#L-VLDL-TG           0.1958      0.068      2.899      0.004         0.063     0.328
#LA/FA              -0.2073      0.067     -3.114      0.002        -0.338    -0.077
#*Lac                 0.2275      0.065      3.527      0.000         0.101     0.354
#Leu                 0.1366      0.067      2.025      0.043         0.004     0.269
#*M-HDL-C            -0.2599      0.068     -3.833      0.000        -0.393    -0.127
#*M-HDL-CE           -0.2536      0.068     -3.754      0.000        -0.386    -0.121
#M-HDL-CE_%         -0.1722      0.068     -2.538      0.011        -0.305    -0.039
#M-HDL-C_%          -0.2235      0.070     -3.176      0.001        -0.361    -0.086
#*M-HDL-FC           -0.2820      0.068     -4.170      0.000        -0.415    -0.149
#M-HDL-FC_%         -0.2614      0.076     -3.423      0.001        -0.411    -0.112
#M-HDL-L            -0.2343      0.070     -3.366      0.001        -0.371    -0.098
#*M-HDL-P            -0.2412      0.068     -3.543      0.000        -0.375    -0.108
#M-HDL-PL           -0.2328      0.070     -3.335      0.001        -0.370    -0.096
#M-HDL-PL_%          0.1856      0.067      2.784      0.005         0.055     0.316
#M-HDL-TG_%          0.2078      0.067      3.121      0.002         0.077     0.338
#M-LDL-TG_%          0.1972      0.068      2.911      0.004         0.064     0.330
#M-VLDL-C            0.1405      0.066      2.139      0.032         0.012     0.269
#M-VLDL-CE_%        -0.2023      0.066     -3.062      0.002        -0.332    -0.073
#M-VLDL-C_%         -0.1381      0.066     -2.102      0.036        -0.267    -0.009
#M-VLDL-FC           0.1931      0.068      2.832      0.005         0.059     0.327
#M-VLDL-FC_%         0.1481      0.060      2.457      0.014         0.030     0.266
#M-VLDL-L            0.2014      0.067      3.022      0.003         0.071     0.332
#M-VLDL-P            0.1862      0.070      2.673      0.008         0.050     0.323
#M-VLDL-PL           0.1853      0.068      2.734      0.006         0.052     0.318
#*M-VLDL-PL_%        -0.2365      0.062     -3.789      0.000        -0.359    -0.114
#M-VLDL-TG           0.2178      0.068      3.214      0.001         0.085     0.351
#*MUFA/FA             0.2839      0.064      4.419      0.000         0.158     0.410
#PC                 -0.1705      0.065     -2.616      0.009        -0.298    -0.043
#PUFA               -0.1300      0.065     -2.003      0.045        -0.257    -0.003
#*PUFA/FA            -0.2508      0.067     -3.728      0.000        -0.383    -0.119
#S-HDL-C            -0.1616      0.067     -2.408      0.016        -0.293    -0.030
#S-HDL-CE           -0.1776      0.067     -2.657      0.008        -0.309    -0.047
#S-HDL-CE_%         -0.1634      0.076     -2.150      0.032        -0.312    -0.014
#S-HDL-C_%          -0.1641      0.070     -2.329      0.020        -0.302    -0.026
#S-HDL-TG            0.2132      0.063      3.383      0.001         0.090     0.337
#*S-HDL-TG_%          0.2908      0.067      4.312      0.000         0.159     0.423
#S-LDL-TG_%          0.2297      0.066      3.464      0.001         0.100     0.360
#*S-VLDL-CE_%        -0.2666      0.070     -3.795      0.000        -0.404    -0.129
#*S-VLDL-C_%         -0.2635      0.068     -3.881      0.000        -0.397    -0.130
#S-VLDL-FC           0.1655      0.065      2.527      0.012         0.037     0.294
#S-VLDL-L            0.1773      0.065      2.707      0.007         0.049     0.306
#S-VLDL-P            0.1749      0.067      2.604      0.009         0.043     0.307
#S-VLDL-PL           0.1608      0.065      2.461      0.014         0.033     0.289
#S-VLDL-PL_%        -0.1700      0.065     -2.613      0.009        -0.298    -0.043
#S-VLDL-TG           0.2353      0.068      3.466      0.001         0.102     0.368
#*S-VLDL-TG_%         0.3068      0.065      4.737      0.000         0.180     0.434
#SM                 -0.2060      0.065     -3.157      0.002        -0.334    -0.078
#Serum-C            -0.1576      0.065     -2.418      0.016        -0.285    -0.030
#Serum-TG            0.2110      0.066      3.202      0.001         0.082     0.340
#*TG/PG               0.2425      0.069      3.505      0.000         0.107     0.378
#TotCho             -0.1656      0.065     -2.547      0.011        -0.293    -0.038
#TotPG              -0.1350      0.065     -2.088      0.037        -0.262    -0.008
#UnSat              -0.1784      0.064     -2.773      0.006        -0.305    -0.052
#VLDL-D              0.2173      0.066      3.271      0.001         0.087     0.348
#VLDL-TG             0.2191      0.067      3.290      0.001         0.089     0.350
#XL-HDL-CE_%         0.2189      0.070      3.130      0.002         0.082     0.356
#XL-HDL-C_%          0.1920      0.068      2.807      0.005         0.058     0.326
#XL-HDL-FC          -0.1662      0.066     -2.511      0.012        -0.296    -0.036
#XL-HDL-L           -0.1688      0.067     -2.504      0.012        -0.301    -0.037
#XL-HDL-P           -0.1774      0.067     -2.658      0.008        -0.308    -0.047
#XL-HDL-PL          -0.2254      0.068     -3.297      0.001        -0.359    -0.091
#XL-HDL-PL_%        -0.1910      0.077     -2.478      0.013        -0.342    -0.040
#XL-HDL-TG_%         0.1785      0.071      2.531      0.011         0.040     0.317
#XL-VLDL-L           0.1485      0.067      2.209      0.027         0.017     0.280
#XL-VLDL-P           0.1408      0.069      2.053      0.040         0.006     0.275
#XL-VLDL-TG          0.1552      0.068      2.288      0.022         0.022     0.288
#XS-VLDL-CE_%       -0.2833      0.066     -4.283      0.000        -0.413    -0.154
#*XS-VLDL-C_%        -0.2620      0.068     -3.866      0.000        -0.395    -0.129
#XS-VLDL-PL_%       -0.1551      0.071     -2.194      0.028        -0.294    -0.017
#XS-VLDL-TG          0.2274      0.066      3.439      0.001         0.098     0.357
#*XS-VLDL-TG_%        0.2960      0.066      4.458      0.000         0.166     0.426


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

variance1 = pca.explained_variance_ratio_
pca = PCA(n_components=6, copy=True)
pca.fit(metabolities_mat_imp)
pca_components = pca.components_

# Look at composition of Principle Components
pca_components_df = pd.DataFrame(pca_components)
pca_components_df.columns = (metabolities_mat_imp.columns).tolist()
pca_components_df.to_csv('PCA_components1.csv')


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



# Run regression
logit_mod = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat.ix[:,0:10]],axis=1), missing = 'drop')
result_mod = logit_mod.fit()
result_mod.summary()
#1, 2, 5 significant

#Bonferroni adjustment 0.008
logit_mod1 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC1'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod1 = logit_mod1.fit()
result_mod1.summary()
#YES, bf yes

logit_mod2 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC2'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod2 = logit_mod2.fit()
result_mod2.summary()
#YES, bf no

logit_mod3 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC3'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod3 = logit_mod3.fit()
result_mod3.summary()
#NO

logit_mod4 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC4'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod4 = logit_mod4.fit()
result_mod4.summary()
#NO

logit_mod5 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC5'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod5 = logit_mod5.fit()
result_mod5.summary()
#YES, bf no

logit_mod6 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC6'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod6 = logit_mod6.fit()
result_mod6.summary()
#NO



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

variance2 = pca.explained_variance_ratio_
pca = PCA(n_components=6, copy=True)
pca.fit(metabolities_mat_imp)
pca_components = pca.components_

# Look at composition of Principle Components
pca_components_df = pd.DataFrame(pca_components)
pca_components_df.columns = (metabolities_mat_imp.columns).tolist()
pca_components_df.to_csv('PCA_components2.csv')

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



# Run regression
logit_mod = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat.ix[:,0:10]],axis=1), missing = 'drop')
result_mod = logit_mod.fit()
result_mod.summary()
#1, 2, 5 significant


#Bonferroni adjustment 0.008
logit_mod1 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC1'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod1 = logit_mod1.fit()
result_mod1.summary()
#YES, bf yes

logit_mod2 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC2'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod2 = logit_mod2.fit()
result_mod2.summary()
#YES, bf no

logit_mod3 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC3'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod3 = logit_mod3.fit()
result_mod3.summary()
#NO

logit_mod4 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC4'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod4 = logit_mod4.fit()
result_mod4.summary()
#NO

logit_mod5 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC5'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod5 = logit_mod5.fit()
result_mod5.summary()
#YES, bf yes

logit_mod6 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC6'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod6 = logit_mod6.fit()
result_mod6.summary()
#NO



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

#0.65 for 5.4 regularisation
#0.66 for 1.6 regularsation
 

##### Analysis without confounders ######

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

variance3 = pca.explained_variance_ratio_
pca = PCA(n_components=6, copy=True)
pca.fit(metabolities_mat_imp)
pca_components = pca.components_

# Look at composition of Principle Components
pca_components_df = pd.DataFrame(pca_components)
pca_components_df.columns = (metabolities_mat_imp.columns).tolist()
pca_components_df.to_csv('PCA_components3.csv')

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



# Run regression
logit_mod = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat.ix[:,0:10]],axis=1), missing = 'drop')
result_mod = logit_mod.fit()
result_mod.summary()
#1, 2, 5 significant

#Bonferroni adjustment 0.008
logit_mod1 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC1'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod1 = logit_mod1.fit()
result_mod1.summary()
#YES, bf yes

logit_mod2 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC2'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod2 = logit_mod2.fit()
result_mod2.summary()
#YES, bf no

logit_mod3 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC3'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod3 = logit_mod3.fit()
result_mod3.summary()
#NO

logit_mod4 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC4'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod4 = logit_mod4.fit()
result_mod4.summary()
#NO

logit_mod5 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC5'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod5 = logit_mod5.fit()
result_mod5.summary()
#YES, bf no

logit_mod6 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC6'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod6 = logit_mod6.fit()
result_mod6.summary()
#NO




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

preds3c = result_mod_no_conf3.predict(pd.concat([PC_data_test['intercept'],PC_data_test.ix[:,0:2],
                                                                PC_data_test.ix[:,3:4],PC_data_test.ix[:,5:6]],axis=1))

#preds = result_mod_no_conf.predict(pd.concat([PC_data_test['intercept'],PC_data_test.ix[:,0:6]],axis=1))

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

variance4 = pca.explained_variance_ratio_
pca = PCA(n_components=6, copy=True)
pca.fit(metabolities_mat_imp)
pca_components = pca.components_

# Look at composition of Principle Components
pca_components_df = pd.DataFrame(pca_components)
pca_components_df.columns = (metabolities_mat_imp.columns).tolist()
pca_components_df.to_csv('PCA_components4.csv')

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



# Run regression
logit_mod = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat.ix[:,0:10]],axis=1), missing = 'drop')
result_mod = logit_mod.fit()
result_mod.summary()
#1, 2, 5 significant

#Bonferroni adjustment 0.008
logit_mod1 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC1'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod1 = logit_mod1.fit()
result_mod1.summary()
#YES, bf yes

logit_mod2 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC2'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod2 = logit_mod2.fit()
result_mod2.summary()
#YES, bf no

logit_mod3 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC3'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod3 = logit_mod3.fit()
result_mod3.summary()
#NO

logit_mod4 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC4'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod4 = logit_mod4.fit()
result_mod4.summary()
#NO

logit_mod5 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC5'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod5 = logit_mod5.fit()
result_mod5.summary()
#YES, bf yes

logit_mod6 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC6'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod6 = logit_mod6.fit()
result_mod6.summary()
#NO




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



##### Analysis without confounders ######

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

variance5 = pca.explained_variance_ratio_
pca = PCA(n_components=6, copy=True)
pca.fit(metabolities_mat_imp)
pca_components = pca.components_

# Look at composition of Principle Components
pca_components_df = pd.DataFrame(pca_components)
pca_components_df.columns = (metabolities_mat_imp.columns).tolist()
pca_components_df.to_csv('PCA_components5.csv')

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



# Run regression
logit_mod = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat.ix[:,0:10]],axis=1), missing = 'drop')
result_mod = logit_mod.fit()
result_mod.summary()
#1, 2, 5 significant

#Bonferroni adjustment 0.008
logit_mod1 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC1'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod1 = logit_mod1.fit()
result_mod1.summary()
#YES, bf yes

logit_mod2 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC2'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod2 = logit_mod2.fit()
result_mod2.summary()
#YES, bf no

logit_mod3 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC3'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod3 = logit_mod3.fit()
result_mod3.summary()
#NO

logit_mod4 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC4'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod4 = logit_mod4.fit()
result_mod4.summary()
#NO

logit_mod5 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC5'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod5 = logit_mod5.fit()
result_mod5.summary()
#YES, bf no

logit_mod6 = sm.Logit(PC_data_mat['CAD50'], pd.concat([PC_data_mat['intercept'],PC_data_mat['PC6'],PC_data_mat.ix[:,6:10]],axis=1), missing = 'drop')
result_mod6 = logit_mod6.fit()
result_mod6.summary()
#NO




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


preds_bin5c = (preds5c > 0.5).astype(int)
truevals = np.array(PC_data_test.ix[:,11])
accuracy5c = ((preds_bin5c== truevals).sum())/(len(PC_data_test)*1.0)
print accuracy5c
0.691056910569




###################################################################################################################
##################### aveage predictions with confounders #####################
###################################################################################################################

#Average variance explained
variance_average = (variance1 + variance2 + variance3 + variance4 + variance5)/5


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




