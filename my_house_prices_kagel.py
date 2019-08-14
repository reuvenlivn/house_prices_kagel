# -*- coding: utf-8 -*-
"""
Created on Sun May 19 08:32:53 2019

@author: reuve
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')

# Load the train and test data in a dataframe
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
 
# MISSING VALUES IMPUTATION

nulls = train.isnull().sum().sort_values(ascending=False)

# drop the columns with more then 1000 nulls
train = train.drop(['Id','PoolQC','MiscFeature','Alley','Fence'],axis = 1)

# in train matrix there are two column 'Fireplaces' and 'FireplaceQu' .The attribute 'FireplaceQu' has
# 690 null values. Comparing columns 'FireplaceQu' and 'Fireplaces' we noticed  that the zeros in 
# 'Fireplaces' column has null value in 'FireplaceQu'. 
#So, we should replace these nulls with "no Fireplace" ('NF')
train['FireplaceQu']=train['FireplaceQu'].fillna('NF')

# 'LotFrontage' colunm has 259 nulls. We replace them with the mean of the colunm
train['LotFrontage'] =train['LotFrontage'].fillna(value=train['LotFrontage'].mean())
#
## ## Attributes related to "GARAGE"
## The columns related to Garage have the same number of null values. It implay that there is  
# a relationship among them.'GarageArea' column has 81 zeros which is equal to 'nulls' in
# the other columns.conclusion: the houses without Garage Area are having 'nulls' at all these columns.
# So, we replace these nulls  with 'No GarageArea'----> 'NG' 
train['GarageType']=train['GarageType'].fillna('NG')
train['GarageCond']=train['GarageCond'].fillna('NG')
train['GarageFinish']=train['GarageFinish'].fillna('NG')
train['GarageYrBlt']=train['GarageYrBlt'].fillna('NG')
train['GarageQual']=train['GarageQual'].fillna('NG')

#
## ## Bsmt (same as the Garage)
train['BsmtExposure']=train['BsmtExposure'].fillna('NB')
train['BsmtFinType2']=train['BsmtFinType2'].fillna('NB')
train['BsmtFinType1']=train['BsmtFinType1'].fillna('NB')
train['BsmtCond']=train['BsmtCond'].fillna('NB')
train['BsmtQual']=train['BsmtQual'].fillna('NB')


## ## MasVnr
train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())
train['MasVnrType'] = train['MasVnrType'].fillna('none')

## ## Electrical
train.Electrical = train.Electrical.fillna('SBrkr')

## # OUTLIERS
num_train = train._get_numeric_data()

def var_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), 
                      x.min(), x.quantile(0.01), x.quantile(0.05),x.quantile(0.10),x.quantile(0.25),
                      x.quantile(0.50),x.quantile(0.75), x.quantile(0.90),x.quantile(0.95), 
                      x.quantile(0.99),x.max()], 
                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,
                         'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])

num_train.apply(lambda x: var_summary(x)).T

sns.boxplot([num_train.LotFrontage])
train['LotFrontage']= train['LotFrontage'].clip_upper(train['LotFrontage'].quantile(0.99)) 

sns.boxplot(num_train.LotArea)
train['LotArea']= train['LotArea'].clip_upper(train['LotArea'].quantile(0.99)) 

sns.boxplot(train['MasVnrArea'])
train['MasVnrArea']= train['MasVnrArea'].clip_upper(train['MasVnrArea'].quantile(0.99))

sns.boxplot(train['BsmtFinSF1']) 
sns.boxplot(train['BsmtFinSF2']) 
train['BsmtFinSF1']= train['BsmtFinSF1'].clip_upper(train['BsmtFinSF1'].quantile(0.99)) 
train['BsmtFinSF2']= train['BsmtFinSF2'].clip_upper(train['BsmtFinSF2'].quantile(0.99)) 

sns.boxplot(train['TotalBsmtSF'])
train['TotalBsmtSF']= train['TotalBsmtSF'].clip_upper(train['TotalBsmtSF'].quantile(0.99))

sns.boxplot(train['1stFlrSF'])
train['1stFlrSF']= train['1stFlrSF'].clip_upper(train['1stFlrSF'].quantile(0.99))

sns.boxplot(train['2ndFlrSF'])
train['2ndFlrSF']= train['2ndFlrSF'].clip_upper(train['2ndFlrSF'].quantile(0.99))

sns.boxplot(train['GrLivArea'])
train['GrLivArea']= train['GrLivArea'].clip_upper(train['GrLivArea'].quantile(0.99))

sns.boxplot(train['BedroomAbvGr'])
train['BedroomAbvGr']= train['BedroomAbvGr'].clip_upper(train['BedroomAbvGr'].quantile(0.99))
train['BedroomAbvGr']= train['BedroomAbvGr'].clip_lower(train['BedroomAbvGr'].quantile(0.01))

sns.boxplot(train['GarageCars'])
train['GarageCars']= train['GarageCars'].clip_upper(train['GarageCars'].quantile(0.99))

sns.boxplot(train['GarageArea'])
train['GarageArea']= train['GarageArea'].clip_upper(train['GarageArea'].quantile(0.99))

sns.boxplot(train['WoodDeckSF'])
train['WoodDeckSF']= train['WoodDeckSF'].clip_upper(train['WoodDeckSF'].quantile(0.99))

sns.boxplot(train['OpenPorchSF'])
train['OpenPorchSF']= train['OpenPorchSF'].clip_upper(train['OpenPorchSF'].quantile(0.99))

sns.boxplot(train['EnclosedPorch'])
train['EnclosedPorch']= train['EnclosedPorch'].clip_upper(train['EnclosedPorch'].quantile(0.99))

sns.boxplot(train['3SsnPorch'])
train['3SsnPorch']= train['3SsnPorch'].clip_upper(train['3SsnPorch'].quantile(0.99))

sns.boxplot(train['ScreenPorch'])
train['ScreenPorch']= train['ScreenPorch'].clip_upper(train['ScreenPorch'].quantile(0.99))

sns.boxplot(train['PoolArea'])
train['PoolArea']= train['PoolArea'].clip_upper(train['PoolArea'].quantile(0.99))

sns.boxplot(train['MiscVal'])
sns.boxplot(train.SalePrice)

train['SalePrice']= train['SalePrice'].clip_upper(train['SalePrice'].quantile(0.99))
train['SalePrice']= train['SalePrice'].clip_lower(train['SalePrice'].quantile(0.01))

num_corr=num_train.corr()
plt.subplots(figsize=(13,10))
sns.heatmap(num_corr,vmax =.8 ,square = True)

k = 14
cols = num_corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(num_train[cols].values.T)
sns.set(font_scale=1.35)
f, ax = plt.subplots(figsize=(10,10))
hm=sns.heatmap(cm, annot = True,vmax =.8, yticklabels=cols.values, xticklabels = cols.values)


# # FEATURE SELECTION
# ## Extracting new Features using PCA
from sklearn.preprocessing import StandardScaler

train_d = pd.get_dummies(train).astype(float)
train_d1 = train_d.drop(['SalePrice'],axis = 1)
y = train_d.SalePrice

# convert all the data into a single scaleusing Standard Scalar method
scaler = StandardScaler()
scaler.fit(train_d1)                
t_train = scaler.transform(train_d1)

from sklearn.decomposition import PCA

pca_hp = PCA(30)
x_fit = pca_hp.fit_transform(t_train)

PCA_variance = np.exp(pca_hp.explained_variance_ratio_)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(x_fit,y)

X_train , X_test, Y_train, Y_test = train_test_split(
        x_fit,
        y,
        test_size=0.20,
        random_state=123)

y_pred = linear.predict(X_test)

from sklearn import metrics

score_1= metrics.r2_score(Y_test, y_pred)
rmse_lin_reg = np.sqrt(metrics.mean_squared_error(Y_test, y_pred))
print('\nlin reg model: score={} , rmse:{}'.format(score_1,rmse_lin_reg ))

from sklearn import metrics
#from sklearn.tree import DecisionTreeRegressor, export_graphviz, export 
from sklearn.tree import DecisionTreeRegressor

#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV

# >>I have used GridSearch cv tofind the hyper parameters, such as n_estimators and depth .
depth_list = list(range(1,20))
for depth in depth_list:
    dt_obj = DecisionTreeRegressor(max_depth=depth)
    dt_obj.fit(X_train, Y_train)
#    print ('depth:', depth, 'R_squared:', metrics.r2_score(Y_test, dt_obj.predict(X_test)))

param_grid = {'max_depth': np.arange(3,20)}
tree = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=10)
tree.fit(X_train, Y_train)

best_param = tree.best_params_

score_2 = tree.best_score_

tree_final = DecisionTreeRegressor(max_depth=best_param.get('max_depth'))
tree_final.fit(X_train, Y_train)
#score_3 = tree_final.score

tree_test_pred = pd.DataFrame({'actual': Y_test, 'predicted': tree_final.predict(X_test)})

score_DT = metrics.r2_score(Y_test, tree_test_pred.predicted)
rmse_DT = np.sqrt(metrics.mean_squared_error(Y_test, tree_test_pred.predicted))
print('\nDecisionTreeRegressor: score={} , rmse:{}'.format(score_DT,rmse_DT))


# ## RANDOM FORESTS

from sklearn.ensemble import RandomForestRegressor

depth_list = list(range(1,20))
best_score = 0
for depth in depth_list:
    dt_obj = RandomForestRegressor(max_depth=depth)
    dt_obj.fit(X_train, Y_train)
    score = metrics.r2_score(Y_test, dt_obj.predict(X_test))
    if (score > best_score):
        best_depth = depth
        best_score = score
        
print ('best depth:{}, best_score:{}'.format(best_depth, best_score))

#radm_clf = RandomForestRegressor(oob_score=True,n_estimators=100,max_depth = 17)
radm_clf = RandomForestRegressor(oob_score=True,max_depth = best_depth)
  
radm_clf.fit( X_train, Y_train )

radm_test_pred = pd.DataFrame( { 'actual': Y_test, 'predicted':radm_clf.predict(X_test) } )

score_4 = metrics.r2_score( radm_test_pred.actual, radm_test_pred.predicted )

rmse_RF = np.sqrt(metrics.mean_squared_error(Y_test, tree_test_pred.predicted))

print('\nDecisionTreeRegressor: score={} , rmse:{}'.format(score_4,rmse_RF))

