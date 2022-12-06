#!/usr/bin/env python
# coding: utf-8

#import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#import dataset
df = pd.read_excel('trainDataset.xls')

#drop ID column from dataset
df = df.iloc[:, 1:]


#REPLACING MISSING VALUES

#replace missing values (999) of each column to the column's median
for column in df:
    df[column]=df[column].replace(999, df[column].median())

#new_df will be without outliers
new_df = df

#drop the columns with booleans/specific grades as they wont have outliers
df = df.drop(['pCR (outcome)', 'ER', 'PgR', 'HER2', 'TrippleNegative', 'ChemoGrade', 'Proliferation', 'HistologyType', 'LNStatus', 'TumourStage'], axis=1)


#REPLACING OUTLIERS

#use IQR method to find outliers
def IQR_outliers(column):
    Q1=df[column].quantile(0.25)
    Q3=df[column].quantile(0.75)
    IQR = Q3-Q1
    df_outliers=df[((df[column]<(Q1-1.5*IQR)) | (df[column]>(Q3+1.5*IQR)))]
    return df_outliers

#replace outliers with median of the column
for column in df:
    df_outliers = IQR_outliers(column)
    
    for i in range((df_outliers.shape[0])):
        label_index = df_outliers.index[i]
        new_df.loc[label_index, column] = df[column].median()
display(new_df)


#NORMALISATION

from sklearn.preprocessing import MinMaxScaler

#namelist of columns that do not need normalisation for classification
drop_classif = ['pCR (outcome)', 'ER', 'PgR', 'HER2', 'TrippleNegative', 'ChemoGrade', 'Proliferation', 'HistologyType', 'LNStatus', 'TumourStage']

#namelist of columns that do not need normalisation for regression
drop_reg = drop_classif + ['RelapseFreeSurvival (outcome)']

def normalize(df, x):
    result = df.copy()
    for feature_name in df.columns:
        if (x == 'classification' and feature_name not in drop_classif) or (x == 'regression' and feature_name not in drop_reg):
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

df_classif = normalize(new_df,'classification')
df_reg = normalize(new_df,'regression')


#FEATURE SELECTION
#Classification 

#separate target variable for classification (PCR)
x_classif = df_classif.drop(labels=['pCR (outcome)'], axis=1)
y_classif = df_classif['pCR (outcome)']

#apply ANOVA for classification
from sklearn.feature_selection import f_classif, SelectKBest
import math

fs = SelectKBest(score_func = f_classif, k= 53)  #k value for when P_value < 0.05
# Apply feature selection
fs.fit(x_classif,y_classif)

#declare variable to put selected features for classification
df_classif_final = pd.DataFrame()

def list_ceil(x):
    return[math.ceil(i) in x]

features_score = pd.DataFrame(fs.scores_)
features_pvalue = pd.DataFrame(np.round(fs.pvalues_,4))
features = pd.DataFrame(x_classif.columns)
feature_score = pd.concat([features,features_score,features_pvalue],axis=1)

# Assign column names
feature_score.columns = ['Input_Features','Score','P_Value']
chosen_features = feature_score.nlargest(53,columns='Score')

# Add selected features into a new dataframe
for feature in chosen_features['Input_Features']:
    df_classif_final = pd.concat([df_classif_final, df_classif[feature]], axis=1)


#Regression

#separate target value for regression (RFS)
x_reg = df_reg.drop(labels=['RelapseFreeSurvival (outcome)'], axis=1)
y_reg = df_reg['RelapseFreeSurvival (outcome)']


# In[11]:


#apply ANOVA for regression
from sklearn.feature_selection import f_regression, SelectKBest
import math

fs = SelectKBest(score_func=f_regression,k=20) #k value for when P_value < 0.05
# Apply feature selection
fs.fit(x_reg,y_reg)

#declare variable to put selected features for classification
df_reg_final = pd.DataFrame()

def list_ceil(x):
    return[math.ceil(i) in x]

features_score = pd.DataFrame(fs.scores_)
features_pvalue = pd.DataFrame(np.round(fs.pvalues_,4))
features = pd.DataFrame(x_reg.columns)
feature_score = pd.concat([features,features_score,features_pvalue],axis=1)

# Assign column names
feature_score.columns = ['Input_Features','Score','P_Value']
chosen_features = feature_score.nlargest(20,columns='Score')

# Add selected features into a new dataframe
for feature in chosen_features['Input_Features']:
    df_reg_final = pd.concat([df_reg_final, df_reg[feature]], axis=1)

