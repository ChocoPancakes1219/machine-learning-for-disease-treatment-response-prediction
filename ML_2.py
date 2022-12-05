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


#NORMALISATION

from sklearn.preprocessing import MinMaxScaler

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

df_normalisation = normalize(new_df)


#FEATURE SELECTION
#Classification 

#separate target variable for classification (PCR)
x_classif = df_normalisation.drop(labels=["pCR (outcome)"], axis=1)
y_classif = df_normalisation["pCR (outcome)"]

#apply ANOVA for classification
from sklearn.feature_selection import f_classif, SelectKBest
import math

fs = SelectKBest(score_func = f_classif, k= 53)  #k value for when P_value < 0.05

# Apply feature selection
fs.fit(x_classif,y_classif)

#declare variable to put selected features for classification
df_classif = pd.DataFrame()

def list_ceil(x):
    return[math.ceil(i) in x]

features_score = pd.DataFrame(fs.scores_)
features_pvalue = pd.DataFrame(np.round(fs.pvalues_,4))
features = pd.DataFrame(x_classif.columns)
feature_score = pd.concat([features,features_score,features_pvalue],axis=1)

# Assign column names
feature_score.columns = ["Input_Features","Score","P_Value"]
chosen_features = feature_score.nlargest(53,columns="Score")

# Add selected features into a new dataframe
for feature in chosen_features['Input_Features']:
    df_classif = pd.concat([df_classif, df_normalisation[feature]], axis=1)


#Regression

#separate target value for regression (RFS)
x_reg = df_normalisation.drop(labels=["RelapseFreeSurvival (outcome)"], axis=1)
y_reg = df_normalisation["RelapseFreeSurvival (outcome)"]

#apply ANOVA for regression
from sklearn.feature_selection import f_regression, SelectKBest
import math

fs = SelectKBest(score_func=f_regression,k=20) #k value for when P_value < 0.05

# Apply feature selection
fs.fit(x_reg,y_reg)

#declare variable to put selected features for classification
df_reg = pd.DataFrame()

def list_ceil(x):
    return[math.ceil(i) in x]

features_score = pd.DataFrame(fs.scores_)
features_pvalue = pd.DataFrame(np.round(fs.pvalues_,4))
features = pd.DataFrame(x_reg.columns)
feature_score = pd.concat([features,features_score,features_pvalue],axis=1)

# Assign column names
feature_score.columns = ["Input_Features","Score","P_Value"]
chosen_features = feature_score.nlargest(20,columns="Score")

# Add selected features into a new dataframe
for feature in chosen_features['Input_Features']:
    df_reg = pd.concat([df_reg, df_normalisation[feature]], axis=1)

