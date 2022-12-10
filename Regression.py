#!/usr/bin/env python
# coding: utf-8

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression


# import dataset
df = pd.read_csv('trainDataset.csv')

#drop ID column from dataset
df = df.iloc[:, 1:]


# DELETE MISSING VALUES

#delete missing values (999) of each column
for column in df:
    df[column]=df[column].replace(999, None)

df = df.dropna()

final_df_classif = df
final_df_reg = df

#namelist of columns that has categorical data or do not need normalisation for classification
drop_classif = ['pCR (outcome)', 'ER', 'PgR', 'HER2', 'TrippleNegative', 'ChemoGrade', 'Proliferation', 'HistologyType', 'LNStatus', 'TumourStage']

#namelist of columns that has categorical data or do not need normalisation for regression
drop_reg = drop_classif + ['RelapseFreeSurvival (outcome)']

#drop the columns with booleans/specific grades as they wont have outliers
df_classif = df.drop(drop_classif, axis=1)
df_reg = df.drop(drop_reg, axis=1)


# REPLACING OUTLIERS

#use IQR method to find outliers
def IQR_outliers(column):
    Q1=df[column].quantile(0.25)
    Q3=df[column].quantile(0.75)
    IQR = Q3-Q1
    df_outliers=df[((df[column]<(Q1-1.5*IQR)) | (df[column]>(Q3+1.5*IQR)))]
    return df_outliers

#delete outliers
def replace_outliers(df1, df2, drop_list):
    for column in df1:
        if column not in drop_list:
            df_outliers = IQR_outliers(column)
            for i in range((df_outliers.shape[0])):
                label_index = df_outliers.index[i]
                df2.loc[label_index, column] = df2[column].median()
    return df2

final_df_classif = replace_outliers(df_classif, final_df_classif, drop_classif)
final_df_reg = replace_outliers(df_reg, final_df_reg, drop_reg)


# NORMALISATION

from sklearn.preprocessing import MinMaxScaler

def normalize(df, x):
    result = df.copy()
    for feature_name in df.columns:
        if (x == 'classification' and feature_name not in drop_classif) or (x == 'regression' and feature_name not in drop_reg):
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

normalised_df_classif = normalize(final_df_classif,'classification')
normalised_df_reg = normalize(final_df_reg,'regression')


# FEATURE SELECTION
# Classification

#separate target variable for classification (PCR)
x_classif = normalised_df_classif.drop(labels=['pCR (outcome)'], axis=1)
y_classif = normalised_df_classif['pCR (outcome)']

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
    df_classif_final = pd.concat([df_classif_final, normalised_df_classif[feature]], axis=1)


# Regression

#separate target value for regression (RFS)
x_reg = normalised_df_reg.drop(labels=['RelapseFreeSurvival (outcome)'], axis=1)
y_reg = normalised_df_reg['RelapseFreeSurvival (outcome)']

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
    df_reg_final = pd.concat([df_reg_final, normalised_df_reg[feature]], axis=1)

print(df_reg.shape)
print(df_reg_final.shape)
# Split data
Regx_train, Regx_test, Regy_train, Regy_test = train_test_split(df_reg_final, y_reg, test_size=0.2, random_state = 42)



# Setting Evaluation Functions
n_folds = 5

# K-fold cross validation
def Regkf(ag, X, y, name):
    scores = cross_val_score(ag, X, y, cv=n_folds,scoring="neg_mean_squared_error")
    print('The {}-fold cross-validation mean squared error score for {} is {:.2f} '.format(n_folds, name, abs(scores.mean())))

# plot
def Regplot(y_test, y_pred, title):
    no = np.array(range(1, len(y_pred)+1))
    plt.plot(no, y_test, color='blue')
    plt.plot(no, y_pred, color = 'red')
    plt.ylabel('RFS')
    plt.xlabel('inputs')
    plt.legend(['RFS (test data)','RFS (predicted data)'], loc=0, frameon=True)
    plt.title(title)
    plt.show()


# Linear Regression
reg = LinearRegression()
reg.fit(Regx_train, Regy_train)
y_pred = reg.predict(Regx_test)
name = 'Linear Regression'
Regkf(reg,x_reg,y_reg, name)
Regplot(Regy_test, y_pred, name)


# SVM
svr = SVR(C=2, epsilon=0.5,gamma=0.25)
svr.fit(Regx_train, Regy_train)
Regy_pred = svr.predict(Regx_test)
name = 'SVM'
Regkf(svr,x_reg,y_reg, name)
Regplot(Regy_test, y_pred, name)

# Settings
svr_cs = 2 ** np.arange(-5, 11, dtype=float)  # Candidates of C
svr_epsilons = 2 ** np.arange(-10, 1, dtype=float)  # Candidates of epsilon
svr_gammas = 2 ** np.arange(-20, 11, dtype=float)  # Candidates of gamma
fold_number = 5  # "fold_number"-fold cross-validation
number_of_training_samples = 1000
number_of_test_samples = 1000


import time
from sklearn.model_selection import train_test_split, GridSearchCV
# Measure time in hyperparameter optimization
start_time = time.time()

# Optimize gamma by maximizing variance in Gram matrix
numpy_autoscaled_Xtrain = np.array(Regx_train)
variance_of_gram_matrix = list()
for svr_gamma in svr_gammas:
    gram_matrix = np.exp(
        -svr_gamma * ((numpy_autoscaled_Xtrain[:, np.newaxis] - numpy_autoscaled_Xtrain).astype(float) ** 2).sum(axis=2))
    variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
optimal_svr_gamma = svr_gammas[np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]

# Optimize epsilon with cross-validation
svr_model_in_cv = GridSearchCV(SVR(kernel='rbf', C=3, gamma=optimal_svr_gamma), {'epsilon': svr_epsilons},
                               cv=fold_number,scoring="neg_mean_squared_error")
svr_model_in_cv.fit(Regx_train, Regy_train)
optimal_svr_epsilon = svr_model_in_cv.best_params_['epsilon']

# Optimize C with cross-validation
svr_model_in_cv = GridSearchCV(SVR(kernel='rbf', epsilon=optimal_svr_epsilon, gamma=optimal_svr_gamma),
                               {'C': svr_cs}, cv=fold_number,scoring="neg_mean_squared_error")
svr_model_in_cv.fit(Regx_train, Regy_train)
optimal_svr_c = svr_model_in_cv.best_params_['C']

# Optimize gamma with cross-validation (optional)
svr_model_in_cv = GridSearchCV(SVR(kernel='rbf', epsilon=optimal_svr_epsilon, C=optimal_svr_c),
                               {'gamma': svr_gammas}, cv=fold_number,scoring="neg_mean_squared_error")
svr_model_in_cv.fit(Regx_train, Regy_train)
optimal_svr_gamma = svr_model_in_cv.best_params_['gamma']

# Check time in hyperparameter optimization
elapsed_time = time.time() - start_time
print("Elapsed time in hyperparameter optimization: {0} [sec]".format(elapsed_time))

# Check optimized hyperparameters
print("C: {0}, Epsion: {1}, Gamma: {2}".format(optimal_svr_c, optimal_svr_epsilon, optimal_svr_gamma))




plt.show()