import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import warnings

# warnings.filterwarnings('ignore')

"""

Regression

"""

# import dataset
df = pd.read_csv('trainDataset.csv')


# Preprocessing
scaler = StandardScaler()

df = df.drop(['ID'], axis=1)
x = df.drop(['RelapseFreeSurvival (outcome)'], axis=1)
y = df['RelapseFreeSurvival (outcome)']

x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns, index=x.index)


# Split data
Regx_train, Regx_test, Regy_train, Regy_test = train_test_split(x, y, test_size=0.2, random_state = 42)


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
    plt.ylabel('RelapseFreeSurvival')
    plt.xlabel('inputs')
    plt.legend(['RelapseFreeSurvival (actual data)','RelapseFreeSurvival (predicted data)'], loc=0, frameon=True)
    plt.title(title)
    plt.show()


# Linear Regression
reg = LinearRegression()
reg.fit(Regx_train, Regy_train)
y_pred = reg.predict(Regx_test)
name = 'Linear Regression'
Regkf(reg,x,y, name)
Regplot(Regy_test, y_pred, name)


# SVM
svr = SVR(C=2, epsilon=9)
svr.fit(Regx_train, Regy_train)
Regy_pred = svr.predict(Regx_test)
name = 'SVM'
Regkf(svr,x,y, name)
Regplot(Regy_test, y_pred, name)


# Decision tree
dtr = DecisionTreeRegressor(max_features= 'auto',
                           max_leaf_nodes= 10,
                           splitter='random',
                           random_state=0)
dtr.fit(Regx_train, Regy_train)
y_pred = dtr.predict(Regx_test)
name = 'Decision Tree'
Regkf(dtr,x,y, name)
Regplot(Regy_test, y_pred, name)


# Neural network
mlpr = MLPRegressor(alpha= 0.002,
                   hidden_layer_sizes = (50,50,50),
                   random_state = 2,
                   learning_rate = 'invscaling',
                   solver = 'sgd')
mlpr.fit(Regx_train, Regy_train)
y_pred = mlpr.predict(Regx_test)
name = 'Neural Network'
Regkf(mlpr,x,y, name)
Regplot(Regy_test, y_pred, name)




