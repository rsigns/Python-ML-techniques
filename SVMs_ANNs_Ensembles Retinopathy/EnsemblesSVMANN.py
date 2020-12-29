#!/usr/bin/env python
# coding: utf-8

# # SVMs, Neural Nets and Ensembles
# 
# This notebook implement SVMs, Neural Nets, and Ensembling methods to classify patients as either having or not having diabetic retinopathy. You can find additional details about the dataset [here](http://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set)

# In[12]:


import warnings
warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


# Read the data from csv file
col_names = []
for i in range(20):
    if i == 0:
        col_names.append('quality')
    if i == 1:
        col_names.append('prescreen')
    if i >= 2 and i <= 7:
        col_names.append('ma' + str(i))
    if i >= 8 and i <= 15:
        col_names.append('exudate' + str(i))
    if i == 16:
        col_names.append('euDist')
    if i == 17:
        col_names.append('diameter')
    if i == 18:
        col_names.append('amfm_class')
    if i == 19:
        col_names.append('label')

data = pd.read_csv("messidor_features.txt", names = col_names)
print(data.shape)
data.head(10)


# ### 1. Data prep

# We will use scaled data from here on since SVMs and NNs both need scaled data. Even though ensembles generally do not, it will be easier to just use the same data since no harm will come of it 
# 
# `sklearn.preprocessing.StandardScaler` will be used to standardize the dataset’s features (mean = 0 and variance = 1)

# In[4]:


dataLabels = data.pop('label')
print(dataLabels)
newScaler = StandardScaler()
newScaler.fit(data)
stdData = newScaler.transform(data)


# ### 2. Support Vector Machines (SVM)

# `sklearn.svm.SVC` (Support Vector Classifier) will be used to classify the data. A 5 fold cross validation grid search will be used to find the best kernel choice.

# In[5]:


grid_params = {'kernel': ['linear','rbf','poly','sigmoid']}

supportClassifier = SVC()
searcher = GridSearchCV(supportClassifier,grid_params, scoring ='accuracy', cv=5)
searcher.fit(stdData,dataLabels)
print(searcher.best_params_)
print("accuracy: "+str(searcher.best_score_))


# We will use the previous optimal kernel to perform another 5 fold cross validation grid search in order to find 'C', the regularization constant of our linear kernel. Because it is a constant, we will use values between 10 and 250 with "jumps" going up by 10

# In[6]:


newGridParams={'C': range(10,251,10)}
newSVC = SVC(kernel = "linear")
CGrid = GridSearchCV(newSVC, newGridParams, scoring = "accuracy", cv = 5)
CGrid.fit(stdData, dataLabels)
print(CGrid.best_params_)
print("accuracy: "+str(CGrid.best_score_))


# ### 3. Neural Networks (NN)

# `sklearn.neural_network.MLPClassifier` will be used to train a multi layer perceptron with a single layer.
# Grid search will be used to find the most accurate combination of hidden layer size and activation function.
# It will be wrapped in a cross_val_score (i.e. nested cross validation) in order to ascertain the generalization error of the model.

# In[7]:


percep = MLPClassifier()
MLPparams={'hidden_layer_sizes': [(10,),(20,),(30,),(40,),(50,),(60,)], 'activation':['logistic','tanh','relu']}
mlpGridSearch = GridSearchCV(percep, MLPparams, scoring ='accuracy',cv=5)
mlpGridSearch.fit(stdData,dataLabels)
newCVScore = cross_val_score(mlpGridSearch,stdData,dataLabels,cv=5)

print("best params:")
print(mlpGridSearch.best_params_)
print("best score:")
print(mlpGridSearch.best_score_)
print(newCVScore)
print("mean acc:")
print(newCVScore.mean())


# ### 4. Ensemble Classifiers
# 
# Ensemble classifiers combine the predictions of multiple base estimators to improve the accuracy of the predictions. One of the key assumptions that ensemble classifiers make is that the base estimators are built independently (so they are diverse).

# **A. Random Forests**
# 
# We will use `sklearn.ensemble.RandomForestClassifier` to classify the data. As per usual a `GridSearchCV` will be used to tune the hyperparameters to get the best results. 
# 
# A cross_val_score with 5-fold CV will again be used to report the generalization error of the model.

# In[8]:


forest = RandomForestClassifier()
forest_params = {'max_depth':range(35,56),'min_samples_leaf':[8,10,12],'max_features':['sqrt','log2']}
forestGrid = GridSearchCV(forest,forest_params,scoring='accuracy',cv=5)
forestGrid.fit(stdData,dataLabels)
forestCVS = cross_val_score(forestGrid,stdData,dataLabels,cv=5)
print(forestGrid.best_params_)
print("accuracy: "+str(forestCVS.mean()))


# **B. AdaBoost**
# 
# Random Forests are a kind of averaging ensemble classifier, where the driving principle is to build several estimators independently and then to average their predictions (by taking a vote). In contrast, there is another class of training ensemble classifiers called *boosting* methods. The boosting algorithm will train further estimators based on the harder to classify examples in order to reduce bias (and therefore increase the predictive value)
# 
# `sklearn.ensemble.AdaBoostClassifier` will be the classifier used on the data. By default, `AdaBoostClassifier` uses decision trees as the base classifiers (but this can be changed), 150 will be used here with a 5 fold cross validation to get an idea of the accuracy

# In[9]:


# your code goes here
booster = AdaBoostClassifier(n_estimators=150)
boosterCVS = cross_val_score(booster,stdData,dataLabels,cv=5)
print("accuracy: "+str(boosterCVS.mean()))


# ### 5. Deploying a final model

# The code below will use the `pickle` package to save the neural net model. This will save the model to a file called finalized_model.sav in the current working directory. 

# In[10]:


import pickle

percep = MLPClassifier()
MLPparams={'hidden_layer_sizes': [(10,),(20,),(30,),(40,),(50,),(60,)], 'activation':['logistic','tanh','relu']}
mlpGridSearch = GridSearchCV(percep, MLPparams, scoring ='accuracy',cv=5)
mlpGridSearch.fit(stdData,dataLabels)

print(mlpGridSearch.best_params_)

final_model = mlpGridSearch

filename = 'finalized_model.sav'
pickle.dump(final_model, open(filename, 'wb'))


# Sample code using the saved neural net classifier and using it to classify a new record 

# In[11]:


record = [[ 0.05905386, 0.2982129, 0.68613149, 0.75078865, 0.87119216, 0.88615694,
  0.93600623, 0.98369184, -0.47426472, -0.57642756, -0.53115361, -0.42789774,
 -0.21907738, -0.20090532, -0.21496782, -0.2080998, 0.06692373, -2.81681183,
 -0.7117194 ]]

 
# loads the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

loaded_model.predict(record)
if(loaded_model.predict(record)[0]==[1]):
    print("Positive for disease")
else:
    print("Negative for disease")

