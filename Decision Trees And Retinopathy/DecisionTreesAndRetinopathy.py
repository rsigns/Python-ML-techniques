#!/usr/bin/env python
# coding: utf-8

# # Decision Trees using `scikit-learn`, `Pandas` and `NumPy`
# 
# In this code we'll attempt to classify patients as either having or not having diabetic retinopathy. For this task we'll be using the Diabetic Retinopathy data set, which contains 1151 instances and 20 attributes (with both continuous and categorical data). Dataset info [here](http://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set).

# In[13]:


import warnings
warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import time


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


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


# ### 1. Data preprocessing  & dimensionality reduction with PCA

# The first block of code below will split the labels of the data into their own data structure (in this case a series/1D array). Next the second block will "standardize" the data such that the features are transformed to have a mean of 0 and a variance of 1. We are doing this as they will be required for PCA later.

# In[16]:


classLabelCol=data.pop("label")


# In[18]:


# StandardScaler "standardizes" the data and returns a NumPy array of the now standardized records
scaler = StandardScaler()
scaledFeatures = (scaler.fit_transform(data))
scaledFeatures


# The code below uses sklearn's `train_test_split` object to split the data into an 80/20 train/test split so we can evaluate the model performance.

# In[28]:


''' Data from train_test_split is returned as a 2D array with the following columns: training set of features,
 testing set of features, training set of labels, testing set of labels  
'''
splitData=train_test_split(scaledFeatures,classLabelCol, test_size=0.2, train_size=0.8)
print("Size of training data set: "+str(len(splitData[0])))
print("Size of test data set: "+str(len(splitData[1])))


# We will be performing PCA in order to reduce the dimensionality of the dataset below. The `pca.explained_variance_ratio` attribute will indicate the percentage of the 95% variance we want to retain is explained by the new dimensional features after the transformation. We will transform the test set into the same dimensional space afterwards.

# In[29]:


# The line below retains 95% variance for the PCA
pca = PCA(n_components=0.95)
pca.fit(splitData[0])
print("The below array shows the variance explained by the new post-PCA columns:\n")
print(pca.explained_variance_ratio_)
reducedTrainingData=pca.transform(splitData[0])
reducedTestingData=pca.transform(splitData[1])


# ### 2. Training Decision Trees in `scikit-learn`

# We will use `DecisionTreeClassifier` with as our split criterion being entropy (default is gini). We will use our newly transformed data that was outputted after using PCA.

# In[30]:


decTree = DecisionTreeClassifier(criterion="entropy")
# Remember that splitData[2] gave us the labels for the training set
decTreeEstimator=decTree.fit(reducedTrainingData,splitData[2])


# Now that we have trained a basic decision tree on a basic train section from a train/test split, we will go ahead and attempt to classify the test data and compare our predictions to the actual labels from that test set.

# In[34]:


estimatedTestClasses=decTree.predict(reducedTestingData)
classifierAccuracy=accuracy_score(splitData[3],estimatedTestClasses)
print("Accuracy of classifier on test data: "+str(classifierAccuracy))


# Because of the numerous hyperparameters used in the decision tree classifier model, we can play around with some values to see how they affect accuracy.

# In[38]:


''' As one can see in the accuracy score below, a few changes can affect the accuracy measure by 5% with some very 
minor tweaks to hyperparameters. Note that this does not mean the model is necessarily well generalizable!
'''
newClassifier = DecisionTreeClassifier(criterion = "gini", max_depth=3, min_samples_leaf=3)
newClassifier.fit(reducedTrainingData, splitData[2])
print(accuracy_score(splitData[3], newClassifier.predict(reducedTestingData)))


# ### 3. Using K-fold Cross Validation
# 
# The 'holdout' method as mentioned previously is not enough to measure the generalized accuracy very well. We will use 'cross validation' in order to do so in a more robust way.

# We will use `sklearn.model_selection.cross_val_score` to perform 10-fold cross validation on our decision tree and report the accuracy. Note the usage of the standardized pre-PCA transformation data. We will use PCA with k-fold cross validation later in another notebook.

# In[40]:


oldDataTree = DecisionTreeClassifier(criterion="entropy")
oldDataCrossVal = cross_val_score(oldDataTree,scaledFeatures,classLabelCol,cv=10)
# The below line will average the accuracy from all the folds in order to get a better idea of accuracy
oldDataCrossVal.mean()


# We want to tune our model's hyperparameters in order to minimize potential overfitting of the model on our test data. To do so we will use the `GridSearchCV` object provided by sklearn. It will calculate the hyperparameter permutation with the best generalizable accuracy from our given data.

# In[46]:


# Hyperparameters to test are give in the dict below
unknownValsDict = {"max_depth":range(5,21), "max_features":range(5,16), "min_samples_leaf":range(5,21)}
gridS = GridSearchCV(oldDataTree, unknownValsDict, scoring = 'accuracy', cv=5)
gridS.fit(scaledFeatures,classLabelCol)
print(gridS.best_params_)


# In[47]:


# new code block to test accuracy of new hyperparameters
bestTree = DecisionTreeClassifier(criterion="entropy", max_depth=7, max_features=14, min_samples_leaf=16)
newCrossVal = cross_val_score(bestTree,scaledFeatures,classLabelCol,cv=5)
newCrossVal.mean()


# The code below will initiate a nested cross validation loop.
# 
# What this does is: the `cross_val_score` splits the data in to train and test sets for the first fold, and it passes the train set into `GridSearchCV`. `GridSearchCV` then splits that set into train and validation sets for k number of folds (the inner CV loop). The hyper-parameters for which the average score over all inner iterations is best, is reported as the `best_params_`, `best_score_`, and `best_estimator_`(best decision tree). This best decision tree is then evaluated with the test set from the `cross_val_score` (the outer CV loop). And this whole thing is repeated for the remaining k folds of the `cross_val_score` (the outer CV loop). 
# 
# We will print out the final accuracy of the model at the end.

# In[48]:


# your code goes here
finalCrossVal = cross_val_score(gridS, scaledFeatures, classLabelCol, cv=5)
finalCrossVal.mean()


# Our accuracy rate isn't very good. We wouldn't want to use this model in the real world to actually diagnose patients due to it onl being accurate ~60% of the time. To improve this accuracy, other algorithms like boosting or bootstrapping could be useful. 

# In[ ]:




