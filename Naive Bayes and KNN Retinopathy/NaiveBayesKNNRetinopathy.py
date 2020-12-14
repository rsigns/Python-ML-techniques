#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes and KNN using scikit-learn
# 
# Note: The dataset used here is the same diabetic retinopathy set [here](http://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set).

# In[4]:


import warnings
warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


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
classLabel = data.pop("label")


# ## Part 1: Naive Bayes Classifier

# The code below uses a naive bayes classifier within a 10-fold cross validation algorithm. Accuracy here is presented as an average of the results of the CV.

# In[7]:


naiveBayesClassifier = GaussianNB(priors=None)
NBCrossVal = cross_val_score(naiveBayesClassifier,data,classLabel,cv=10)
print("Accuracy: "+str(NBCrossVal.mean()))


# `cross_val_predict` will return the predictions for every record when it was within the test set using the model. The first printed `NBMatrix` object will be a confusion matrix for these predictions, with the layout as follows for the array indexes: 
# * (0,0) = true negatives (value of 501)
# * (0,1) = false positives (value of 39)
# * (1,0) = true positives (value of 420)
# * (1,1) = false negatives (value of 191)
# 
# Below this matrix is a table of error measures for the predictions from the classifier based on this matrix, with the rows (0 or 1) being doesn't have retinopathy and does have retinopathy respectively.

# In[8]:


pred = cross_val_predict(naiveBayesClassifier,data,classLabel)
NBMatrix = confusion_matrix(classLabel,pred)
print(NBMatrix)
classRep = classification_report(classLabel,pred)
print(classRep)


# Below is an implementation of a ROC (Reciever Operating Characteristic) curve for the predictions made by the NB classifier to get good info on the specificity vs sensitivity tradeoff.

# Note:
# * FPR = false positive rate
# * TPR = true positive rate

# In[9]:


trainData, testData, trainLabels, testLabels =train_test_split(data, classLabel, test_size=0.2, train_size=0.8)
# training set data is at 0 index of array and test data is at 1 index of array
naiveBayesClassifier.fit(trainData, trainLabels)
probArray = naiveBayesClassifier.predict_proba(testData)
newROC = roc_curve(testLabels,probArray[:,1])
fpr, tpr = newROC[0], newROC[1]
print("Area under the curve: "+str(roc_auc_score(testLabels,probArray[:,1])))

plt.plot([0,1],[0,1],'k--') #plot the diagonal line
plt.plot(fpr, tpr, label='NB') #plot the ROC curve
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC Curve Naive Bayes')
plt.show()


# ## Part 2: K Nearest Neighbor (KNN) Classifier

# The dataset will be normalized (between 0 and 1) in order to use a KNN classifier, this is necessary because the usage of distance (usually Euclidean) between points with KNN classifiers leads to class imbalances if normalization is NOT used. `MinMaxScaler` is used to achieve this.

# In[10]:


scaler = MinMaxScaler()
scaler.fit(data)
scaledData = scaler.transform(data)
scaledData


# A `k = 5` classifier will be used within a 10-fold CV and some error measures will be printed in order to get an idea of the effectiveness of this classifier

# In[11]:


KNN = KNeighborsClassifier()
KNN.fit(scaledData, classLabel)
KNNpred = cross_val_predict(KNN,scaledData,classLabel, cv=10)
print(classification_report(classLabel,KNNpred))


# We will tune hyperparameters for `k` using grid search below:

# In[13]:


grid_params = {'n_neighbors': range(1, 31)}
optimalk = GridSearchCV(KNN, grid_params, cv=10, scoring='accuracy')
optimalk.fit(scaledData,classLabel)
print("Best Value of k: ", str(optimalk.best_params_['n_neighbors']))
print(optimalk.best_params_)


# Below are the new error measures for the new `k = 23` classifier. Note the improvement (albeit small) over the previous placeholder value of 5.

# In[14]:


newKNN = KNeighborsClassifier(n_neighbors = 23)
newKNN.fit(scaledData,classLabel)
newKNNpred = cross_val_predict(newKNN,scaledData,classLabel, cv = 10)
print(classification_report(classLabel,newKNNpred))


# Below, a nested 10 fold cross validation is used with our prior grid search object. This will prevent overfitting, and will give us a good idea of the generalization error of our model. The arrays+measures printed below are in the following order of scoring categories:
# * Accuracy
# * Recall
# * Precision

# In[34]:


#yaga = cross_val_predict(optimalk, scaledData, classLabel, cv = 10)
yaga = cross_val_score(optimalk, scaledData, classLabel, cv = 10)
yaga2 = cross_val_score(optimalk, scaledData, classLabel, cv = 10, scoring='recall')
yaga3 = cross_val_score(optimalk, scaledData, classLabel, cv = 10, scoring='precision')

#print(classification_report(classLabel,yaga))
# accuracy
print(yaga)
print(yaga.mean())

# recall
print(yaga2)
print(yaga2.mean())

#precision
print(yaga3)
print(yaga3.mean())


# Below, the code is using a PCA transformation on the data within a cross validation. However, the difference is that a pipeline must be used in order to apply this properly. The code as set up will first apply the PCA to the training data, and THEN the test data will be transformed into that space. The KNN classifier will be applied right after the PCA within the folds on both the train AND the test data. This is all done in order to avoid the curse of dimensionality (which can also be done with feature engineering/selection).

# In[29]:


from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pca = PCA()
knn = KNeighborsClassifier()
pipe = Pipeline(steps=[('pca', pca), ('knn', knn)])

#Parameters of pipelines are set using ‘__’ separated parameter names:
param_grid = {
    'pca__n_components': list(range(1, 19)),
    'knn__n_neighbors': list(range(1, 30)),
}
#Pipeline is passed into the GridSearchCV below for 5-fold 
gridz = GridSearchCV(pipe, param_grid=param_grid, scoring="accuracy", cv = 5)
gridz.fit(scaledData,classLabel)
print(gridz.best_params_)
print(gridz.best_score_)


# As can be seen above, we are left with an accuracy of ~66% which can be significantly improved upon but is nearly 10% better than when we started with only a few basic tweaks.

# In[ ]:




