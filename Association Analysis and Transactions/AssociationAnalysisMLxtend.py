#!/usr/bin/env python
# coding: utf-8

# # Association Analysis with the apriori algorithm
# 
# The data belongs to a bakery called "The Bread Basket", located in the historic center of Edinburgh. The dataset contains more than 9000 transactions from the bakery. The file contains the following columns:
# 
# - Date. Categorical variable that tells us the date of the transactions (YYYY-MM-DD format). The column includes dates from 30/10/2016 to 09/04/2017.
# 
# - Time. Categorical variable that tells us the time of the transactions (HH:MM:SS format).
# 
# - Transaction. Quantitative variable that allows us to differentiate the transactions. The rows that share the same value in this field belong to the same transaction.
# 
# - Item. Categorical variable with the products.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Reading the dataset from file
def load_dataset(file_name):
    with open(file_name, 'r') as f:
        content = f.readlines()[1:]
    transactions = []
    prev_tid = -1
    for t in content:
        t = t.strip().split(',')[-2:]
        tid = t[0]
        item = t[1]
        if prev_tid != tid:
            prev_tid = tid
            transactions.append([item])
        else:
            transactions[-1].append(item)
    return transactions


# In[4]:


dataset = load_dataset('BreadBasket_DMS.csv')
#dataset is a 2D list
print("Num transactions:", len(dataset))
#line below prints the first 10 transactions
dataset[:10]


# The code below displays the top 10 selling items by frequency in a bar chart

# In[5]:


from collections import *
from itertools import *

#Counter object below will "flatten" the dataset and display it in frequent order
c = Counter(chain.from_iterable(dataset)).most_common(10)
c = list(c)

transactionValues=[]
items = []

#lines below add top 10 frequent item names to items[] and their values to transactionValues[]
for i in c:
    items.append(i[0])
    transactionValues.append(i[1])

#graphing code
y_pos = np.arange(len(items))
y_pos = y_pos[::-1]
plt.barh(y_pos, transactionValues, align='center', alpha=0.5)
plt.yticks(y_pos, items)
print(c)
print(items)
print(transactionValues)
print(y_pos)


# `mlxtend.preprocessing.TransactionEncoder` will be used to transform `dataset` into an array format suitable for the `mlxtend` library. 
# 
# `TransactionEncoder` learns unique items from the dataset and transforms each transaction into a one-hot encoded boolean numpy array. For example, the resulting encoded dataset will be represented by something like this, where each row is a transaction:
# <img src="table.png">

# In[6]:


newEncoder = TransactionEncoder()
newEncoder.fit(dataset)
nearray = newEncoder.transform(dataset)
#line below just sample line that reverses the transform 
#newEncoder.inverse_transform(nearray[:5])


# Code below will convert the numpy array to a pandas dataframe

# In[7]:


df = pd.DataFrame(nearray, columns=newEncoder.columns_)
print(df.head(1))


# `mlxtend.frequent_patterns.apriori` is used to generate the frequent itemsets with minimum support of 1%, also displayed is their support values

# In[8]:


x=apriori(df, min_support=0.01)
x


# Now all frequent maximal datasets and their support values will be displayed below

# In[11]:


nonMax=[]
for index, row in x.iterrows():
    for index, row2 in x.iterrows():
        if((row['itemsets'].issubset(row2['itemsets'])) and (row['itemsets'])!=(row2['itemsets'])):
            nonMax.append(row['itemsets'])
unique=list(set(nonMax))

df2 = x.copy(deep=True)
df2=df2[~df2.itemsets.isin(unique)]
df2


# `mlxtend.frequent_patterns.association_rules` will now be used to calculate rules with a confidence level of 0.25 for the frequent itemsets generated earlier

# In[12]:


from mlxtend.frequent_patterns import association_rules

association_rules(x, metric='confidence', min_threshold=0.25)


# The code below will generate the frequent itemsets with minimum support of 0.5% and plot the number of rules generated with respect to the confidence threshold by varying its value between 0 and 1 with increments of 0.1 in order to get an idea of where to select our cutoff to balance number of rules with confidence values

# In[13]:


def newRange(x, y, jump):
  while x < y:
    yield x
    x += jump

ruleList = []
confidenceThreshold = []
for i in newRange(0, 1, .1):
    num = 0
    num = len(association_rules(x, metric="confidence", min_threshold=i))
    ruleList.append(num)
    confidenceThreshold.append(i)
ruleDF = pd.DataFrame({'Rules':ruleList, 'Confidence':confidenceThreshold})
ruleDF.plot(x="Confidence", y="Rules", kind="scatter")


# Because of the "drop" at a confidence level of 0.5, going past this may not be very useful for us given that the number of rules returned drops to a very low level and so the "predictive value" of the algorithm will be small. 0.5 is a good balance point

# In[14]:


association_rules(x, metric='confidence', min_threshold=0.50)
# An interesting note about the rules is that they all have a single consequent in common, that being Coffee in this case

