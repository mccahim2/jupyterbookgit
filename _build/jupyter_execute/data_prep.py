#!/usr/bin/env python
# coding: utf-8

# # Data preparation for clustering

# For clustering using the k-means method we will need to follow a few steps:
# * **Normalise the data:** The process of converting an actual range of values into a standard range of values
# * **Find a Similarity Measure:**
# * **Interpret Results:**

# In[1]:


import pandas as pd
import numpy as np


# Importing cleaned data

# In[2]:


all_data = pd.read_csv('data/all_data.csv')
all_data=all_data.drop(["Unnamed: 0"], axis=1) # Drop Unnamed: 0 column
all_data.head()


# get_dummies is a data manipulation function used to convert categorical data into indicator variables
# 
# :::{note}
# Machine learning models require all input and output variables to be numeric
# :::

# In[3]:


all_data = pd.get_dummies(all_data, columns=["Study_Type"])


# In[4]:


all_data.head()


# ### Normalizing data

# As for K-means, often it is not sufficient to normalize only mean. One normalizes data equalizing variance along different features as K-means is sensitive to variance in data, and features with larger variance have more emphasis on result. So for K-means, I would recommend using StandardScaler for data preprocessing.
# 
# Don't forget also that k-means results are sensitive to the order of observations, and it is worth to run algorithm several times, shuffling data in between, averaging resulting clusters and running final evaluations with those averaged clusters centers as starting points.

# Normalizing data is a common step for data preparation for clustering

# In[5]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
all_data[['Total', 'Amount_won', 'Amount_lost', '1', '2', '3', '4']] = scaler.fit_transform(all_data[['Total', 'Amount_won', 'Amount_lost', '1', '2', '3', '4']])


# In[6]:


all_data.head()


# In[7]:


normalise_data = all_data
normalise_data.head(1)


# ### Elbow Method

# In[8]:


from sklearn.cluster import KMeans
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(normalise_data)
    distortions.append(kmeanModel.inertia_)


# In[9]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[ ]:





# In[ ]:




