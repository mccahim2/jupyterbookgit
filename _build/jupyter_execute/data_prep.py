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

df = pd.read_csv('data/all_data.csv')
df=df.drop(["Unnamed: 0"], axis=1) # Drop Unnamed: 0 column


# Changing categorical data entries can be done by One-hot encoding.
# 
# **One-hot encoding** is a method during data preparation for converting categorical data variables so they can be provided ot machine learning algorithims to improve predictions
# 
# get_dummies is a data manipulation function used to convert categorical data into indicator variables
# 
# :::{note}
# Machine learning models require all input and output variables to be numeric
# :::
# 
# It is impossible to do k-means clustering on a 

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


# **Exporting Data to CSV file**

# In[10]:


normalise_data.to_csv('data/normalise.csv')


# Show why I normalised vs standarised

# ### Standardising data Test

# In[11]:


df = pd.get_dummies(df, columns=["Study_Type"])


# In[12]:


from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(df)


# In[13]:


X_scaled = scaler.transform(df)


# In[14]:


X_scaled


# In[15]:


df.head(1)


# In[16]:


sd = pd.DataFrame(X_scaled, columns=['Total', 'No_participants', 'Amount_won', 'Amount_won', '1', '2', '3', '4', 'Study_Type_Fridberg', 'Study_Type_Horstmann', 'Study_Type_Kjome', 'Study_Type_Maia', 'Study_Type_Premkumar', 'Study_Type_Steingroever2011', 'Study_Type_SteingroverInPrep', 'Study_Type_Wetzels', 'Study_Type_Wood', 'Study_Type_Worthy'])
sd


# In[17]:


from sklearn.cluster import KMeans
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(sd)
    distortions.append(kmeanModel.inertia_)


# In[18]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# ### Conclusion

# 
