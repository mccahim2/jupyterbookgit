#!/usr/bin/env python
# coding: utf-8

# # K-Means Clustering

# ### Table of contents

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
get_ipython().run_line_magic('matplotlib', 'inline')


# Read in normalised data

# In[2]:


df = pd.read_csv('data/normalise.csv')
df=df.drop(["Unnamed: 0"], axis=1) # Drop Unnamed: 0 column
df.head()


# In[3]:


df.head()


# :::{note}
# This correlation matrix is used to find highly correlated variables
# :::

# In[4]:


plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


# ### PCA

# To avoid the curse of dimensionality. The curse of dimensionality is the problem caused by the exponential increase in volume associated with adding extra dimensions to Euclidean space. To avoid this we can perform Principal Component analysis (PCA) to our data frame. This will leave use with an array of components.
# 
# PCA is believed to imporve the performance for cluster analysis
# 
# For the purpose of this assignment I will select 2 features to be used for clustering

# In[5]:


pca = PCA(2)
df_tester = pca.fit_transform(df)


# In[6]:


df_tester


# The newly optained PCA scores will be incorporated in the the K-means algorithm

# The next step of the clustering process is to determine a value for K. This can be done by:<br />
# * **The Elbow Method**
# * **Silhouette score**
# 
# These two methods will be used to find the best value for K to use for clustering

# ### Methods for finding K Value

# In[7]:


# Elbow(distortion and inertia) method and silhouette method


# When the distortions are plotted and the plot looks like an arm then the [“elbow”](https://predictivehacks.com/k-means-elbow-method-code-for-python/)(the point of inflection on the curve) is the best value of k.

# The formula for the Elbow method can be seen here: <br />
# $$
#     Sum Squared Errors = \sum_{i=1}^{N} {(y_i - ŷ_i)^2}
# $$

# The formula for the Silhouette Score method can be seen here: <br />
# $$
#     s_{i} = \frac{b_{i} - a_{i}}{max(b_{i}, a_{i})}
# $$

# In[8]:


wcss=[]
for i in range(1,21):
    k_means_pca=KMeans(n_clusters=i, init="k-means++", random_state=42)
    k_means_pca.fit(df_tester)
    wcss.append(k_means_pca.inertia_)


# In[9]:


plt.figure(figsize=(10,8))
plt.plot(range(1,21), wcss, marker="o", linestyle="--")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("K-means with PCA Clustering")
plt.show()


# In[10]:


for n in range(2, 21):
    km = KMeans(n_clusters=n)
    km.fit_predict(df_tester)
    score = silhouette_score(df, km.labels_, metric='euclidean')
    print('N = ' + str(n) + ' Silhouette Score: %.3f' % score)


# From looking at the Elbow blot and 

# 
# The formula for the Elbow method can be seen here: <br />
# $$
#     s_{i} = \frac{b_{i} - a_{i}}{max(b_{i}, a_{i})}
# $$

# From the various methods above the optimal value for K to bus used for clustering will be 4

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 
