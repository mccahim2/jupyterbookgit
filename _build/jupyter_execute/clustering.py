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
get_ipython().run_line_magic('matplotlib', 'inline')


# Read in normalised data

# In[2]:


df = pd.read_csv('data/normalise.csv')
df=df.drop(["Unnamed: 0"], axis=1) # Drop Unnamed: 0 column
df.head()


# In[3]:


df.head()


# The first step of the clustering process is to determine a value for K. This can be done by:<br />
# * **The Elbow Method**
# * **Silhouette score**
# * **SilhouetteVisualizer**
# 
# Three methods will be used to find the best value for K to use for clustering

# ### Methods for finding K Value

# In[4]:


# Elbow(distortion and inertia) method and silhouette method


# When the distortions are plotted and the plot looks like an arm then the [“elbow”](https://predictivehacks.com/k-means-elbow-method-code-for-python/)(the point of inflection on the curve) is the best value of k.

# The formula for the Elbow method can be seen here: <br />
# $$
#     SSE = \sum_{i=1}^{N} {(y_i - ŷ_i)^2}
# $$

# In[5]:


from sklearn.cluster import KMeans
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df)
    distortions.append(kmeanModel.inertia_)


# In[6]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# silhouette score is used to to measure the degree of seperation between clusters
# 
# The formula for the Elbow method can be seen here: <br />
# $$
#     s_{i} = \frac{b_{i} - a_{i}}{max(b_{i}, a_{i})}
# $$

# In[7]:


for n in range(2, 11):
    km = KMeans(n_clusters=n)
#
# Fit the KMeans model
# Have to pick subset of columns as Study column is in string format
    km.fit_predict(df)
#
# Calculate Silhoutte Score
#
    score = silhouette_score(df, km.labels_, metric='euclidean')
#
# Print the score
#
    print('N = ' + str(n) + ' Silhouette Score: %.3f' % score)


# Perform Comparative Analysis to Determine Best Value of K using Silhouette Plot

# In[8]:


from yellowbrick.cluster import SilhouetteVisualizer

fig, ax = plt.subplots(2, 2, figsize=(15,8))
for i in [2, 3, 4, 5]:
    '''
    Create KMeans instance for different number of clusters
    '''
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 2)
    '''
    Create SilhouetteVisualizer instance with KMeans instance
    Fit the visualizer
    '''
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(df)


# From the various methods above the optimal value for K to bus used for clustering will be 2

# A scatter matrix is an array of scatter plots used to assess the relatsionship between many variables at once

# In[9]:


pd.plotting.scatter_matrix(df, figsize=(50,50), hist_kwds=dict(bins=50), cmap="Set1")
plt.show()


# Looking at this initial scatter matrix it is clear to see there is a lot of noise.
# 
# I will drop the study name columns to make findings more clear

# In[10]:


df.drop(columns=['No_participants', 'Study_Type_Fridberg', 'Study_Type_Horstmann', 'Study_Type_Kjome', 'Study_Type_Maia', 'Study_Type_Premkumar', 'Study_Type_Steingroever2011', 'Study_Type_SteingroverInPrep', 'Study_Type_Wetzels', 'Study_Type_Wood', 'Study_Type_Worthy'], inplace=True)


# In[11]:


df.head()


# In[12]:


pd.plotting.scatter_matrix(df, figsize=(15,15), hist_kwds=dict(bins=50), cmap="Set1")
plt.show()


# :::{note}
# This correlation matrix is used to find highly correlated variables
# :::

# In[13]:


plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


# From my initial data we saw that of all the decks picked, deck no 2 was picked the most. It might be interesting to do some clustering on deck no2 for Amount_won, Amount_lost and totals compared to the other decks

# ### Main Cluster Analysis

# :::{note}
# kmeans.cluster_centers_ returns the coordinates of the centers 
# :::

# In[14]:


KMeans(n_clusters=2).fit(df[["Total", "2"]]).cluster_centers_


# In[15]:


kmeans_margin_standard = KMeans(n_clusters=2).fit(df[["Total", "2"]])
centroids_betas_standard = kmeans_margin_standard.cluster_centers_


# In[16]:


plt.figure(figsize=(16,8))
plt.scatter(df['Total'], df['2'], c= kmeans_margin_standard.labels_, cmap = "Set1", alpha=0.5)
plt.scatter(centroids_betas_standard[:, 0], centroids_betas_standard[:, 1], c='blue', marker='x')
plt.title('K-Means cluster for all Totals: Selection - Deck 2')
plt.xlabel('Total')
plt.ylabel('Deck 2 Selected')
plt.show()


# Curse of Dimensionality

# In[17]:


kmeans.inertia_


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




