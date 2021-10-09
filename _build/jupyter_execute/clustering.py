#!/usr/bin/env python
# coding: utf-8

# # K-Means Clustering

# In[1]:


from sklearn.cluster import KMeans


# Elbow method is used to find the best value to use for k

# In[2]:


from sklearn.cluster import KMeans
kmeans = KMeans(3)
clusters = kmeans.fit_predict(customers_norm)
labels = pd.DataFrame(clusters)
labeledCustomers = pd.concat((customers,labels),axis=1)
labeledCustomers = labeledCustomers.rename({0:'labels'},axis=1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




