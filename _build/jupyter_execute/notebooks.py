#!/usr/bin/env python
# coding: utf-8

# # Data Exploration

# The first step of the data exploration phase is to import any necessary packages

# In[1]:


import pandas as pd
import numpy as np


# For the purpose of this assignment their are several different data sets to be read in and explored.<br />
# There are 12 data sets to read in.
# They are as follows:
# * **Wins:** These data sets show how much money participants win during each part of the trial
# * **Losses:** These data sets show how much money participants lose during each part of the trial
# * **Choice:** These data sets indicate what deck participants chose during each part of the trial
# * **Index:** These data sets contain the name of the first author of the study that reports the data of the corresponding participant.

# ### Importing data sets

# In[2]:


win_95 = pd.read_csv('data/wi_95.csv')
win_100 = pd.read_csv('data/wi_100.csv')
win_150 = pd.read_csv('data/wi_150.csv')


# In[3]:


loss_95 = pd.read_csv('data/lo_95.csv')
loss_100 = pd.read_csv('data/lo_100.csv')
loss_150 = pd.read_csv('data/lo_150.csv')


# In[4]:


choice_95 = pd.read_csv('data/choice_95.csv')
choice_100 = pd.read_csv('data/choice_100.csv')
choice_150 = pd.read_csv('data/choice_150.csv')


# In[5]:


index_95 = pd.read_csv('data/index_95.csv')
index_100 = pd.read_csv('data/index_100.csv')
index_150 = pd.read_csv('data/index_150.csv')


# The next process is to clean the above data

# Data cleaning is a very important part of the data exploration process as it will identify and remove errors for machine learning processes in the future

# ### Data Cleaning

# My first data cleaning step is to check for null values in the data sets

# In[6]:


win_95.isna().sum().sum() + win_100.isna().sum().sum() + win_150.isna().sum().sum()


# There are no null values in the wins data sets
# 

# In[7]:


loss_95.isna().sum().sum() + loss_100.isna().sum().sum() + loss_150.isna().sum().sum()


# There are no null values in the losses data sets

# In[8]:


choice_95.isna().sum().sum() + choice_100.isna().sum().sum() + choice_150.isna().sum().sum()


# There are no null values in the choices data sets
# 

# In[9]:


index_95.isna().sum().sum() + index_100.isna().sum().sum() + index_150.isna().sum().sum()


# There are no null values in the choices data sets

# ### Data Analysis/Exploration

# Join all wins together, add column for what trial they were part of and also add how many partcipants were in that study
# 
# Join all losses together, add column for what trial they were part of and also add how many partcipants were in that study

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




