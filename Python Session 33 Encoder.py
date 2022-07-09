#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
import csv


# In[8]:


ds=pd.read_csv('empl.csv')


# In[9]:


ds


# In[10]:


df=ds.drop(['Name'],axis=1)


# In[11]:


df


# In[12]:


sns.heatmap(df.isnull())


# In[13]:


df.isnull().sum()


# In[14]:


df['Age']=df['Age'].fillna((df['Age'].median()))


# In[15]:


df


# In[16]:


df=df.replace(np.NAN,df['Salary'].mean())


# In[17]:


df


# In[18]:


from sklearn.impute import SimpleImputer


# In[19]:


imp=SimpleImputer(missing_values=np.nan, strategy='mean')


# In[20]:


df['Age']=imp.fit_transform(df['Age'].values.reshape(-1,1))
df['Salary']=imp.fit_transform(df['Salary'].values.reshape(-1,1))


# In[21]:


df


# In[22]:


le=LabelEncoder()


# In[23]:


list1=['City','Country']
for val in list1:
    df[val]=le.fit_transform(df[val].astype(str))


# In[24]:


df


# In[25]:


city_dummies=pd.get_dummies(df.City)
city_dummies


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




