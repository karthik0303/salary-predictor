#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


# In[17]:


df = pd.read_excel("D:/Sample.xlsx")


# In[18]:


df.head()


# ## Pairplot (One code for all)

# In[19]:


sns.pairplot(df)


# In[35]:


plt.figure(figsize=(15,7))
sns.heatmap(df.corr())


# ## Sorting variables into dependent and independent features

# In[20]:


X = df[["SSC Percentage", "HSC Percentage", "Graduation Percentage"]]


# In[21]:


X.head()


# In[22]:


y = df["Salary"]


# In[23]:


y.head()


# ## Creating a simple Linear Regression Model

# In[13]:


from sklearn.linear_model import LinearRegression


# In[15]:


reg = LinearRegression()


# In[16]:


reg.fit(X,y)


# ## Saving model to disk

# In[24]:


pickle.dump(reg, open("model.pkl", "wb"))


# ## Let's try and check a prediction from our model

# In[25]:


model = pickle.load(open("model.pkl", "rb"))


# In[36]:


print(model.predict([[85.6,69.7,60.77]]))


# In[ ]:




