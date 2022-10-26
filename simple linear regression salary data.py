#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# In[2]:


data=pd.read_csv(r"C:\Users\Lovely_Ray\Desktop\data science\Assignment 4\Salary_Data.csv") # Q2=Salary data model prediction


# In[3]:


data


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.corr() # correlation computation.


# In[7]:


sns.distplot(data['YearsExperience']) # drawing density plot 


# In[8]:


sns.distplot(data['Salary'])


# In[9]:


plt.scatter(x='YearsExperience', y='Salary', data=data) # scatter plot


# In[10]:


sns.regplot(x=data['YearsExperience'], y=data['Salary']) # Regression plot


# In[11]:


model = smf.ols("Salary~YearsExperience",data = data).fit()


# In[12]:


model.params # computation of model parameters


# In[13]:


model.tvalues, model.pvalues # finding t values & p values


# In[14]:


model.rsquared , model.rsquared_adj # finding r squared  value. model accuracy is quite high with 95%.


# In[16]:


model.summary()


# In[17]:


# we will employ RMSE to check model accuracy


# In[18]:


x=data['YearsExperience']
y=data['Salary']


# In[21]:


x_train,x_test,y_train,y_test = train_test_split(x,y,train_size =0.7,test_size = 0.3,random_state = 100)


# In[22]:


x_train.shape


# In[23]:


x_test.shape


# In[26]:


x_train_sm = sm.add_constant(x_train)
model = sm.OLS(y_train, x_train_sm).fit()


# In[27]:


print(model.summary())


# In[28]:


plt.scatter(x_train,y_train)
plt.plot(x_train,25200+x_train * 9731.2038)
plt.show()


# In[30]:


y_train_pred = model.predict(x_train_sm)


# In[31]:


residual = (y_train - y_train_pred)


# In[32]:


sns.distplot(residual)


# In[33]:


sns.scatterplot(x_train,residual)


# In[34]:


x_test_sm= sm.add_constant(x_test)


# In[35]:


y_pred = model.predict(x_test_sm)


# In[36]:


RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
RMSE


# In[37]:


r2_score(y_test,y_pred)


# In[39]:


# so rsquare value has been improved to 96%.


# In[ ]:




