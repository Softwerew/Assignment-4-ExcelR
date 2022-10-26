#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf


# In[2]:


data=pd.read_csv(r"C:\Users\Lovely_Ray\Desktop\data science\Assignment 4\delivery_time.csv")# Q1- predicting delivery time using sorting time


# In[ ]:





# In[4]:


data.describe()


# In[5]:


data.info()


# In[7]:


sns.distplot(data['Sorting Time']) # drawing density plot for variable x


# In[8]:


sns.distplot(data['Delivery Time']) # drawing density plot for variable y


# In[9]:


plt.scatter(x='Sorting Time',y='Delivery Time',data=data) # drawing scatter plot


# In[10]:


sns.regplot(x="Sorting Time", y="Delivery Time", data=data) # drawing regression plot


# In[11]:


data.corr()# calculating correlation


# In[12]:


#Correlation coefficient value r = 0.825997 indicates that there is a strong correlation between independent variable and dependent variable


# In[13]:


data=data.rename({'Delivery Time':'delivery_time', 'Sorting Time':'sorting_time'},axis=1) # renaming the data as part of feature engineering
data


# In[14]:


model = smf.ols("delivery_time~sorting_time",data = data).fit() # building model


# In[15]:


model.params # calculating model parameters


# In[18]:


model.tvalues, model.pvalues # model t values and p values


# In[19]:


r_sq, r_sq_adj = (model.rsquared, model.rsquared_adj)
print("R2: ", r_sq)
print("R2_Adj: ", r_sq_adj) # calculating rsquared values


# In[21]:


# implimentation of Log transformation for improvement of model
data['log_sorting_time'] = np.log(data['sorting_time'])
data


# In[22]:


data_log = data[['delivery_time','log_sorting_time']]
data_log


# In[23]:


sns.regplot(x="log_sorting_time", y="delivery_time", data=data_log) # drawing regression plot


# In[24]:


log_model = smf.ols('delivery_time~log_sorting_time', data=data_log).fit()


# In[25]:


data['sqrt_sorting_time'] = np.sqrt(data['sorting_time'])
data


# In[26]:


data_sqrt = data[['delivery_time','sqrt_sorting_time']]
data_sqrt


# In[27]:


sns.regplot(x="sqrt_sorting_time", y="delivery_time", data=data_sqrt) # drawing regression plot


# In[28]:


sqrt_model = smf.ols('delivery_time~sqrt_sorting_time', data=data_sqrt).fit()


# In[30]:


model.summary() #model testing


# In[31]:


log_model.summary()


# In[32]:


sqrt_model.summary() #sqrt model has got better rsquared value.


# In[37]:


newdata=pd.Series([6,8]) # sample data


# In[38]:


newdata


# In[39]:


data_pred=pd.DataFrame(newdata,columns=['sorting_time'])
data_pred


# In[40]:


data_pred['sqrt_sorting_time']=np.sqrt(data_pred['sorting_time'])


# In[41]:


sqrt_model.predict(data_pred['sqrt_sorting_time'])# automatic model prediction using sample data.


# In[42]:


sqrt_model.predict(data['sqrt_sorting_time']) # implementing new model to the whole data.


# In[ ]:




