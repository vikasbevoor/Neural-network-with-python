#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the necessary libraries


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


# importing the dataset


# In[7]:


conc = pd.read_csv(r"D:\Data science\Assignments docs\Neural Networks\concrete.csv")
conc.head()


# In[6]:


conc.shape


# In[5]:


conc.describe()


# In[8]:


conc.info()


# ## Exploratory data analysis

# In[9]:


conc.isna().sum()


# #### There are NA values in the dataset

# ### Univariate analysis

# In[10]:


# PLotting histograms to check the distribution of data


# In[11]:


for feature in conc.columns:
    data = conc.copy()
    data[feature].hist(bins=10)
    plt.xlabel(feature)
    plt.ylabel("count")
    plt.title(feature)
    plt.show()


# #### Only few of the columns are normally distributed

# In[12]:


# PLotting boxplots to check the outliers if any


# In[14]:


for feature in conc.columns:
    data = conc.copy()
    conc.boxplot(column=feature)
    plt.ylabel(feature)
    plt.title(feature)
    plt.show()


# #### Some of the variables have few outliers and other variables dont have any outliers

# ### Bi-variate analysis

# In[16]:


# Scatter plots input variables with output variable


# In[17]:


for feature in conc.columns:
    if feature != "strength":
        data = conc.copy()
        plt.plot(data[feature], data['strength'], "bo")
        plt.xlabel(feature)
        plt.ylabel('strength')
        plt.title("Scatter plot")
        plt.show()


# #### The variation of output variable with many input variables was insignificant 

# ### Correlation matrix

# In[18]:


conc.corr()


# #### There are no input variables with higher correlation between each other

# In[19]:


# Seperating input and output variables


# In[20]:


X = conc.drop(columns = "strength", axis=1)
Y = conc["strength"]


# In[21]:


X.head()


# In[22]:


Y.head()


# ## Feature scaling

# In[25]:


from sklearn.preprocessing import StandardScaler


# In[26]:


scaler = StandardScaler()


# In[27]:


scaler.fit(X)


# In[28]:


X = scaler.transform(X)


# In[23]:


# Splitting the data into train and test


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2)


# ## Model building

# In[33]:


from sklearn.neural_network import MLPRegressor


# In[34]:


mlp = MLPRegressor(hidden_layer_sizes=(50,30), random_state = 1, max_iter = 100)


# In[35]:


mlp.fit(x_train,y_train)


# In[36]:


# Predicted values


# In[37]:


pred_train = mlp.predict(x_train)
pred_test = mlp.predict(x_test)


# In[41]:


# plot predicted data 
plt.plot(y_test,pred_test,"bo")  
plt.title('MLP') 
plt.xlabel('Observed values') 
plt.ylabel('Predicted values') 
plt.show()


# In[42]:


# RMSE values


# In[44]:


from sklearn import metrics


# In[45]:


print('RMSE of training data:', np.sqrt(metrics.mean_squared_error(y_train,pred_train)))


# In[46]:


print('RMSE of training data:', np.sqrt(metrics.mean_squared_error(y_test,pred_test)))


# #### RMSE values for both train and test are similar and also RMSE values are low

# ## Hyper parameter tuning to improve accuracy

# In[92]:


mlp = MLPRegressor(hidden_layer_sizes=(60,30,10), random_state = 1, max_iter = 500, learning_rate="adaptive")


# In[93]:


mlp.fit(x_train,y_train)


# In[94]:


# Predicted values


# In[95]:


pred_train = mlp.predict(x_train)
pred_test = mlp.predict(x_test)


# In[96]:


# plot predicted data 
plt.plot(y_test,pred_test,"bo")  
plt.title('MLP') 
plt.xlabel('Observed values') 
plt.ylabel('Predicted values') 
plt.show()


# In[97]:


# RMSE values


# In[98]:


from sklearn import metrics


# In[99]:


print('RMSE of training data:', np.sqrt(metrics.mean_squared_error(y_train,pred_train)))


# In[100]:


print('RMSE of training data:', np.sqrt(metrics.mean_squared_error(y_test,pred_test)))


# #### After hyper parameter tuning the least RMSE obtained is 5.36
# #### The tuned parameters are learning rate = "adaptive", max_iter = 500 with 3 hidden_layers of 60,30,10 neurons
