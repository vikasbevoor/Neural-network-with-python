#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the necessary libraries


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


fire = pd.read_csv(r"D:\Data science\Assignments docs\Neural Networks\forestfires.csv")
fire.head()


# #### In the data, the month and day variables are already endoded and thier dummy columns has been added. Hence removing original 'month' and 'day column'

# In[7]:


fire.drop(["month","day"], axis=1, inplace=True)


# In[8]:


fire.head()


# In[9]:


fire.shape


# In[10]:


fire.describe()


# In[11]:


fire.info()


# ## Exploratory data analysis

# In[17]:


# Checking for NA values


# In[18]:


fire.isna().sum()


# #### There are no NA values in the dataset

# In[16]:


# Considering only the numerical columns neglecting dummy variable columns


# In[15]:


num_values = fire[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']]


# ### Univariate analysis

# In[12]:


# plotting the histograms to check the distribution of data


# In[19]:


for feature in num_values.columns:
    data = num_values.copy()
    data[feature].hist(bins=15)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


# #### All the columns are not normally distributed

# In[20]:


# Checking the outliers by using boxplots


# In[22]:


for feature in num_values.columns:
    data = num_values.copy()
    num_values.boxplot(column=feature)
    plt.ylabel(feature)
    plt.title(feature)
    plt.show()


# #### Many variables consists of lot of outliers especially "area"

# ### Bivariate analysis

# In[23]:


# Scatter plots of input feature with output feature


# In[24]:


for feature in num_values:
    sns.boxplot("size_category",feature, data=fire, palette='hls')
    plt.show()


# #### There is no considerable changes in the two categories of output variables with input variables

# In[ ]:


# Count plot of target variable


# In[25]:


sns.countplot("size_category", data=fire , palette='hls')


# In[9]:


fire["size_category"].value_counts()


# #### Dataset consists of more "small" category values compared to "large"

# ### Correlation matrix

# In[28]:


fire.corr()


# #### The input variables "DMC" and "DC" have higher correlation among all the variables

# #### The date and month dummy variable columns hardl have any influence on the size of the fire, hence dropping these columns

# In[26]:


# Splitting the data into input and output variables


# In[30]:


X = fire[num_values.columns]
Y = fire["size_category"]


# ## Feature scaling

# In[32]:


from sklearn.preprocessing import StandardScaler


# In[33]:


scaler = StandardScaler()


# In[34]:


scaler.fit(X)


# In[35]:


X = scaler.transform(X)


# In[37]:


# Splitting the data into train and test


# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.25)


# In[40]:


x_train.shape, x_test.shape


# ## Model building

# In[41]:


from sklearn.neural_network import MLPClassifier


# In[42]:


mlp = MLPClassifier(hidden_layer_sizes=(60,30), random_state=1)


# In[43]:


mlp.fit(x_train,y_train)


# In[44]:


# predcited values


# In[45]:


pred_train = mlp.predict(x_train)
pred_test = mlp.predict(x_test)


# In[46]:


# Confusion matrix


# In[50]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[48]:


confusion_matrix(y_train, pred_train)


# In[49]:


confusion_matrix(y_test, pred_test)


# In[51]:


# Checking the accuracy


# In[52]:


accuracy_score(y_train, pred_train)


# In[53]:


accuracy_score(y_test, pred_test)


# #### The obtained accuracy is good, but tuning hyper parameters to improve accuracy

# In[87]:


mlp = MLPClassifier(hidden_layer_sizes=(60,30), max_iter=500, learning_rate="adaptive", random_state=1)


# In[88]:


mlp.fit(x_train,y_train)


# In[89]:


# predcited values


# In[90]:


pred_train = mlp.predict(x_train)
pred_test = mlp.predict(x_test)


# In[91]:


# Confusion matrix


# In[92]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[93]:


confusion_matrix(y_train, pred_train)


# In[94]:


confusion_matrix(y_test, pred_test)


# In[95]:


# Checking the accuracy


# In[96]:


accuracy_score(y_train, pred_train)


# In[97]:


accuracy_score(y_test, pred_test)


# #### The accuracy has been improved to 95.3 % from 93.07%
# #### The tuned hyper parameters are learning_rate="adaptive", max_iter=500 and the hidden layers are two with 60 and 30 neurons in each layer respectively
