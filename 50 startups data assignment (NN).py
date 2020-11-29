#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary libraries


# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Importing the dataset


# In[5]:


strt = pd.read_csv(r"D:\Data science\Assignments docs\Neural Networks\50_Startups.csv")
strt.head()


# In[6]:


strt.shape


# In[7]:


strt.describe()


# In[8]:


strt.columns


# In[9]:


strt.info()


# ## Exploratory data analysis

# In[10]:


# Checking the missing values in the dataset


# In[11]:


strt.isna().sum()


# #### There are no NA values in the dataset

# ### Univariate analysis

# In[12]:


# PLotting the histograms


# In[13]:


strt.columns


# In[35]:


for feature in strt.columns:
    if feature != "State":
        data = strt.copy()
        data[feature].hist(bins=15)
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.title(feature)
        plt.show()


# #### Most of the features are normally distributed

# In[29]:


# PLotting boxplots


# In[33]:


for feature in strt.columns:
    if feature != "State":
        data = strt.copy()
        strt.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()


# #### There are no outliers in the features

# In[40]:


# Count plot for state


# In[41]:


sns.countplot("State", data=strt, palette ="hls")


# #### Counts of all the three states is almost equal in the dataset

# ### Bivariate analysis

# In[ ]:


# Scatter plots of input feature with output feature


# In[39]:


for feature in strt.columns:
    if feature != "State":
        if feature != "Profit":
            data = strt.copy()
            plt.plot(data[feature], data['Profit'], "bo")
            plt.xlabel(feature)
            plt.ylabel('Profit')
            plt.title("Scatter plot")
            plt.show()


# #### R&D spend has good correlation with the output variable and other variables have lesser variation

# In[42]:


# Plotting categorical value states with output variable profit


# In[43]:


strt.groupby('State')['Profit'].median().plot.bar(); plt.xlabel('State'); plt.ylabel('Profit')


# #### There is hardly any change in the output variable for three different states

# ### Correlation matrix

# In[44]:


strt.corr()


# #### Input variables R&D Spend and Marketing spend have high correlation of around 72%

# ### Converting output variable "Profit" from continuos variable to categorical variable

# In[47]:


strt["Profit"].describe()


# #### Converting the Profit into two categories "High" and "Low"
# #### Value for "High" will be > 140000

# In[48]:


profit = pd.cut(strt.Profit, bins=[10000,140000,200000], labels = ["Low", "High"])


# In[50]:


strt = strt.drop(["Profit"], axis=1)


# In[51]:


# Adding categorized profit variable


# In[52]:


strt["Profit"] = profit


# In[53]:


strt.head()


# In[55]:


# Checking the value counts


# In[54]:


strt.Profit.value_counts()


# In[56]:


# Encoding the "State" variable


# In[57]:


states = pd.get_dummies(strt["State"], drop_first=True)


# In[60]:


strt = pd.concat([strt, states], axis=1)


# In[63]:


strt = strt.drop("State", axis=1)


# In[64]:


strt.head()


# In[72]:


# Splitting the data into input and output variables


# In[65]:


X = strt.drop(columns=["Profit"],axis=1)
Y = strt["Profit"]


# In[66]:


X.head()


# In[67]:


Y.head()


# In[71]:


# Spliting the data into train and test


# In[69]:


from sklearn.model_selection import train_test_split


# In[70]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# ## Feature scaling

# In[73]:


from sklearn.preprocessing import StandardScaler


# In[74]:


Scaler = StandardScaler()


# In[75]:


Scaler.fit(x_train)


# In[76]:


x_train = Scaler.transform(x_train)
x_test= Scaler.transform(x_test)


# ## Model building

# In[78]:


from sklearn.neural_network import MLPClassifier


# In[79]:


mlp = MLPClassifier(hidden_layer_sizes=(50,30))


# In[80]:


mlp.fit(x_train, y_train)


# In[81]:


# predicting the values


# In[82]:


pred_train = mlp.predict(x_train)
pred_test = mlp.predict(x_test)


# In[83]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[84]:


# Confusion matrix


# In[85]:


confusion_matrix(y_train, pred_train)


# In[86]:


confusion_matrix(y_test, pred_test)


# In[87]:


# Checking the accuracy


# In[88]:


accuracy_score(y_train, pred_train)


# In[89]:


accuracy_score(y_test, pred_test)


# #### Accuracy obtained is good, but still tuning hyper parameter and checking the accuracy

# ## Hyper parameter tuning

# In[100]:


mlp = MLPClassifier(hidden_layer_sizes=(50,30),activation="logistic")


# In[101]:


mlp.fit(x_train, y_train)


# In[102]:


# predicting the values


# In[103]:


pred_train = mlp.predict(x_train)
pred_test = mlp.predict(x_test)


# In[104]:


# Confusion matrix


# In[105]:


confusion_matrix(y_train, pred_train)


# In[106]:


confusion_matrix(y_test, pred_test)


# In[107]:


# Checking the accuracy


# In[108]:


accuracy_score(y_train, pred_train)


# In[109]:


accuracy_score(y_test, pred_test)


# #### There is no improvement in the accuracy with this activation method

# In[180]:


mlp = MLPClassifier(hidden_layer_sizes=(50,30), max_iter=500,activation="identity")


# In[181]:


mlp.fit(x_train, y_train)


# In[182]:


# predicting the values


# In[183]:


pred_train = mlp.predict(x_train)
pred_test = mlp.predict(x_test)


# In[184]:


# Confusion matrix


# In[185]:


confusion_matrix(y_train, pred_train)


# In[186]:


confusion_matrix(y_test, pred_test)


# In[187]:


# Checking the accuracy


# In[188]:


accuracy_score(y_train, pred_train)


# In[189]:


accuracy_score(y_test, pred_test)


# #### The accuracy is not improving beyond 90%, which is good for a smaller dataset like this.
# #### The default parameter model with two hidden layer of 50 and 30 neurons is giving highest accuracy of 90%
