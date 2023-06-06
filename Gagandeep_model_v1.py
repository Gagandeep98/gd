#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the necessary libraries:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[2]:


#Load the dataset using the provided URL:

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
data = pd.read_csv(url, header=None)


# In[3]:


#Separate the features (X) and the target variable (y):

X = data.iloc[:, 2:]  # Select all columns starting from the 3rd column as features
y = data.iloc[:, 1]   # Select the 2nd column as the target variable


# In[4]:


#Split the dataset into training and testing sets:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


#Perform feature scaling on the feature variables:

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[6]:


#Create an instance of the Support Vector Machine (SVM) classifier:

model = SVC(random_state=42)


# In[12]:


#Fit the model to the training data:

model.fit(X_train, y_train)


# In[13]:


#Make predictions on the testing set:

y_pred = model.predict(X_test)


# In[17]:


#Evaluate the model's performance:

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

