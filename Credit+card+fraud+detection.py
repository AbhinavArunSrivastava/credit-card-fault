#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


df=pd.read_csv('credit.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


#dont need ID
df=df.drop('ID', axis=1)


# In[7]:


df.head()


# In[8]:


# Checking for null values
df.isnull().sum()


# In[9]:


#UNIVARIATE ANALYSIS: Looking at the type of data present in different columns
cat_cols=['SEX','EDUCATION','MARRIAGE','default.payment.next.month']

fig,ax=plt.subplots(1,4,figsize=(25,5))

for cols,subplots in zip(cat_cols,ax.flatten()):
    sns.countplot(x=df[cols],ax=subplots)
    
#SEX: Gender (1=male, 2=female)
#EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
#MARRIAGE: Marital status (1=married, 2=single, 3=others)
#Default payment (1=yes, 0=no)


# In[10]:


# Vizualizing the imbalance 

yes=(((df['default.payment.next.month']==1).sum())/len(df['default.payment.next.month']))*100
no=(((df['default.payment.next.month']==0).sum())/len(df['default.payment.next.month']))*100

x=[yes,no]

plt.pie(x,labels=['Yes','No'],colors=['red', 'green'],radius=2,autopct='%1.0f%%')
plt.title('default.payment.next.month')
plt.show()


# In[11]:


df['default.payment.next.month'].value_counts(normalize=True)


# In[12]:


X=df.drop('default.payment.next.month',axis=1)
y=df['default.payment.next.month']
print(X)
df.EDUCATION.unique()


# In[13]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)


# In[14]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()


# In[15]:


models=rfc.fit(X_train.values,y_train)


# In[16]:


predictions=models.predict(X_test)


# In[17]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,predictions))


# In[19]:


import pickle
pickle.dump(models,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


# In[ ]:




