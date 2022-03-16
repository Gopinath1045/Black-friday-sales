#!/usr/bin/env python
# coding: utf-8

# In[110]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
pd.pandas.set_option('display.max_columns',None)


# In[111]:


df1=pd.read_csv("G:/Black friday sales/train.csv") #train data
df2=pd.read_csv("G:/Black friday sales/test.csv")  #test data
sub=pd.read_csv("G:/Black friday sales/sample.csv")  #sample submission file


# In[112]:


sns.heatmap(df1.isnull(), yticklabels=False )


# In[113]:


sns.heatmap(df2.isnull(), yticklabels=False )


# In[114]:


categorical_feature=[feature for feature in df1.columns if df1[feature].dtypes=='O' ]
categorical_features2=[feature for feature in df2.columns if df2[feature].dtype=='O']


# In[115]:


for c in categorical_feature:
    print ('\nFrequency of Categories for varible %s'%c)
    print (df1[c].value_counts())


# In[116]:


for c in categorical_feature:
    print ('\nFrequency of Categories for varible %s'%c)
    print (df2[c].value_counts())


# In[117]:


df1['City_Category']=df1['City_Category'].replace({'B':0,'C':1,'A':2})
df2['City_Category']=df2['City_Category'].replace({'B':0,'C':1,'A':2})
df1['Gender']=df1['Gender'].map({'M':0,'F':1})
df2['Gender']=df2['Gender'].map({'M':0,'F':1})
df1['Stay_In_Current_City_Years']=df1['Stay_In_Current_City_Years'].replace({1:0,2:1,3:2,'4+':3,0:4})
df2['Stay_In_Current_City_Years']=df2['Stay_In_Current_City_Years'].replace({1:0,2:1,3:2,'4+':3,0:4})


# In[118]:


df1['Age']=df1['Age'].replace({'26-35':0,'36-45':1,'18-25':2,'46-50':3,'51-55':4,'55+':5,'0-17':6})
df2['Age']=df2['Age'].replace({'26-35':0,'36-45':1,'18-25':2,'46-50':3,'51-55':4,'55+':5,'0-17':6})


# In[120]:


df1['Product_Category_2'].fillna(df1['Product_Category_2'].mean(),inplace=True)
df2['Product_Category_2'].fillna(df2['Product_Category_2'].mean(),inplace=True)


# In[121]:


df1['Product_Category_3'].fillna(df1['Product_Category_3'].mean(),inplace=True)
df2['Product_Category_3'].fillna(df2['Product_Category_3'].mean(),inplace=True)


# In[122]:


sns.heatmap(df2.isnull(), yticklabels=False )


# In[123]:


sns.heatmap(df1.isnull(), yticklabels=False )


# In[124]:


plt.figure(figsize=(16,8))
sns.heatmap(df1.corr(),annot=True,cmap='viridis')


# In[125]:


plt.figure(figsize=(16,8))
sns.heatmap(df2.corr(),annot=True,cmap='viridis')


# In[126]:


#now we will split the data


# In[164]:


X=df1.drop(['User_ID','Product_ID','Purchase'],axis=1)


# In[165]:


y=df1['Purchase']


# In[166]:


#split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=14)


# In[167]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


# In[168]:


y_pred=model.predict(X_test)
y_pred_train=model.predict(X_train)


# In[169]:


from sklearn.metrics import mean_squared_error, r2_score
mse_test=mean_squared_error(y_test,y_pred)
mse_train=mean_squared_error(y_train,y_pred_train)


# In[170]:


mse_test


# In[171]:


mse_train


# In[172]:


r2_score_test=r2_score(y_test,y_pred)
r2_score_train=r2_score(y_train,y_pred_train)


# In[173]:


r2_score_test


# In[174]:


r2_score_train


# In[175]:


rmse_test=mean_squared_error(y_test,y_pred,squared=False)
rmse_train=mean_squared_error(y_train,y_pred_train,squared=False)


# In[176]:


rmse_test


# In[177]:


rmse_train


# In[182]:


from sklearn.model_selection import GridSearchCV
parameters = {'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]}
grid = GridSearchCV(model,parameters, cv=None)
grid.fit(X_train, y_train)
print ("r2_score / variance : ", grid.best_score_)
print("Residual sum of squares: %.2f"
              % np.mean((grid.predict(X_test) - y_test) ** 2))


# In[153]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
dtr=DecisionTreeRegressor()
dtr.fit(X_train,y_train)
cross_val_score(dtr, X_train, y_train, cv=10)


# In[154]:


y_pred=dtr.predict(X_test)
y_pred_train=dtr.predict(X_train)


# In[155]:


rmse_test=mean_squared_error(y_test,y_pred)**.5
rmse_train=mean_squared_error(y_train,y_pred_train)**.5


# In[156]:


from sklearn.ensemble import RandomForestRegressor
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor()
rf.fit(X_train, y_train)


# In[157]:


y_pred=rf.predict(X_test)
y_pred_train=rf.predict(X_train)


# In[158]:


rmse_test=mean_squared_error(y_test,y_pred, squared=False)
rmse_train=mean_squared_error(y_train,y_pred_train,squared=False)


# In[159]:


#getting submission file


# In[99]:


df3=df2.drop(['User_ID','Product_ID'],axis=1)


# In[100]:


sub.head()


# In[101]:


y_pred3 = model.predict(df3)


# In[102]:


y_pred3 


# In[103]:


df=pd.DataFrame(y_pred3,columns=['Purchase'])


# In[105]:


df.head()


# In[106]:


sub['Purchase']=df['Purchase']
sub.head()


# In[107]:


sub.shape


# In[108]:


sub.to_csv('G:/Black friday sales/submitted.csv',index=False)


# In[ ]:




