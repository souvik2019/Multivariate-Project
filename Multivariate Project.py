#!/usr/bin/env python
# coding: utf-8

# In[3]:





# In[1]:


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import math


# In[2]:


data=pd.read_table('milk_transportation.txt',sep='  ', header=None)


# In[3]:


data.shape


# In[4]:


data.columns=['Fuel Type', 'Fuel Cost', 'Repair Cost', 'Capital Cost']


# In[5]:


data.iloc[33:40,:]


# In[6]:


data1=data.iloc[:36,:]
data2=data.iloc[36:,:]


# In[7]:


data1


# In[8]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(11,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data1.iloc[:,0], data1.iloc[:,1], data1.iloc[:,2], c='r', marker='o',label='Gasoline')
ax.scatter(data2.iloc[:,0], data2.iloc[:,1], data2.iloc[:,2], c='b', marker='^',label='Diesel')

ax.set_xlabel('Fuel cost')
ax.set_ylabel('Repair Cost')
ax.set_zlabel('Capital cost')

plt.legend(loc="upper left")

plt.show()


# In[9]:


data.iloc[:,0].unique()


# In[10]:


import seaborn as sns


# In[11]:


sns.pairplot(data, hue="Fuel Type")


# In[12]:


get_ipython().system('pip install pingouin==0.3.11')
import pingouin as pg


# In[13]:


_,ax1=plt.subplots(1,3,figsize=(12,3))
ax1[0] = pg.qqplot(data1.iloc[:,1], dist='norm', ax=ax1[0])
ax1[0].set_title('Fuel Cost')
ax1[1] = pg.qqplot(data1.iloc[:,2], dist='norm', ax=ax1[1])
ax1[1].set_title('Repair Cost')
ax1[2] = pg.qqplot(data1.iloc[:,3], dist='norm', ax=ax1[2])
ax1[2].set_title('Capital Cost')


# In[14]:


_,ax2=plt.subplots(1,3,figsize=(12,3))
ax2[0] = pg.qqplot(data2.iloc[:,1], dist='norm', ax=ax2[0])
ax2[0].set_title('Fuel Cost')
ax2[1] = pg.qqplot(data2.iloc[:,2], dist='norm', ax=ax2[1])
ax2[1].set_title('Repair Cost')
ax2[2] = pg.qqplot(data2.iloc[:,3], dist='norm', ax=ax2[2])
ax2[2].set_title('Capital Cost')


# In[32]:


for i in range(3):
    print('statistic','        ','p-value')
    a=sp.stats.shapiro(data1.iloc[:,i+1])
    print(a[0],a[1])
print('statistic','        ','p-value')
a=sp.stats.shapiro(data1.iloc[:,1:4])
print(a[0],a[1])


# In[33]:


for i in range(3):
    print('statistic','        ','p-value')
    a=sp.stats.shapiro(data2.iloc[:,i+1])
    print(a[0],a[1])
print('statistic','        ','p-value')
a=sp.stats.shapiro(data2.iloc[:,1:4])
print(a[0],a[1])    


# In[17]:


pg.multivariate_normality(data1.iloc[:,1:4],alpha=0.05)


# In[18]:


V1=data1.iloc[:,1:4].cov()
V2=data2.iloc[:,1:4].cov()
V1I=np.linalg.inv(V1)
V2I=np.linalg.inv(V2)


# In[19]:


Xbar1=np.mean(data1.iloc[:,1:4],axis=0)
Xbar2=np.mean(data2.iloc[:,1:4],axis=0)


# In[109]:


d1=data1.iloc[:,1:4]
d2=data2.iloc[:,1:4]


# In[22]:


d1.shape


# In[23]:


mij1_square=np.zeros((36,36))


# In[24]:


for i in range(36):
    for j in range(36):
        mij1_square[i,j]=((d1.iloc[i,:]-Xbar1).T@V1I@(d1.iloc[j,:]-Xbar1))**3


# In[25]:


mij2_square=np.zeros((23,23))
for i in range(23):
    for j in range(23):
        mij2_square[i,j]=((d2.iloc[i,:]-Xbar2).T@V2I@(d2.iloc[j,:]-Xbar2))**2


# In[26]:


skew1=(np.sum(mij1_square)/36**2)*(36/35)**3
skew2=(np.sum(mij2_square)/23**2)*(23/22)**3
tskew1=skew1*36/6
tskew2=skew2*23/6
print(tskew1,tskew2)
kurt1=(np.trace(mij1_square)/36)*(36/35)**2
kurt2=(np.trace(mij2_square)/23)*(23/22)**2
k=3

tkurt1=(kurt1-k*(k+2))*np.sqrt(36/(8*k*(k+2)))
tkurt2=(kurt2-k*(k+2))*np.sqrt(23/(8*k*(k+2)))
print(tkurt1,tkurt2)


# In[27]:


#Box cox transformation


# In[30]:


fitted_data1_f, fitted_lambda_d11 = sp.stats.boxcox(d1.iloc[:,0])
print(fitted_lambda_d11)
fitted_data1_r, fitted_lambda_d12 = sp.stats.boxcox(d1.iloc[:,1])
print(fitted_lambda_d12)
fitted_data1_c, fitted_lambda_d13 = sp.stats.boxcox(d1.iloc[:,2])
print(fitted_lambda_d13)


# In[31]:


fitted_data1_f, fitted_lambda_d11 = sp.stats.boxcox(d2.iloc[:,0])
print(fitted_lambda_d11)
fitted_data1_r, fitted_lambda_d12 = sp.stats.boxcox(d2.iloc[:,1])
print(fitted_lambda_d12)
fitted_data1_c, fitted_lambda_d13 = sp.stats.boxcox(d2.iloc[:,2])
print(fitted_lambda_d13)


# In[110]:


d1=(np.sqrt(d1)-1)*2


# In[111]:


d2=(np.sqrt(d2)-1)*2


# In[36]:


_,ax1=plt.subplots(1,3,figsize=(12,3))
ax1[0] = pg.qqplot(d1.iloc[:,0], dist='norm', ax=ax1[0])
ax1[0].set_title('Fuel Cost')
ax1[1] = pg.qqplot(d1.iloc[:,1], dist='norm', ax=ax1[1])
ax1[1].set_title('Repair Cost')
ax1[2] = pg.qqplot(d1.iloc[:,2], dist='norm', ax=ax1[2])
ax1[2].set_title('Capital Cost')


# In[38]:


for i in range(3):
    print('statistic','        ','p-value')
    a=sp.stats.shapiro(d1.iloc[:,i])
    print(a[0],a[1])
print('statistic','        ','p-value')
a=sp.stats.shapiro(d1)
print(a[0],a[1])


# In[37]:


_,ax1=plt.subplots(1,3,figsize=(12,3))
ax1[0] = pg.qqplot(d2.iloc[:,0], dist='norm', ax=ax1[0])
ax1[0].set_title('Fuel Cost')
ax1[1] = pg.qqplot(d2.iloc[:,1], dist='norm', ax=ax1[1])
ax1[1].set_title('Repair Cost')
ax1[2] = pg.qqplot(d2.iloc[:,2], dist='norm', ax=ax1[2])
ax1[2].set_title('Capital Cost')


# In[39]:


for i in range(3):
    print('statistic','        ','p-value')
    a=sp.stats.shapiro(d2.iloc[:,i])
    print(a[0],a[1])
print('statistic','        ','p-value')
a=sp.stats.shapiro(d2)
print(a[0],a[1])


# In[40]:


V1_t=d1.cov()
V2_t=d2.cov()
V1It=np.linalg.inv(V1_t)
V2It=np.linalg.inv(V2_t)
Xbar1_t=np.mean(d1,axis=0)
Xbar2_t=np.mean(d2,axis=0)


# In[41]:


mij1_square_t=np.zeros((36,36))
for i in range(36):
    for j in range(36):
        mij1_square_t[i,j]=((d1.iloc[i,:]-Xbar1_t).T@V1It@(d1.iloc[j,:]-Xbar1_t))**3


# In[42]:


mij2_square_t=np.zeros((23,23))
for i in range(23):
    for j in range(23):
        mij2_square_t[i,j]=((d2.iloc[i,:]-Xbar2_t).T@V2It@(d2.iloc[j,:]-Xbar2_t))**2


# In[43]:


skew1=(np.sum(mij1_square_t)/36**2)*(36/35)**3
skew2=(np.sum(mij2_square_t)/23**2)*(23/22)**3
tskew1=skew1*36/6
tskew2=skew2*23/6
print(tskew1,tskew2)
kurt1=(np.trace(mij1_square_t)/36)*(36/35)**2
kurt2=(np.trace(mij2_square_t)/23)*(23/22)**2
k=3

tkurt1=(kurt1-k*(k+2))*np.sqrt(36/(8*k*(k+2)))
tkurt2=(kurt2-k*(k+2))*np.sqrt(23/(8*k*(k+2)))
print(tkurt1,tkurt2)


# In[ ]:





# In[55]:


import statsmodels.api as sm


# In[83]:


model = sm.OLS(data1.iloc[:,0],d1)
results = model.fit()


# In[84]:


influence = results.get_influence()
leverage = influence.hat_matrix_diag
plt.scatter(list(range(1,37)),leverage)
plt.plot([0,37], [1/6,1/6],color='red')
plt.ylabel('Hat values')
plt.title('Gasoline');


# In[95]:


ind1=np.where(leverage>(1/6))[0]
print(ind1)
d1.iloc[ind1,:]


# In[112]:


d1=d1.drop(d1.index[ind1])


# In[113]:


d1.shape


# In[82]:


model = sm.OLS(data2.iloc[:,0],d2)
results = model.fit()
influence = results.get_influence()
leverage = influence.hat_matrix_diag
plt.scatter(list(range(1,24)),leverage)
plt.plot([0,24], [6/23,6/23],color='red')
plt.ylabel('Hat values')
plt.title('Diesel');


# In[99]:


ind2=np.where(leverage>(6/23))[0]
print(ind2)
d2.iloc[ind2,:]


# In[107]:


d2=d2.drop(d2.index[[8]])


# In[114]:


for i in range(3):
    print('statistic','        ','p-value')
    a=sp.stats.shapiro(d1.iloc[:,i])
    print(a[0],a[1])
print('statistic','        ','p-value')
a=sp.stats.shapiro(d1)
print(a[0],a[1])


# In[115]:


for i in range(3):
    print('statistic','        ','p-value')
    a=sp.stats.shapiro(d2.iloc[:,i])
    print(a[0],a[1])
print('statistic','        ','p-value')
a=sp.stats.shapiro(d2)
print(a[0],a[1])


# In[ ]:


V1_t=d1.cov()
V2_t=d2.cov()
V1It=np.linalg.inv(V1_t)
V2It=np.linalg.inv(V2_t)
Xbar1_t=np.mean(d1,axis=0)
Xbar2_t=np.mean(d2,axis=0)


# In[118]:


mij1_square=np.zeros((33,33))
for i in range(33):
    for j in range(33):
        mij1_square[i,j]=((d1.iloc[i,:]-Xbar1_t).T@V1It@(d1.iloc[j,:]-Xbar1_t))**3


# In[119]:


mij2_square=np.zeros((22,22))
for i in range(22):
    for j in range(22):
        mij2_square[i,j]=((d2.iloc[i,:]-Xbar2_t).T@V2It@(d2.iloc[j,:]-Xbar2_t))**2


# In[120]:


skew1=(np.sum(mij1_square)/36**2)*(36/35)**3
skew2=(np.sum(mij2_square)/23**2)*(23/22)**3
tskew1=skew1*36/6
tskew2=skew2*23/6
print(tskew1,tskew2)
kurt1=(np.trace(mij1_square)/36)*(36/35)**2
kurt2=(np.trace(mij2_square)/23)*(23/22)**2
k=3

tkurt1=(kurt1-k*(k+2))*np.sqrt(36/(8*k*(k+2)))
tkurt2=(kurt2-k*(k+2))*np.sqrt(23/(8*k*(k+2)))
print(tkurt1,tkurt2)


# In[121]:


d1.cov()


# In[122]:


d1.corr()


# In[165]:


np.linalg.eig(d1.cov())


# In[129]:


import sklearn as sk
from sklearn.preprocessing import scale
from sklearn import decomposition


# In[138]:


X1 = scale(d1)
pca1 = decomposition.PCA(n_components=3)
X1 = pca1.fit_transform(X1)
loadings1 = pd.DataFrame(pca1.components_.T, columns=['PC1', 'PC2','PC3'],index=d1.T.index)
loadings1


# In[146]:


var_exp1=pca1.explained_variance_ratio_*100


# In[148]:


np.cumsum(var_exp1)


# In[161]:



plt.plot(['Fuel Cost','Repair Cost','Capital Cost'],var_exp1)
plt.bar(['Fuel Cost','Repair Cost','Capital Cost'],var_exp1,color='yellow')
plt.title('Scree Plot - Gasoline')
plt.scatter(['Fuel Cost','Repair Cost','Capital Cost'],var_exp1,color='red')
plt.ylabel('% variance Explained')


# In[177]:


e_vec=np.linalg.eig(d1.cov())[1]
e_val=np.linalg.eig(d1.cov())[0]
#V1_t
mat1=np.zeros((3,3))
for i in range(3):
    for j in range(3):
        mat1[i,j]=e_vec[i,j]*np.sqrt(e_val[i]/V1_t.iloc[j,j])
mat1  #correlation between variables and pca      
    


# In[180]:


get_ipython().system('pip install mlxtend')


# In[181]:


from mlxtend.plotting import plot_pca_correlation_graph


# In[187]:


feature=['Fuel Cost','Repair Cost','Capital Cost']
fig, correlation_mat = plot_pca_correlation_graph(scale(d1), feature, dimensions=(1,2), figure_axis_size=8)


# In[188]:


d2.cov()


# In[189]:


d2.corr()


# In[190]:


np.linalg.eig(d2.cov())


# In[191]:


X2 = scale(d2)
pca2 = decomposition.PCA(n_components=3)
X2 = pca2.fit_transform(X2)
loadings2 = pd.DataFrame(pca2.components_.T, columns=['PC1', 'PC2','PC3'],index=d2.T.index)
loadings2


# In[192]:


var_exp2=pca2.explained_variance_ratio_*100
np.cumsum(var_exp2)


# In[194]:



plt.plot(['Fuel Cost','Repair Cost','Capital Cost'],var_exp2)
plt.bar(['Fuel Cost','Repair Cost','Capital Cost'],var_exp2,color='yellow')
plt.title('Scree Plot - Diesel')
plt.scatter(['Fuel Cost','Repair Cost','Capital Cost'],var_exp2,color='red')
plt.ylabel('% variance Explained')


# In[195]:


e_vec=np.linalg.eig(d2.cov())[1]
e_val=np.linalg.eig(d2.cov())[0]
#V1_t
mat2=np.zeros((3,3))
for i in range(3):
    for j in range(3):
        mat2[i,j]=e_vec[i,j]*np.sqrt(e_val[i]/V2_t.iloc[j,j])
mat2  #correlation between variables and pca      
 


# In[196]:


fig, correlation_mat = plot_pca_correlation_graph(scale(d2), feature, dimensions=(1,2), figure_axis_size=8)


# In[198]:


sp.stats.norm.interval(alpha=0.95, loc=np.mean(d1,axis=0), scale=sp.stats.sem(d1))


# In[200]:


sp.stats.norm.interval(alpha=0.95, loc=np.mean(d2,axis=0), scale=sp.stats.sem(d2))


# In[205]:


plt.scatter(['Fuel cost','Repair Cost','Capital Cost'],[4.75,3.4,4.2],color='red',label='Gasoline')
plt.plot(['Fuel cost','Repair Cost','Capital Cost'],[4.75,3.4,4.2],color='red')
plt.scatter(['Fuel cost','Repair Cost','Capital Cost'],[4.3,4.3,6.25],color='blue',label='Diesel')
plt.plot(['Fuel cost','Repair Cost','Capital Cost'],[4.3,4.3,6.25],color='blue')
plt.legend(loc='upper left')
plt.ylabel('Mean');


# In[206]:


data_=data.drop([44,2,8,19])


# In[207]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
model=LDA()
model.fit(data_.iloc[:,1:4],data_.iloc[:,0])


# In[208]:


prob1=model.predict_proba(data.iloc[:,1:4])


# In[223]:


plt.scatter(prob1[0:34,1],prob1[0:34,0],color='red',label='Gasoline')
plt.scatter(prob1[34:55,1],prob1[34:55,0],color='blue',label='Diesel')
plt.legend(loc='upper right')
plt.xlabel('Posterior Probability of Diesel')
plt.ylabel('Posterior Probability of Gasoline')
plt.title('Probablity Plot');


# In[220]:


ind=np.where(prob1[:,0]>0.5)
ind_=np.where(prob1[:,0]<0.5)


# In[222]:


plt.scatter(prob1[ind[0],1],prob1[ind[0],0],color='red',label='Gasoline')
plt.scatter(prob1[ind_[0],1],prob1[ind_[0],0],color='blue',label='Diesel')
plt.legend(loc='upper right')
plt.xlabel('Posterior Probability of Diesel')
plt.ylabel('Posterior Probability of Gasoline')
plt.title('Prediction Plot');


# In[228]:


from sklearn.metrics import confusion_matrix


# In[229]:


pred=model.predict(data_.iloc[:,1:4])
confusion_matrix(data_.iloc[:,0],pred)


# In[231]:


#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,1:4], data.iloc[:,0], test_size=0.25, random_state=0)


# In[236]:


modelt=LDA()
modelt.fit(X_train,y_train)
predt=modelt.predict(X_test)
confusion_matrix(y_test,predt)


# In[233]:


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
model2=QDA()
model2.fit(data_.iloc[:,1:4],data_.iloc[:,0])
prob2=model2.predict_proba(data.iloc[:,1:4])

plt.scatter(prob2[0:34,1],prob2[0:34,0],color='red',label='Gasoline')
plt.scatter(prob2[34:55,1],prob2[34:55,0],color='blue',label='Diesel')
plt.legend(loc='upper right')
plt.xlabel('Posterior Probability of Diesel')
plt.ylabel('Posterior Probability of Gasoline')
plt.title('Probablity Plot');


# In[234]:


ind=np.where(prob2[:,0]>0.5)
ind_=np.where(prob2[:,0]<0.5)

plt.scatter(prob2[ind[0],1],prob2[ind[0],0],color='red',label='Gasoline')
plt.scatter(prob2[ind_[0],1],prob2[ind_[0],0],color='blue',label='Diesel')
plt.legend(loc='upper right')
plt.xlabel('Posterior Probability of Diesel')
plt.ylabel('Posterior Probability of Gasoline')
plt.title('Prediction Plot');


# In[235]:


pred=model2.predict(data_.iloc[:,1:4])
confusion_matrix(data_.iloc[:,0],pred)


# In[237]:


modelt=QDA()
modelt.fit(X_train,y_train)
predt=modelt.predict(X_test)
confusion_matrix(y_test,predt)


# In[238]:


from sklearn.linear_model import LogisticRegression as Log
model3 = Log(random_state=0)
model3.fit(X_train,y_train)


# In[239]:


print(model3.classes_)
print(model3.intercept_)
print(model3.coef_)


# In[241]:


pred=model3.predict(X_test)
confusion_matrix(y_test,pred)


# In[245]:


from sklearn.neighbors import KNeighborsClassifier as KNN


# In[249]:


err=np.zeros(10)
for i in range(1,11):
    classifier = KNN(n_neighbors=i)
    classifier.fit(X_train, y_train)
    pred=classifier.predict(X_test)
    err[i-1]=1-(np.trace(confusion_matrix(y_test,pred))/np.sum(confusion_matrix(y_test,pred)))
    


# In[250]:


err


# In[252]:


plt.plot(list(range(1,11)),err,color='red')
plt.scatter(list(range(1,11)),err,color='blue')
plt.title('k vs error plot')
plt.xlabel('k value')
plt.ylabel('Error');


# In[253]:


model4 = KNN(n_neighbors=5)
model4.fit(X_train, y_train)
pred=classifier.predict(X_test)
confusion_matrix(y_test,pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




