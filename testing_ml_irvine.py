
# coding: utf-8

# In[2]:


import csv
import pickle
import pandas as pd
import numpy as np

import requests
import json

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
import sklearn.preprocessing
from sklearn import preprocessing


# In[6]:


data=pd.read_csv("1.csv")
data.head()


# In[7]:


data.columns


# In[8]:


data=data.drop_duplicates(subset=['ADDRESS'], keep='first')
data


# In[9]:


data=data.rename(columns={'URL (SEE http://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)':'url',
                          'PROPERTY TYPE':'type',
                         'CITY':'city',
                         'ZIP':'zip',
                         'PRICE':'price',
                         'BEDS':'beds',
                         'BATHS':'baths',
                         'SQUARE FEET':'sqrft',
                         'LOT SIZE':'lot',
                         'YEAR BUILT':'built',
                         'DAYS ON MARKET':'dom',
                         '$/SQUARE FEET':'$/sqrft',
                         'HOA/MONTH':'hoa',
                         'LATITUDE':'lat',
                         'LONGITUDE':'lon'})


# In[10]:


data['full_address'] = data['ADDRESS'] + ", " + data['city'] + ", " + data['STATE']
data.head()


# In[91]:


# api_key='AIzaSyAOjSf4Tk_StWcxTANG_2Sih0IN19W9cSI'
# url="https://maps.googleapis.com/maps/api/geocode/json?address={}&key={}"
# url


# In[92]:


# lat_list=[]
# lon_list=[]


# In[93]:



# for i in data.full_address:
#     response=requests.get(url.format(i,api_key)).json()
#     print(json.dumps(response, indent=4, sort_keys=True))
#     lat=response["results"][0]["geometry"]["location"]["lat"]
#     lat_list.append(lat)
#     lon=response["results"][0]["geometry"]["location"]["lng"]
#     lon_list.append(lon)



# In[94]:


# len(lat_list)


# In[95]:


# data['lat_backup']=pd.Series(lat_list)
# data['lon_backup']=pd.Series(lon_list)
# data


# In[11]:


data=data[['type','city','zip','price','beds','baths','sqrft','lot','built',
          'dom','$/sqrft','hoa','lat','lon']]


# In[12]:


data.describe()


# In[13]:


data.info()


# In[14]:


data['type']=data['type'].replace('Single Family Residential','sfr')
data['type']=data['type'].replace('Condo/Co-op','condo')
data['type']=data['type'].replace('Townhouse','thr')
data['type']=data['type'].replace('Multi-Family (2-4 Unit)','mfr')


# In[15]:


data.isnull().sum()


# In[16]:


data=data[data['built'].notnull()]
data.head()


# In[17]:


print(data.isnull().sum())
from numpy import nan
data[data['hoa'].isnull()]


# In[18]:


#pass 0 for hoa of NaN homes with yeaer before 2000
mask=(data['hoa'].isnull()) & (data['built']<2000)
data['hoa']=data['hoa'].mask(mask,0)


# In[19]:


data.info()


# In[20]:


data=data.set_index('zip')
data['lot medians']=data.groupby('zip')['lot'].median()
data.head()


# In[21]:


mask1=(data['lot'].isnull())
data['lot']=data['lot'].mask(mask1,data['lot medians'])
data.head()


# In[22]:


del data['lot medians']


# In[23]:


data=data.reset_index()
data.info()


# In[24]:


data.shape


# # Multicollinearity check

# In[25]:


correlations=data.corr()
plt.subplots(figsize=(10,8))
sns.heatmap(correlations,annot=True)
fig=plt.figure()
plt.show()


# beds
# baths
# sqrft
# lot
# per_sqrft
# zipcode
# types
#yr built
#hoa


#multi-collinearity: beds and sqrft/baths and sqrft/beds and baths


# In[26]:


plt.subplots(figsize=(20,8))
sns.distplot(data['price'],fit=stats.norm)

(mu,sigma)=stats.norm.fit(data['price'])
plt.legend(['Normal Distribution Params mu={} and sigma={}'.format(mu,sigma)],loc='best')
plt.ylabel('frequency')

fig=plt.figure()
plt.show()


# In[27]:


mini=data['built'].min()
maxi=data['built'].max()
print(mini,maxi)

decades_no=[]
for i in data.built:
    decades=(i-mini)/10
#     print(decades)
    decades_no.append(decades)
    
data['train_built']=pd.Series(decades_no)

data['train_built']=data['train_built'].round(0)
data.head()


# In[28]:


len(data)


# # Pickled Cleaned Irvine DF Pre-Inference

# In[120]:



# data.to_pickle('irvine_data.pk1')
infile=open('irvine_data.pk1','rb')
train=pickle.load(infile)

train.head()


# In[121]:


len(train)


# # Flask Functions for Front End:
# ## Sizer Assist for Pred DF +  Min Built Return

# In[122]:


def train_flask():
    infile=open('irvine_data.pk1','rb')
    train=pickle.load(infile)
    
    cols=['zip','type','train_built','beds','baths','sqrft','lot','$/sqrft']
    x=train[cols]
    
    train['$/sqrft']=np.log1p(train['$/sqrft'])
    train['sqrft']=np.log1p(train['sqrft'])
    train['lot']=np.log1p(train['lot'])
    x=pd.get_dummies(x,columns=['zip','type','train_built'])

    return x



# In[123]:


def min_built():
    infile=open('irvine_data.pk1','rb')
    train=pickle.load(infile)
    
    #for integrating: load all pickle files
    #output is a list of minimums
    
    irvine_mini=int(train['built'].min())
    
    return irvine_mini

print(train['built'].min())
type(int(train['built'].min()))

# def min_built():
#     infile=open('irvine_data.pk1','rb')
#     train=pickle.load(infile)
#     f=open('whatever')
#     train_tust=pickle.load(f)
    
#     #for integrating: load all pickle files
#     #output is a list of minimums
    
#     irvine_mini=train['built'].min()
#     tustin_mini=train_tustin['built'].min()
    
#     return [irvine_mini,tustin_mini]


# # Inference Tests

# In[124]:


# mini=train['built'].min()
# maxi=train['built'].max()
# print(mini,maxi)

# decades_no=[]
# for i in train.built:
#     decades=(i-mini)/10
# #     print(decades)
#     decades_no.append(decades)
    
# train['train_built']=pd.Series(decades_no)

# train['train_built']=train['train_built'].round(0)
# train.head()


# In[125]:


anova_data=train[['price','train_built']]

# anova_data['train_built']=anova_data['train_built'].round(0)
# bin_series=anova_data['train_built'].value_counts()

##bin without series:
bins=pd.unique(anova_data.train_built.values)
f_test_data={grp:anova_data['price'][anova_data.train_built==grp] for grp in bins}
print(bins)

from scipy import stats

F, p=stats.f_oneway(f_test_data[0.],f_test_data[9.],f_test_data[11.],f_test_data[7.],f_test_data[10.],
                    f_test_data[12.],f_test_data[8.])

print(F,p)


k=len(pd.unique(anova_data.train_built.values))
N=len(anova_data.values)
n=anova_data['train_built'].value_counts()

#F-static: btw/within variability

DFbetween = k - 1
DFwithin = N - k
DFtotal = N - 1
print(f"degrees of freedom between: {DFbetween}")
print(f"degrees of freedom within: {DFwithin}")
print(f"degrees of freedom total: {DFtotal}")


#reject null, not all group means are equal, variance exists, include year built in ML



# In[126]:


import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

##non-zero HOA data df prep for reg. analysis, p-value, 95% CI
hoa_f_prep=train[train['hoa'].notnull()]
# hoa_f_prep.info()
dep_var=hoa_f_prep['price']
indep_var=hoa_f_prep['hoa']

indep_var=indep_var.values.reshape(-1,1)

# define the model
model = LinearRegression()

# fit the model to training data
model.fit(indep_var, dep_var)

#run p-test
params = np.append(model.intercept_,model.coef_)
predictions = model.predict(indep_var)


new_indep_var = pd.DataFrame({"Constant":np.ones(len(indep_var))}).join(pd.DataFrame(indep_var))
MSE = (sum((dep_var-predictions)**2))/(len(new_indep_var)-len(new_indep_var.columns))

var_b = MSE*(np.linalg.inv(np.dot(new_indep_var.T,new_indep_var)).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params/ sd_b

p_values =[2*(1-stats.t.cdf(np.abs(i),(len(new_indep_var)-1))) for i in ts_b]

sd_b = np.round(sd_b,3)
ts_b = np.round(ts_b,3)
p_values = np.round(p_values,3)
params = np.round(params,4)

p_test_df = pd.DataFrame()
p_test_df["Coefficients"],p_test_df["Standard Errors"],p_test_df["t values"],p_test_df["Probabilites"] = [params,sd_b,ts_b,p_values]
print(p_test_df)

# predict
dep_var_pred = model.predict(indep_var)

print(r2_score(dep_var, dep_var_pred))

#low r2 value, despite low p-val, t-statistic lookup conclusive, we look for precise predictions for upcoming ML section, it is statistically safe to disregard hoa inferentially


# In[127]:


train['price']=np.log1p(train['price'])


# In[128]:


plt.subplots(figsize=(20,8))
sns.distplot(train['price'],fit=stats.norm)


(mu,sigma)=stats.norm.fit(train['price'])

plt.legend(['Normal Distribution Params mu={} and sigma={}'.format(mu,sigma)],loc='best')
plt.ylabel('frequency')

fig=plt.figure()
stats.probplot(train['price'],plot=plt)

plt.show()


# In[129]:


# cols=['zip','type','beds','baths','sqrft','lot','$/sqrft','train_built']

cols=['zip','train_built','type','beds','baths','sqrft','lot']
x=train[cols]
y=train['price']



# In[10]:


# y=np.log1p(y)

# plt.subplots(figsize=(20,8))
# sns.distplot(y,fit=stats.norm)


# (mu,sigma)=stats.norm.fit(y)

# plt.legend(['Normal Distribution Params mu={} and sigma={}'.format(mu,sigma)],loc='best')
# plt.ylabel('frequency')

# fig=plt.figure()
# stats.probplot(y,plot=plt)

# plt.show()


# In[130]:


train['$/sqrft']=np.log1p(train['$/sqrft'])


plt.subplots(figsize=(5,5))
sns.distplot(train['$/sqrft'],fit=stats.norm)


(mu,sigma)=stats.norm.fit(train['$/sqrft'])

plt.legend(['Normal Distribution Params mu={} and sigma={}'.format(mu,sigma)],loc='best')
plt.ylabel('frequency')

fig=plt.figure()
stats.probplot(train['$/sqrft'],plot=plt)

plt.show()


# In[131]:


train['sqrft']=np.log1p(train['sqrft'])


plt.subplots(figsize=(5,5))
sns.distplot(train['sqrft'],fit=stats.norm)


(mu,sigma)=stats.norm.fit(train['sqrft'])

plt.legend(['Normal Distribution Params mu={} and sigma={}'.format(mu,sigma)],loc='best')
plt.ylabel('frequency')

fig=plt.figure()
stats.probplot(train['sqrft'],plot=plt)

plt.show()


# In[132]:


train['lot']=np.log1p(train['lot'])


plt.subplots(figsize=(5,5))
sns.distplot(train['lot'],fit=stats.norm)


(mu,sigma)=stats.norm.fit(train['lot'])

plt.legend(['Normal Distribution Params mu={} and sigma={}'.format(mu,sigma)],loc='best')
plt.ylabel('frequency')

fig=plt.figure()
stats.probplot(train['lot'],plot=plt)

plt.show()


# In[133]:


train[cols].head()


# In[134]:


x.head()


# In[135]:


x=pd.get_dummies(x,columns=['zip','type','train_built'])
x.head()


# In[136]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.20,random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score


# In[137]:


# define the model
model = LinearRegression()

# fit the model to training data
model.fit(x_train, y_train)

# predict
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)


# In[138]:


print("The R^2 score for training data is", r2_score(y_train, y_train_pred))
print("The R^2 score for testing data is", r2_score(y_test, y_test_pred))


# In[139]:


print("The train RMSE is ", mean_squared_error(y_train, y_train_pred)**0.5)
print("The test RMSE is ", mean_squared_error(y_test, y_test_pred)**0.5)


# In[140]:


dff=pd.DataFrame({"true_values": y_train, "predicted": y_train_pred, "residuals": y_train - y_train_pred})
dff


# # Check normality of residuals for IV

# In[141]:


plt.subplots(figsize=(5,5))
# plt.subplots(1,2,sharex='none')
# sns.distplot(dff['residuals'],fit=stats.norm)
# plt.subplots(1,2,sharex='none')
# stats.probplot(dff['residuals'],plot=plt)

# fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=False,sharey=False)
sns.distplot(dff['residuals'],fit=stats.norm)
(mu,sigma)=stats.norm.fit(dff['residuals'])

plt.legend(['Normal Distribution Params mu={} and sigma={}'.format(mu,sigma)],loc='best')
plt.ylabel('frequency')

fig=plt.figure()
stats.probplot(dff['residuals'],plot=plt)

plt.show()


# In[142]:


dff['true_values'].max()


# In[143]:


from sklearn.linear_model import Lasso, Ridge, ElasticNet


# In[144]:


# define the model
lasso = Lasso(random_state=42)

# fir the model to the data
lasso.fit(x_train, y_train)

# predictions
y_pred_lasso = lasso.predict(x_test)

RMSE_lasso = mean_squared_error(y_test, y_pred_lasso)**0.5
r2_lasso = r2_score(y_test, y_pred_lasso)

print(RMSE_lasso)
print(r2_lasso)


# In[145]:


# define the model
ridge = Ridge(random_state=42)

# fir the model to the data
ridge.fit(x_train, y_train)

y_train_pred=ridge.predict(x_train)   ##this one
# predictions
y_pred_ridge = ridge.predict(x_test)

RMSE_ridge = mean_squared_error(y_test, y_pred_ridge)**0.5
r2_ridge = r2_score(y_test, y_pred_ridge)

RMSE_ridge_train = mean_squared_error(y_train, y_train_pred)**0.5 #this
r2_train=r2_score(y_train, y_train_pred) #this

print(RMSE_ridge)
print(r2_ridge)

print(RMSE_ridge_train)
print(r2_train)


# In[146]:


ridge


# In[147]:


np.expm1(model.predict(x_test.iloc[0].values.reshape(1,-1)))


# In[148]:


x_test.iloc[0]


# In[149]:


np.expm1(y_test.iloc[0])


# In[150]:


# import pickle
# ridge_pickle = open("irvine_ridge_model.pkl","wb")
# pickle.dump(ridge, ridge_pickle)


# In[151]:


ridge_model = open("irvine_ridge_model.pkl","rb")
ridge = pickle.load(ridge_model)


# In[152]:


beds = []
baths = []
sqrft = []
lot = []
# per_sqrft = []
zipcode = ""
types = ""
year_built=""

beds.append(input("Bedrooms: "))
baths.append(input("Bathrooms: "))
sqrft.append(input("Squarefeet: "))
lot.append(input("Lot Size: "))
# per_sqrft.append(input("$'s per Square Feet': "))
city=input("City: ")
zipcode = input("Zipcode: ")
types = input("House Type: ")
year_built=input("Built: ")


# In[69]:


int_year_built=int(year_built)


# In[155]:


# def min_built():
#     infile=open('irvine_data.pk1','rb')
#     train=pickle.load(infile)
#     f=open('whatever')
#     train_tust=pickle.load(f)
    
#     #for integrating: load all pickle files
#     #output is a list of minimums
    
#     irvine_mini=train['built'].min()
#     tustin_mini=train_tustin['built'].min()
    
#     return [irvine_mini,tustin_mini]
temp=min_built()
def temp_bin(num):
    temp_yr_bin=round((num-temp)/10,0)
    return temp_yr_bin

# def binned_year(num):
    
#     minimums=min_built()
    
#     if city=="Irvine" or city=="irvine":
#         city_min=minimum[0]
#     elif city=="tustin" or 'Tustin'
#         city_min=minimum[1]
#         #etc

#     binned_yr=round((num-city_min)/10,0)
    
#     return binned_yr
print(temp_bin(int_year_built))
print(type(temp_bin(int_year_built)))


# In[162]:


user_dictionary={'zip':zipcode,'type':types,'built':str(temp_bin(int_year_built)),'beds':beds,'baths':baths,'sqrft':sqrft,'lot':lot}
user_df=pd.DataFrame(user_dictionary)
user_df_fit=pd.get_dummies(user_df,columns=['zip','type','built'])


# In[163]:


type(user_dictionary['built'])


# In[164]:


user_df_fit


# In[166]:


x.columns

for i in x.columns:
    if i in user_df_fit.columns:
        pass
    else:
        user_df_fit[i]=0

# user_df_fit=user_df_fit.set_index('$/sqrft')

user_df_fit


# In[167]:


# np.expm1(ridge.predict(user_df_fit))
np.expm1(ridge.predict(user_df_fit))


# np.expm1(model.predict(x_test.iloc[0].values.reshape(1,-1)))


# In[99]:


# form = HouseForms(csrf_enabled=False)
# if (request.method == "POST") and (form.validate()):

ridge_model = open("irvine_ridge_model.pkl","rb")
ridge = pickle.load(ridge_model)

user_dictionary={'zip':[92618],'built':[10],'type':['sfr'],'beds':[4],'baths':[3],
'sqrft':[5000],'lot':[8000],'$/sqrft':[500]}

user_df=pd.DataFrame(user_dictionary)
user_df_fit=pd.get_dummies(user_df,columns=['zip','type','built'])
n=train_flask()
for i in n.columns:
    if i in user_df_fit.columns:
        pass
    else:
        user_df_fit[i]=0
print(user_df_fit)
user_df_fit=user_df_fit.set_index('beds')

prediction = np.expm1(ridge.predict(user_df_fit))
# prediction=model.predict(user_df_fit)
# p = round(prediction[0],2)
print(prediction)
# else:


# In[119]:


print(int(prediction))


# In[101]:


user_df_fit.columns

