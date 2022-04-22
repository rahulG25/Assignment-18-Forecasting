#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime,time
import warnings
import itertools
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
from pylab import rcParams
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smf


# # Data collection and description

# In[2]:


coca = pd.read_excel("CocaCola_Sales_Rawdata.xlsx")


# In[3]:


coca1 = coca.copy()


# In[4]:


coca1.head().T


# In[5]:


coca1.isnull().sum()


# In[6]:


coca1.dtypes


# In[7]:


coca1.describe().T


# In[8]:


temp = coca1.Quarter.str.replace(r'(Q\d)_(\d+)', r'19\2-\1')


# In[9]:


coca1['quater'] = pd.to_datetime(temp).dt.strftime('%b-%Y')


# In[10]:


coca1.head()


# In[11]:


coca1 = coca1.drop(['Quarter'], axis=1)


# In[12]:


coca1.reset_index(inplace=True)


# In[13]:


coca1['quater'] = pd.to_datetime(coca1['quater'])


# In[14]:


coca1 = coca1.set_index('quater')


# In[15]:


coca1.head()


# In[16]:


coca1['Sales'].plot(figsize=(20, 8),color='green',marker='o')
plt.show()


# In[17]:


for i in range(2,10,2):
    coca1["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)


# In[18]:


ts_add = seasonal_decompose(coca1.Sales,model="additive")
fig = ts_add.plot()
plt.show()


# In[19]:


ts_mul = seasonal_decompose(coca1.Sales,model="multiplicative")
fig = ts_mul.plot()
plt.show()


# In[20]:


ts_mul = seasonal_decompose(coca1.Sales,model="multiplicative")
fig = ts_mul.plot()
plt.show()


# In[21]:


tsa_plots.plot_pacf(coca1.Sales, lags=10,color='black')


# In[22]:


tsa_plots.plot_acf(coca1.Sales, lags=30,color='red')


# # Building Time series forecasting with ARIMA

# In[23]:


X = coca1['Sales'].values


# In[24]:


size = int(len(X) * 0.66)


# In[25]:


train, test = X[0:size], X[size:len(X)]


# In[26]:


model = ARIMA(train, order=(5,1,0))


# In[27]:


model_fit = model.fit(disp=0)


# In[28]:


print(model_fit.summary())


# 
# This summarizes the coefficient values used as well as the skill of the fit on the on the in-sample observations
# 

# In[29]:


residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde',color='red')
pyplot.show()
print(residuals.describe())


# The plot of the residual errors suggests that there may still be some trend information not captured by the model
# The results show that indeed there is a bias in the prediction (a non-zero mean in the residuals)

# # Rolling Forecast ARIMA Model

# In[30]:


history = [x for x in train]


# In[31]:


predictions = list()


# In[32]:


for t in range(len(test)):
	model = ARIMA(history, order=(1,1,1))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))


# In[33]:


error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)


# In[34]:


pyplot.plot(test)
pyplot.plot(predictions, color='orange')
pyplot.show()


# A line plot is created showing the expected values (blue) compared to the rolling forecast predictions (red). We can see the values show some trend and are in the correct scale
# 

# # Comparing Multiple Models

# In[35]:


coca2 = pd.get_dummies(coca, columns = ['Quarter'])


# In[36]:


coca2.columns = ['Sales','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q3','Q3','Q3','Q3','Q3','Q3','Q3','Q3','Q3','Q3','Q4','Q4','Q4','Q4','Q4','Q4','Q4','Q4','Q4','Q4']


# In[37]:


coca2.head().T


# In[38]:


t= np.arange(1,43)


# In[39]:


coca2['t'] = t


# In[40]:


coca2['t_sq'] = coca2['t']*coca2['t']


# In[41]:


log_Sales=np.log(coca2['Sales'])


# In[42]:


coca2['log_Sales']=log_Sales


# In[43]:


coca2.head().T


# In[44]:


train1, test1 = np.split(coca2, [int(.67 *len(coca2))])


# In[53]:


linear= smf.ols('Sales ~ t',data=train1).fit()
predlin=pd.Series(linear.predict(pd.DataFrame(test1['t'])))
rmselin=np.sqrt((np.mean(np.array(test1['Sales'])-np.array(predlin))**2))
rmselin


# In[54]:


quad=smf.ols('Sales~t+t_sq',data=train1).fit()
predquad=pd.Series(quad.predict(pd.DataFrame(test1[['t','t_sq']])))
rmsequad=np.sqrt(np.mean((np.array(test1['Sales'])-np.array(predquad))**2))
rmsequad


# In[55]:


expo=smf.ols('log_Sales~t',data=train1).fit()
predexp=pd.Series(expo.predict(pd.DataFrame(test1['t'])))
rmseexpo=np.sqrt(np.mean((np.array(test1['Sales'])-np.array(np.exp(predexp)))**2))
rmseexpo


# In[57]:


additive= smf.ols('Sales~ Q1+Q2+Q3+Q4',data=train1).fit()
predadd=pd.Series(additive.predict(pd.DataFrame(test1[['Q1','Q2','Q3','Q4']])))
rmseadd=np.sqrt(np.mean((np.array(test1['Sales'])-np.array(predadd))**2))
rmseadd


# In[59]:


addlinear= smf.ols('Sales~t+Q1+Q2+Q3+Q4',data=train1).fit()
predaddlinear=pd.Series(addlinear.predict(pd.DataFrame(test1[['t','Q1','Q2','Q3','Q4']])))
rmseaddlinear=np.sqrt(np.mean((np.array(test1['Sales'])-np.array(predaddlinear))**2))
rmseaddlinear


# In[61]:


addquad=smf.ols('Sales~t+t_sq+Q1+Q2+Q3+Q4',data=train1).fit()
predaddquad=pd.Series(addquad.predict(pd.DataFrame(test1[['t','t_sq','Q1','Q2','Q3','Q4']])))
rmseaddquad=np.sqrt(np.mean((np.array(test1['Sales'])-np.array(predaddquad))**2))
rmseaddquad


# In[63]:


mulsea=smf.ols('log_Sales~Q1+Q2+Q3+Q4',data=train1).fit()
predmul= pd.Series(mulsea.predict(pd.DataFrame(test1[['Q1','Q2','Q3','Q4']])))
rmsemul= np.sqrt(np.mean((np.array(test1['Sales'])-np.array(np.exp(predmul)))**2))
rmsemul


# In[65]:


mullin= smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data=train1).fit()
predmullin= pd.Series(mullin.predict(pd.DataFrame(test1[['t','Q1','Q2','Q3','Q4']])))
rmsemulin=np.sqrt(np.mean((np.array(test1['Sales'])-np.array(np.exp(predmullin)))**2))
rmsemulin


# In[67]:


mul_quad= smf.ols('log_Sales~t+t_sq+Q1+Q2+Q3+Q4',data=train1).fit()
pred_mul_quad= pd.Series(mul_quad.predict(test1[['t','t_sq','Q1','Q2','Q3','Q4']]))
rmse_mul_quad=np.sqrt(np.mean((np.array(test1['Sales'])-np.array(np.exp(pred_mul_quad)))**2))
rmse_mul_quad


# # Conclusion

# In[68]:


output = {'Model':pd.Series(['rmse_mul_quad','rmseadd','rmseaddlinear','rmseaddquad','rmseexpo','rmselin','rmsemul','rmsemulin','rmsequad']),
          'Values':pd.Series([rmse_mul_quad,rmseadd,rmseaddlinear,rmseaddquad,rmseexpo,rmselin,rmsemul,rmsemulin,rmsequad])}


# In[69]:


rmse=pd.DataFrame(output)


# In[70]:


print(rmse)


# Additive seasonality with quadratic trend has the best RMSE value
