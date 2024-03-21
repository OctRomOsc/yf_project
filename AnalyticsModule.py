# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:55:09 2021

Edited on Tue Feb 27 15:18:36 2024

Title: Analytics and Forecasting Module For Ticker Stock Price

"""

'''Cell 0: Import Packages'''
import os,sys
import glob
import psycopg2
from configparser import ConfigParser
import time
import pandas as pd
import numpy as np
import math
import seaborn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.datasets import make_regression
from sklearn.metrics import silhouette_score
from sklearn.metrics import mean_absolute_error
import tqdm
import copy


#%%
'''Cell 0b: Helper Functions'''
# os.chdir(r"D:\Professional\Data Science Finance Project\yf_project")
from Database_Connect import config, connect


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE)
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse[0]



#%%
'''Cell 1: Pulling Ticker data from PostgreSQL server'''
Ticker='aapl'
query1=f'''SELECT * FROM {Ticker}_history;'''

### Outputs dataframe "data" which contains historical Ticker data (along with column_names) ###
connect(query1)

#%%
'''Cell 2: Defining Modelling Variables and Plotting historical adjusted stock closing price '''

data=data.interpolate() #NaN values found for 1981-08-10
# data.dropna(inplace=True)

data_stats=data.describe()
#####################
# For time series models which cannot reconcile missing date values, # 
# carry over previous closing price for weekends and holidays #
#####################

fullrange_datarange = pd.DataFrame(
    {"date": pd.date_range(data.date.min(), data.date.max(), freq="D").date}
)
fullrange_data = (
    pd.concat([data, fullrange_datarange], sort=False)
    .drop_duplicates(subset='date',keep="first")
    .sort_values("date")
)

fullrange_data=fullrange_data.fillna(fullrange_data.ffill())

#####################

X=data.loc[:, data.columns!='adj_close']
X=X.drop(['date'], axis=1)
X.index=pd.to_datetime(data['date'])
Y=data[['adj_close']]
Y.index=pd.to_datetime(data['date'])

chars=seasonal_decompose(Y, period = int(len(Y)/2))
f1=plt.figure(1)
plt.subplot(211)
plt.plot(Y)
plt.subplot(212)
plt.plot(Y.iloc[9850:])
f1.show()

f2=plt.figure(2)
plt.subplot(313)
chars.plot()
f2.show()


X_input=X.values
Y_input=Y.values
#%%
################################################################
## Cells 3a-3i define, train, and evaluate regression methods ##
################################################################
'''Cell 3a: Linear Regression'''

linreg = LinearRegression()
linreg.fit(X_input[:11715],Y_input[:11715])
print(linreg.coef_, linreg.intercept_)

pred=linreg.predict(X_input[11715:])
y_pred=pd.DataFrame(pred,index=Y.index[11715:])

# f=plt.figure(1)
plt.subplot(211)
plt.plot(Y, label='Train')
plt.plot(y_pred, label='Test')

plt.subplot(212)
plt.plot(Y[11715:], label='Train')
plt.plot(y_pred, label='Test')
plt.legend()
plt.title('Linear Regression')

linreg_mae=mean_absolute_error(Y_input[11715:], y_pred)
linreg_mape=mean_absolute_percentage_error(Y_input[11715:], y_pred)
linreg_rmse=calculate_rmse(Y_input[11715:], y_pred)

print(f'MAE: {round(linreg_mae,3)}\nMAPE: {round(linreg_mape,3)}%\nRMSE: '+ 
      f'{round(linreg_rmse,3)}')

# f.show()

#%%
'''Cell 3b: Gradient Boost'''

gradboost = GradientBoostingRegressor(random_state=5,n_estimators=100)
gradboost.fit(X_input[:11715],Y_input[:11715])
# print(gradboost.coef_, gradboost.intercept_)

pred=gradboost.predict(X_input[11715:])
y_pred=pd.DataFrame(pred,index=Y.index[11715:])

# f=plt.figure(1)
plt.subplot(211)
plt.plot(Y, label='Train')
plt.plot(y_pred, label='Test')

plt.subplot(212)
plt.plot(Y[11715:], label='Train')
plt.plot(y_pred, label='Test')
plt.legend()
plt.title('Gradient Boost')

gradboost_mae=mean_absolute_error(Y_input[11715:], y_pred)
gradboost_mape=mean_absolute_percentage_error(Y_input[11715:], y_pred)
gradboost_rmse=calculate_rmse(Y_input[11715:], y_pred)

print(f'MAE: {round(gradboost_mae,3)}\nMAPE: {round(gradboost_mape,3)}%\nRMSE: '+ 
      f'{round(gradboost_rmse,3)}')

# f.show()

#%%
'''Cell 3c: Adaptive Boost'''

adaboost = AdaBoostRegressor(random_state=5,n_estimators=10)
adaboost.fit(X_input[:11715],Y_input[:11715])
# print(adaboost.coef_, adaboost.intercept_)

pred=adaboost.predict(X_input[11715:])
y_pred=pd.DataFrame(pred,index=Y.index[11715:])

# f=plt.figure(1)
plt.subplot(211)
plt.plot(Y, label='Train')
plt.plot(y_pred, label='Test')

plt.subplot(212)
plt.plot(Y[11715:], label='Train')
plt.plot(y_pred, label='Test')
plt.legend()
plt.title('Adaptive Boost')

adaboost_mae=mean_absolute_error(Y_input[11715:], y_pred)
adaboost_mape=mean_absolute_percentage_error(Y_input[11715:], y_pred)
adaboost_rmse=calculate_rmse(Y_input[11715:], y_pred)

print(f'MAE: {round(adaboost_mae,3)}\nMAPE: {round(adaboost_mape,3)}%\nRMSE: '+ 
      f'{round(adaboost_rmse,3)}')

# f.show()

#%%
'''Cell 3d: Random Forest'''

randomforest = RandomForestRegressor(max_depth=6,min_samples_leaf=5,n_estimators=1000,random_state=5)
randomforest.fit(X_input[:11715],np.ravel(Y_input[:11715]))
# print(randomforest.coef_, randomforest.intercept_)

pred=randomforest.predict(X_input[11715:])
y_pred=pd.DataFrame(pred,index=Y.index[11715:])

# f=plt.figure(1)
plt.subplot(211)
plt.plot(Y, label='Train')
plt.plot(y_pred, label='Test')

plt.subplot(212)
plt.plot(Y[11715:], label='Train')
plt.plot(y_pred, label='Test')
plt.legend()
plt.title('Random Forest')

randomforest_mae=mean_absolute_error(Y_input[9000:], y_pred)
randomforest_mape=mean_absolute_percentage_error(Y_input[9000:], y_pred)
randomforest_rmse=calculate_rmse(Y_input[9000:], y_pred)

print(f'MAE: {round(randomforest_mae,3)}\nMAPE: {round(randomforest_mape,3)}%\nRMSE: '+ 
      f'{round(randomforest_rmse,3)}')

# f.show()
#%%
'''Cell 3e: Single-hidden layer Feed-Forward Neural Network Regression'''


X_tens=torch.tensor(X_input)
Y_tens=torch.tensor(Y_input)
X_train=X_tens[:11715]
y_train=Y_tens[:11715]    


# training parameters
n_epochs = 5 # number of epochs to run
batch_size = 100  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)    

model=nn.Sequential(nn.Linear(len(X_train.T), 10, bias=True),nn.ReLU(),nn.Linear(10, 1, bias=True))
torch.manual_seed(5)

loss_fn = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=0.01)

# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []

for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            X_batch = X_train[start:start+batch_size].to(torch.float32)
            y_batch = y_train[start:start+batch_size].to(torch.float32)
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            opt.zero_grad()
            loss.backward()
            # update weights
            opt.step()
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    model.eval()
    y_pred = model(X_tens[11715:].to(torch.float32))
    mse = loss_fn(y_pred, Y_tens[11715:])
    mse = float(mse)
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())


y_pred=pd.DataFrame(y_pred.detach().numpy(),index=Y.index[9000:])


plt.subplot(211)
plt.plot(Y, label='Train')
plt.plot(y_pred, label='Test')

plt.subplot(212)
plt.plot(Y[11715:], label='Train')
plt.plot(y_pred, label='Test')
plt.legend()
plt.title('Neural Network')

neuralnet_mae=mean_absolute_error(Y_input[9000:], y_pred)
neuralnet_mape=mean_absolute_percentage_error(Y_input[9000:], y_pred)
neuralnet_rmse=calculate_rmse(Y_input[9000:], y_pred)

print(f'MAE: {round(neuralnet_mae,3)}\nMAPE: {round(neuralnet_mape,3)}%\nRMSE: '+ 
      f'{round(neuralnet_rmse,3)}')

#%%
'''Cell 3f: Moving Average Model'''


X_full=fullrange_data.loc[:, data.columns!='adj_close']
X_full=X_full.drop(['date'], axis=1)
X_full.index=pd.to_datetime(fullrange_data['date'])
Y_full=fullrange_data[['adj_close']]
Y_full.index=pd.to_datetime(fullrange_data['date'])


X_full_input=X_full.values
Y_full_input=Y_full.values

movavg = ARIMA(endog=Y_full_input[:11715] , order=(0,0,1)).fit()
print(movavg.summary())

pred=movavg.predict(start_date=str(Y_full.index[11715].strftime('%Y-%m-%d')), end_date=str(Y_full.index[-1].strftime('%Y-%m-%d')))
# pred=movavg.predict(start=11715, end=14643, dynamic=True)
y_pred=pd.DataFrame(pred[8657:],index=Y_full.index[11715:])

# f=plt.figure(1)
plt.subplot(211)
plt.plot(Y_full, label='Train')
plt.plot(y_pred, label='Test')

plt.subplot(212)
plt.plot(Y_full[11715:], label='Train')
plt.plot(y_pred, label='Test')
plt.legend()
plt.title('Moving Average')

movavg_mae=mean_absolute_error(Y_full_input[11715:], y_pred)
movavg_mape=mean_absolute_percentage_error(Y_full_input[11715:], y_pred)
movavg_rmse=calculate_rmse(Y_full_input[11715:], y_pred)

print(f'MAE: {round(movavg_mae,3)}\nMAPE: {round(movavg_mape,3)}%\nRMSE: '+ 
      f'{round(movavg_rmse,3)}')

#%%
'''Cell 3g: ARIMA Model'''



X_full=fullrange_data.loc[:, data.columns!='adj_close']
X_full=X_full.drop(['date'], axis=1)
X_full.index=pd.to_datetime(fullrange_data['date'])
Y_full=fullrange_data[['adj_close']]
Y_full.index=pd.to_datetime(fullrange_data['date'])


X_full_input=X_full.values
Y_full_input=Y_full.values

arima = ARIMA(endog=Y_full_input[:11715] , order=(1,1,1)).fit()
print(arima.summary())

pred=arima.predict(start_date=str(Y_full.index[11715].strftime('%Y-%m-%d')), end_date=str(Y_full.index[-1].strftime('%Y-%m-%d')))
y_pred=pd.DataFrame(pred[8657:],index=Y_full.index[11715:])

# f=plt.figure(1)
plt.subplot(211)
plt.plot(Y_full, label='Train')
plt.plot(y_pred, label='Test')

plt.subplot(212)
plt.plot(Y_full[11715:], label='Train')
plt.plot(y_pred, label='Test')
plt.legend()
plt.title('ARIMA')

arima_mae=mean_absolute_error(Y_full_input[11715:], y_pred)
arima_mape=mean_absolute_percentage_error(Y_full_input[11715:], y_pred)
arima_rmse=calculate_rmse(Y_full_input[11715:], y_pred)

print(f'MAE: {round(arima_mae,3)}\nMAPE: {round(arima_mape,3)}%\nRMSE: '+ 
      f'{round(arima_rmse,3)}')

#%%
'''Cell 3h: SARIMAX Model'''



X_full=fullrange_data.loc[:, data.columns!='adj_close']
X_full=X_full.drop(['date'], axis=1)
X_full.index=pd.to_datetime(fullrange_data['date'])
Y_full=fullrange_data[['adj_close']]
Y_full.index=pd.to_datetime(fullrange_data['date'])


X_full_input=X_full.values
Y_full_input=Y_full.values

sarimax = SARIMAX(endog=Y_full_input[:11715] , order=(1,1,1), seasonal_order=(1,1,1,14)).fit()
print(sarimax.summary())

pred=sarimax.predict(start_date=str(Y_full.index[11715].strftime('%Y-%m-%d')), end_date=str(Y_full.index[-1].strftime('%Y-%m-%d')))
y_pred=pd.DataFrame(pred[8657:],index=Y_full.index[11715:])

# f=plt.figure(1)
plt.subplot(211)
plt.plot(Y_full, label='Train')
plt.plot(y_pred, label='Test')

plt.subplot(212)
plt.plot(Y_full[11715:], label='Train')
plt.plot(y_pred, label='Test')
plt.legend()
plt.title('SARIMAX')

sarimax_mae=mean_absolute_error(Y_full_input[11715:], y_pred)
sarimax_mape=mean_absolute_percentage_error(Y_full_input[11715:], y_pred)
sarimax_rmse=calculate_rmse(Y_full_input[11715:], y_pred)

print(f'MAE: {round(sarimax_mae,3)}\nMAPE: {round(sarimax_mape,3)}%\nRMSE: '+ 
      f'{round(sarimax_rmse,3)}')
#%%
'''Cell 3i: Holt-Winters' Exponential Smoothing Model'''



X_full=fullrange_data.loc[:, data.columns!='adj_close']
X_full=X_full.drop(['date'], axis=1)
X_full.index=pd.to_datetime(fullrange_data['date'])
Y_full=fullrange_data[['adj_close']]
Y_full.index=pd.to_datetime(fullrange_data['date'])


X_full_input=X_full.values
Y_full_input=Y_full.values

expsmo = ExponentialSmoothing(Y_full_input[:11715], seasonal='additive', seasonal_periods=7).fit(smoothing_level=0.5)#, seasonal='additive', seasonal_periods=7).fit()
print(expsmo.summary())

pred=expsmo.predict(start=11715, end=len(Y_full_input)-1)
y_pred=pd.DataFrame(pred,index=Y_full.index[11715:])

# f=plt.figure(1)
plt.subplot(211)
plt.plot(Y_full, label='Train')
plt.plot(y_pred, label='Test')

plt.subplot(212)
plt.plot(Y_full[11715:], label='Train')
plt.plot(y_pred, label='Test')
plt.legend()
plt.title('Holt-Winters')

expsmo_mae=mean_absolute_error(Y_full_input[11715:], y_pred)
expsmo_mape=mean_absolute_percentage_error(Y_full_input[11715:], y_pred)
expsmo_rmse=calculate_rmse(Y_full_input[11715:], y_pred)

print(f'MAE: {round(expsmo_mae,3)}\nMAPE: {round(expsmo_mape,3)}%\nRMSE: '+ 
      f'{round(expsmo_rmse,3)}')
#%%
###############################################################
## Cells 4a-4e predict on trained models, producing forecast ##
###############################################################
'''Cell 4a: Linear Regression'''
#%%
'''Cell 5: Statistical Analysis of Forecasts'''