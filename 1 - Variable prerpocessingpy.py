#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 14:02:21 2023

@author: Basil
"""
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas_ta as ta
import pyti
from ta.volatility import average_true_range
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import random
from scipy.stats import normaltest

#Set seed
random.seed(1)

data = pd.read_csv("/Users/Basil/Thesis2023/Thesis phase/^ALL CODE/^FINAL INPUT DATA/Experiment 1/SPY.csv")
data2 = pd.read_csv("/Users/Basil/Thesis2023/Thesis phase/^ALL CODE/^FINAL INPUT DATA/Experiment 1/SPY.csv")


#Visualise the data for Adj Close
data2['Date'] = pd.to_datetime(data2['Date'])

data2.set_index('Date', inplace=True)
# Create the plot
plt.figure(figsize=(6, 6))
plt.plot(data2.index, data2['Adj Close'], color='blue')

# Customize the plot
plt.title('Adjusted stock price over time')
plt.xlabel('Date')
plt.ylabel('Adjusted price')
plt.grid(True)
plt.tight_layout()
plt.show()

#Visualise the data for Volume
# Create the plot
plt.figure(figsize=(6, 6))
plt.plot(data2.index, data2['Volume'], color='blue')

# Customize the plot
plt.title('Trading volume over time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.grid(True)
plt.tight_layout()
plt.show()







#Here, we follow the setup as described in the thesis document
#Open, High, Low, Adj Close, Volume are already present in the dataset

#Create daily returns
adjprice = data['Adj Close']
data['daily_return'] = adjprice.pct_change(1)
daily_return = adjprice.pct_change(1)


#Create 5 Day Momentum
data['momentum_5d'] = data['Adj Close'].pct_change(periods=5)


#Create 14 Day Simple Moving Average
data['ma_14'] = data['Adj Close'].rolling(window=14).mean()

#Create 14 Day Exponential Moving Average
#N = number of days in EMA set at 14
#k = 2 / (N+1) set at 0.4 (as done by Sethia and Raut)
#length = length of input
#Define the function for EMA
def ema_helper(prices, N, k, length):
    if len(prices) == length-N:
        return prices[0]
    res_ema = [p for p in prices[:N]] # this keeps the ema
    for t in range(N, length):
        res_ema.append(prices[t] * k + res_ema[t-1] * (1 - k))
    return res_ema
data['ema_14'] = pd.Series(ema_helper(adjprice,14,0.4,len(adjprice)))


#Create 14 Day Bollinger Bands
def get_sma(prices, rate):
    return prices.rolling(rate).mean()
sma = get_sma(adjprice, 14) #Get 14 day SMA
def get_sma(prices, rate):
    return prices.rolling(rate).mean()
def get_bollinger_bands(prices, rate=14):
    sma = get_sma(adjprice, rate) #Get SMA for 20 days
    std = prices.rolling(rate).std() #Get rolling standard deviation for 20 days
    bollinger_up = sma + std * 2 #Calculate top band
    bollinger_down = sma - std * 2 #Calculate bottom band
    return bollinger_up, bollinger_down
data['bollinger_up'], data['bollinger_down'] = get_bollinger_bands(adjprice)


#Create 14 Day Fast, Slow and Smoothed Slow Stochastic Indicators
k_period = 14
d_period = 3
#Add a "n_high" column with max value of previous 14 periods
n_high = data['High'].rolling(k_period).max()
#Add an "n_low" column with min value of previous 14 periods
n_low = data['Low'].rolling(k_period).min()
#Use the min/max values to calculate the %k line
k = (data['Close'] - n_low) * 100 / (n_high - n_low)
#Use the %k to calculates a SMA over the past 3 values of %k
d = k.rolling(d_period).mean()
#Use d as input to calculate the smoothed slow
ds = d.rolling(d_period).mean()
#Attach to the dataframe
data['fast_indc'] = k
data['slow_indc'] = d
data['smooth_indc'] = ds


#Create Past 8 Weekly Returns
data['ret_past8weeks'] = adjprice.pct_change(periods=40).rolling(window=40).sum()


#Create Past 2 Monthly Returns
data['ret_past2months'] = adjprice.pct_change(periods=42).rolling(window=42).sum()


#Create 14 Day Moving Average Convergence Divergence
exp1 = data['Adj Close'].ewm(span=12, adjust=False).mean()
exp2 = data['Adj Close'].ewm(span=26, adjust=False).mean()
data['macd14'] = exp1 - exp2
data['signal_line_14'] = data['macd14'].ewm(span=9, adjust=False).mean()
data['hist_macd14'] = data['macd14'] - data['signal_line_14']


#Create 21 Day Moving Average Convergence Divergence
exp1 = data['Adj Close'].ewm(span=12, adjust=False).mean()
exp2 = data['Adj Close'].ewm(span=33, adjust=False).mean()
data['macd21'] = exp1 - exp2
data['signal_line_21'] = data['macd21'].ewm(span=9, adjust=False).mean()
data['hist_macd21'] = data['macd21'] - data['signal_line_21']


#Create Pivot Points, Support Levels & Resistance Levels
pp = (data['High'] + data['Low'] + data['Close']) / 3
data['pp'] = pp
data['res1'] = (2 * data['pp']) - data['Low']
data['sup1'] = (2 * data['pp']) - data['High']
data['res2'] = data['pp'] + (data['High'] - data['Low'])
data['sup2'] = data['pp'] - (data['High'] - data['Low'])
data['res3'] = data['High'] + 2*(data['pp'] - data['Low'])
data['sup3'] = data['Low'] - 2*(data['High'] - data['pp'])


#Create 14 Day True Average
atr = average_true_range(data['High'], data['Low'], data['Close'], window=14)
atr = atr.replace(0,np.nan)
data['true_avg_14'] = atr


#Create 14 Day Relative Strength
rsi = ta.rsi(adjprice, period=14)
data['rsi'] = rsi


#Create On Balance Volume
data['obv'] = np.where(data['Close'] > data['Close'].shift(1), data['Volume'], 
np.where(data['Close'] < data['Close'].shift(1), -data['Volume'], 0)).cumsum()


#Create 7, 14 & 21 Day Up & Down Trending Fibonacci Retracement at 38.2, 50, 61.8%
#7 Days
highs = data['High'].rolling(7).max()
lows = data['Low'].rolling(7).min()

diff = np.array(highs - lows)

data['up_levels7_382'] = highs - (diff * 0.382)
data['down_levels7_382'] = lows + (diff * 0.382)
data['up_levels7_50'] = highs - (diff * 0.5)
data['down_levels7_50'] = lows + (diff * 0.5)
data['up_levels7_618'] = highs - (diff * 0.618)
data['down_levels7_618'] = lows + (diff * 0.618)

#14 Days
highs = data['High'].rolling(14).max()
lows = data['Low'].rolling(14).min()

diff = np.array(highs - lows)

data['up_levels14_382'] = highs - (diff * 0.382)
data['down_levels14_382'] = lows + (diff * 0.382)
data['up_levels14_50'] = highs - (diff * 0.5)
data['down_levels14_50'] = lows + (diff * 0.5)
data['up_levels14_618'] = highs - (diff * 0.618)
data['down_levels14_618'] = lows + (diff * 0.618)

#21 Days
highs = data['High'].rolling(21).max()
lows = data['Low'].rolling(21).min()

diff = np.array(highs - lows)

data['up_levels21_382'] = highs - (diff * 0.382)
data['down_levels21_382'] = lows + (diff * 0.382)
data['up_levels21_50'] = highs - (diff * 0.5)
data['down_levels21_50'] = lows + (diff * 0.5)
data['up_levels21_618'] = highs - (diff * 0.618)
data['down_levels21_618'] = lows + (diff * 0.618)


#Create 3 Day Rate Of Change
data['rate_change3'] = data['Adj Close'].pct_change(periods=3)

del adjprice, atr, d, d_period, daily_return, diff, ds, exp1, exp2, highs, k, k_period, lows, n_high, n_low, pp, rsi, sma


#Set the date column as index
data.set_index('Date', inplace=True)


#Check out the data
print(data.head()) #NaN values are present
print(data.tail())

#Handle NaN values
data = data.dropna()


#Create the variables which we want to predict
months = data['Adj Close'].shift(-21)
y_1month = (months > data['Adj Close']).astype(int)
months = data['Adj Close'].shift(-42)
y_2month = (months > data['Adj Close']).astype(int)
months = data['Adj Close'].shift(-63)
y_3month = (months > data['Adj Close']).astype(int)
months = data['Adj Close'].shift(-126)
y_6month = (months > data['Adj Close']).astype(int)

data.to_csv('/Users/Basil/Thesis2023/Thesis phase/^ALL CODE/^STORE DATASETS/First Experiment/data_processed.csv', index=False)

#Apply Z-Score Standardisation
data_standard = data.apply(zscore)

#Minimax normalisation 
scaler = MinMaxScaler()
data_nrm = pd.DataFrame(scaler.fit_transform(data_standard), columns=data_standard.columns)

#----------------------------------------------------------------------------#

data_nrm.to_csv('/Users/Basil/Thesis2023/Thesis phase/^ALL CODE/^STORE DATASETS/First Experiment/data_nrm.csv', index=False)
y_1month.to_csv('/Users/Basil/Thesis2023/Thesis phase/^ALL CODE/^STORE DATASETS/First Experiment/y_1month.csv', index=False)
y_2month.to_csv('/Users/Basil/Thesis2023/Thesis phase/^ALL CODE/^STORE DATASETS/First Experiment/y_2month.csv', index=False)
y_3month.to_csv('/Users/Basil/Thesis2023/Thesis phase/^ALL CODE/^STORE DATASETS/First Experiment/y_3month.csv', index=False)
y_6month.to_csv('/Users/Basil/Thesis2023/Thesis phase/^ALL CODE/^STORE DATASETS/First Experiment/y_6month.csv', index=False)

