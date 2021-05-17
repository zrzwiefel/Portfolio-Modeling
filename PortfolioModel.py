#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import fredapi
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlfinlab.portfolio_optimization.modern_portfolio_theory import CriticalLineAlgorithm
from datetime import datetime
from os import path


# In[ ]:


fred = fredapi.Fred(api_key='**********')


# In[ ]:


def fill_nan(series, iteration):
    
    closest_future_entry = []
    future_index = 0 
    
    closest_past_entry = []
    past_index = 0
    
    while closest_future_entry == []:
        if not math.isnan(series[iteration + future_index]):
            closest_future_entry.append(series[iteration + future_index])
        else:
            future_index += 1
            
    while closest_past_entry == []:
        if not math.isnan(series[iteration - past_index]):
            closest_past_entry.append(series[iteration - past_index])
        else:
            past_index += -1
    
    series[iteration] = (closest_future_entry[0] + closest_past_entry[0])/2


# In[ ]:


class FREDseries:
    #For series which are requested from the FRED API
    def __init__(self, series_name):
        self.name = series_name
        self.data = fred.get_series(series_name)
        self.series_type = 'FREDseries'
        for j in range(len(self.data)):
            if math.isnan(self.data[j]):
                fill_nan(self.data, j)
                
class MarketSeries:
    #Market or security series read from a local file
    def __init__(self, series_name, file_path):
        self.name = series_name
        #file_path =  'individual_stocks_5yr\\'+ series_name + '_data.csv'
        self.df = pd.read_csv(file_path)    
        date = self.df['date']
        close = self.df['close']      
        data = pd.Series(data=close, name=series_name)
        data.index = [datetime.strptime(d, '%Y-%m-%d') for d in date]
        self.data = data
        self.series_type = 'MarketSeries'
        
        
class Asset(FREDseries, MarketSeries):
    #Takes either a FRED API series or local file series, along with a price amount of that asset held
    def __init__(self, asset, amount, series_type):
        self.amount = amount
        if series_type == 'FREDseries':
            FREDseries.__init__(self, series_name = asset)
        elif series_type == 'MarketSeries':
            MarketSeries.__init__(self, series_name = asset)
        else:
            print('Invalid series')

        
class Portfolio:
    #Argument is a list of Asset class objects
    def __init__(self, holdings):
        self.holdings = holdings
        self.amounts = [i.amount for i in holdings]
        self.price_hist = pd.concat([n.data for n in holdings], axis=1).dropna()
        self.data_diff_log = pd.concat([np.log(n.data).diff() for n in holdings], axis=1).dropna()
        self.data_diff = pd.concat([n.data.diff() for n in holdings], axis=1).dropna()
    
        
    def add_asset(self, ast):
        self.holdings.append(ast)
        self.amounts.append(ast.amount)
        self.price_hist = pd.concat([n.data for n in self.holdings], axis=1).dropna()
        self.data_diff_log = pd.concat([np.log(n.data).diff() for n in self.holdings], axis=1).dropna()
        self.data_diff = pd.concat([n.data.diff() for n in self.holdings], axis=1).dropna()
        
    #Monte Carlo simulation using the Gaussian copula
    def sim_series(self, length, log=False):  
           
        def gen_data(self, length, log=False):
            sim_data = []
            uncorr_data = [np.random.normal(size=length) for i in range(len(self.holdings))]
            if log:
                uncorr_data = np.asarray(uncorr_data).transpose()
                transf_matrix = np.dot(np.sqrt(np.diag(self.cov_eigenvalues(log))),
                                             np.linalg.inv(self.cov_eigenvectors(log)))
                for row in range(length):
                    sim_data.append(list(np.dot(uncorr_data[row,], transf_matrix)))
                return np.array(sim_data)
        
            else:
                uncorr_data = np.asarray(uncorr_data).transpose()
                transf_matrix = np.dot(np.sqrt(np.diag(self.cov_eigenvalues())),
                                             np.linalg.inv(self.cov_eigenvectors()))
                for row in range(length):
                    sim_data.append(list(np.dot(uncorr_data[row,], transf_matrix)))
                return np.array(sim_data)
        
        if log:
            sim_change = gen_data(self, length, log)
            price = np.log(self.price_hist.iloc[self.data_diff_log.shape[0]].values)
            sim_price = np.zeros(shape=(length, len(self.holdings)))
            for row in range(length):
                price += sim_change[row,]
                sim_price[row] = price
            return sim_price
        else:
            sim_change = gen_data(self, length)
            price = self.price_hist.iloc[self.data_diff.shape[0]].values
            sim_price = np.zeros(shape=(length, len(self.holdings)))
            for row in range(length):
                price += sim_change[row,]
                sim_price[row] = price
            return sim_price

    #Geometric mean of the once-differenced market prices
    def diff_mean(self, log=False):
        if log:
            return [np.mean(self.data_diff_log.iloc[:,i]) for i in range(len(self.holdings))]
        else:
            return [np.mean(self.data_diff.iloc[:,i]) for i in range(len(self.holdings))]

    #Standard deviation of the once-differenced market prices
    def diff_sd(self, log=False):
        return np.sqrt(np.diagonal(self.cov_df(log)))


    #Covariance dataframe
    def cov_df(self, log=False):
        if log:
            return self.data_diff_log.cov()
        else:
            return self.data_diff.cov()

    #Eigenvalues of the covariance matrix
    def cov_eigenvalues(self, log=False):
        cov_ev = np.linalg.eigvals(self.cov_df(log))
        cov_ev = [max(ev, 0) for ev in cov_ev]
        return cov_ev

    #Eigenvectors of the covariance matrix
    def cov_eigenvectors(self, log=False):
        return np.linalg.eig(self.cov_df(log))[1]

    #Relative price amount of each asset held
    @property
    def amount_ratio(self, index = False):
        if index == True:
            return {i.name: round(i.amount/self.total_inv, 3) for i in self.holdings}
        else:
            return [round(i.amount/self.total_inv, 3) for i in self.holdings]

    @amount_ratio.getter
    def amount_ratio(self):
        return [round(i.amount/self.total_inv, 3) for i in self.holdings]

    #Total amount invested
    @property
    def total_inv(self):
        return sum(list(self.amounts))

    @total_inv.setter
    def total_inv(self, added_inv):
        self.amounts = [i.amount + added_inv*self.amount_ratio[i.name] for i in self.holdings]


    @property
    def corr_df(self):
        return self.data_diff_log.corr()

    #Critical line algorithm for optimal portfolios. Common solutions include:
    #'efficient_frontier', 'max_sharpe', 'min_volatility'
    def CLA(self, solution):
        cla = CriticalLineAlgorithm()
        asset_names = [i.name for i in self.holdings]

        cla.allocate(asset_prices=self.price_hist, solution=solution, asset_names = asset_names)
        cla.weights.sort_values(by=0, ascending=False, axis=1)
        return cla

    #Variance curve of the portfolio as different amounts are held between two assets
    def var_curve(self, asset1_loc, asset2_loc):
        #assetx_loc is numerical position within portfolio
        weight = np.arange(0, 1, 0.01).tolist()
        
        def weighted_return(asset1_loc, asset2_loc, wght):
            asset1_return = [0]
            asset2_return = [0]
            for j in self.data_diff_log[asset1_loc]:
                asset1_return.append(wght*(j))
            for j in self.data_diff_log[asset2_loc]:
                asset2_return.append((1-wght)*j)
            return np.array(asset1_return) + np.array(asset2_return)
            
        def weighted_var(asset1_loc, asset2_loc, wght, log=False):
            asset1_var = self.cov_df(log).iloc[asset1_loc,asset1_loc]
            asset2_var = self.cov_df(log).iloc[asset2_loc,asset2_loc]
            asset_corr = self.corr_df.iloc[asset1_loc, asset2_loc]
            var = (wght*asset1_var)**2 + ((1-wght)*asset2_var)**2
            + 2*wght*(1-wght)*np.sqrt(asset1_var)*np.sqrt(asset2_var)*asset_corr
            return var
 
        weight_var = []
        weight_returns = []
        for w in weight:
            asset_return = weighted_return(asset1_loc, asset2_loc, w)
            weight_var.append([w, weighted_var(asset1_loc, asset2_loc, w)])
            weight_returns.append([w, np.mean(asset_return)])
        weight_var = pd.DataFrame(weight_var)
        weight_returns = pd.DataFrame(weight_returns)
        
        plt.plot(weight_var[1], weight_returns[1])