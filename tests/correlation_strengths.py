import sys, os

import numpy as np

sys.path.append('../python')
from util import *
from scipy.signal import correlate
import seaborn as sns

ticker_dict, tickerSymbols = get_categorical_tickers()
start="2010-01-01"
end="2019-12-31"
industries = ['energy','materials','utilities','financials','estate']
tickers = ['SHEL','RIO','NEE','BRK-A','JPM']
samples = len(tickers)
data_values = []
for ticks in tickers:
    values = get_tick_values(ticks, start, end)
    data_values.append(values)

data_values = np.array(data_values)
cc_matrix = np.zeros((samples,samples))
rmse_matrix = np.zeros((samples,samples))
cc_matrix[np.arange(samples),np.arange(samples)] = 0.5

for i in range(1,samples):
    for j in range(i):
        cc_matrix[i,j] = cross_corr(data_values[i,:],data_values[j,:])

cc_matrix += cc_matrix.T
model = [[],[],[],[],[]]
for i in range(samples):
    tickeranalysis = tickers[i]
    tickerList = ticker_dict[industries[i]]
    tickerList.remove(tickeranalysis)
    model[i] = LSTM_Model_MS(tickerSymbol=tickeranalysis, start=start, end=end, depth=0, naive=True,
                           tickerSymbolList=tickerList, sameTickerTestTrain=True)
    model[i].model_workflow()
    for j in range(samples):
        print('\r Computing ticker: %s for model: %s'%(tickers[j],industries[i]),end='')
        model[i].tickerSymbol = tickers[j]
        model[i].prepare_workflow()
        model[i].infer_values(model[i].xtest, model[i].ytest)
        rmse_matrix[i,j] = model[i].RMS_error_update

print()
plt.figure(1)
sns.heatmap(cc_matrix, vmin=cc_matrix.reshape(-1,).min(), vmax=cc_matrix.reshape(-1,).max(),
    center=(cc_matrix.reshape(-1,).max()+cc_matrix.reshape(-1,).min())/2,
    cmap=sns.diverging_palette(20, 220, n=200),square=True)
plt.xticks(np.arange(0.75,samples+0.75,1),labels = tickers,rotation=45,horizontalalignment='right')
plt.yticks(np.arange(0.25,samples+0.25,1),labels = tickers,rotation=45,horizontalalignment='right')
plt.figure(2)
sns.heatmap(rmse_matrix, vmin=rmse_matrix.reshape(-1,).min(), vmax=rmse_matrix.reshape(-1,).max(),
    center=(rmse_matrix.reshape(-1,).max()+rmse_matrix.reshape(-1,).min())/2,
    cmap=sns.diverging_palette(20, 220, n=200),square=True)
plt.xticks(np.arange(0.75,samples+0.75,1),labels = tickers,rotation=45,horizontalalignment='right')
plt.yticks(np.arange(0.25,samples+0.25,1),labels = tickers,rotation=45,horizontalalignment='right')
plt.show()