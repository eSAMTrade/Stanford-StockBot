import gtab
import numpy as np
import pandas as pd
import yfinance as yf
my_path = "../GTAB_banks"
t = gtab.GTAB(dir_path=my_path)

tickerSymbol = 'GOOG'

query = 'Google stock'
trend_points = pd.DataFrame()
last_val = 1
start =  "2010-01-01"
end = "2020-12-31"
tickerData = yf.Ticker(tickerSymbol)
tickerDf = yf.download(tickerSymbol, start=start, end=end)
tickerDf = tickerDf['Adj Close']
for i in range(0, 11):
  # Run for Jan to Jun of one year
  timeframe_str = str(2010+i) + "-01-01 " + str(2010+i) + "-07-01"
  t.set_active_gtab("google_anchorbank_geo=US_timeframe=" + timeframe_str + ".tsv")
  nq = t.new_query(query)
  aa = nq['max_ratio'].copy()
  if i == 0:
    last_val = aa[-1:].values
    aa = aa[:-1]
    trend_points = aa.copy()
  else:
    aa = aa * (last_val / aa[0])
    last_val = aa[-1:].values
    aa = aa[:-1]
    trend_points = pd.concat([trend_points, aa])

  # Run for Jul to Dec of one year
  timeframe_str = str(2010+i) + "-07-01 " + str(2011+i) + "-01-01"
  t.set_active_gtab("google_anchorbank_geo=US_timeframe=" + timeframe_str + ".tsv")
  nq = t.new_query(query)
  aa = nq['max_ratio'].copy()
  aa = aa * (last_val / aa[0])
  last_val = aa[-1:].values
  aa = aa[:-1]
  trend_points = pd.concat([trend_points, aa])

trend_points = trend_points[tickerDf.index]
tset = np.concatenate([trend_points.values.reshape([-1,1]),tickerDf.values.reshape([-1,1])],axis=1)
