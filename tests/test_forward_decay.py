import sys, os
sys.path.append('../python')
from util import *
from copy import deepcopy

'''
Other initializers of the class the can be mentioned along with the default values:
past_history = 60       # number of days in the past you want to look at
forward_look = 1        # number of days forward that you want to predict
train_test_split = 0.8
batch_size = 30
epochs = 50
steps_per_epoch = 200   #
validation_steps = 50    # Steps taken while validating over the dev set
verbose = 0             # Whether you want tensorflow to print out the training gunk
depth = 1               # Depth of the stacked LSTM (I could get rid of naive in the future)
naive = False           # Should've called it a better name but it just refers to one LSTM
values = 200            # Future days that you want to plot for (computed one day at a time)
'''


ticker_dict, tickerSymbols = get_categorical_tickers()
start="2010-01-01"
end="2019-12-31"
fwd = 20
past = 60
##############################################
tickeranalysis = tickerSymbols[0]
#LSTM_1 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0, naive = True)
#LSTM_1.full_workflow_and_plot()

LSTM_2 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0, naive = True, forward_look = fwd, past_history = past)
LSTM_2.full_workflow_and_plot()

plt.figure(3)
plot_handle =np.zeros(fwd)
for i in range(LSTM_2.values - fwd - 1):
    plot_handle += (1/(LSTM_2.values - fwd - 1))*LSTM_2.pred[np.linspace(i,fwd+i-1,fwd,dtype=int),np.linspace(fwd-1+i,i,fwd,dtype=int)]

plt.plot(plot_handle)
#plt.plot(LSTM_2.pred[np.linspace(0,fwd-1,fwd,dtype=int),np.linspace(fwd-1,0,fwd,dtype=int)])

plt.show()
'''
LSTM_3 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0, naive = False, forward_look = fwd, past_history = past)
LSTM_3.full_workflow_and_plot()

plt.figure(5)
plt.plot(LSTM_3.pred[np.linspace(0,fwd-1,fwd,dtype=int),np.linspace(fwd-1,0,fwd,dtype=int)])
plt.show()

LSTM_4 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 1, naive = False, forward_look = fwd, past_history = past)
LSTM_4.full_workflow_and_plot()

plt.figure(7)
plt.plot(LSTM_4.pred[np.linspace(0,fwd-1,fwd,dtype=int),np.linspace(fwd-1,0,fwd,dtype=int)])
plt.show()
'''