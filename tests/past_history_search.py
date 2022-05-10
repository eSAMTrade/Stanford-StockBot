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
##############################################
tickeranalysis = tickerSymbols[0]
LSTM_base = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0, naive = True, plot_values = False)
past = np.linspace(10,100,46,dtype = int)
RMSE = np.zeros(len(past))
for i in range(len(past)):
    LSTM = deepcopy(LSTM_base)
    LSTM.past_history = past[i]
    LSTM.full_workflow()
    RMSE[i] = LSTM.RMS_error

plt.plot(past, RMSE)
plt.show()

