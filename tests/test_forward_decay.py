import sys, os

import matplotlib.pyplot as plt

sys.path.append('../python')
from util import *
from copy import deepcopy
from scipy.optimize import curve_fit
def optimize(x,a,b):
    return 1.0 + a*x/b
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

LSTM_1 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0, naive = True, forward_look = fwd, past_history = past, custom_loss = True)
LSTM_1.full_workflow_and_plot()
LSTM_1.plot_bot_decision()
plt.clf()
LSTM_2 = LSTM_Model(tickerSymbol = tickeranalysis, start = start, end = end, depth = 0, naive = True, forward_look = fwd, past_history = past)
LSTM_2.full_workflow_and_plot()
LSTM_2.plot_bot_decision()
plt.clf()

############ Plotting the value convergence as we move closer to target prediction day ################
plt.figure()
plot_handle =np.zeros(fwd)
for i in range(LSTM_2.values - fwd - 1):
    plot_vec = LSTM_2.pred[np.linspace(i,fwd+i-1,fwd,dtype=int),np.linspace(fwd-1,0,fwd,dtype=int)]
    plot_handle += (1/(LSTM_2.values - fwd - 1))*plot_vec/plot_vec[-1]

np.savez('../save_mat/plot_conv.npz',conv = plot_handle[np.linspace(len(plot_handle)-1,0,len(plot_handle),dtype=int)])
popt,_ = curve_fit(optimize,np.linspace(0,len(plot_handle)-1,len(plot_handle),dtype=int),plot_handle[np.linspace(len(plot_handle)-1,0,len(plot_handle),dtype=int)])
a,b=popt
plt.plot(plot_handle[np.linspace(len(plot_handle)-1,0,len(plot_handle),dtype=int)]) #
plt.plot(np.linspace(0,len(plot_handle)-1,len(plot_handle),dtype=int),optimize(np.linspace(0,len(plot_handle)-1,len(plot_handle),dtype=int),a,b),'r--',label=r'$w=1+%.2f x/ %.2f$'%(a,b))
plt.ylabel('Normalized stock price')
plt.legend(fontsize=12)
plt.xlabel('Days away from prediction')
plt.savefig('../images/Correction_factor.png')
plt.show()
