import sys, os
sys.path.append('../python')
from util import *

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
tickeranalysis = ticker_dict['energy'][0]
tickerList = ticker_dict['energy']
tickerList.remove(tickeranalysis)
tickerNameDict = get_company_names()
tickerNameDict = tickerNameDict['energy']
# tickeranalysis = 'GOOG'
# tickerList = ['GOOG']
tickerName = tickerNameDict[tickeranalysis]
tickerNameDict.pop(tickeranalysis)

#Manually popping out bad keywords
tickerNameDict.pop('TTE')
tickerList.remove('TTE')
tickerNameDict.pop('PBR')
tickerList.remove('PBR')
tickerNameDict.pop('SNP')
tickerList.remove('SNP')

LSTM_1 = LSTM_Model_MS_GT(tickerSymbol = tickeranalysis, tickerName = tickerName,
                          start = start, end = end,
                          depth = 0, naive = True, sameTickerTestTrain = True, verbose = True,
                          tickerSymbolList = tickerList, tickerNameDict = tickerNameDict)

# LSTM_1.get_ticker_values()
# LSTM_1.prepare_test_train()
LSTM_1.full_workflow_and_plot()
LSTM_1.plot_bot_decision()

LSTM_2 = LSTM_Model_MS_GT(tickerSymbol = tickeranalysis, tickerName = tickerName,
                          start = start, end = end,
                          depth = 0, naive = True, sameTickerTestTrain = False, verbose = True,
                          tickerSymbolList = tickerList, tickerNameDict = tickerNameDict)

# LSTM_1.get_ticker_values()
# LSTM_1.prepare_test_train()
LSTM_2.full_workflow_and_plot()
LSTM_2.plot_bot_decision()