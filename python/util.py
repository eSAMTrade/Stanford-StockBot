import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow import keras
import gtab

def get_categorical_tickers():
    '''
    This Function returns a dictionary of tickers for different industry types
    :return:
    ticker_dict: Dictionary of 9 different industry types with over 8 tickers each
    tickerSymbols: Set of three tickers
    '''
    ticker_dict = {}
    all_tickers = []
    ticker_dict['energy'] = ['XOM', 'CVX', 'SHEL', 'PTR', 'TTE', 'BP', 'PBR', 'SNP', 'SLB', 'VLO']
    ticker_dict['materials'] = ['BHP', 'LIN', 'RIO', 'DD', 'SHW', 'CTA-PB', 'APD']
    ticker_dict['industrials'] = ['UPS', 'HON', 'LMT', 'BA', 'GE', 'MMM', 'RTX', 'CAT', 'WM', 'ABB', 'ETN', 'EMR',
                                  'FDX', 'TRI']
    ticker_dict['utilities'] = ['NEE', 'DUK', 'NGG', 'AEP', 'XEL','AWK' ,'ETR', 'PCG']
    ticker_dict['healthcare'] = ['UNH', 'JNJ', 'PFE', 'NVO', 'TMO', 'MRK', 'AZN', 'NVS', 'DHR', 'AMGN', 'CVS', 'GSK',
                                 'ZTS', 'GILD']
    ticker_dict['financials'] = ['BRK-A', 'V', 'JPM', 'BAC', 'MA', 'WFC', 'C-PJ', 'MS', 'RY', 'AXP']
    ticker_dict['discretionary'] = ['AMZN', 'TSLA', 'HD', 'BABA', 'TM', 'NKE', 'MCD', 'SBUX', 'F', 'MAR', 'GM', 'ORLY',
                                    'LILI', 'HMC', 'CMG', 'HLT']
    ticker_dict['staples'] = ['WMT', 'PG', 'KO', 'COST', 'PEP', 'BUD', 'UL', 'TGT', 'MDLZ', 'CL', 'DG', 'KHC', 'KDP',
                              'HSY']
    ticker_dict['IT'] = ['AAPL', 'MSFT', 'TSM', 'NVDA', 'AVGO', 'CSCO', 'ORCL', 'ACN', 'ADBE', 'INTC', 'CRM', 'TXN',
                         'QCOM', 'AMD', 'IBM', 'SONY', 'AMAT', 'INFY', 'ADI', 'MU', 'LRCX']
    ticker_dict['communication'] = ['GOOG', 'FB', 'DIS', 'VZ', 'CMCSA', 'TMUS', 'T', 'NFLX', 'SNAP', 'VOD', 'BAIDU',
                                    'TWTR', 'EA']
    ticker_dict['estate'] = ['PLD', 'AMT', 'CCI', 'EQIX', 'SPG', 'DLR', 'WELL', 'EQR', 'AVB', 'WY', 'INVH', 'MAA']
    ticker_keys = []
    for key in ticker_dict.keys():
        ticker_keys.append(key)
        all_tickers.append(ticker_dict[key])
    ticker_dict['all'] = all_tickers
    tickerSymbols = ['BRK-A', 'GOOG', 'MSFT']
    return ticker_dict, tickerSymbols

def get_company_names():
    '''
    Get a dictionary of search strings corresponding to different ticker labels
    :return:
    ticker_dict: Dictionary of search strings given a stock ticker
    '''
    ticker_dict = {}
    all_tickers = []
    ticker_dict['energy'] = {'XOM': 'Exxon Mobil', 'CVX': 'Chevron', 'SHEL': 'Shell', 'PTR': 'PetroChina',
                             'TTE': 'TotalEnergies', 'BP': 'BP', 'PBR': 'Petroleo Brasileiro',
                             'SNP': 'China Petroleum', 'SLB': 'Schlumberger', 'VLO': 'Valero'}
    '''
    ticker_dict['materials'] = ['BHP', 'LIN', 'RIO', 'DD', 'SHW', 'CTA-PB', 'APD']
    ticker_dict['industrials'] = ['UPS', 'HON', 'LMT', 'BA', 'GE', 'MMM', 'RTX', 'CAT', 'WM', 'ABB', 'ETN', 'EMR',
                                  'FDX', 'TRI']
    ticker_dict['utilities'] = ['NEE', 'DUK', 'NGG', 'AEP', 'XEL','AWK' ,'ETR', 'PCG']
    ticker_dict['healthcare'] = ['UNH', 'JNJ', 'PFE', 'NVO', 'TMO', 'MRK', 'AZN', 'NVS', 'DHR', 'AMGN', 'CVS', 'GSK',
                                 'ZTS', 'GILD']
    ticker_dict['financials'] = ['BRK-A', 'V', 'JPM', 'BAC', 'MA', 'WFC', 'C-PJ', 'MS', 'RY', 'AXP']
    ticker_dict['discretionary'] = ['AMZN', 'TSLA', 'HD', 'BABA', 'TM', 'NKE', 'MCD', 'SBUX', 'F', 'MAR', 'GM', 'ORLY',
                                    'LILI', 'HMC', 'CMG', 'HLT']
    ticker_dict['staples'] = ['WMT', 'PG', 'KO', 'COST', 'PEP', 'BUD', 'UL', 'TGT', 'MDLZ', 'CL', 'DG', 'KHC', 'KDP',
                              'HSY']
    ticker_dict['IT'] = ['AAPL', 'MSFT', 'TSM', 'NVDA', 'AVGO', 'CSCO', 'ORCL', 'ACN', 'ADBE', 'INTC', 'CRM', 'TXN',
                         'QCOM', 'AMD', 'IBM', 'SONY', 'AMAT', 'INFY', 'ADI', 'MU', 'LRCX']
    ticker_dict['communication'] = ['GOOG', 'FB', 'DIS', 'VZ', 'CMCSA', 'TMUS', 'T', 'NFLX', 'SNAP', 'VOD', 'BAIDU',
                                    'TWTR', 'EA']
    ticker_dict['estate'] = ['PLD', 'AMT', 'CCI', 'EQIX', 'SPG', 'DLR', 'WELL', 'EQR', 'AVB', 'WY', 'INVH', 'MAA']
    ticker_keys = []
    for key in ticker_dict.keys():
        ticker_keys.append(key)
        all_tickers.append(ticker_dict[key])
    ticker_dict['all'] = all_tickers
    '''
    return ticker_dict

def cross_corr(a,b):
    '''
    Compute the cross-correlation between
    :param a: Time-series data of first stock
    :param b: Time-series data of second stock
    :return: Cross-correlation of the two stocks that are input
    '''
    return (a*b).sum()/((a**2).sum()*(b**2).sum())**0.5

def get_tick_values(tickerSymbol, start, end):
    '''
    Function to extract the time series data
    :param tickerSymbol: String of stock ticker
    :param start: String of starting date of the time-series data
    :param end: String of ending date of the time-series data
    :return: type(list): Time series data
    '''
    tickerData = yf.Ticker(tickerSymbol)
    tickerDf = yf.download(tickerSymbol, start=start, end=end)
    tickerDf = tickerDf['Adj Close']
    data = tickerDf
    return data.values

def get_control_vector(val):
    '''
    Returns the mask of day instances where stock purchase/sell decisions are to be made
    :param val: Input array of stock values
    :return: np.array of decisions maks labels (-2/0/2)
    '''
    return np.diff(np.sign(np.diff(val)))

def buy_and_sell_bot(val,controls):
    '''
    Returns the growth of investment over time as function of the input decision mask and the stock values
    :param val: np.array of the actual stock value over time
    :param controls: np.array of the control mask to make purchase/sell decisions
    :return: np.array of percentage growth value of the invested stock
    '''
    inv = []
    curr_val = 100
    inds = np.where(controls)[0]
    buy_inds = np.where(controls>0)[0]
    sell_inds = np.where(controls<0)[0]
    max_limit = sell_inds[-1] if sell_inds[-1]>buy_inds[-1] else buy_inds[-1]
    for i in range(buy_inds[0]+2):
        inv.append(curr_val)
    for i in range(buy_inds[0],max_limit+1):
        if controls[i]>0:
            buy_val = val[i+1]
        elif controls[i]<0:
            sell_val = val[i+1]
            curr_val = curr_val*sell_val/buy_val
        inv.append(curr_val)
    if max_limit+1!=len(controls):
        for i in range(len(controls)-max_limit-1):
            inv.append(curr_val)
    return inv


class LSTM_Model():
    '''
    Class to train and infer stock price for one particular stock
    '''
    def __init__(self,tickerSymbol, start, end,
                 past_history = 60, forward_look = 1, train_test_split = 0.8, batch_size = 30,
                 epochs = 50, steps_per_epoch = 200, validation_steps = 50, verbose = 0, infer_train = True,
                 depth = 1, naive = False, values = 200, plot_values = True, plot_bot = True,
                 custom_loss = False):
        '''
        Initialize parameters for the class
        :param tickerSymbol: String of Ticker symbol to train on
        :param start: String of start date of time-series data
        :param end: String of end date of time-series data
        :param past_history: Int of past number of days to look at
        :param forward_look: Int of future days to predict at a time
        :param train_test_split: Float of fraction train-test split
        :param batch_size: Int of mini-batch size
        :param epochs: Int of total number of epochs in training
        :param steps_per_epoch: Int for total number of mini-batches to run over per epoch
        :param validation_steps: Int of total number of steps to use while validating with the dev set
        :param verbose: Int to decide to print training stage results
        :param infer_train: Flag to carry out prediction on training set
        :param depth: Int to decide depth of stacked LSTM
        :param naive: Flag for deciding if we need a Vanila model
        :param values: Int for number of days to predict for by iteratively updating the time-series histroy
        :param plot_values: Flag to plot
        :param plot_bot: Flag to plot the investment growth by the decision making bot
        :param custom_loss: Flag to decude if custom loss function needs to be applied
        '''
        self.tickerSymbol = tickerSymbol
        self.start = start
        self.end = end
        self.past_history = past_history
        self.forward_look = forward_look
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.verbose = verbose
        self.values = values
        self.depth = depth
        self.naive = naive
        self.plot_values = plot_values
        self.plot_bot = plot_bot
        self.infer_train = infer_train
        self.custom_loss = custom_loss

    def custom_loss_def(self,y_true,y_pred):
        '''
        Definition of the custom loss function
        :param y_true: Np.array of true value
        :param y_pred: np.array of predicted value
        :return: Customised loss function as tf.tensor
        '''
        self.weights = np.float32(1.0 + 0.1*np.linspace(0,self.forward_look-1,self.forward_look)/20.0)
        return tf.math.reduce_mean(tf.math.square(tf.multiply(self.weights,y_true - y_pred)))

    def data_preprocess(self, dataset, iStart, iEnd, sHistory, forward_look=1):
        '''
        Preprocess the data to make either the test set or the train set
        :param dataset: np.array of time-series data
        :param iStart: int of index start
        :param iEnd: int of index end
        :param sHistory: int number of days in history that we need to look at
        :param forward_look: int of number of days in the future that needs to predicted
        :return: returns a list of test/train data
        '''
        self.data = []
        self.target = []
        iStart += sHistory
        if iEnd is None:
            iEnd = len(dataset) - forward_look + 1
        for i in range(iStart, iEnd):
            indices = range(i - sHistory, i)  # set the order
            if forward_look > 1:
                fwd_ind = range(i, i + forward_look)
                fwd_entity = np.asarray([])
                fwd_entity = np.append(fwd_entity, dataset[fwd_ind])
            reshape_entity = np.asarray([])
            reshape_entity = np.append(reshape_entity, dataset[
                indices])  # Comment this out if there are multiple identifiers in the feature vector
            self.data.append(np.reshape(reshape_entity, (sHistory, 1)))  #
            if forward_look > 1:
                self.target.append(np.reshape(fwd_entity, (forward_look, 1)))
            else:
                self.target.append(dataset[i])
        self.data = np.array(self.data)
        self.target = np.array(self.target)

    def plot_history_values(self):
        '''
        Plots time-series data of the chosen ticker
        '''
        tickerData = yf.Ticker(self.tickerSymbol)
        tickerDf = yf.download(self.tickerSymbol, start=self.start, end=self.end)
        tickerDf = tickerDf['Adj Close']
        data = tickerDf
        y = data
        y.index = data.index
        y.plot()
        plt.title(f"{self.tickerSymbol}")
        plt.ylabel("price")
        plt.show()

    def get_ticker_values(self):
        '''
        Get ticker values in a list
        '''
        tickerData = yf.Ticker(self.tickerSymbol)
        tickerDf = yf.download(self.tickerSymbol, start=self.start, end=self.end)
        tickerDf = tickerDf['Adj Close']
        data = tickerDf
        self.y = data.values

    def prepare_test_train(self):
        '''
        Create the dataset from the extracted time-series data
        '''
        training_size = int(self.y.size * self.train_test_split)
        training_mean = self.y[:training_size].mean()  # get the average
        training_std = self.y[:training_size].std()  # std = a measure of how far away individual measurements tend to be from the mean value of a data set.
        self.y = (self.y - training_mean) / training_std  # prep data, use mean and standard deviation to maintain distribution and ratios
        self.data_preprocess(self.y, 0, training_size, self.past_history, forward_look = self.forward_look)
        self.xtrain, self.ytrain = self.data, self.target
        self.data_preprocess(self.y, training_size, None, self.past_history, forward_look = self.forward_look)
        self.xtest, self.ytest = self.data, self.target

    def create_p_test_train(self):
        '''
        Prepare shuffled train and test data
        '''
        BATCH_SIZE = self.batch_size
        BUFFER_SIZE = self.y.size
        p_train = tf.data.Dataset.from_tensor_slices((self.xtrain, self.ytrain))
        self.p_train = p_train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()
        p_test = tf.data.Dataset.from_tensor_slices((self.xtest, self.ytest))
        self.p_test = p_test.batch(BATCH_SIZE).repeat()

    def model_LSTM(self):
        '''
        Create the stacked LSTM model and train it using the shuffled train set
        '''
        self.model = tf.keras.models.Sequential()
        if self.naive:
            self.model.add(tf.keras.layers.LSTM(20, input_shape = self.xtrain.shape[-2:]))
        else:
            self.model.add(tf.keras.layers.LSTM(20, return_sequences=True, input_shape = self.xtrain.shape[-2:]))
        for i in range(self.depth):
            self.model.add(tf.keras.layers.LSTM(20, return_sequences=True))
        if self.naive is False:
            self.model.add(tf.keras.layers.LSTM(20))
        self.model.add(tf.keras.layers.Dense(self.forward_look))
        if self.custom_loss:
            self.model.compile(optimizer='Adam',
                      loss=self.custom_loss_def, metrics=['mse'])
        else:
            self.model.compile(optimizer='Adam',
                               loss='mse', metrics=['mse'])

        self.create_p_test_train()
        self.hist = self.model.fit(self.p_train, epochs = self.epochs, steps_per_epoch = self.steps_per_epoch,
                  validation_data = self.p_test, validation_steps = self.validation_steps,
                  verbose = self.verbose)

    def infer_values(self, xtest, ytest, ts):
        '''
        Infer values by using the test set
        :param xtest: test dataset
        :param ytest: actual value dataset
        :param ts: tikcer symbol
        :return:
        '''
        self.pred = []
        self.pred_update = []
        self.usetest = xtest.copy()
        if self.infer_train:
            self.pred_train = []
            self.pred_update_train = []
            self.usetest_train = self.xtrain.copy()
        for i in range(self.values):
            self.y_pred = self.model.predict(xtest[i,:,:].reshape(1,xtest.shape[1],xtest.shape[2]))[0][:]
            self.y_pred_update = self.model.predict(self.usetest[i,:,:].reshape(1,xtest.shape[1],xtest.shape[2]))[0][:]
            self.pred.append(self.y_pred)
            self.pred_update.append(self.y_pred_update)
            self.usetest[np.linspace(i+1,i+self.past_history-1,self.past_history-1,dtype=int),np.linspace(self.past_history-2,0,self.past_history-1,dtype=int),:] =  self.y_pred_update[0]
            if self.infer_train:
                self.y_pred_train = self.model.predict(self.xtrain[i, :, :].reshape(1, self.xtrain.shape[1], self.xtrain.shape[2]))[0][:]
                self.y_pred_update_train = \
                self.model.predict(self.usetest_train[i, :, :].reshape(1, self.xtrain.shape[1], self.xtrain.shape[2]))[0][:]
                self.pred_train.append(self.y_pred_train)
                self.pred_update_train.append(self.y_pred_update_train)
                self.usetest_train[np.linspace(i + 1, i + self.past_history - 1, self.past_history - 1, dtype=int),
                np.linspace(self.past_history - 2, 0, self.past_history - 1, dtype=int), :] = self.y_pred_update_train[0]
        self.pred = np.array(self.pred)
        self.pred_update = np.array(self.pred_update)
        self.RMS_error = self.hist.history['val_mse'][-1]
        self.RMS_error_train = self.hist.history['mse'][-1]
        if self.infer_train:
            self.pred = np.array(self.pred)
            self.pred_update_train = np.array(self.pred_update_train)
        if self.forward_look>1:
            self.RMS_error_update = (np.mean(((self.ytest[:self.values - 1, 0, 0] - self.pred_update[1:, 0]) / (
            self.ytest[:self.values - 1, 0, 0])) ** 2)) ** 0.5/self.batch_size
            if self.infer_train:
                self.RMS_error_update_train = (np.mean(((self.ytrain[:self.values - 1, 0, 0] - self.pred_update_train[1:, 0]) / (
                    self.ytrain[:self.values - 1, 0, 0])) ** 2)) ** 0.5/self.batch_size
        else:
            self.RMS_error_update = (np.mean(
                ((self.ytest[:self.values - 1] - self.pred_update[1:]) / (self.ytest[:self.values - 1])) ** 2)) ** 0.5/self.batch_size
            if self.infer_train:
                self.RMS_error_update_train = (np.mean(((self.ytrain[:self.values - 1] - self.pred_update_train[1:]) / (
                    self.ytrain[:self.values - 1])) ** 2)) ** 0.5/self.batch_size

    def plot_test_values(self):
        '''
        Plot predicted values against actual values
        '''
        plt.figure()
        if self.forward_look>1:
            plt.plot(self.yt[:self.values-1,0,0],label='actual (%s)'%self.ts)
            plt.plot(self.pred[1:,0],label='predicted (%s)'%self.ts)
            plt.plot(self.pred_update[1:,0],label='predicted (update)')
            plt.xlabel("Days")
            plt.ylabel("Normalized stock price")
            plt.title('The relative RMS error is %f' % self.RMS_error)
            plt.legend()
            plt.savefig('../images/Stock_prediction_%d_%d_%d_%d_%s_%s.png' % (
            self.depth, int(self.naive), self.past_history, self.forward_look, self.ts, int(self.custom_loss)))
            plt.figure()
            plt.plot(self.pred[1:, 0]-self.pred_update[1:,0], label='difference (%s)' % self.ts)
            plt.xlabel("Days")
            plt.ylabel("Prediction difference")
            plt.savefig('../images/Difference_%d_%d_%d_%d_%s_%s.png' % (
            self.depth, int(self.naive), self.past_history, self.forward_look, self.ts, int(self.custom_loss)))
        else:
            plt.plot(self.yt[:self.values-1],label='actual (%s)'%self.ts)
            plt.plot(self.pred[1:],label='predicted (%s)'%self.ts)
            plt.plot(self.pred_update[1:],label='predicted (update)')
            plt.xlabel("Days")
            plt.ylabel("Normalized stock price")
            plt.title('The relative RMS error is %f' % self.RMS_error)
            plt.legend()
            plt.savefig('../images/Stock_prediction_%d_%d_%d_%d_%s_%s.png'%(
            self.depth,int(self.naive), self.past_history, self.forward_look, self.ts, int(self.custom_loss)))
            plt.figure()
            plt.plot(self.pred[1:] - self.pred_update[1:], label='difference (%s)' % self.ts)
            plt.xlabel("Days")
            plt.ylabel("Prediction difference")
            plt.savefig('../images/Difference_%d_%d_%d_%d_%s_%s.png' % (
            self.depth, int(self.naive), self.past_history, self.forward_look, self.ts, int(self.custom_loss)))
        print('The relative test RMS error is %f'%self.RMS_error)
        print('The relative test RMS error for the updated dataset is %f' % self.RMS_error_update)
        if self.infer_train:
            print('The relative train RMS error is %f' % self.RMS_error_train)
            print('The relative train RMS error for the updated dataset is %f' % self.RMS_error_update_train)
            
    def arch_plot(self):
        '''
        Plot the network architecture
        '''
    	dot_img_file = '../images/LSTM_arch_depth%d_naive%d.png' %( self.depth, int(self.naive))
    	tf.keras.utils.plot_model(self.model, to_file=dot_img_file, show_shapes=True)
    

    def full_workflow(self, model = None):
        '''
        Workflow to carry out the entire process end-to-end
        :param model: Choose which model to use to predict inferred values
        :return:
        '''
        self.get_ticker_values()
        self.prepare_test_train()
        self.model_LSTM()
        if model is None:
            self.xt = self.xtest
            self.yt = self.ytest
            self.ts = self.tickerSymbol
        else:
            self.xt = model.xtest
            self.yt = model.ytest
            self.ts = model.tickerSymbol
        self.infer_values(self.xt, self.yt, self.ts)
        # self.arch_plot()

    def full_workflow_and_plot(self, model = None):
        '''
        Workflow to carry out the entire process end-to-end
        :param model: Choose which model to use to plot inferred values
        :return:
        '''
        self.full_workflow(model = model)
        self.plot_test_values()
        

    def plot_bot_decision(self):
        '''
        calculate investment growth from the inferred prediction value and plot the resulting growth
        '''
        if self.forward_look > 1:
            ideal = self.yt[:self.values - 1, 0, 0]
            pred = np.asarray(self.pred[1:, 0]).reshape(-1,)
            pred_update = np.asarray(self.pred_update[1:, 0]).reshape(-1,)
        else:
            ideal = self.yt[:self.values - 1]
            pred = np.asarray(self.pred[1:]).reshape(-1,)
            pred_update = np.asarray(self.pred_update[1:]).reshape(-1,)
        control_ideal = get_control_vector(ideal)
        control_pred = get_control_vector(pred)
        control_pred_update = get_control_vector(pred_update)
        bot_ideal = buy_and_sell_bot(ideal, control_ideal)
        bot_pred = buy_and_sell_bot(ideal, control_pred)
        bot_pred_update = buy_and_sell_bot(ideal, control_pred_update)
        plt.figure()
        plt.plot(bot_ideal, label='Ideal case (%.2f)'%bot_ideal[-1])
        plt.plot(bot_pred, label='From prediction (%.2f)'%bot_pred[-1])
        plt.plot(bot_pred_update, label='From prediction (updated) (%.2f)'%bot_pred_update[-1])
        plt.plot(ideal / ideal[0] * 100.0, label='Stock value(%s)' % self.ts)
        plt.xlabel("Days")
        plt.ylabel("Percentage growth")
        plt.legend()
        plt.savefig('../images/Bot_prediction_%d_%d_%d_%d_%s_%s.png' % (self.depth, int(self.naive), self.past_history, self.forward_look, self.ts, int(self.custom_loss)))


class LSTM_ED_Model():
    def __init__(self,tickerSymbol, start, end,
                 past_history = 60, forward_look = 1, train_test_split = 0.8, batch_size = 30,
                 epochs = 50, steps_per_epoch = 200, validation_steps = 50, verbose = 0,
                 depth = 1, naive = False, values = 200, tickerSymbolList = None, LSTM_latent_dim = 20):
        self.tickerSymbol = tickerSymbol
        self.start = start
        self.end = end
        self.past_history = past_history
        self.forward_look = forward_look
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.verbose = verbose
        self.values = values
        self.depth = depth
        self.naive = naive
        self.LSTM_latent_dim = LSTM_latent_dim;
        if tickerSymbolList == None:
            self.tickerSymbolList = [tickerSymbol]
        else:
            self.tickerSymbolList = tickerSymbolList

    def data_preprocess(self, dataset, iStart, iEnd, sHistory, forward_look=1):
        self.data_enc = []
        self.data_dec = []
        self.target = []
        iStart += sHistory
        if iEnd is None:
            iEnd = len(dataset) - forward_look
        for i in range(iStart, iEnd):
            indices_x = range(i - sHistory, i)  # set the order
            indices_x_dec = range(i-1, i + forward_look-1)
            indices_y_dec = range(i, i + forward_look)
            reshape_entity_x = np.asarray([])
            reshape_entity_x = np.append(reshape_entity_x, dataset[indices_x])  # Comment this out if there are multiple identifiers in the feature vector
            reshape_entity_x_dec = np.asarray([])
            reshape_entity_x_dec = np.append(reshape_entity_x_dec, dataset[indices_x_dec])  # Comment this out if there are multiple identifiers in the feature vector
            reshape_entity_y_dec = np.asarray([])
            reshape_entity_y_dec = np.append(reshape_entity_y_dec, dataset[indices_y_dec])  # Comment this out if there are multiple identifiers in the feature vector
            self.data_enc.append(np.reshape(reshape_entity_x, (sHistory, 1)))  #
            self.data_dec.append(np.reshape(reshape_entity_x_dec, (forward_look, 1)))
            self.target.append(np.reshape(reshape_entity_y_dec, (forward_look, 1)))
        self.data_enc = np.array(self.data_enc)
        self.data_dec = np.array(self.data_dec)
        self.target = np.array(self.target)

    def plot_history_values(self):
        tickerData = yf.Ticker(self.tickerSymbol)
        tickerDf = yf.download(self.tickerSymbol, start=self.start, end=self.end)
        tickerDf = tickerDf['Adj Close']
        data = tickerDf
        y = data
        y.index = data.index
        y.plot()
        plt.title(f"{self.tickerSymbol}")
        plt.ylabel("price")
        plt.show()

    def get_ticker_values(self, option = 0):
        if option == 0:
            tickerData = yf.Ticker(self.tickerSymbol)
            tickerDf = yf.download(self.tickerSymbol, start=self.start, end=self.end)
            tickerDf = tickerDf['Adj Close']
            data = tickerDf
            self.y = data.values
        else:
            # Write code for multiple tickers. Code below is not enough
            tickerData = yf.Ticker(self.tickerSymbol)
            tickerDf = yf.download(self.tickerSymbol, start=self.start, end=self.end)
            tickerDf = tickerDf['Adj Close']
            data = tickerDf
            self.y = data.values


    def prepare_test_train(self):
        training_size = int(self.y.size * self.train_test_split)
        training_mean = self.y[:training_size].mean()  # get the average
        training_std = self.y[:training_size].std()  # std = a measure of how far away individual measurements tend to be from the mean value of a data set.
        self.y = (self.y - training_mean) / training_std  # prep data, use mean and standard deviation to maintain distribution and ratios
        self.data_preprocess(self.y, 0, training_size, self.past_history, forward_look = self.forward_look)
        self.xtrain, self.xtrain_dec, self.ytrain = self.data_enc, self.data_dec, self.target
        self.data_preprocess(self.y, training_size, None, self.past_history, forward_look = self.forward_look)
        self.xtest, self.xtest_dec, self.ytest = self.data_enc, self.data_dec, self.target

    def create_p_test_train(self):
        BATCH_SIZE = self.batch_size
        BUFFER_SIZE = self.y.size
        p_train = tf.data.Dataset.from_tensor_slices(((self.xtrain, self.xtrain_dec), self.ytrain))
        self.p_train = p_train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()
        p_test = tf.data.Dataset.from_tensor_slices(((self.xtest, self.xtest_dec), self.ytest))
        self.p_test = p_test.batch(BATCH_SIZE).repeat()

    def model_LSTM(self):
        latent_dim = self.LSTM_latent_dim
        encoder_inputs = keras.Input(shape=(None, 1))
        encoder = keras.layers.LSTM(latent_dim, return_state=True)  # Number of latent dimensions, defaults to 20
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = keras.Input(shape=(None, 1))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = keras.layers.Dense(1, activation=None)
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # configure model for training.
        # self.model.compile(optimizer='Adam',
        #               loss='mean_absolute_percentage_error')
        self.model.compile(optimizer='Adam',
                           loss='mse', metrics=['mse'])
        self.create_p_test_train()
        self.history  = self.model.fit(self.p_train,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 steps_per_epoch=self.steps_per_epoch,
                                 validation_data=self.p_test,
                                 validation_steps=self.validation_steps,
                                 verbose=self.verbose)

    def model_inference_LSTM(self):
        latent_dim = self.LSTM_latent_dim

        encoder_inputs = self.model.input[0]  # input_1
        encoder_outputs, state_h_enc, state_c_enc = self.model.layers[2].output  # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
        self.encoder_model = keras.Model(encoder_inputs, encoder_states)

        decoder_inputs = self.model.input[1]  # input_2
        decoder_state_input_h = keras.Input(shape=(latent_dim,))
        decoder_state_input_c = keras.Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = self.model.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = self.model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = keras.Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
        )

    def infer_values(self, xtest, ytest, ts):
        self.pred = []
        self.pred_update = []
        self.usetest = xtest.copy()

        # Predict encoder for x_test to get decoder inputs. Iteratively predict decoder output.
        # This one is incomplete.
        states_value = self.encoder_model.predict(xtest[0:1,:,:])
        decoder_input = xtest[0:1, -1, :]  # choosing the most recent value to feed the decoder
        for i in range(self.values):
            new_pred, h, c = self.decoder_model.predict([decoder_input] + states_value)
            y_pred = new_pred.reshape((-1, 1))
            decoder_input = new_pred
            states_value = [h, c]
            self.pred_update.append(y_pred)
        self.pred_update = np.array(self.pred_update)

        # Predict encoder for x_test to get decoder inputs and use it for the decoder for one extra day.
        states_value = self.encoder_model.predict(xtest)
        decoder_input = xtest[:, -1, :]  # choosing the most recent value to feed the decoder
        new_pred, h, c = self.decoder_model.predict([decoder_input] + states_value)
        y_pred = new_pred[:self.values,0:1,0:1]
        self.pred = y_pred


        # for i in range(self.values):
        #     states_value = self.encoder_model.predict(xtest[i:i+1, :, :])
        #     decoder_input = xtest[i:i+1, -1, :]  # choosing the most recent value to feed the decoder
        #     new_pred, h, c = self.decoder_model.predict([decoder_input] + states_value)
        #     y_pred = new_pred.reshape((-1, 1))
        #     states_value = [h, c]
        #     self.pred.append(y_pred)
        # self.pred = np.array(self.pred)

        if self.forward_look>1:
            self.RMS_error = self.history.history['mse'][-1]
        else:
            self.RMS_error = self.history.history['mse'][-1]

    def plot_test_values(self):
        plt.figure()
        if self.forward_look > 1:
            print("Sorry, still working on this\n")
            # plt.plot(self.ytest[:self.values, 0, 0],'*-', label='actual (%s)' % self.ts)
            # plt.plot(self.pred[:,0,0],'o-', label='predicted (%s)' % self.ts)
            # # plt.plot(self.pred_update[1:, 0], label='predicted (update)')
            # plt.xlabel("Days")
            # plt.ylabel("Normalized stock price")
            # plt.title('The relative RMS error is %f' % self.RMS_error)
            # plt.legend()
            # plt.savefig('../images/ED_Stock_prediction_%d_%d_%d_%d_%s.png' % (
            #     self.depth, int(self.naive), self.past_history, self.forward_look, self.ts))
            # plt.figure()
            # plt.plot(self.pred[1:, 0]-self.pred_update[1:,0], label='difference (%s)' % self.ts)
        else:
            plt.plot(self.ytest[:self.values, 0, 0], '-', label='actual (%s)' % self.ts)
            plt.plot(self.pred[:, 0, 0], '-', label='predicted (%s)' % self.ts)
            plt.plot(self.pred_update[:, 0, 0], label='predicted (update)')
            plt.xlabel("Days")
            plt.ylabel("Normalized stock price")
            plt.title('The relative RMS error is %f' % self.RMS_error)
            plt.legend()
            plt.savefig('../images/ED_Stock_prediction_%d_%d_%d_%d_%s.png' % (
                self.depth, int(self.naive), self.past_history, self.forward_look, self.ts))

        print('The relative RMS error is %f' % self.RMS_error)

    def arch_plot(self):
    	dot_img_file = '../images/LSTM_ED_arch_depth%d_naive%d.png' %( self.depth, int(self.naive))
    	tf.keras.utils.plot_model(self.model, to_file=dot_img_file, show_shapes=True)
    	        
    def full_workflow(self, model=None):
        self.get_ticker_values()
        self.prepare_test_train()
        self.model_LSTM()
        self.model_inference_LSTM()
        if model is None:
            self.xt = self.xtest
            self.yt = self.ytest
            self.ts = self.tickerSymbol
        else:
            self.xt = model.xtest
            self.yt = model.ytest
            self.ts = model.tickerSymbol
        self.infer_values(self.xt, self.yt, self.ts)
        # self.arch_plot()

    def full_workflow_and_plot(self, model=None):
        self.full_workflow(model=model)
        self.plot_test_values()

    def plot_bot_decision(self):
        if self.forward_look > 1:
            ideal = self.ytest[:self.values - 1, 0, 0]
            pred = np.asarray(self.pred[0,1:, 0]).reshape(-1, )
            # pred_update = np.asarray(self.pred_update[1:, 0]).reshape(-1, )
        else:
            ideal = self.ytest[:self.values - 1,0,0]
            pred = np.asarray(self.pred[0,1:,0]).reshape(-1, )
            # pred_update = np.asarray(self.pred_update[1:]).reshape(-1, )
        control_ideal = get_control_vector(ideal)
        control_pred = get_control_vector(pred)
        # control_pred_update = get_control_vector(pred_update)
        bot_ideal = buy_and_sell_bot(ideal, control_ideal)
        bot_pred = buy_and_sell_bot(ideal, control_pred)
        # bot_pred_update = buy_and_sell_bot(ideal, control_pred_update)
        plt.figure()
        plt.plot(bot_ideal, label='Ideal case (%.2f)' % bot_ideal[-1])
        plt.plot(bot_pred, label='From prediction (%.2f)' % bot_pred[-1])
        # plt.plot(bot_pred_update, label='From prediction (updated) (%.2f)' % bot_pred_update[-1])
        plt.plot(ideal / ideal[0] * 100.0, label='Stock value(%s)' % self.ts)
        plt.xlabel("Days")
        plt.ylabel("Percentage growth")
        plt.legend()
        plt.savefig('../images/ED_Bot_prediction_%d_%d_%d_%d.png' % (
        self.depth, int(self.naive), self.past_history, self.forward_look))


class LSTM_Model_MS():
    def __init__(self,tickerSymbol, start, end,
                 past_history = 60, forward_look = 1, train_test_split = 0.8, batch_size = 30,
                 epochs = 50, steps_per_epoch = 200, validation_steps = 50, verbose = 0, infer_train = True,
                 depth = 1, naive = False, values = 200, plot_values = True, plot_bot = True, tickerSymbolList = None, sameTickerTestTrain = True):
        self.tickerSymbol = tickerSymbol
        self.start = start
        self.end = end
        self.past_history = past_history
        self.forward_look = forward_look
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.verbose = verbose
        self.values = values
        self.depth = depth
        self.naive = naive
        self.plot_values = plot_values
        self.plot_bot = plot_bot
        self.infer_train = infer_train
        self.sameTickerTestTrain = sameTickerTestTrain
        if tickerSymbolList == None:
            self.tickerSymbolList = [tickerSymbol]
        else:
            self.tickerSymbolList = tickerSymbolList

    def data_preprocess(self, dataset, iStart, iEnd, sHistory, forward_look=1):
        data = []
        target = []
        iStart += sHistory
        if iEnd is None:
            iEnd = len(dataset) - forward_look + 1
        for i in range(iStart, iEnd):
            indices = range(i - sHistory, i)  # set the order
            if forward_look > 1:
                fwd_ind = range(i, i + forward_look)
                fwd_entity = np.asarray([])
                fwd_entity = np.append(fwd_entity, dataset[fwd_ind])
            reshape_entity = np.asarray([])
            reshape_entity = np.append(reshape_entity, dataset[
                indices])  # Comment this out if there are multiple identifiers in the feature vector
            data.append(np.reshape(reshape_entity, (sHistory, 1)))  #
            if forward_look > 1:
                target.append(np.reshape(fwd_entity, (forward_look, 1)))
            else:
                target.append(dataset[i])
        data = np.array(data)
        target = np.array(target)
        return data, target

    def plot_history_values(self):
        tickerData = yf.Ticker(self.tickerSymbol)
        tickerDf = yf.download(self.tickerSymbol, start=self.start, end=self.end)
        tickerDf = tickerDf['Adj Close']
        data = tickerDf
        y = data
        y.index = data.index
        y.plot()
        plt.title(f"{self.tickerSymbol}")
        plt.ylabel("price")
        plt.show()

    def get_ticker_values(self):
        self.y_all = []
        for tickerSymbol in self.tickerSymbolList:
            tickerData = yf.Ticker(tickerSymbol)
            tickerDf = yf.download(tickerSymbol, start=self.start, end=self.end)
            tickerDf = tickerDf['Adj Close']
            data = tickerDf
            self.y_all.append(data.values)
            self.maxTestValues = len(data.values) - int(len(data.values) * self.train_test_split)
        if self.sameTickerTestTrain == False: # This indicates self.tickerSymbol is the test ticker and self.tickerSymbolList is the training set
            tickerData = yf.Ticker(self.tickerSymbol)
            tickerDf = yf.download(self.tickerSymbol, start=self.start, end=self.end)
            tickerDf = tickerDf['Adj Close']
            data = tickerDf
            self.ytestSet = data.values
            self.maxTestValues = len(data.values) - int(len(data.values) * self.train_test_split)


    def prepare_test_train(self):
        self.y_size = 0
        if self.sameTickerTestTrain == True: # For each ticker, split data into train and test set. Test and validation are the same
            self.xtrain = []
            self.ytrain = []
            self.xtest = []
            self.ytest = []
            for y in self.y_all:
                training_size = int(y.size * self.train_test_split)
                training_mean = y[:training_size].mean()  # get the average
                training_std = y[:training_size].std()  # std = a measure of how far away individual measurements tend to be from the mean value of a data set.
                y = (y - training_mean) / training_std  # prep data, use mean and standard deviation to maintain distribution and ratios
                data, target = self.data_preprocess(y, 0, training_size, self.past_history, forward_look = self.forward_look)
                self.xtrain.append(data)
                self.ytrain.append(target)
                data, target = self.data_preprocess(y, training_size, None, self.past_history, forward_look = self.forward_look)
                self.xtest.append(data)
                self.ytest.append(target)
                self.y_size += y.size

            self.xtrain = np.concatenate(self.xtrain)
            self.ytrain = np.concatenate(self.ytrain)
            self.xtest = np.concatenate(self.xtest)
            self.ytest = np.concatenate(self.ytest)
            self.xt = self.xtest.copy()
            self.yt = self.ytest.copy()
        else: # For each ticker, data into train set only. Split test ticker data into validation and test sets
            self.xtrain = []
            self.ytrain = []
            for y in self.y_all:
                training_size = int(y.size)
                training_mean = y[:training_size].mean()  # get the average
                training_std = y[:training_size].std()  # std = a measure of how far away individual measurements tend to be from the mean value of a data set.
                y = (y - training_mean) / training_std  # prep data, use mean and standard deviation to maintain distribution and ratios
                data, target = self.data_preprocess(y, 0, training_size, self.past_history, forward_look=self.forward_look)
                self.xtrain.append(data)
                self.ytrain.append(target)
                self.y_size += y.size

            self.xtrain = np.concatenate(self.xtrain)
            self.ytrain = np.concatenate(self.ytrain)

            y = self.ytestSet
            validation_size = int(y.size * self.train_test_split)
            validation_mean = y[:validation_size].mean()  # get the average
            validation_std = y[:validation_size].std()  # std = a measure of how far away individual measurements tend to be from the mean value of a data set.
            y = (y - validation_mean) / validation_std
            data, target = self.data_preprocess(y, 0, validation_size, self.past_history, forward_look=self.forward_look)
            self.xtest = data
            self.ytest = target
            data, target = self.data_preprocess(y, validation_size, None, self.past_history, forward_look=self.forward_look)
            self.xt = data
            self.yt = target


    def create_p_test_train(self):
        BATCH_SIZE = self.batch_size
        BUFFER_SIZE = self.y_size
        p_train = tf.data.Dataset.from_tensor_slices((self.xtrain, self.ytrain))
        self.p_train = p_train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()
        p_test = tf.data.Dataset.from_tensor_slices((self.xtest, self.ytest))
        self.p_test = p_test.batch(BATCH_SIZE).repeat()

    def model_LSTM(self):
        self.model = tf.keras.models.Sequential()
        if self.naive:
            self.model.add(tf.keras.layers.LSTM(20, input_shape = self.xtrain.shape[-2:]))
        else:
            self.model.add(tf.keras.layers.LSTM(20, return_sequences=True, input_shape = self.xtrain.shape[-2:]))
        for i in range(self.depth):
            self.model.add(tf.keras.layers.LSTM(20, return_sequences=True))
        if self.naive is False:
            self.model.add(tf.keras.layers.LSTM(20))
        self.model.add(tf.keras.layers.Dense(self.forward_look))

        self.model.compile(optimizer='Adam',
                      loss='mse', metrics=['mse'])
        self.create_p_test_train()
        self.hist = self.model.fit(self.p_train, epochs = self.epochs, steps_per_epoch = self.steps_per_epoch,
                  validation_data = self.p_test, validation_steps = self.validation_steps,
                  verbose = self.verbose)

    def infer_values(self, xtest, ytest, ts = None):
        self.pred = []
        self.pred_update = []
        self.usetest = xtest.copy()
        if self.infer_train:
            self.pred_train = []
            self.pred_update_train = []
            self.usetest_train = self.xtrain.copy()
        for i in range(self.values):
            self.y_pred = self.model.predict(xtest[i,:,:].reshape(1,xtest.shape[1],xtest.shape[2]))[0][:]
            self.y_pred_update = self.model.predict(self.usetest[i,:,:].reshape(1,xtest.shape[1],xtest.shape[2]))[0][:]
            self.pred.append(self.y_pred)
            self.pred_update.append(self.y_pred_update)
            self.usetest[np.linspace(i+1,i+self.past_history-1,self.past_history-1,dtype=int),np.linspace(self.past_history-2,0,self.past_history-1,dtype=int),:] =  self.y_pred_update[0]
            if self.infer_train:
                self.y_pred_train = self.model.predict(self.xtrain[i, :, :].reshape(1, self.xtrain.shape[1], self.xtrain.shape[2]))[0][:]
                self.y_pred_update_train = \
                self.model.predict(self.usetest_train[i, :, :].reshape(1, self.xtrain.shape[1], self.xtrain.shape[2]))[0][:]
                self.pred_train.append(self.y_pred_train)
                self.pred_update_train.append(self.y_pred_update_train)
                self.usetest_train[np.linspace(i + 1, i + self.past_history - 1, self.past_history - 1, dtype=int),
                np.linspace(self.past_history - 2, 0, self.past_history - 1, dtype=int), :] = self.y_pred_update_train[0]
        self.pred = np.array(self.pred)
        self.pred_update = np.array(self.pred_update)
        self.RMS_error = self.hist.history['val_mse'][-1]
        self.RMS_error_train = self.hist.history['mse'][-1]
        if self.infer_train:
            self.pred = np.array(self.pred)
            self.pred_update = np.array(self.pred_update)
        if self.forward_look > 1:
            self.RMS_error_update = (np.mean(((self.ytest[:self.values - 1, 0, 0] - self.pred_update[1:, 0]) / (
                self.ytest[:self.values - 1, 0, 0])) ** 2)) ** 0.5 / self.batch_size
            if self.infer_train:
                self.RMS_error_update_train = (np.mean(
                    ((self.ytrain[:self.values - 1, 0, 0] - self.pred_update_train[1:, 0]) / (
                        self.ytrain[:self.values - 1, 0, 0])) ** 2)) ** 0.5 / self.batch_size
        else:
            self.RMS_error_update = (np.mean(
                ((self.ytest[:self.values - 1] - self.pred_update[1:]) / (
                self.ytest[:self.values - 1])) ** 2)) ** 0.5 / self.batch_size
            if self.infer_train:
                self.RMS_error_update_train = (np.mean(((self.ytrain[:self.values - 1] - self.pred_update_train[1:]) / (
                    self.ytrain[:self.values - 1])) ** 2)) ** 0.5 / self.batch_size

    def plot_test_values(self):
        plt.figure()
        if self.forward_look>1:
            plt.plot(self.yt[:self.values-1,0,0],label='actual (%s)'%self.ts)
            plt.plot(self.pred[1:,0],label='predicted (%s)'%self.ts)
            plt.plot(self.pred_update[1:,0],label='predicted (update)')
            plt.xlabel("Days")
            plt.ylabel("Normalized stock price")
            plt.title('The relative RMS error is %f' % self.RMS_error)
            plt.legend()
            plt.savefig('../images/MultiStock_prediction_%d_%d_%d_%d_%s.png' % (
            self.depth, int(self.naive), self.past_history, self.forward_look, self.ts))
            plt.figure()
            plt.plot(self.pred[1:, 0]-self.pred_update[1:,0], label='difference (%s)' % self.ts)
            plt.xlabel("Days")
            plt.ylabel("Prediction difference")
            plt.savefig('../images/MSDifference_%d_%d_%d_%d_%s.png' % (
            self.depth, int(self.naive), self.past_history, self.forward_look, self.ts))
        else:
            plt.plot(self.yt[:self.values-1],label='actual (%s)'%self.ts)
            plt.plot(self.pred[1:],label='predicted (%s)'%self.ts)
            plt.plot(self.pred_update[1:],label='predicted (update)')
            plt.xlabel("Days")
            plt.ylabel("Normalized stock price")
            plt.title('The relative RMS error is %f' % self.RMS_error)
            plt.legend()
            plt.savefig('../images/MultiStock_prediction_%d_%d_%d_%d_%s.png'%(
            self.depth,int(self.naive), self.past_history, self.forward_look, self.ts))
            plt.figure()
            plt.plot(self.pred[1:] - self.pred_update[1:], label='difference (%s)' % self.ts)
            plt.xlabel("Days")
            plt.ylabel("Prediction difference")
            plt.savefig('../images/MSDifference_%d_%d_%d_%d_%s.png' % (
            self.depth, int(self.naive), self.past_history, self.forward_look, self.ts))
        print('The relative test RMS error is %f'%self.RMS_error)
        print('The relative test RMS error for the updated dataset is %f' % self.RMS_error_update)
        if self.infer_train:
            print('The relative train RMS error is %f' % self.RMS_error_train)
            print('The relative train RMS error for the updated dataset is %f' % self.RMS_error_update_train)

    def full_workflow(self, model = None):
        self.get_ticker_values()
        self.prepare_test_train()
        self.model_LSTM()
        if model is None:
            self.ts = self.tickerSymbol
        else:
            self.xt = model.xtest
            self.yt = model.ytest
            self.ts = model.tickerSymbol
        if self.sameTickerTestTrain == True:
            self.ts = 'Ensemble'

        self.infer_values(self.xt, self.yt, self.ts)

    def model_workflow(self):
        self.get_ticker_values()
        self.prepare_test_train()
        self.model_LSTM()

    def prepare_test(self):
        training_size = int(self.ytemp.size * self.train_test_split)
        training_mean = self.ytemp[:training_size].mean()  # get the average
        training_std = self.ytemp[:training_size].std()  # std = a measure of how far away individual measurements tend to be from the mean value of a data set.
        self.ytemp = (self.ytemp - training_mean) / training_std  # prep data, use mean and standard deviation to maintain distribution and ratios
        data, target = self.data_preprocess(self.ytemp, training_size, None, self.past_history, forward_look = self.forward_look)
        self.xtest, self.ytest = data, target

    def get_tick_values(self):
        tickerData = yf.Ticker(self.tickerSymbol)
        tickerDf = yf.download(self.tickerSymbol, start=self.start, end=self.end)
        tickerDf = tickerDf['Adj Close']
        data = tickerDf
        self.ytemp = data.values

    def prepare_workflow(self):
        self.get_tick_values()
        self.prepare_test()

    def full_workflow_and_plot(self, model = None):
        self.full_workflow(model = model)
        self.plot_test_values()

    def plot_bot_decision(self):
        if self.forward_look > 1:
            ideal = self.yt[:self.values - 1, 0, 0]
            pred = np.asarray(self.pred[1:, 0]).reshape(-1,)
            pred_update = np.asarray(self.pred_update[1:, 0]).reshape(-1,)
        else:
            ideal = self.yt[:self.values - 1]
            pred = np.asarray(self.pred[1:]).reshape(-1,)
            pred_update = np.asarray(self.pred_update[1:]).reshape(-1,)
        control_ideal = get_control_vector(ideal)
        control_pred = get_control_vector(pred)
        control_pred_update = get_control_vector(pred_update)
        bot_ideal = buy_and_sell_bot(ideal, control_ideal)
        bot_pred = buy_and_sell_bot(ideal, control_pred)
        bot_pred_update = buy_and_sell_bot(ideal, control_pred_update)
        plt.figure()
        plt.plot(bot_ideal, label='Ideal case (%.2f)'%bot_ideal[-1])
        plt.plot(bot_pred, label='From prediction (%.2f)'%bot_pred[-1])
        plt.plot(bot_pred_update, label='From prediction (updated) (%.2f)'%bot_pred_update[-1])
        plt.plot(ideal / ideal[0] * 100.0, label='Stock value(%s)' % self.ts)
        plt.xlabel("Days")
        plt.ylabel("Percentage growth")
        plt.legend()
        plt.savefig('../images/MSBot_prediction_%d_%d_%d_%d_%s.png' % (self.depth, int(self.naive), self.past_history, self.forward_look, self.ts))


class LSTM_Model_MS_GT():
    def __init__(self,tickerSymbol, tickerName, start = '2010-01-01', end = '2020-12-31',
                 past_history = 60, forward_look = 1, train_test_split = 0.8, batch_size = 30,
                 epochs = 50, steps_per_epoch = 200, validation_steps = 50, verbose = False, infer_train = True,
                 depth = 1, naive = False, values = 200, plot_values = True, plot_bot = True, tickerSymbolList = None, tickerNameDict = None, sameTickerTestTrain = True):
        self.tickerSymbol = tickerSymbol
        self.tickerName = tickerName
        self.start = start
        self.end = end
        self.past_history = past_history
        self.forward_look = forward_look
        self.train_test_split = train_test_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.verbose = verbose
        self.values = values
        self.depth = depth
        self.naive = naive
        self.plot_values = plot_values
        self.plot_bot = plot_bot
        self.infer_train = infer_train
        self.sameTickerTestTrain = sameTickerTestTrain
        if tickerSymbolList == None:
            self.tickerSymbolList = [tickerSymbol]
            self.tickerNameDict = {tickerSymbol: tickerName}
        else:
            self.tickerSymbolList = tickerSymbolList
            self.tickerNameDict = tickerNameDict

    def data_preprocess(self, dataset, iStart, iEnd, sHistory, forward_look=1):
        data = []
        target = []
        iStart += sHistory
        if iEnd is None:
            iEnd = len(dataset) - forward_look
        for i in range(iStart, iEnd):
            indices = range(i - sHistory, i)  # set the order
            if forward_look > 1:
                fwd_ind = range(i, i + forward_look)
                fwd_entity = np.asarray([])
                fwd_entity = np.append(fwd_entity, dataset[fwd_ind,:])
            reshape_entity = np.asarray([])
            reshape_entity = np.append(reshape_entity, dataset[
                indices,:])  # Comment this out if there are multiple identifiers in the feature vector
            data.append(np.reshape(reshape_entity, (sHistory, 2)))  #
            if forward_look > 1:
                target.append(np.reshape(fwd_entity, (forward_look, 2)))
            else:
                target.append(dataset[i,:])
        data = np.array(data)
        target = np.array(target)
        return data, target

    def plot_history_values(self):
        tickerData = yf.Ticker(self.tickerSymbol)
        tickerDf = yf.download(self.tickerSymbol, start=self.start, end=self.end)
        tickerDf = tickerDf['Adj Close']
        data = tickerDf
        y = data
        y.index = data.index
        y.plot()
        plt.title(f"{self.tickerSymbol}")
        plt.ylabel("price")
        plt.show()

    def get_gtrends_data(self, tickerName):
        my_path = "../GTAB_banks"
        t = gtab.GTAB(dir_path=my_path)
        trend_points = pd.DataFrame()
        last_val = 1
        query = tickerName + ' stock'

        for i in range(0, 11):
            # Run for Jan to Jun of one year
            timeframe_str = str(2010 + i) + "-01-01 " + str(2010 + i) + "-07-01"
            t.set_active_gtab("google_anchorbank_geo=US_timeframe=" + timeframe_str + ".tsv")
            nq = t.new_query(query)
            trend_val = nq['max_ratio'].copy()
            if i == 0:
                last_val = trend_val[-1:].values
                trend_val = trend_val[:-1]
                trend_points = trend_val.copy()
            else:
                trend_val = trend_val * (last_val / trend_val[0])
                last_val = trend_val[-1:].values
                trend_val = trend_val[:-1]
                trend_points = pd.concat([trend_points, trend_val])

            # Run for Jul to Dec of one year
            timeframe_str = str(2010 + i) + "-07-01 " + str(2011 + i) + "-01-01"
            t.set_active_gtab("google_anchorbank_geo=US_timeframe=" + timeframe_str + ".tsv")
            nq = t.new_query(query)
            trend_val = nq['max_ratio'].copy()
            trend_val = trend_val * (last_val / trend_val[0])
            last_val = trend_val[-1:].values
            trend_val = trend_val[:-1]
            trend_points = pd.concat([trend_points, trend_val])
        return trend_points


    def get_ticker_values(self):
        self.y_all = []
        for tickerSymbol in self.tickerSymbolList:
            tickerData = yf.Ticker(tickerSymbol)
            tickerDf = yf.download(tickerSymbol, start=self.start, end=self.end)
            tickerDf = tickerDf['Adj Close']
            data = tickerDf.values
            trend_points = self.get_gtrends_data(self.tickerNameDict[tickerSymbol])
            trend_points = trend_points[tickerDf.index]
            tset = np.concatenate([data.reshape([-1, 1]), trend_points.values.reshape([-1, 1])], axis=1)
            self.y_all.append(tset)
            self.maxTestValues = len(data) - int(len(data) * self.train_test_split)
        if self.sameTickerTestTrain == False: # This indicates self.tickerSymbol is the test ticker and self.tickerSymbolList is the training set
            tickerData = yf.Ticker(self.tickerSymbol)
            tickerDf = yf.download(self.tickerSymbol, start=self.start, end=self.end)
            tickerDf = tickerDf['Adj Close']
            data = tickerDf.values
            trend_points = self.get_gtrends_data(self.tickerNameDict[tickerSymbol])
            trend_points = trend_points[tickerDf.index]
            tset = np.concatenate([data.reshape([-1, 1]), trend_points.values.reshape([-1, 1])], axis=1)
            self.ytestSet = tset
            self.maxTestValues = len(data.values) - int(len(data.values) * self.train_test_split)


    def prepare_test_train(self):
        self.y_size = 0
        if self.sameTickerTestTrain == True: # For each ticker, split data into train and test set. Test and validation are the same
            self.xtrain = []
            self.ytrain = []
            self.xtest = []
            self.ytest = []
            for y in self.y_all:
                training_size = int(len(y) * self.train_test_split)
                training_mean = y[:training_size].mean(axis=0)  # get the average
                training_std = y[:training_size].std(axis=0)  # std = a measure of how far away individual measurements tend to be from the mean value of a data set.
                y = (y - training_mean) / training_std  # prep data, use mean and standard deviation to maintain distribution and ratios
                data, target = self.data_preprocess(y, 0, training_size, self.past_history, forward_look = self.forward_look)
                self.xtrain.append(data)
                self.ytrain.append(target)
                data, target = self.data_preprocess(y, training_size, None, self.past_history, forward_look = self.forward_look)
                self.xtest.append(data)
                self.ytest.append(target)
                self.y_size += y.size

            self.xtrain = np.concatenate(self.xtrain)
            self.ytrain = np.concatenate(self.ytrain)
            self.xtest = np.concatenate(self.xtest)
            self.ytest = np.concatenate(self.ytest)
            self.xt = self.xtest.copy()
            self.yt = self.ytest.copy()
        else: # For each ticker, data into train set only. Split test ticker data into validation and test sets
            self.xtrain = []
            self.ytrain = []
            for y in self.y_all:
                training_size = int(y.size)
                training_mean = y[:training_size].mean(axis=0)  # get the average
                training_std = y[:training_size].std(axis=0)  # std = a measure of how far away individual measurements tend to be from the mean value of a data set.
                y = (y - training_mean) / training_std  # prep data, use mean and standard deviation to maintain distribution and ratios
                data, target = self.data_preprocess(y, 0, training_size, self.past_history, forward_look=self.forward_look)
                self.xtrain.append(data)
                self.ytrain.append(target)
                self.y_size += y.size

            self.xtrain = np.concatenate(self.xtrain)
            self.ytrain = np.concatenate(self.ytrain)

            y = self.ytestSet
            validation_size = int(y.size * self.train_test_split)
            validation_mean = y[:validation_size].mean()  # get the average
            validation_std = y[:validation_size].std()  # std = a measure of how far away individual measurements tend to be from the mean value of a data set.
            y = (y - validation_mean) / validation_std
            data, target = self.data_preprocess(y, 0, validation_size, self.past_history, forward_look=self.forward_look)
            self.xtest = data
            self.ytest = target
            data, target = self.data_preprocess(y, validation_size, None, self.past_history, forward_look=self.forward_look)
            self.xt = data
            self.yt = target


    def create_p_test_train(self):
        BATCH_SIZE = self.batch_size
        BUFFER_SIZE = self.y_size
        p_train = tf.data.Dataset.from_tensor_slices((self.xtrain, self.ytrain))
        self.p_train = p_train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()
        p_test = tf.data.Dataset.from_tensor_slices((self.xtest, self.ytest))
        self.p_test = p_test.batch(BATCH_SIZE).repeat()

    def model_LSTM(self):
        self.model = tf.keras.models.Sequential()
        if self.naive:
            self.model.add(tf.keras.layers.LSTM(20, input_shape = self.xtrain.shape[-2:]))
        else:
            self.model.add(tf.keras.layers.LSTM(20, return_sequences=True, input_shape = self.xtrain.shape[-2:]))
        for i in range(self.depth):
            self.model.add(tf.keras.layers.LSTM(20, return_sequences=True))
        if self.naive is False:
            self.model.add(tf.keras.layers.LSTM(20))
        self.model.add(tf.keras.layers.Dense(2 * self.forward_look))

        self.model.compile(optimizer='Adam',
                      loss='mse', metrics=['mse'])
        self.create_p_test_train()
        self.hist = self.model.fit(self.p_train, epochs = self.epochs, steps_per_epoch = self.steps_per_epoch,
                  validation_data = self.p_test, validation_steps = self.validation_steps,
                  verbose = self.verbose)
        print('Model Built')

    def infer_values(self, xtest, ytest, ts = None):
        self.pred = []
        self.pred_update = []
        self.usetest = xtest.copy()
        if self.infer_train:
            self.pred_train = []
            self.pred_update_train = []
            self.usetest_train = self.xtrain.copy()
        for i in range(self.values):
            self.y_pred = self.model.predict(xtest[i,:,:].reshape(1,xtest.shape[1],xtest.shape[2]))[0][:]
            self.y_pred_update = self.model.predict(self.usetest[i,:,:].reshape(1,xtest.shape[1],xtest.shape[2]))[0][:]
            self.pred.append(self.y_pred)
            self.pred_update.append(self.y_pred_update)
            self.usetest[np.linspace(i+1,i+self.past_history-1,self.past_history-1,dtype=int),np.linspace(self.past_history-2,0,self.past_history-1,dtype=int),:] =  self.y_pred_update[:]
            if self.infer_train:
                self.y_pred_train = self.model.predict(self.xtrain[i, :, :].reshape(1, self.xtrain.shape[1], self.xtrain.shape[2]))[0][:]
                self.y_pred_update_train = \
                self.model.predict(self.usetest_train[i, :, :].reshape(1, self.xtrain.shape[1], self.xtrain.shape[2]))[0][:]
                self.pred_train.append(self.y_pred_train)
                self.pred_update_train.append(self.y_pred_update_train)
                self.usetest_train[np.linspace(i + 1, i + self.past_history - 1, self.past_history - 1, dtype=int),
                np.linspace(self.past_history - 2, 0, self.past_history - 1, dtype=int), :] = self.y_pred_update_train[0]
        self.pred = np.array(self.pred)
        self.pred_update = np.array(self.pred_update)
        self.RMS_error = self.hist.history['val_mse'][-1]
        self.RMS_error_train = self.hist.history['mse'][-1]
        if self.infer_train:
            self.pred = np.array(self.pred)
            self.pred_update = np.array(self.pred_update)
        if self.forward_look > 1:
            self.RMS_error_update = (np.mean(((self.ytest[:self.values - 1, 0, 0] - self.pred_update[1:, 0]) / (
                self.ytest[:self.values - 1, 0, 0])) ** 2)) ** 0.5 / self.batch_size
            if self.infer_train:
                self.RMS_error_update_train = (np.mean(
                    ((self.ytrain[:self.values - 1, 0, 0] - self.pred_update_train[1:, 0]) / (
                        self.ytrain[:self.values - 1, 0, 0])) ** 2)) ** 0.5 / self.batch_size
        else:
            self.RMS_error_update = (np.mean(
                ((self.ytest[:self.values - 1] - self.pred_update[1:]) / (
                self.ytest[:self.values - 1])) ** 2)) ** 0.5 / self.batch_size
            if self.infer_train:
                self.RMS_error_update_train = (np.mean(((self.ytrain[:self.values - 1] - self.pred_update_train[1:]) / (
                    self.ytrain[:self.values - 1])) ** 2)) ** 0.5 / self.batch_size

    def plot_test_values(self):
        plt.figure()
        if self.forward_look>1:
            plt.plot(self.yt[:self.values-1,0,0],label='actual (%s)'%self.ts)
            plt.plot(self.pred[1:,0],label='predicted (%s)'%self.ts)
            plt.plot(self.pred_update[1:,0],label='predicted (update)')
            plt.xlabel("Days")
            plt.ylabel("Normalized stock price")
            plt.title('The relative RMS error is %f' % self.RMS_error)
            plt.legend()
            plt.savefig('../images/MultiStock_prediction_%d_%d_%d_%d_%s.png' % (
            self.depth, int(self.naive), self.past_history, self.forward_look, self.ts))
            plt.figure()
            plt.plot(self.pred[1:, 0]-self.pred_update[1:,0], label='difference (%s)' % self.ts)
            plt.xlabel("Days")
            plt.ylabel("Prediction difference")
            plt.savefig('../images/MSDifference_%d_%d_%d_%d_%s.png' % (
            self.depth, int(self.naive), self.past_history, self.forward_look, self.ts))
        else:
            plt.plot(self.yt[:self.values-1,0],label='actual (%s)'%self.ts)
            plt.plot(self.pred[1:,0],label='predicted (%s)'%self.ts)
            plt.plot(self.pred_update[1:,0],label='predicted (update)')
            plt.xlabel("Days")
            plt.ylabel("Normalized stock price")
            plt.title('The relative RMS error is %f' % self.RMS_error)
            plt.legend()
            plt.savefig('../images/MultiStock_prediction_%d_%d_%d_%d_%s.png'%(
            self.depth,int(self.naive), self.past_history, self.forward_look, self.ts))
            plt.figure()
            plt.plot(self.pred[1:,0] - self.pred_update[1:,0], label='difference (%s)' % self.ts)
            plt.xlabel("Days")
            plt.ylabel("Prediction difference")
            plt.savefig('../images/MSDifference_%d_%d_%d_%d_%s.png' % (
            self.depth, int(self.naive), self.past_history, self.forward_look, self.ts))
        print('The relative test RMS error is %f'%self.RMS_error)
        print('The relative test RMS error for the updated dataset is %f' % self.RMS_error_update)
        if self.infer_train:
            print('The relative train RMS error is %f' % self.RMS_error_train)
            print('The relative train RMS error for the updated dataset is %f' % self.RMS_error_update_train)

    def full_workflow(self, model = None):
        self.get_ticker_values()
        self.prepare_test_train()
        self.model_LSTM()
        if model is None:
            self.ts = self.tickerSymbol
        else:
            self.xt = model.xtest
            self.yt = model.ytest
            self.ts = model.tickerSymbol
        if self.sameTickerTestTrain == True:
            self.ts = 'Ensemble'

        self.infer_values(self.xt, self.yt, self.ts)

    def model_workflow(self):
        self.get_ticker_values()
        self.prepare_test_train()
        self.model_LSTM()

    def prepare_test(self):
        training_size = int(self.ytemp.size * self.train_test_split)
        training_mean = self.ytemp[:training_size].mean()  # get the average
        training_std = self.ytemp[:training_size].std()  # std = a measure of how far away individual measurements tend to be from the mean value of a data set.
        self.ytemp = (self.ytemp - training_mean) / training_std  # prep data, use mean and standard deviation to maintain distribution and ratios
        data, target = self.data_preprocess(self.ytemp, training_size, None, self.past_history, forward_look = self.forward_look)
        self.xtest, self.ytest = data, target

    def get_tick_values(self):
        tickerData = yf.Ticker(self.tickerSymbol)
        tickerDf = yf.download(self.tickerSymbol, start=self.start, end=self.end)
        tickerDf = tickerDf['Adj Close']
        data = tickerDf
        self.ytemp = data.values

    def prepare_workflow(self):
        self.get_tick_values()
        self.prepare_test()

    def full_workflow_and_plot(self, model = None):
        self.full_workflow(model = model)
        self.plot_test_values()

    def plot_bot_decision(self):
        if self.forward_look > 1:
            ideal = self.yt[:self.values - 1, 0, 0]
            pred = np.asarray(self.pred[1:, 0]).reshape(-1,)
            pred_update = np.asarray(self.pred_update[1:, 0]).reshape(-1,)
        else:
            ideal = self.yt[:self.values - 1]
            pred = np.asarray(self.pred[1:, 0]).reshape(-1,)
            pred_update = np.asarray(self.pred_update[1:, 0]).reshape(-1,)
        control_ideal = get_control_vector(ideal)
        control_pred = get_control_vector(pred)
        control_pred_update = get_control_vector(pred_update)
        bot_ideal = buy_and_sell_bot(ideal, control_ideal)
        bot_pred = buy_and_sell_bot(ideal, control_pred)
        bot_pred_update = buy_and_sell_bot(ideal, control_pred_update)
        plt.figure()
        plt.plot(bot_ideal, label='Ideal case (%.2f)'%bot_ideal[-1])
        plt.plot(bot_pred, label='From prediction (%.2f)'%bot_pred[-1])
        plt.plot(bot_pred_update, label='From prediction (updated) (%.2f)'%bot_pred_update[-1])
        plt.plot(ideal / ideal[0] * 100.0, label='Stock value(%s)' % self.ts)
        plt.xlabel("Days")
        plt.ylabel("Percentage growth")
        plt.legend()
        plt.savefig('../images/MSBot_prediction_%d_%d_%d_%d_%s.png' % (self.depth, int(self.naive), self.past_history, self.forward_look, self.ts))

