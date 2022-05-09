import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


def get_categorical_tickers():
    ticker_dict = {}
    all_tickers = []
    ticker_dict['energy'] = ['XOM', 'CVX', 'SHEL', 'PTR', 'TTE', 'BP', 'PBR', 'SNP', 'SLB', 'VLO']
    ticker_dict['materials'] = ['BHP', 'LIN', 'RIO', 'DOW', 'DD', 'CTA-PB', 'SHW', 'NTR', 'APD']
    ticker_dict['industrials'] = ['UPS', 'HON', 'LMT', 'BA', 'GE', 'MMM', 'RTX', 'CAT', 'WM', 'ABB', 'ETN', 'EMR',
                                  'FDX', 'TRI']
    ticker_dict['utilities'] = ['NEE', 'DUK', 'DCUE', 'NGG', 'AEP', 'EXL', 'XEL', 'WEV', 'AWK', 'ETR', 'PCG']
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


class LSTM_Model():
    def __init__(self,tickerSymbol, start, end,
                 past_history = 60, forward_look = 1, train_test_split = 0.8, batch_size = 30,
                 epochs = 50, steps_per_epoch = 200, validation_steps = 50, verbose = 0,
                 depth = 1, naive = False, values = 200):
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

    def data_preprocess(self, dataset, iStart, iEnd, sHistory, forward_look=1):
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
        self.xtrain, self.ytrain = self.data, self.target
        self.data_preprocess(self.y, training_size, None, self.past_history, forward_look = self.forward_look)
        self.xtest, self.ytest = self.data, self.target

    def create_p_test_train(self):
        BATCH_SIZE = self.batch_size
        BUFFER_SIZE = self.y.size
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
                      loss='mse')
        self.create_p_test_train()
        self.model.fit(self.p_train, epochs = self.epochs, steps_per_epoch = self.steps_per_epoch,
                  validation_data = self.p_test, validation_steps = self.validation_steps,
                  verbose = self.verbose)

    def infer_values(self, xtest, ytest, ts):
        self.pred = []
        self.pred_update = []
        self.usetest = xtest.copy()
        for i in range(self.values):
            self.y_pred = self.model.predict(xtest[i,:,:].reshape(1,xtest.shape[1],xtest.shape[2]))[0][:]
            # y_pred_update = model.predict(usetest[i,:,:].reshape(1,xtest.shape[1],xtest.shape[2]))[0][0]
            self.pred.append(self.y_pred)
            # pred_update.append(y_pred_update)
            # usetest[i+1,-1,:] = ytest[i]
            # usetest[np.linspace(i+1,i+past_history,past_history,dtype=int),np.linspace(past_history-1,0,past_history,dtype=int),:] =  y_pred_update[0]
            # print(xtest[i,-values+1:,:].T,y_pred)
            # print(usetest[i,-values+1:,:].T,xtest[i,-values+1:,:].T)
        plt.figure()
        self.pred = np.array(self.pred)

    def plot_test_values(self):
        if self.forward_look>1:
            plt.plot(self.yt[:self.values-1,0,0],label='actual (%s)'%self.ts)
            plt.plot(self.pred[1:,0],label='predicted (%s)'%self.ts)
            # plt.plot(pred_update[1:],label='predicted (update)')
            self.RMS_error = (np.mean(((self.yt[:self.values-1,0,0]-self.pred[1:,0])/(self.yt[:self.values-1,0,0]))**2))**0.5
        else:
            plt.plot(self.yt[:self.values-1],label='actual (%s)'%self.ts)
            plt.plot(self.pred[1:],label='predicted (%s)'%self.ts)
            # plt.plot(pred_updates[1:],label='predicted (update)')
            self.RMS_error = (np.mean(((self.yt[:self.values-1]-self.pred[1:])/(self.yt[:self.values-1]))**2))**0.5
        plt.title('The relative RMS error is %f'%self.RMS_error)
        plt.legend()
        print('The relative RMS error is %f'%self.RMS_error)

    def full_workflow(self, model = None):
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

    def full_workflow_and_plot(self, model = None):
        self.full_workflow(model = model)
        self.plot_test_values()


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
            indices_x_dec = range(i-1, i + sFuture-1)
            indices_y_dec = range(i, i + sFuture)
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
        self.xtrain_enc, self.xtrain_dec, self.ytrain = self.data_enc, self.data_dec, self.target
        self.data_preprocess(self.y, training_size, None, self.past_history, forward_look = self.forward_look)
        self.xtest_enc, self.xtest_dec, self.ytest = self.data_enc, self.data_dec, self.target

    def create_p_test_train(self):
        BATCH_SIZE = self.batch_size
        BUFFER_SIZE = self.y.size
        p_train = tf.data.Dataset.from_tensor_slices(((self.xtrain_enc, self.xtrain_dec), self.ytrain))
        self.p_train = p_train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()
        p_test = tf.data.Dataset.from_tensor_slices(((self.xtest_enc, self.xtest_dec), self.ytest))
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
        self.model.compile(optimizer='Adam',
                      loss='mean_absolute_percentage_error')
        self.create_p_test_train()
        self.history = self.model.fit(p_train,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 steps_per_epoch=self.steps_per_epoch,
                                 validation_data=self.p_test,
                                 validation_steps=self.validation_steps,
                                 verbose=self.verbose)


        # self.model = tf.keras.models.Sequential()
        # if self.naive:
        #     self.model.add(tf.keras.layers.LSTM(self.LSTM_latent_dim, input_shape = self.xtrain.shape[-2:]))
        # else:
        #     self.model.add(tf.keras.layers.LSTM(self.LSTM_latent_dim, return_sequences=True, input_shape = self.xtrain.shape[-2:]))
        # for i in range(self.depth):
        #     self.model.add(tf.keras.layers.LSTM(self.LSTM_latent_dim, return_sequences=True))
        # if self.naive is False:
        #     self.model.add(tf.keras.layers.LSTM(self.LSTM_latent_dim))
        # self.model.add(tf.keras.layers.Dense(self.forward_look))
        #
        # self.model.compile(optimizer='Adam',
        #               loss='mse')
        # self.create_p_test_train()
        # self.model.fit(self.p_train, epochs = self.epochs, steps_per_epoch = self.steps_per_epoch,
        #           validation_data = self.p_test, validation_steps = self.validation_steps,
        #           verbose = self.verbose)
    def model_inference_LSTM(self):
        latent_dim = self.LSTM_latent_dim

        encoder_inputs = self.model.input[0]  # input_1
        encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
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
    def plot_test_values(self, xtest, ytest):
        self.pred = []
        self.pred_update = []
        self.usetest = xtest.copy()

        for i in range(self.values):
            self.y_pred = self.model.predict(xtest[i,:,:].reshape(1,xtest.shape[1],xtest.shape[2]))[0][0]
            # y_pred_update = model.predict(usetest[i,:,:].reshape(1,xtest.shape[1],xtest.shape[2]))[0][0]
            self.pred.append(self.y_pred)
            # pred_update.append(y_pred_update)
            # usetest[i+1,-1,:] = ytest[i]
            # usetest[np.linspace(i+1,i+past_history,past_history,dtype=int),np.linspace(past_history-1,0,past_history,dtype=int),:] =  y_pred_update[0]
            # print(xtest[i,-values+1:,:].T,y_pred)
            # print(usetest[i,-values+1:,:].T,xtest[i,-values+1:,:].T)
        plt.figure()
        if self.forward_look>1:
            plt.plot(ytest[:self.values-1,0,0],label='actual')
            plt.plot(self.pred[1:],label='predicted')
            # plt.plot(pred_update[1:],label='predicted (update)')
            self.RMS_error = (np.mean(((ytest[:self.values-1,0,0]-self.pred[1:])/(ytest[:self.values-1,0,0]))**2))**0.5
        else:
            plt.plot(ytest[:self.values-1],label='actual')
            plt.plot(self.pred[1:],label='predicted')
            # plt.plot(pred_update[1:],label='predicted (update)')
            self.RMS_error = (np.mean(((ytest[:self.values-1]-self.pred[1:])/(ytest[:self.values-1]))**2))**0.5
        plt.legend()
        print('The relative RMS error is %f'%self.RMS_error)

    def full_workflow(self):
        self.get_ticker_values()
        self.prepare_test_train()
        self.model_LSTM()

    def full_workflow_and_plot(self, xtest = None, ytest = None):
        self.full_workflow()
        if xtest is None:
            self.xt = self.xtest
        else:
            self.xt = xtest
        if ytest is None:
            self.yt = self.ytest
        else:
            self.yt = ytest
        self.plot_test_values(self.xt, self.yt)
