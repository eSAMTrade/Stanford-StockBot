## Stock price prediction
We intend to leverage historical stock data to predict the stock prices for the subsequent days. This repository consists of our computational experiments surrounding the model architecture and the feature preparation that we used to converge onto the final production level code.

## Description
We use an LSTM based approach for the initial prediction. We begin with single LSTMs and stacked LSTMs for our analysis. We then develop an encoder-decoder system to predict future stock prediction values. On converging onto to the ideal LSTM architecture, we then use stock ticker data from related industries and google trends data to make a more accurate prediction. We then develop a decision making bot that makes buy and sell decisions depending on the stock closing prices to maximize profits.  


## Installation
Use the following lines to get all the dependencies setup

```
cd code-directory
git clone https://gitlab.com/cs230_stock_analysis/stock_price_prediction.git
cd stock_price_prediction
python3 -m pip install -r py_requirements.txt
cd tests
```

## Usage: Tests
The predictions using a series of LSTM networks and for different tickers can be obtained by running (also to produce Figs.1, 2a, 2b and 6 from the final report):
```
python3 train_single_test.py
```
The encoder-decoder results from the report in Fig. 3 can be reproduced by:
```
python3 train_single_test_ED.py
```
Using multiple stocks of a given industry to train model and then carry out inference, as shown in Fig. 4a can be reproduced by:
```
python3 train_categorical_test.py
```
The cross-correlation between stocks and effect fo inferring stock prices from models trained on different industries, as shown in Figs. 4b and 4c can be reproduced by:
```
python3 correlation_strengths.py
```
The comparison between predictions using the custom loss function and the regular loss function while predicting multiple dayds in advance, as shwon in Fig. 5 can be reproduced by:
```
python3 test_forward_decay.py
```
The google-trends data along with the stock ticker data to predict stock prices, as shown in Fig. 7, can be reproduced by:
```
python3 train_categorical_GT_test.py
```
## Support
For any support regarding the implementation of the source code, contact the developers at: shaswatm@stanford.edu, avijay@stanford.edu or nandan99@gmail.com

## Contributing
We are open to suggestions and contributions to the repository after June 2022 (end of CS230 course).

## Authors and acknowledgment
Authored by Shaswat Mohanty, Anirudh Vijay and Nandagopan Gopakumar. We acknowledge the teaching team of Andrew Ng and Kian Katanforosh for equipping us with the tools to work on this project. We thank Elaine Sui and Vincent Liu for their suggestions and guidance in the initial phases of the project.

## Project status
Developement is currently ongoing.
