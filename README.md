## Stock price prediction
We intend to leverage historical stock data to predict the stock prices for the subsequent days. This repository consists of our computational experiments surrounding the model architecture and the feature preparation that we used to converge onto the final production level code.

## Description
We use an LSTM based approach for the initial prediction. We begin with single LSTMs and stacked LSTMs for our analysis. We then develop an encoder-decoder system to predict future stock prediction values. 


## Installation
Use the following lines to get all the dependencies setup

```
cd code-directory
git clone https://gitlab.com/cs230_stock_analysis/stock_price_prediction.git
cd stock_price_prediction
python3 -m pip install -r py_requirements.txt
```

## Usage
The predictions using a series of LSTM networks and for different tickers can be obtained by running:
```
python3 train_single_test.py
```

The convergence in the stock values as we move closer to the prediction day (in the case of multiple days being predicted at once) can be obtained using:
```
python3 test_forward_decay.py
```

## Support
For any support regarding the implementation of the source code, contact the developers at: shaswatm@stanford.edu, avijay@stanford.edu or nandan99@gmail.com

## Roadmap
* Use related/all ticker values to predict a single stock value so that the perfromance of related or all stocks can be used to develop a more generalizable model.
* Develop a bot that makes decisions based on closing prices and see how that compares against stock performances and popular ETFs. 


## Contributing
We are open to suggestions and contributions to the repository after June 2022.

## Authors and acknowledgment
Authored by Shaswat Mohanty, Anirudh Vijay and Nandagopan Gopakumar.


## Project status
Developement is currently ongoing.
