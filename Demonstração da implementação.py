from Finance import Portfolio

import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as fig
import pandas_datareader.data as web
import yfinance as yf
yf.pdr_override()
import scipy.optimize as solver

#Ações avaliadas
tickers = ["PETR4.SA","VALE3.SA", "BBAS3.SA", "BBDC4.SA", "ITUB4.SA",
           "ELET6.SA", "EMBR3.SA", "TAEE11.SA"]

#Obtendo cotações
cotacoes = web.get_data_yahoo(tickers, start = "2017-01-01", end = "2021-12-31")["Adj Close"]
cotacoes = cotacoes.fillna(method = 'ffill')

#Criar o objeto carteira
carteria = Portfolio(cotacoes)

#Calcular os pesos dos ativos para cada carteira
pesos_min_risco = carteria.portfolio_minimo_risco()
pesos_max_sharpe = carteria.portfolio_max_sharp()
pesos_hrp = carteria.portfolio_HRP()

#Chamar função para realizar backtesting
base_retorno = carteria.backtesting(data_inicio_analise = pd.to_datetime("2019-01-01"))


        
        