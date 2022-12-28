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
carteira = Portfolio(cotacoes)


#Funções Básicas

#Calcular o retorno e volatilidade anual de cada ativo
retorno_anual = carteira.calcular_retorno_anual()
volatilidade_anual = carteira.calcular_volatilidade_anual()

#Calcular o índice Sharpe de cada ativo
sharpe = carteira.calcular_sharpe()

#Calcular o máximo drawdown de cada ativo
max_drawdown = carteira.max_drawdown()

#Calcular o drawdown de um ativo específico
carteira.drawdown(coluna = ['VALE3.SA'])

#Mostrar uma tabela resumo para os indicadores calculados acima
tabela_resumo = carteira.tabela_resumo()

#Otimização de Portfolios

#Determinar os pesos para alocação de cada portfolio
pesos_min_risco = carteira.portfolio_minimo_risco()
pesos_max_sharpe = carteira.portfolio_max_sharp()
pesos_hrp = carteira.portfolio_HRP()

#Avaliar o desempenho de cada portfolio
retornos = carteira.avaliar_desempenho_portfolio(pesos_min_risco, "Portfolio MInimo Risco")

#Simulação

#Realizar backtesting
tabela_retornos = carteira.backtesting(data_inicio_analise = pd.to_datetime("2019-01-01"))


        
        
