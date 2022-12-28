#Importação das bibliotecas
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as fig
import pandas_datareader.data as web
import yfinance as yf
yf.pdr_override()
import scipy.optimize as solver

class Portfolio:
    
    def __init__(self, dataset, calcular_retornos_diarios = True):
        
        self.dataset = dataset
        self.ri = dataset.pct_change(1).dropna() if calcular_retornos_diarios == True else dataset
        
    def calcular_retorno_anual(self):

        n_periodos = len(self.ri)
        retorno_anualizado = (1+self.ri).prod()**(252/n_periodos)-1
        retorno_anualizado = retorno_anualizado.reset_index()
        retorno_anualizado = retorno_anualizado.rename(columns = {0:"Retorno",'index':"Ativos"}).set_index('Ativos')
        
        return retorno_anualizado
        
    def calcular_volatilidade_anual(self):
        
        vol_anualizado = self.ri.std()*(252**0.5)
        vol_anualizado = vol_anualizado.reset_index()
        vol_anualizado = vol_anualizado.rename(columns = {0:"Risco",'index':"Ativos"}).set_index('Ativos')
        
        return vol_anualizado
        
    def calcular_sharpe(self, taxa_livre_risco = 0):
        
        #Calcular o risco das carteiras
        retorno = self.calcular_retorno_anual() - taxa_livre_risco
        risco = self.calcular_volatilidade_anual()
        sharpe = retorno.merge(risco, on = "Ativos", how = "inner")
        sharpe['Sharpe'] = sharpe['Retorno']/sharpe['Risco']
        sharpe = sharpe.drop(columns = ['Retorno', 'Risco'])
        
        return sharpe
    
    def drawdown(self,coluna, grafico = True):
        
        df_drawdown = self.ri[[coluna]]
        df_drawdown["Retorno Acumulado"] = 1000*(1+self.ri[[coluna]]).cumprod()
        df_drawdown["Pico"] = df_drawdown["Retorno Acumulado"].cummax()
        df_drawdown["Drawdown"] = (df_drawdown["Retorno Acumulado"] - df_drawdown["Pico"])/df_drawdown["Pico"]

        df_drawdown = df_drawdown.drop(columns = coluna)

        if grafico == True:
            df_drawdown[["Retorno Acumulado","Pico"]].plot()

        return df_drawdown
    
    def max_drawdown(self):
        
        max_drawdown = pd.DataFrame()
        for coluna in self.ri.columns:
            
            temp_drawdown = self.drawdown(coluna, grafico = False)
            temp_drawdown = temp_drawdown.reset_index()[["Date","Drawdown"]].min()
            temp_drawdown = pd.DataFrame(temp_drawdown).rename(columns = {0:coluna,"Date":"Date Drawdown"}).T
            max_drawdown = max_drawdown.append(temp_drawdown)
            
        max_drawdown = max_drawdown.reset_index().rename(columns = {"Date":"Date Drawdown","index":"Ativos"})
        max_drawdown = max_drawdown.set_index("Ativos")
            
        return max_drawdown
    
    def tabela_resumo(self, taxa_livre_risco = 0):
        
        #Calcular o risco das carteiras
        retorno = self.calcular_retorno_anual()
        risco = self.calcular_volatilidade_anual()
        sharpe = self.calcular_sharpe(taxa_livre_risco)
        max_drawdown = self.max_drawdown()
        
        tabela_resumo = pd.DataFrame()
        tabela_resumo = retorno
        tabela_resumo = tabela_resumo.merge(risco, on = 'Ativos', how = 'inner')
        tabela_resumo = tabela_resumo.merge(sharpe, on = 'Ativos', how = 'inner')
        tabela_resumo = tabela_resumo.merge(max_drawdown, on = 'Ativos', how = 'inner')
        
        return tabela_resumo
    
    
    
    def portfolio_minimo_risco(self):
        
        """
        
        Função para calcular o portfolio otimizado conforme porposto por
        Markowitz (1952) para a minimização do risco da carteira
        
        Recebe como parâmetros:
            Um dataframe com as cotações dos ativos, onde o index é a data da cotação
            e cada ativo em uma coluna diferente
            
        Retorna:
            Um Array contendo os pesos que deverão ser alocados em cada ativo
        
        """
        
        n = len(self.ri.columns)
        
        #Calculando retorno dos ativos e covariância da carteira
        sigma = self.ri.cov()*252 #covariância da carteria
        
        #Função objetivo que minimiza o risco
        def f_obj(peso):
            return np.sqrt(np.dot(peso.T,np.dot(sigma,peso)))
        
        #Definição do chute inicial
        x0 = np.array([1.0/(n) for x in range(n)])
        
        #Definição dos limites de posição de cada ativo
        bounds = tuple((0,1) for x in range(n))
        
        #Definição das restrições
        constraints = [{'type':'eq', 'fun': lambda x: sum(x)-1}]
        
        #Executar Solver
        outcome = solver.minimize(f_obj, x0, constraints = constraints, 
                                  bounds = bounds, method = 'SLSQP')
        
        #Formatar Dataframe com os pesos dos ativos
        pesos = outcome['x']
        pesos = pd.DataFrame(pesos, index = self.ri.columns, columns = ['Pesos'])

        return pesos
    
    def portfolio_max_sharp(self):
        
        """
        
        Função para calcular o portfolio otimizado conforme porposto por
        Markowitz (1952) para a maximação do sharpe da carteira
        
        Recebe como parâmetros:
            Um dataframe com as cotações dos ativos, onde o index é a data da cotação
            e cada ativo em uma coluna diferente
            
        Retorna:
            Um Array contendo os pesos que deverão ser alocados em cada ativo
            
        """
        
        n = len(self.ri.columns)
        
        #Calculando retorno dos ativos e covariância da carteira  
        rets = self.calcular_retorno_anual()
        sigma = self.ri.cov()*252 #covariância da carteria
        
        def port_ret(peso):
            return np.sum(rets*peso)
        
        def port_vol(peso):
            return np.sqrt(np.dot(peso.T,np.dot(sigma,peso)))
            
        #Função objetivo que minimiza o risco
        def f_obj(peso):
            return -port_ret(peso)/port_vol(peso)
        
        #Definição do chute inicial
        x0 = np.array([1.0/(n) for x in range(n)])
        
        #Definição dos limites de posição de cada ativo
        bounds = tuple((0,1) for x in range(n))
        
        #Definição das restrições
        constraints = [{'type':'eq', 'fun': lambda x: sum(x)-1}]
        
        #Executar Solver
        outcome = solver.minimize(f_obj, x0, constraints = constraints, 
                                  bounds = bounds, method = 'SLSQP')
        
        #Formatar Dataframe com os pesos dos ativos
        pesos = outcome['x']
        pesos = pd.DataFrame(pesos, index = self.dataset.columns, columns = ['Pesos'])

        return pesos
    
    def portfolio_HRP(self):
        
        from Portfolio_HRP import correlDist,getQuasiDiag,getRecBipart
        import scipy.cluster.hierarchy as sch
        
        cov = self.ri.cov()*252
        corr = self.ri.corr()
    
        #Pegar a distância da matriz de correlação
        dist=correlDist(corr)
        
        #Criar clusters
        link=sch.linkage(dist,'single')
        
        #Ordenar os itens do cluster pela distância
        sortIx=getQuasiDiag(link)
        sortIx=corr.index[sortIx].tolist() # recover labels
        
        #Calcular alocação
        pesos=getRecBipart(cov,sortIx)
        pesos = pd.DataFrame(pesos, index = self.dataset.columns, columns = ['Pesos'])
        
        return pesos
    
    def avaliar_desempenho_portfolio(self, pesos, nome):
        
        """
        
        Função para realizar a avaliação do portfolio
        
        Recebe como parâmetros:
            Um dataframe com as cotações dos ativos, onde o index é a data da cotação
            e cada ativo em uma coluna diferente
            Um Array contendo os pesos que deverão ser alocados em cada ativo
            Uma String contendo o nome do portfolio avaliado
            
        Retorna:
            Retorna um dataframe contendo os retornos do portfolio avaliado durante
            o período
            
        """
        
        pesos = np.array(pesos)[:,0]
        ri_analise_ponderada = pesos*self.ri
        ri_analise_ponderada[nome] = ri_analise_ponderada.sum(axis = 1)
        ri_analise_ponderada = ri_analise_ponderada[[nome]]
        
        return ri_analise_ponderada

    def backtesting(self, data_inicio_analise, duracao_carteria = 30, 
                    avaliacao_historico = 365, taxa_livre_risco = 0):
        
        
        """
        
        Função para realizar o backtesting da alocação dos portfolios abaixo:
            Portfolio com o mínimo Risco
            Portfolio máximo sharpe
            Portfolio com pesos iguais
            Portfolio com pesos alaetórios
            
        Recebe como parâmetros:
            Um dataframe com as cotações dos ativos, onde o index é a data da cotação
            e cada ativo em uma coluna diferente
            Uma data (timestamps) para o inicio da análise
            Um INT definindo quanto tempo essa carteira será mantida
            UM INT definindo um período para a avaliação do histórico
            
        Retorno:
            Retorna um dataframe contendo os retornos de cada portfolio ao longo do
            período avaliado
        
        """
        
        #Definindo Df para armazenar o resultado dos portfolios
        base_retorno = pd.DataFrame()
        
        #definindo uma data limite de avaliação
        data_limite = max(self.dataset.index)
        
        #Loop para realizar a análise ao longo do tempo
        while data_inicio_analise <= data_limite: 
            
            #Definindo data limite para a faixa de análise
            data_fim_analise = data_inicio_analise + np.timedelta64(duracao_carteria, 'D')
            
            #Definindo período de tempo para construção de carteira
            data_inicio_amostra = data_inicio_analise - np.timedelta64(avaliacao_historico, 'D')
            data_fim_amostra = data_inicio_analise - np.timedelta64(1, 'D')
            
            #Base de cotaçoes para formação da carteira
            base_amostra = self.dataset[data_inicio_amostra:data_fim_amostra]
            
            #Criar um objeto para calcular os pesos de carteira
            carteria = Portfolio(base_amostra)
            
            #Calculando carteira que minimiza o riscos    
            pesos_minimo_risco = carteria.portfolio_minimo_risco()
            
            #Calculando carteira que maximiza o sharpe
            pesos_max_sharpe = carteria.portfolio_max_sharp()
            
            #Calculando carteira HRP
            pesos_hrp = carteria.portfolio_HRP()
            
            #Base de cotações para análise do desempenho da carteira           
            base_analise = self.dataset[data_inicio_analise:data_fim_analise]
                        
            #Criar um objeto para para avaliar o desempenho de cada carteira
            ri_analise = Portfolio(base_analise)
            
            #função para avaliar o desempenho do portofolio
            desempenho_portfolio_min_risco = ri_analise.avaliar_desempenho_portfolio(pesos_minimo_risco,"Portfolio minimo risco")
            
            #Criar base para armazenar os vários retornos de portfolios
            retornos = desempenho_portfolio_min_risco
            
            #função para avaliar o desempenho do portofolio
            desempenho_portfolio_max_sharpe = ri_analise.avaliar_desempenho_portfolio(pesos_max_sharpe,"Portfolio maximo sharpe")
            
            #Adicionar o desempenho do portfólio com pesos iguais a base de retorno
            retornos = retornos.merge(desempenho_portfolio_max_sharpe, how = 'left', on = 'Date')
            
            #função para avaliar o desempenho do portofolio
            desempenho_portfolio_hrp = ri_analise.avaliar_desempenho_portfolio(pesos_hrp,"Portfolio HRP")
            
            #Adicionar o desempenho do portfólio com pesos iguais a base de retorno
            retornos = retornos.merge(desempenho_portfolio_hrp, how = 'left', on = 'Date')
            
            #Substituir a data do início da análise
            data_inicio_analise = data_fim_analise
            
            #Adicionar retornos obtidos na base de retorno de todos os períodos avaliados
            base_retorno = base_retorno.append(retornos)
        
        
        base_retorno = Portfolio(base_retorno, calcular_retornos_diarios = False)
        
        tabela_resumo = base_retorno.tabela_resumo(taxa_livre_risco)

        #Calcular retornos acumulados
        base_retorno_acumulado = base_retorno.ri.cumsum()
        base_retorno_acumulado.plot()
        
        return tabela_resumo
