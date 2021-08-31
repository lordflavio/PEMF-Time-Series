
import numpy as np


def normalise_interval(minimo, maximo, serie):
	from sklearn.preprocessing import MinMaxScaler
	scaler = MinMaxScaler(feature_range=(minimo, maximo))
	scaler = scaler.fit(serie)
	normalized = scaler.transform(serie)
	return normalized, scaler 
	
	
	
def desnorm_interval(serie_norm, serie_real, minimo, maximo):
	norm, scaler = normalise_interval(minimo, maximo, serie_real)
	inversed = scaler.inverse_transform(serie_norm)
	return inversed



def normalise(serie):
    minimo = min(serie)
    maximo = max(serie)
    y = (serie - minimo) / (maximo - minimo)
    return y


def desnorm(serie_atual, serie_real):
    import pandas as pd
    minimo = min(serie_real)
    maximo = max(serie_real)
    
    serie = (serie_atual * (maximo - minimo)) + minimo
    
    return list(serie) 


def create_windows(serie, n_in=3,n_out=1, dropnan=True):
    import pandas as pd

    serie = pd.DataFrame(serie)
    n_vars = 1  if type(serie) is list else serie.shape[1]
    df = pd.DataFrame(serie)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))

    for i in range(0, n_out):
        cols.append(df.shift(-i))

    agg = pd.concat(cols, axis=1)

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


def select_lag_acf(serie, max_lag):
    from statsmodels.tsa.stattools import acf
    x = serie[0: max_lag+1]
    
    acf_x, confint = acf(serie, nlags=max_lag, alpha=.05, fft=False,
                             unbiased=False)
    
    
    limiar_superior = confint[:, 1] - acf_x
    limiar_inferior = confint[:, 0] - acf_x

    lags_selecionados = []
    
    for i in range(1, max_lag+1):

        
        if acf_x[i] >= limiar_superior[i] or acf_x[i] <= limiar_inferior[i]:
            lags_selecionados.append(i-1)  #-1 por conta que o lag 1 em python é o 0
    
    #caso nenhum lag seja selecionado, essa atividade de seleção para o gridsearch encontrar a melhor combinação de lags
    if len(lags_selecionados)==0:


        print('NENHUM LAG POR ACF')
        lags_selecionados = [i for i in range(max_lag)]

    print('LAGS', lags_selecionados)

    #inverte o valor dos lags para usar na lista de dados
    lags_selecionados = [max_lag - (i+1) for i in lags_selecionados]



    return lags_selecionados

def split_serie_less_lags(serie, perc_train, perc_val = 0):
    import numpy as np
    
    series = serie.values
        
    train_size = np.fix(len(series) *perc_train)
    train_size = train_size.astype(int)
    
    if perc_val > 0:
        
        val_size = np.fix(len(serie) *perc_val).astype(int)
        
        x_train = series[0:train_size]
        x_val = series[train_size:train_size+val_size]        
        x_test = series[(train_size+val_size):-1]

        return x_train, x_test, x_val
        
    else:
        
                
        x_train = series[0:train_size+1]
        x_test = series[train_size:-1]
        

        return x_train, x_test
		
def select_validation_sample(serie, perc_val):
    tam = len(serie)
    val_size = np.fix(tam *perc_val).astype(int)
    return serie[0:tam-val_size,:],  serie[tam-val_size:-1,:]



