from PEMF.PEMF import PEMF
from surrogate import Surrogate
import numpy as np
from preprocessamento import *
import pandas as pd

serie_name = 'airline'
print('Série:', serie_name)
end = './dataset/'+serie_name+'.txt'
data = pd.read_csv(end, delimiter=' ', header=None)
serie = data[0]

serie_normalizada = normalise(serie)

train, test = split_serie_less_lags(serie_normalizada, 0.75)

max_lag = 20
lags_acf = select_lag_acf(serie_normalizada, max_lag)
max_sel_lag = lags_acf[0]

train_lags = create_windows(train, max_sel_lag+1)

test_data = create_windows(test, max_sel_lag+1)

train_data,  val_data = select_validation_sample(train_lags, 0.34)

x_train = train_data[:,0:-1]
x_train = x_train[:,lags_acf]
y_train = train_data[:,-1]
x_val = val_data[:,0:-1]
x_val = x_val[:,lags_acf]
y_val = val_data[:,-1]

surrogate_trainer = Surrogate("deca", 0.1, 'Gaussian')
PEMF(surrogate_trainer,x_train,y_train)




