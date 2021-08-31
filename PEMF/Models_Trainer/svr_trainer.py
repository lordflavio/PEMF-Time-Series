from sklearn.svm import SVR
import numpy as np

def svr_trainer(X,Y, HP,Kernel):
    if (Kernel == 'rbf'):
        svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        svr.fit(X, Y)
        return svr
    elif (Kernel == 'linear'):
        svr = SVR(kernel='linear', C=100, gamma='auto')
        svr.fit(X, Y)
        return svr
    elif (Kernel == 'poly'):
        svr = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
                  coef0=1)
        svr.fit(X, Y)
        return svr
    elif (Kernel == 'sigmoid'):
        svr = SVR(kernel='sigmoid', C=100, gamma='auto', degree=3, epsilon=.1,
                  coef0=1)
        svr.fit(X, Y)
        return svr


def predict_svr(svr, X):
    trained_model = svr.predict(X)
    return trained_model





