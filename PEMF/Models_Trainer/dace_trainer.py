import numpy as np

from pydacefit.corr import corr_gauss, corr_cubic, corr_exp, corr_expg, corr_spline, corr_spherical
from pydacefit.dace import DACE, regr_linear, regr_quadratic
from pydacefit.regr import regr_constant


def dece_treiner(X,Y,HP,Kernel):

    if(Kernel == 'Gaussian'):
        regression = regr_constant
        correlation = corr_gauss
        dacefit = DACE(regr=regression, corr=correlation,
                       theta=HP, thetaL=0.00001, thetaU=20)
        dacefit.fit(X,Y)
        return dacefit

    elif(Kernel == 'Exponential'):
        regression = regr_constant
        correlation = corr_exp
        dacefit = DACE(regr=regression, corr=correlation,
                       theta=HP, thetaL=0.00001, thetaU=20)
        dacefit.fit(X, Y)
        return dacefit
    elif(Kernel == 'Cubic'):
        regression = regr_constant
        correlation = corr_cubic
        dacefit = DACE(regr=regression, corr=correlation,
                       theta=HP, thetaL=0.00001, thetaU=20)
        dacefit.fit(X, Y)
        return dacefit
    elif(Kernel == 'Linear'):
        regression = regr_constant
        correlation = corr_spline
        dacefit = DACE(regr=regression, corr=correlation,
                       theta=HP, thetaL=0.00001, thetaU=20)
        dacefit.fit(X, Y)
        return dacefit
    elif(Kernel == 'Spherical'):
        regression = regr_constant
        correlation = corr_spherical
        dacefit = DACE(regr=regression, corr=correlation,
                       theta=HP, thetaL=0.00001, thetaU=20)
        dacefit.fit(X, Y)
        return dacefit

def predict_deca(deca, X):
    trained_model = deca.predict(X)
    return trained_model





