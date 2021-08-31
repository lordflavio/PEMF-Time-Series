import numpy as np
from PEMF.Models_RBF.redbas import redbas

def rbf_reg(x_in,y_out,C_SP,C_RP,kernelType):
    x = x_in
    mu = y_out

    n_p = x.sharp[0]       # size(x, 2)   number of data points
    n_dim = x.sharp[1]     # size(x, 1)  number of dimensions
    a = np.zeros(n_p,n_p)
    for i in range(n_p):
        for j in range(n_p):
            a[i,j] = redbas(x[:,i],x[:,j],C_SP,kernelType)

    rbf_coeff = np.linalg.inv(a+C_RP*(np.eye(n_p)))@mu

    return rbf_coeff
