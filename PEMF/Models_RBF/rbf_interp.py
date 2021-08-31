import numpy as np
from PEMF.Models_RBF.redbas import redbas
def rbf_interp(x_in,y_out,c,kernelType):
    x = x_in
    mu = y_out

    n_p = x.sharp[0]       # size(x, 2)   number of data points
    n_dim = x.sharp[1]     # size(x, 1)  number of dimensions
    a = np.zeros(n_p,n_p)
    for i in range(n_p):
        for j in range(n_p):
            a[i,j] = redbas(x[:,i],x[:,j],c,kernelType)

    rbf_coeff = np.linalg.lstsq(a, mu)
    return rbf_coeff
