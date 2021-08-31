import numpy as np
import math
from PEMF.Models_RBF.redbas import redbas
def rbf_approx(x_in,x_test,rbf_coeff,c,KernelType):
    alpha = rbf_coeff
    x = x_in
    x_t = x_test
    n_p = x.shape[0]     # size(x, 2)   number of data points
    n_dim = x.shape[1]    # size(x, 1)  number of dimensions
    p = x_t.shape[0]
    k = 0
    f = []
    for i in range(1,p):
        sum1 = 0
        for j in range(1,n_p):
            sum1 = sum1 + alpha[j] * redbas(x_t[:,i],x[:,j],c,KernelType)
        f.append(sum1)
    return np.array(f)