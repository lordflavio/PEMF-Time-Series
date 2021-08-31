import numpy as np
from PEMF.Models_RBF.rbf_interp import rbf_interp
from PEMF.Models_RBF.rbf_approx import rbf_approx

def rbf_trainer(X,X_data, Y_data, C, kernel_type):

    rbf_coeff = rbf_interp(X_data, Y_data, C, kernel_type)
    trained_model = rbf_approx(X_data, X, rbf_coeff, C, kernel_type)
    return trained_model
