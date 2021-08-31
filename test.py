# Find an optimum regression pipeline

import pandas as pd
import numpy as np
from sklearn.gaussian_process.kernels import Matern, Sum, ExpSineSquared
from SKSurrogate import *

config = {
    # Regressors
    "sklearn.linear_model.LinearRegression": {"normalize": Categorical([True, False])},
    "sklearn.kernel_ridge.KernelRidge": {
        "alpha": Real(1.0e-4, 10.0),
        "kernel": Categorical(
            [
                Sum(Matern(), ExpSineSquared(l, p))
                for l in np.logspace(-2, 2, 10)
                for p in np.logspace(0, 2, 10)
            ]
        ),
    },
    # Preprocesssors
    "sklearn.preprocessing.StandardScaler": {
        "with_mean": Categorical([True, False]),
        "with_std": Categorical([True, False]),
    },
    #"sklearn.preprocessing.Normalizer": {"norm": Categorical(["l1", "l2", "max"])},
}
import warnings

warnings.filterwarnings("ignore", category=Warning)

df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
    sep="\t",
    names=["Frequency", "Angle", "length", "velocity", "thickness", "level"],
)
X = df.drop("level", axis=1).values
y = df["level"].values

A = AML(
    config=config,
    length=3,
    check_point="./",
    verbose=2,
    scoring="neg_mean_squared_error",
)
A.eoa_fit(X, y, max_generation=5, num_parents=8)
print(A.get_top(5))
