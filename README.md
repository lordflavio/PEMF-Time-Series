# PEMF-Time-Series
Predictive Estimation of Model Fidelity (PEMF) - Adapted for use in time series.

Predictive Estimation of Model Fidelity (PEMF) is a model-independent approach to measure the fidelity of surrogate models or metamodels, such as Kriging, Radial Basis Functions (RBF), Support Vector Regression (SVR), and Neural Networks. It can be perceived as a novel sequential and predictive implementation of K-fold cross-validation. PEMF takes as input a model trainer (e.g., RBF-multiquadric or Kriging-Linear), sample data on which to train the model, and hyper-parameter values (e.g., shape factor in RBF) to apply to the model. 

As output, it provides a predicted estimate of the median and/or the maximum error in the surrogate model. PEMF has been reported to be more accurate and robust than typical leave-one-out cross-validation, in providing surrogate model error measures (for various benchmark functions). 

The current version of PEMF has been implemented with RBF (included in this package), Kriging (DACE package), and SVR (Libsvm package), PEMF (has been and) can be readily used for the following purposes:

1. Surrogate model validation 
2. Surrogate model uncertainty analysis
3. Surrogate model selection 4. 
4. Surrogate-based optimization (to guide sequential sampling) Other perceived broader applications of PEMF include testing of machine learning models and uncertainty analysis with data-driven models (and other areas where leave-one-out or k-fold cross-validation is typically used).
