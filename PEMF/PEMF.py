import numpy as np
from scipy.optimize import curve_fit
from numpy.random import default_rng
import pandas as pd
from scipy import stats

def PEMF( surrogate_trainer,X,Y):
    n_pnts = len(X) # size(x, 2)   number of data points
    n_var  = len(X[0]) # size(x, 1)  number of dimensions

    error_type = 'median'
    n_pnts_final = np.floor(max(0.05 * n_pnts, 3))
    n_pnts_step = n_pnts_final
    n_steps = 4
    n_permutations = 40
    verbosity = 'none'

    check_input(surrogate_trainer,X,Y,error_type,verbosity,n_pnts_final,n_pnts_step,n_steps,n_permutations)

    LB = np.zeros(n_var)
    UB = np.zeros(n_var)
    for j in range(n_var):
        LB[j] = min(X[:, j])
        UB[j] = max(X[:, j])

    y_ref = np.std(Y)

    data_X = X
    data_Y = Y
    data = pd.DataFrame([X,Y])

    PEMF_Error_max = np.zeros(n_steps) # Preallocate
    PEMF_Error_med = np.zeros(n_steps) # Preallocate
    n_train = np.zeros(n_steps) # Preallocate
    MedianTest = np.zeros((n_steps, n_permutations)) # Preallocate
    MaxTest = np.zeros((n_steps, n_permutations)) #Preallocate
    med_params = np.zeros((n_steps, 2))
    max_params = np.zeros((n_steps, 2))

    for i in range(n_steps):
        n_train[i] = n_pnts - (n_pnts_final + (i - 1) * n_pnts_step)
       # Training and Test Points for all combinatio in i - th step
        M_Combination = np.zeros((n_permutations, int(n_train[i])),dtype=int) # Preallocate for speed
        for i_NC in range(n_permutations):
            M_Combination[i_NC,:]=np.random.choice(n_pnts, int(n_train[i]),replace=False)

        for j in range(n_permutations):
        # Separate training points and test points:
            training_data_X = []
            training_data_Y = []
            for l in range(len(M_Combination)):
                # training_data[l]=data_X[int(M_Combination[j][l])]
                training_data_X.append(data_X[int(M_Combination[j][l])])
                training_data_Y.append(data_Y[int(M_Combination[j][l])])
            df_X = pd.DataFrame(training_data_X)
            df_Y = pd.DataFrame(training_data_Y)
            # % Define Training and Test Points (X and Y)
            x_train = df_X.values
            y_train = df_Y.values
            n_tests = len(data_X)
            x_test = data_X
            y_test = data_Y

            # % Train Model and Test it
            if(verbosity == 'high'):
                trained_model = surrogate_trainer.fit(x_train, y_train)
            else:
                trained_model = surrogate_trainer.fit(x_train, y_train)

            RAE = np.zeros(n_tests) # RAE - Relative Absolute Error

            for k in range(n_tests):
                y_predicted = trained_model.predict(x_test)
                RAE[k] = np.abs((y_test[k] - y_predicted[k])) #rever y_ref

            #Calculate Median/Max of RAE
            MedianTest[i,j] = np.median(RAE)
            MaxTest[i,j] = np.max(RAE)

        if (error_type == 'median' or error_type == 'both'):
            # MODE - MED % Remove Outlier in Med(RAE) and fit to log - normal
            parmhat = lognfit_outliers(MedianTest[i,:], 70)
            med_params[i,:] = parmhat
            # Calculate mode of distribution
            PEMF_Error_med[i] = np.exp(parmhat[0] - (parmhat[1])**2)
        # end if median
        if (error_type == 'median' or error_type == 'both'):
            # MODE - MED % Remove Outlier in Med(RAE) and fit to log - normal
            parmhat = lognfit_outliers(MedianTest[i,:], 70)
            max_params[i,:] = parmhat
            # Calculate mode of distribution
            PEMF_Error_max[i] = np.exp(parmhat[0] - (parmhat[1])**2)
        # end if median

        if (verbosity == 'low' or verbosity == 'high'):
            tot = n_steps * n_permutations
            curr = i * n_permutations
            print('Iter %d: %d of %d intermediate models evaluated\n',i, curr, tot)

    n_train = np.flipud(n_train[:])
    PEMF_Error_med = np.flipud(PEMF_Error_med[:])
    PEMF_Error_max = np.flipud(PEMF_Error_max[:])
    MaxTest = np.flipud(MaxTest)
    MedianTest = np.flipud(MedianTest)
    max_params = np.flipud(max_params)
    med_params = np.flipud(med_params)


    RMSE_MedianE = []
    RMSE_MaxE = []

    for model_type in range(1,3):
        RMe = SelectRegression(n_train,PEMF_Error_med,model_type,n_pnts)
        RMSE_MedianE.append(RMe[:])
        RMa = SelectRegression(n_train,PEMF_Error_max,model_type,n_pnts)
        RMSE_MaxE.append(RMa[:])

    # Median Error
    model_id = min(RMSE_MedianE)
    model_type_med = RMSE_MedianE.index(model_id)
    select_med = SelectRegression(n_train, PEMF_Error_med, model_type_med, n_pnts)
    MedianPrediction = select_med[0]
    v_med = select_med[1]
    CorrelationParameterMedian = SmoothnessCriteria(n_train, PEMF_Error_med, model_type_med)

    if (abs(CorrelationParameterMedian) >= 0.90):
        PEMF_MedError_return = MedianPrediction
        x_med = [n_train,n_pnts]
        # print(PEMF_MedError_return)
    else:
        PEMF_MedError_return = PEMF_Error_med[n_steps - 1]
        x_med = [n_train,n_train[len(n_train)-1]]
        if (error_type == 'both' or error_type == 'median'):
            print('\nSmoothness criterion violated for predicition of median error.\n')
            print('K-fold estimate is used from last iteration.\n\n')

    # Maximum Error
    model_id = min(RMSE_MaxE)
    model_type_med = RMSE_MaxE.index(model_id)
    select_max = SelectRegression(n_train, PEMF_Error_max, model_type_med, n_pnts)
    MaxPrediction = select_max[0]
    v_med = select_max[1]
    CorrelationParameterMax = SmoothnessCriteria(n_train, PEMF_Error_max, model_type_med)

    if (abs(CorrelationParameterMax) >= 0.90):
        PEMF_MaxError_return = MaxPrediction
        x_med = [n_train,n_pnts]
    else:
        PEMF_MedError_return = PEMF_Error_max[n_steps-1]
        x_med = [n_train,n_train[len(n_train)-1]]
        if (error_type == 'both' or error_type == 'max'):
            print('\nSmoothness criterion violated for predicition of maximum error.\n')
            print('K-fold estimate is used from last iteration.\n\n')

    # Return desired type of error
    if(error_type =='median'):
        PEMF_Error = PEMF_MedError_return;
        if(verbosity =='none'):
            print('\nPEMF_Error (median): %f\n\n',PEMF_Error)
    elif(error_type =='max'):
        PEMF_Error = PEMF_MaxError_return
        if(verbosity == 'none'):
            print('\nPEMF_Error (max): %f\n\n',PEMF_Error)
    elif(error_type =='both'):
        PEMF_Error = [PEMF_MedError_return,PEMF_MaxError_return]
        if( verbosity =='none'):
            print('\nPEMF_Error (median): %f\n\n',PEMF_Error[0])
            print('\nPEMF_Error (max): %f\n\n',PEMF_Error[1])
    return 0 #PEMF_Error

def check_input(surrogate_trainer, X,Y, error_type, verbosity, n_pnts_final, n_pnts_step, n_steps, n_permutations):
    # % Check X and Y data are the right size
    if(len(Y) != len(X)):
        print('X and Y must have the same number of columns')
        exit()
    #% Check is there enough data
    min_pnts_first_step = 3
    if(len(Y)< n_steps*n_pnts_step + min_pnts_first_step):   #verificar aqui!
        print('Not enough data points.  Use less steps, a smaller step size, or provide more data')
        exit()
    #% Check n_permutations is greater than 9
    if(n_permutations < 10 ):
        print('At least 10 permutations are required per iteration')
        exit()

    try:
        model = surrogate_trainer(X, Y)
    except:
        print("PEMF surrogate_trainer does not match expected format")


    try:
        model(X[1,:])
    except:
        print("PEMF surrogate_trainer does not match expected format")

    if (min(Y) / max(Y) < 5 * 10 ** -3):
        print('Data spans much more than 2 orders of magnitude.  PEMF uses relative error.')

def lognfit_outliers(dat, outlier_percent):
    # YP = np.percentile(dat,outlier_percent)
    # l = 0
    # CT = np.zeros(len(dat))
    # for i in range(len(dat)):
    #     if(dat[i] < YP):
    #         CT[l] = dat[i]
    #         l+=1
    # # CT[l:(l-1)] = []
    #
    # CP = CT
    # mu = np.median(CP)
    # sigma = np.std(CP)
    # n = len(CP)
    #
    # Meanmat = np.tile(mu,(n,1))
    # Sigmamat = np.tile(sigma,(n,1))
    # outliers = CP - Meanmat[0] > 3 * Sigmamat[0]
    # CP[np.any(outliers, axis= 2),:] = []
    # norm_mu, norm_sig = stats.norm.fit(np.log(CP[:]))

    parmhat  = stats.norm.fit(np.log(dat))
    return parmhat

def SmoothnessCriteria(x,y,iSType):
# Returns the smothness of the fit for a given regression model
    if(iSType==1):
        r = np.corrcoef(x,np.log(y))
    elif(iSType==2):
        r = np.corrcoef(np.log(x), np.log(y))
    Rho = r[0,1] # r[1,2]
    return Rho

def SelectRegression(X,Y,iSType,NinP):
    # Helper function with a few modes
    # Able to fit PEMF error to two different regression models depending on the value of isType
    # Able to predict the next value in the regression (ErrorPrediction)
    # Able to give the model fit parameters (VCoe = [a,b])

    n_X = len(X)

    # 1. Exponential Fit Model   Y=a*exp(b*X)
    if (iSType == 1):
        ff = curve_fit(func,X,Y)
        a11= ff[0][0]
        b11= ff[0][1]
        # c11= ff[0][2]
        ErrorPrediction = a11 * np.exp(b11 * NinP)

        data = Y
        estimate = []
        for i in range(n_X):
            estimate.append(a11*np.exp(b11*X[i]))

        RMSE = Grmse(data,estimate)

        VCoe = [a11,b11]

    if (iSType == 2):
        a11,b11 = powerFit(X,Y)
        ErrorPrediction = a11 * (NinP)**(b11)

        data = Y
        estimate = []
        for i in range(n_X):
            estimate.append(a11 * X[i] ** b11)

        RMSE = Grmse(data, estimate)

        VCoe = [a11, b11]

    return [ErrorPrediction,RMSE,VCoe]

def Grmse(data,estimate):
    # Function to calculate root mean square error from a data vector or matrix
    # I = ~isnan(data) & ~isnan(estimate);
    # data = data(I); estimate = estimate(I);
    rI= 0
    n = len(data)
    for i in range(n):
        rI = rI + (data[i] - estimate[i])**2
    rI = rI/n
    r = np.sqrt(rI)
    return r

def powerFit(Y,X):
    # Performs a regression fit with a Power fit model Y = a*X^b
    n = len(X)
    z = np.zeros(n)

    for i in range(n):
        z[i] = np.log(Y[i])
    w = np.zeros(n)

    for i in range(n):
        w[i] = np.log(X[i])
    wav = sum(w) / n
    zav = sum(z) / n

    swz = 0
    sww = 0

    for i in range(n):
        swz = swz + w[i]*z[i]*zav
        sww = sww + (w[i])**2 - wav**2
    a1 = swz/sww
    a0 = zav - a1 * wav
    a = np.exp(a0)
    b = a1

    return a,b

def func(x, a, b, c):
    # Generate data with an exponential trend and then fit the data using a single-term exponential
    #  Coefficients (with 95% confidence bounds): a: [2.021,1.89, 2.151] b [ -0.1812,-0.2104,-0.152]
    return a * np.exp(b * x) + c
