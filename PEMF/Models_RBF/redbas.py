import numpy as np
import math
def redbas(x1,x2,c,kernelType):
    d = 0
    if(kernelType == 'Linear'):
        d = np.linalg.norm(x1 - x2)
    elif (kernelType == 'Cubic'):
        d = (np.linalg.norm(x1 - x2)**3)
    elif (kernelType == 'Tps'):
        r = np.linalg.norm(x1 - x2)
        if(r < 1e-200):
            d = 0
        else:
            d = r**2*math.log(r)
    elif (kernelType == 'Gaussian'):
        d = np.exp(-(np.linalg.norm(x1 - x2))**2/(2*c))
    elif (kernelType == 'Multiquadric'):
        d = ((np.linalg.norm(x1 - x2))**2+c**2)**0.5
    return d

