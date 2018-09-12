import numpy as np
from numpy.linalg import norm as norm
from scipy.special import erfc

def get_quantiles(fmin, m, s):
    '''
    Quantiles of the Gaussian distribution useful to determine the acquisition function values
    :param fmin: current minimum.
    :param m: vector of means.
    :param s: vector of standard deviations. 
    '''
    if isinstance(s, np.ndarray):
        s[s<1e-10] = 1e-10
    elif s< 1e-10:
        s = 1e-10
    u = (fmin-m)/s
    phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
    Phi = 0.5 * erfc(-u / np.sqrt(2))    
    return (phi, Phi, u)

def EI(x, fmin=None, model=None, n=None, d=None):
    '''
    Expected improvement with gradient
    '''
    m, v = model.predict(x)
    v = np.clip(v, 1e-10, np.inf)
    s = np.sqrt(v)
    phi, Phi, u = get_quantiles(fmin, m, s)
    f_acqu = s * (u * Phi + phi)
    dmdx, dvdx = model.predictive_gradients(x)
    dmdx = dmdx[:,:,0]
    dsdx = dvdx / (2*np.sqrt(v))
    df_acqu = dsdx * phi - Phi * dmdx
    return -f_acqu, -df_acqu.T

def LCB(x, fmin=None, model = None, n=None, d=None):
    '''
    Lower confidence bound with gradient
    '''
    m, v = model.predict(x)
    v = np.clip(v, 1e-10, np.inf)
    s = np.sqrt(v)
    eta = 0.1
    exploration_weight = np.sqrt(2.*np.log((n**(d/2.+2.))*(np.pi**2.)/(3.*eta)))
    f_acqu = -m  + exploration_weight * s
    
    dmdx, dvdx = model.predictive_gradients(x)
    dmdx = dmdx[:,:,0]
    dsdx = dvdx / (2*s)
    df_acqu = -dmdx + exploration_weight * dsdx    
    return -f_acqu, -df_acqu.T 

def PI(x, fmin=None, model = None, n=None, d=None):
    '''
    Probability of improvement with gradient
    '''
    m, v = model.predict(x)
    v = np.clip(v, 1e-10, np.inf)
    s = np.sqrt(v)
    phi, Phi, u = get_quantiles(fmin, m, s)    
    f_acqu = Phi

    dmdx, dvdx = model.predictive_gradients(x)
    dmdx = dmdx[:,:,0]
    dsdx = dvdx / (2*np.sqrt(v))    
    df_acqu = -(phi/s)* (dmdx + dsdx * u)
    return -f_acqu, -df_acqu.T