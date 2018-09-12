from __future__ import division
import sigopt_funcs as sf
import os.path
import numpy as np
import numpy.random as rndm
import scipy.linalg
from scipy.stats import multivariate_normal
from optimization import BlackBoxFunction
import itertools
from BO_interface import NN

class Normalizer(BlackBoxFunction):
    '''
    Normalizes input space between [0,1] in all dimensions and the output between [-1,0]
    '''
    def __init__(self, func):
        self.func = func
        self.dim = self.func.dim
        self.us_fmin = self.func.fmin
        self.deviation = self.func.fmax-self.func.fmin
        bounds_array, lengths = self.tuplebounds_2_arrays(self.func.bounds) 
        self.lengths = lengths
        self.fmin = 0.0
        self.fmax = 1.0
        self.bounds = sf.lzip([0] * self.dim, [1] * self.dim)
        self.us_bounds = bounds_array
        self.min_loc = (self.func.min_loc-self.us_bounds[:,0])/self.lengths
    
    def tuplebounds_2_arrays(self, bounds):
        bounds_array = np.zeros((self.dim,2))
        lengths = np.zeros((self.dim))
        for i in range(self.dim):
            bounds_array[i,0] = bounds[i][0]
            bounds_array[i,1] = bounds[i][1]
            lengths[i] = bounds[i][1]- bounds[i][0]
        return bounds_array, lengths

    def get_dim(self):
        return self.dim
  
    def do_evaluate(self, x):
        return (self.func.do_evaluate( (x*self.lengths + self.us_bounds[:,0]) ) - self.us_fmin)/self.deviation-1.0
    
    def do_evaluate_clean(self,x):
        return (self.func.do_evaluate_clean( (x*self.lengths + self.us_bounds[:,0]) ) - self.us_fmin)/self.deviation-1.0
        
    def get_maximum(self):
        tuples = list(itertools.product(*zip([0]*self.dim,[1]*self.dim)))
        res = [self.do_evaluate(tuples[i]) for i in range(len(tuples))]
        return max(res)
    
    def confirm(self):
        print("min: {}, max: {}".format(self.do_evaluate(self.min_loc), self.get_maximum()))
        
class Gaussian(BlackBoxFunction):
    def __init__(self, dim=1, num_peaks=1, seed=None, safe_limit=0.2, non_centrality=0.0):
        if seed is not None:
            np.random.seed(seed)
        self.num_peaks = num_peaks
        self.num_evals = 0
        self.verify = True
        self.dim = dim
        if num_peaks == 1:
            self.weights = np.array([1])
        else:
            self.weights = np.random.rand(num_peaks)+np.finfo(float).eps
        fl = np.random.randint(2, size=(num_peaks))
        d = np.random.randint(dim, size=(num_peaks))
        #self.centers = np.random.rand(num_peaks, dim)*(0.5-safe_limit-non_centrality) + safe_limit +fl*(0.5-safe_limit+non_centrality)
        self.centers = np.random.rand(num_peaks, dim)*(1.-2.*safe_limit)+safe_limit
        for i in range(num_peaks):
            self.centers[i,d[i]] = np.random.rand(1, 1)*(0.5-safe_limit-non_centrality) + safe_limit + fl[i]*(0.5-safe_limit+non_centrality)
        og = [scipy.linalg.orth(np.random.randn(dim,dim)) for i in range(num_peaks)]
        self.variances = [np.dot(np.dot(og[i], np.diag(np.random.rand(dim)*0.7+0.3)), og[i].T)*(dim / 20 ) for i in range(num_peaks)]
        mins = [self.do_evaluate(self.centers[i,:]) for i in range(num_peaks)]
        ind = np.argmin( [self.do_evaluate(self.centers[i,:]) for i in range(num_peaks)] )
        self.fmin = mins[ind]
        self.min_loc = self.centers[ind,:]
        self.fmax = self.get_maximum()
        self.bounds = sf.lzip([0] * self.dim, [1] * self.dim)

    def get_maximum(self):
        tuples = list(itertools.product(*zip([0]*self.dim,[1]*self.dim)))
        res = [self.do_evaluate(tuples[i]) for i in range(len(tuples))]
        return max(res)
        
    def do_evaluate(self, x):
        x = np.array(x)
        return np.sum( [-self.weights[i]*scipy.stats.multivariate_normal.pdf(x, self.centers[i,:], self.variances[i]) for i in range(self.num_peaks)] )
    
    def do_evaluate_clean(self, x):
        return self.do_evaluate(x)
        
def function_of_dimension(funcs, dim):
    ret = []
    for func in funcs:
        try:
            ret += [func(dim)]
        except AssertionError:
            pass
    return ret
        
def get_sigopt_functions(max_dim):
    func_names = [sf.Exponential, sf.Giunta, sf.Leon, sf.McCourt14, sf.McCourt15, sf.McCourt16,\
                 sf.McCourt17, sf.McCourt27, sf.McCourt28, sf.Plateau, sf.Problem04, sf.Problem06, sf.Problem13, sf.Problem15, sf.Problem18, sf.Problem20, sf.Sargan, sf.Schwefel01, sf.Schwefel06,\
                 sf.Schwefel20, sf.Schwefel36, sf.Sphere, sf.Step, sf.SumPowers, sf.Ursem04, sf.Quadratic]
    funcs = []
    for dim in range(max_dim):
        funcs += [function_of_dimension(func_names, dim+1)]
    return funcs

def get_gaussian_functions(num, max_dim, num_peaks, safe_limit=0.2, non_centrality=0.0):
    func_list = []
    num_per_dim = int(num/max_dim)
    return [get_gaussian_functions_of_dim(num_per_dim, i+1, num_peaks, safe_limit=safe_limit, non_centrality=non_centrality) for i in range(max_dim)]

def get_gaussian_functions_of_dim(num, dim=1, num_peaks=1,safe_limit=0.2, non_centrality=0.0):
    return [Gaussian(dim = dim, num_peaks=num_peaks, seed=i, safe_limit=safe_limit, non_centrality=non_centrality) for i in range(num)]

def get_nn_functions(num_functions):
    return [[NN(i)] for i in range(num_functions)]

def noisify_functions(func_list, noise_level):
    if isinstance(func_list, list):
        return [noisify_functions(func, np.sqrt(noise_level)) for func in func_list]
    else:
        return sf.Noisifier(func_list, 'add', noise_level)
    
def normalize_functions(func_list):
    if isinstance(func_list, list):
        return [normalize_functions(func) for func in func_list]
    else:
        return Normalizer(func_list)