import GPy
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from get_factorial import get_factorial
from acquisitions import EI, LCB, PI
import copy
import time
from pylab import *

#Dirty hack to make GPy work for us (we want N to be 1, not [[1]]) (bug in GPy!)
def logpdf_link(self, inv_link_f, y, Y_metadata=None):
    N = np.ones(y.shape) if Y_metadata is None else Y_metadata.get('trials', np.ones(y.shape))
    np.testing.assert_array_equal(N.shape, y.shape)

    nchoosey = special.gammaln(N+1) - special.gammaln(y+1) - special.gammaln(N-y+1)
    Ny = N-y
    t1 = np.zeros(y.shape)
    t2 = np.zeros(y.shape)
    t1[y>0] = y[y>0]*np.log(inv_link_f[y>0])
    t2[Ny>0] = Ny[Ny>0]*np.log(1.-inv_link_f[Ny>0])
    
    return nchoosey + t1 + t2
GPy.likelihoods.Binomial.logpdf_link = logpdf_link #This is the dirty part


class BlackBoxFunction:
    '''
    All functions to be optimized with BayesianOptimization class
    or DerivativeBayesianOptimization class must inherit this class
    '''
    def __init__(self):
        pass

    def get_dim(self):
        '''
        Should return the size of the input space
        '''
        raise NotImplementedError

    def do_evaluate(self, x):
        '''
        returns the possibly stochastic evaluation for given x
        '''
        raise NotImplementedError

    def do_evaluate_clean(self, x):
        '''
        If possible, returns the noiseless evaluation for given x
        '''
        return None

def get_model_kernel(dim):
    '''
    Returns RBF kernel with very naive priors
    Don't use this function for your own applications
    '''
    ker_sexp = GPy.kern.RBF(input_dim=dim, variance=0.15, lengthscale=0.15, ARD=False)
    prior = GPy.priors.Gaussian(0.5,0.5)
    ker_sexp.lengthscale.unconstrain()
    ker_sexp.lengthscale.set_prior(prior)
    ker_sexp.lengthscale.constrain_positive()
    prior = GPy.priors.Gaussian(0.2,0.5)
    ker_sexp.variance.unconstrain()
    ker_sexp.variance.set_prior(prior)
    ker_sexp.variance.constrain_positive()
    return ker_sexp

def get_model_likelihood(noise = 0.0):
    '''
    Returns gaussian likelihood with fixed noise.
    '''
    lik = GPy.likelihoods.Gaussian(variance=noise)
    if noise < 0.00001:
        lik.variance.constrain_fixed(value=1e-6,warning=True,trigger_parent=True)
    else:
        lik.variance.constrain_fixed(value=noise,warning=True,trigger_parent=True)
    return lik

def initial_x(dim, num_factorials=1):
    '''
    Returns locations of initial X using full factorial design and assuming that we are bounded between [0,1] in all dimensions
    '''
    delta = 0.5/(num_factorials+1)
    f = get_factorial(dim)
    x = np.empty((0,dim))
    for i in range(num_factorials):
        x = np.append(x, f*delta*(i+1) + 0.5, axis=0)
    return x

class BayesianOptimization(object):
    def __init__(self, func, X, acquisition_function=EI, kernel_function=get_model_kernel, likelihood_function=get_model_likelihood, bounds=None, max_iter=25, num_acquisition_starts=20, print_progress=0, min_point_point_dist=0.005):
        '''
        Performs regular Bayesian Optimization
        parameters:
        func: A member of a class that inherits BlackBoxFunction-class: function that is optimized
        X: Locations of initial points that are evaluated before BO starts
        acquisition_function: Acquisition function used to select next points, see acquisitions.py for options
        kernel_function: Function that returns covariance function of GP given the size of the input space
        likelihood_function: Function that returns likelihood function of GP
        bounds: Optimization bounds defaults to [0,1] in all dimensions. Array of size (d,2)
        max_iter: Number of maximum iterations
        num_acquisition_starts: number of starting points for optimization of acquisition function
        print_progress: 0=print nothing, 1=print iteration and iteration result, 2=print details of iterations, 3=print also how much time different parts of optimization took
        min_point_point_dist: Minimum distance between two trainign samples
        '''
        assert(isinstance(X, np.ndarray))
        self.func = func
        self.dim = self.func.get_dim()
        assert(X.shape[1] == self.dim)
        self.X = X
        self.Y = np.array([self.func.do_evaluate(self.X[i,:]) for i in range(self.X.shape[0])]).reshape((-1,1))
        self.acq = acquisition_function
        self.kernel_function=kernel_function
        self.likelihood_function=likelihood_function
        self.max_iter = max_iter
        self.num_acquisition_starts = num_acquisition_starts
        self.min_point_point_dist = min_point_point_dist
        self.print_progress=print_progress
        if bounds is None:
            bounds = np.array([[0,1] for i in range(self.dim)])
        self.bounds = bounds
        self.f_min = []
        self.f_min_ref = []
        self.x_min = []
   
    def optimize(self):
        '''
        Optimizes the black box funxction
        '''
        self._update_model()
        for i in range(self.max_iter):
            if(self.print_progress > 0):
                    print("Iteration {}".format(i))
            n = self._get_size()
            j=0
            while self._get_size() == n:
                start = time.time()
                x_new = self._maximize_acquisition()
                end = time.time()
                if(self.print_progress > 2): print("Maximizing acq. took: {}".format(str(end-start)))
                start = time.time()
                j = j+1
                if(j>10):
                    if(self.print_progress > 1): print("Added more than 10 derivative sign observations, forcing regular observation")
                    self._add_point(x_new, force=True)
                else:
                    self._add_point(x_new, force=False)
                end = time.time()
                if(self.print_progress > 2): print("Adding point to the model took: {}".format(str(end-start)))
            if(self.print_progress > 2): self.print_model()
            self._collect_metrics()
            if(self.print_progress > 0): 
                print("best so far: {} ".format(self.f_min[-1]), end="") 
                if self.f_min_ref[-1] is not None:
                    print("(true function value at this location: {})".format(self.f_min_ref[-1]), end="")
                print("")
        return self.X, self.Y

    def _get_size(self):
        '''
        Returns the number of training data points
        '''
        return self.X.shape[0]
    
    def _update_model(self):
        '''
        Updates the GP with thecurrent training data 
        '''
        kern = self.kernel_function(self.dim)
        lik = self.likelihood_function()
        self.model = GPy.core.GP(X = self.X, Y = self.Y, kernel=kern, likelihood=lik)
        self.model.optimize() 
    
    def _maximize_acquisition(self):
        '''
        Maximizes acquisition function with L-BFGS-B method using num_points starting points
        '''
        x_best = None
        preds, _ = self.model.predict(self.X)
        acq_n = lambda x: self.acq(np.array([x]), fmin = min(preds), model = self.model, n=self.X.shape[0], d=self.dim)
        best = np.inf
        for i in range(self.num_acquisition_starts):
            x = np.random.rand(1, self.dim)
            opt = minimize(acq_n, x, method='L-BFGS-B', bounds = tuple((self.bounds[i,0], self.bounds[i,1]) for i in range(self.dim) ), jac=True, tol=1e-50)
            temp,_ = acq_n(opt.x)
            if temp < best:
                x_best = opt.x
                best = temp
        return np.array([x_best])
    
    def _deviate_point(self, x_new):
        '''
        Deviates point x_new to avoid points that are too close to each other
        '''
        X = self.X
        if isinstance(X, list):
            X = np.vstack(X)
        if isinstance(x_new, list):
            x_new = np.vstack(x_new)
        for i in range(X.shape[0]):
            dist = norm(X[i,:]-x_new)
            if dist < 1e-6: #To avoid devision by zero
                if x_new[0,0] < self.min_point_point_dist:
                    x_new[0,0] += self.min_point_point_dist
                else:
                    x_new[0,0] -= self.min_point_point_dist
                if(self.print_progress > 0): print("There already was a point at that location: deviating")
            elif dist < self.min_point_point_dist:
                x_new += (X[i,:]-x_new)/dist*self.min_point_point_dist
                if(self.print_progress > 0): print("There already was a point near that location: deviating")
        return x_new
    
    def _collect_metrics(self):
        '''
        Collects the current state of the algorithm to vectors
        '''
        preds, _ = self.model.predict(self.X)
        ind = np.argmin(preds)
        self.f_min = self.f_min + [preds[ind,0]]
        self.f_min_ref = self.f_min_ref + [self.func.do_evaluate_clean(self.X[ind,:])]
        self.x_min = self.x_min + [ind]
        
    def _add_point(self, x_new, force=False):
        '''
        adds x_new to the model
        '''
        x_new = self._deviate_point(x_new, print_progress=self.print_progress)
        self.X = np.append(self.X, x_new, axis=0)
        self.Y = np.append(self.Y, np.array([[self.func.do_evaluate(x_new)]]), axis=0)
        self._update_model()
        if(self.print_progress > 1): print("Added the input point: {}".format(x_new))
 
    def print_model(self):
        '''
        Prints GP model (see details in GPy documentation)
        '''
        print(self.model)
    
    def plot_model(self, name):
        '''
        Plots true function and all training points if the problem is 1 or 2 dimensional
        '''
        preds, _ = self.model.predict(self.X)
        acq_n = lambda x: self.acq(x, fmin = min(preds), model = self.model, n=self.X.shape[0], d=self.dim)
        if self.dim==2:
            self._plot_model_base_2d(name, self.func.do_evaluate, self.model.predict, self.X, extra=None)
        elif self.dim==1:
            self._plot_model_base_1d(name, self.func.do_evaluate, self.model.predict, self.X, extra=None)
        
    def _plot_model_base_2d(self,save_name, func, model, acquisitions, extra=None ):
        '''
        saves an image of the BO iterations under the save_name. The plot includes contours of func, points in acquisitions as dots and points in extra as crosses
        '''
        v = np.linspace(0,1,20)
        xv0, xv1 = np.meshgrid(v, v, sparse=False, indexing='ij')

        xt = np.array([xv0.reshape((-1)), xv1.reshape((-1))]).T
        yt = np.array([func(xt[i,:]) for i in range(xt.shape[0])])
        
        f, (ax) = plt.subplots(1, 1, sharex=True, sharey=True)
        
        CS = ax.contourf(xv0, xv1, yt.reshape((20,20)), title='real function',cmap=plt.cm.get_cmap('binary_r', 10))
        
        ax.scatter(acquisitions[:,0], acquisitions[:,1], marker='.')
        
        
        if extra is not None:
            ax.scatter(extra[:,0], extra[:,1],  marker='+')
        
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(save_name, bbox_inches='tight', pad_inches=0)

    def _plot_with_uncertainty(self, x, y, color='r', linestyle='-', fill=True, label='', axis=None):
        '''
        Plots GP with uncertainty
        '''
        if axis is None:
            f, axis = plt.subplots()
        main, = axis.plot(x, y[1,:], color=color, linestyle=linestyle, label=label)
    
        lower, upper = y[0,:], y[2,:]
        axis.plot(x, lower, color=color, alpha=0.15, linestyle=linestyle)
        axis.plot(x, upper, color=color, alpha=0.15, linestyle=linestyle)

        if fill:
            axis.fill_between(x.ravel(), lower.ravel(), upper.ravel(), color=color, alpha=0.05, linestyle=linestyle)
        return main
     
    def _plot_model_base_1d(self,save_name, func, model, acquisitions, extra=None ):
        '''
        saves an image of the BO iterations under the save_name. The plot includes contours of func, posterior fit of model, points in acquisitions as dots and points in extra as crosses
        '''
        xt = np.atleast_2d(np.linspace(0,1,100)).T

        yt = np.array([func(xt[i]) for i in range(xt.shape[0])])
        
        ya = np.array([func(acquisitions[i]) for i in range(acquisitions.shape[0])])
        
        f, (ax) = plt.subplots(1, 1, sharex=True, sharey=True)
        
        ax.plot(xt, yt)
        ym, cm = model(xt)
        self._plot_with_uncertainty(xt, np.array([ym-2*cm, ym, ym+2*cm]), color='r', axis=ax)
        
        ax.scatter(acquisitions, ya, marker = '*')
        
        
        if extra is not None:
            ye = np.array([func(extra[i]) for i in range(extra.shape[0])])
            ax.scatter(extra, ye,  marker='+')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    
class DerivativeBayesianOptimization(BayesianOptimization):
    def __init__(self, func, X, acquisition_function=EI, kernel_function=get_model_kernel, likelihood_function=get_model_likelihood, bounds=None, max_iter=25, num_acquisition_starts=20, print_progress=0, min_virtual_dist=0.01, min_point_dist=0.01, adaptive=False, min_point_point_dist=0.005):
        '''
        Performs Bayesian optimization with derivative sign observations.
        Parameters that differ from BayesianOptimization (see rest fromBayesianOptimization):
        min_virtual_dist: minimum idstance between virtual derivative observations
        min_point_dist: minimum distance between virtual derivative observation and regualr observation
        adaptive: if True, always before adding a virtual derivative sign observation it is checked if the data supports this.
        '''
        super(DerivativeBayesianOptimization, self).__init__(func, X, acquisition_function=acquisition_function, kernel_function=get_model_kernel, likelihood_function=get_model_likelihood, bounds=bounds, max_iter=max_iter, num_acquisition_starts= num_acquisition_starts, print_progress=print_progress, min_point_point_dist=min_point_point_dist)
        
        self.X = [self.X] + [np.empty((0,self.dim))]*self.dim
        self.Y = [self.Y] + [np.empty((0,1))]*self.dim

        self.derivn = []
        self.adaptive = adaptive
        self.min_virtual_dist=min_virtual_dist
        self.min_point_dist=min_point_dist

    def _collect_metrics(self):
        '''
        Appends the current state of the optimization to the vectors that track the state
        These vectors are:
        f_min: estimated minimum
        f_min_ref: true function value at the location of the estimated minimum (only used if we know the ground truth)
        x_min: index of the training data point that is the minimum
        derivn: number of derivative sign observations used
        '''
        preds, _ = self.model.predict([self.X[0]])
        ind = np.argmin(preds)
        self.f_min = self.f_min + [preds[ind,0]]
        self.f_min_ref = self.f_min_ref + [self.func.do_evaluate_clean(self.X[0][ind,:])]
        self.x_min = self.x_min + [ind]
        self.derivn = self.derivn + [np.concatenate(self.X[1:], axis=0).shape[0]]
        
    def _get_size(self):
        '''
        returns the number of non virtual observations
        '''
        return self.X[0].shape[0]

    def _add_point(self, x_new, force=False):
        '''
        Updates the model with one point. If force is true, the next point will not be a virtual derivative observation
        If force is false, the added point might be virtual derivative observation if it is close to the border
        If point is close to any other point in the model, it is deviated before it is added
        '''
        x_border, border, dist, sign = self._give_border(x_new)
        virtual_dist, dim, obs = self._give_distance_to_virtual_observation(x_new)
        data_support=True
        if self.adaptive:
            data_support = (np.absolute(self._get_point_monotonicity(x_new, border) - sign) < 0.1) #True if data supports that sign of the derivative observation should be same as proposed
        #Virtual derivative observation is added only if the point to be added is far enough from already present virtual observations,
        #close enough to the border, next virtual observation is not forced and data supports the sign of the observation
        if (virtual_dist > self.min_virtual_dist) and (dist < self.min_point_dist) and (force is False) and  data_support:
            if (dist < 0.01):
                if(self.print_progress > 1): print("Distance to border is smaller than treshold")
            x_border = self._deviate_point(x_border)
            self.X[border+1] = np.append(self.X[border+1], x_border, axis=0)
            self.Y[border+1] = np.append(self.Y[border+1], np.array([[sign]]), axis=0)
            self._update_model()
            if(self.print_progress > 0): print("Added the derivative sign observation to: {} of sign: {}".format(x_border, sign))
        else:
            if force is True:
                if(self.print_progress > 1): print("Forcing the following point")
            if (virtual_dist) < 0.01 and (self.adaptive==True):
                self.X[dim+1] = np.delete(self.X[dim+1], obs ,0)
                self.Y[dim+1] = np.delete(self.Y[dim+1], obs ,0)
                if(self.print_progress > 0): print("Deleted virtual observation that was close by")
            x_new = self._deviate_point(x_new)
            self.X[0] = np.append(self.X[0], x_new, axis=0)
            self.Y[0] = np.append(self.Y[0], np.array([[self.func.do_evaluate(x_new)]]), axis=0)
            self._update_model()        
            if(self.print_progress > 0): print("Added the input point: {}".format(x_new))

    def _give_border(self, x):
        '''
        Returns a projection of the given point to the nearest border of the optimization area
        first returned vector is the projected points, second is index of the projected dimension,
        third is the distance between the projected and original point and fourth is the sign of
        the virtual derivative observation that should be added to the projected point.
        '''
        mid = np.sum(self.bounds, axis=1)/2.
        x_new = np.copy(x)
        tmp = np.zeros(x.shape)
        tmp[x<=mid] = (x-self.bounds[:,0].T)[x<=mid]
        tmp[x>mid] = (self.bounds[:,1].T-x)[x>mid]
        ind = np.argmin(tmp)
        x_new[:,ind] = self.bounds[ind,0] if x[:,ind] <= mid[ind] else self.bounds[ind,1]
        sign = -1. if x[:,ind] <= mid[ind] else +1.
        return x_new, ind, tmp[:,ind], sign
    
    def _give_distance_to_virtual_observation(self, x):
        '''
        Returns the smallest distance to virtual derivative observation of given point (and index of that observation)
        '''
        X = self.X[1:]
        min_dist = 1.
        i,j = None, None
        for dim in range(len(X)):
            for obs in range(X[dim].shape[0]):
                dist = norm(X[dim][obs,:]-x)
                if dist < min_dist:
                    min_dist=dist
                    i,j = dim, obs
        return min_dist, i, j

    def _update_model(self):
        '''
        Updates the gp model with current training data and virtual derivative observations
        '''
        kern = self.kernel_function(self.dim) 
        kern_list = [kern] + [GPy.kern.DiffKern(kern,i) for i in range(self.dim)]
        
        lik_list = [self.likelihood_function()]
        probit = GPy.likelihoods.Binomial(gp_link = GPy.likelihoods.link_functions.ScaledProbit(nu=1000))
        lik_list += [probit for i in range(self.dim)]
        start = time.time()
        self.model = GPy.models.MultioutputGP(X_list = self.X, Y_list = self.Y, kernel_list=kern_list, likelihood_list=lik_list, inference_method=GPy.inference.latent_function_inference.EP())
        end = time.time()
        if(self.print_progress > 2): print("Creating GP took: {}".format(str(end-start)))
        start = time.time()
        self.model.optimize()
        end = time.time()
        if(self.print_progress > 2): print("Optimizing GP took: {}".format(str(end-start)))

    def _get_point_monotonicity(self, x, dim):
        '''
        Given location x, gives the direction of gradient sign that the already existing data supports the most
        '''
        x_new =  np.c_[x, np.array([[dim+1]])]
        ind=np.array([dim+1])
        tmp = {'output_index': ind, 'trials': np.ones(ind.shape)}
        y = [-1, 0, 1]
        p = [None]*3
        for i in range(len(y)):
            p[i] = self.model.log_predictive_density(x_new, np.array([[y[i]]]), Y_metadata = tmp)
        k = y[np.argmax(p)]
        if(self.print_progress > 0): print("Data says that the sign of the derivative observation should be: {} (Probabilities of signs [-1,0,1]={})".format(k, exp(p).T))
        return k

    def print_model(self):
        '''
        Prints GP model (see details in GPy documentation)
        '''
        print(self.model)
        
    def _maximize_acquisition(self):
        '''
        Maximizes acquisition function with L-BFGS-B method using num_points starting points
        '''
        x_best = None
        preds, _ = self.model.predict([self.X[0]])
        acq_n = lambda x: self.acq([np.array([x])], fmin = min(preds), model = self.model,  n=self.X[0].shape[0], d=self.dim)
        best = np.inf
        for i in range(self.num_acquisition_starts):
            x = np.random.rand(1,self.dim)
            opt = minimize(acq_n, x, method='L-BFGS-B', bounds = tuple((self.bounds[i,0], self.bounds[i,1]) for i in range(self.dim) ), jac=True, tol=1e-50 )
            temp, _ = acq_n(opt.x)
            if (temp < best):
                x_best = opt.x
                best = temp
        return np.array([x_best])
    
    def plot_model(self, name):
        '''
        Plots true function and all training points if the problem is 1 or 2 dimensional
        '''
        model = lambda x: self.model.predict([x])
        extra = np.concatenate(self.X[1:], axis=0)
        if self.dim == 2:
            self._plot_model_base_2d(name, self.func.do_evaluate, model, self.X[0], extra=extra)
        elif self.dim==1:
            self._plot_model_base_1d(name, self.func.do_evaluate, model, self.X[0], extra=extra)
        
if __name__ == "__main__":
    import optimization
    import util
    import acquisitions
    l = util.get_gaussian_functions(1,1,1)
    l_new = util.normalize_functions(l)
    bo = DerivativeBayesianOptimization(func = l_new[0][0], X=initial_x(1), acquisition_function=acquisitions.EI, max_iter=10, print_progress=1, adaptive=True)
    X,Y = bo.optimize()
    bo.plot_model("test.png")
