import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import seaborn as snb
from dataclasses import dataclass


from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.optimize import minimize
from jax import value_and_grad
from jax import grad
from jax.scipy.stats import norm
from jax import hessian
# from jax.misc.optimizers import adam
from jax.flatten_util import ravel_pytree

def log_npdf(x, m, v):
    return -0.5*(x-m)**2/v - 0.5*jnp.log(2*jnp.pi*v)




#######################################################################################
# Neural network model with MAP inference
# Adapted from: # https://github.com/HIPS/autograd/blob/master/examples/neural_net.py
########################################################################################

class NeuralNetworkMAP(object):

    def __init__(self, X, y, layer_sizes, likelihood, alpha=1., step_size=0.01, max_itt=1000, seed=0):

        # data
        self.X = X
        self.y = y

        # model and optimization parameters
        self.likelihood = likelihood(y)
        self.layer_sizes = layer_sizes
        self.step_size = step_size
        self.max_itt = max_itt
        self.alpha = alpha

        # random number genration
        self.seed = seed
        self.key = random.PRNGKey(self.seed)
        
        # initialize parameters and optimize
        self.params = self.init_random_params()
        self.optimize_adam()


    def init_random_params(self):
        """Build a list of (weights, biases) tuples,
        one for each layer in the net."""
        parameters = []
        for m, n in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.key, subkey = random.split(self.key)
            w_key, b_key = random.split(subkey, 2)
            weight_matrix =jnp.sqrt(2/n) * random.normal(w_key, shape=(m, n))
            bias_vector = jnp.sqrt(2/n) * random.normal(b_key, shape=(n))
            parameters.append((weight_matrix, bias_vector))

        return parameters

    def neural_net_predict(self, params, inputs):
        """Implements a deep neural network for classification.
        params is a list of (weights, bias) tuples.
        inputs is an (N x D) matrix.
        returns logits."""
        for W, b in params:
            outputs = jnp.dot(inputs, W) + b
            inputs = jnp.tanh(outputs)
        return outputs# - logsumexp(outputs, axis=1, keepdims=True)

    def predict(self, inputs):
        return self.neural_net_predict(self.params, inputs)

    def log_prior(self, params):
        # implement a Gaussian prior on the weights
        flattened_params, _ = ravel_pytree(params)
        return  jnp.sum(log_npdf(flattened_params, 0., 1/self.alpha))
    
    def log_likelihood(self, params):     
        y = self.neural_net_predict(params, self.X)
        return self.likelihood.log_lik(y.ravel())

    def log_posterior(self, params):
        return self.log_prior(params) + self.log_likelihood(params)

    def optimize_adam(self, b1=0.9, b2=0.999, eps=1e-8):
    
        # Define training objective and gradient of objective using autograd.
        def objective(params, iter):
            return -self.log_posterior(params)
            
        objective_grad = grad(objective)
        params_flat, unflatten = ravel_pytree(self.params)
        params = self.params

        m = jnp.zeros(len(params_flat))
        v = jnp.zeros(len(params_flat))

        for itt in range(self.max_itt):

            # compute gradient and flatten
            g = objective_grad(params, itt)
            g, _ = ravel_pytree(g)

            # ADAM update rules
            m = (1 - b1) * g + b1 * m  # First  moment estimate.
            v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
            mhat = m / (1 - b1 ** (itt + 1))  # Bias correction.
            vhat = v / (1 - b2 ** (itt + 1))
            params_flat = params_flat - self.step_size * mhat / (jnp.sqrt(vhat) + eps)
            params = unflatten(params_flat)

        self.params = params

        return self

#######################################################################################
# Helper function for sampling multivariate Gaussians
########################################################################################


def generate_samples(key, mean, K, M, jitter=1e-8):
    """ returns M samples from a zero-mean Gaussian process with kernel matrix K
    
    arguments:
    K      -- NxN kernel matrix
    M      -- number of samples (scalar)
    jitter -- scalar
    returns NxM matrix
    """
    
    L = jnp.linalg.cholesky(K + jitter*jnp.identity(len(K)))
    zs = random.normal(key, shape=(len(K), M))
    fs = mean + jnp.dot(L, zs)
    return fs


#######################################################################################
# Kernels
########################################################################################

@dataclass
class Hyperparameters(object):
    kappa:          float = 1.0 # magnitude, positive scalar (default=1.0)
    lengthscale:    float = 1.0 # characteristic lengthscale, positive scalar (default=1.0)
    sigma:          float = 1.0 # noise std. dev., positive scalar (default=1.0)

    def to_array(self):
        """ return hyperparameters as flat JaX-array (to be used later) """
        return jnp.array([self.kappa, self.lengthscale, self.sigma])
    
    @staticmethod
    def from_array(hyper_array):
        """ creates Hyperparameter object from flat JaX-array (or list) of hyperparameters (to be used later) """
        kappa, lengthscale, sigma = hyper_array
        return Hyperparameters(kappa, lengthscale, sigma)
    
    def __repr__(self):
        """ for reporting hyperparameter values """
        return f'Hyperparameters(kappa={self.kappa:3.2f}, lengthscale={self.lengthscale:3.2f}, sigma={self.sigma:3.2f})'


# in the code below tau represents the distance between to input points, i.e. tau = ||x_n - x_m||.
def squared_exponential(tau, hyperparameters):
    return hyperparameters.kappa**2*jnp.exp(-0.5*tau**2/hyperparameters.lengthscale**2)

def matern12(tau, hyperparameters):
    return hyperparameters.kappa**2*jnp.exp(-tau/hyperparameters.lengthscale)

def matern32(tau, hyperparameters):
    return hyperparameters.kappa**2*(1 + jnp.sqrt(3)*tau/hyperparameters.lengthscale)*jnp.exp(-jnp.sqrt(3)*tau/hyperparameters.lengthscale)

class StationaryIsotropicKernel(object):

    def __init__(self, kernel_fun):
        """
            the argument kernel_fun must be a function of two arguments kernel_fun(||tau||, hyperparameters), e.g. 
            squared_exponential = lambda tau, hyper: hyper.kappa**2*np.exp(-0.5*tau**2/hyper.lengthscale**2).
        """
        self.kernel_fun = kernel_fun

    def construct_kernel(self, X1, X2, hyperparameters, jitter=1e-8):
        """ compute and returns the NxM kernel matrix between the two sets of input X1 (shape NxD) and X2 (MxD) using the stationary and isotropic covariance function specified by self.kernel_fun
    
        arguments:
            X1              -- NxD matrix
            X2              -- MxD matrix or None
            hyperparameters -- Hyperparameter object compatible with self.kernel_fun
            jitter          -- non-negative scalar
        
        returns
            K               -- NxM matrix    
        """

        # extract dimensions 
        N, M = X1.shape[0], X2.shape[0]

        ##############################################
        # Your solution goes here
        ##############################################
        # <solution_block>
        # compute all the pairwise distances efficiently (can also be done using nested for loops)
        dists = jnp.sqrt(jnp.sum((jnp.expand_dims(X1, 1) - jnp.expand_dims(X2, 0))**2, axis=-1))
        
        # squared exponential covariance function
        K = self.kernel_fun(dists, hyperparameters)
        
        # add jitter to diagonal for numerical stability
        if len(X1) == len(X2) and jnp.allclose(X1, X2):
            K = K + jitter*jnp.identity(len(X1))
        # <eos_block>
        ##############################################
        # End of solution
        ##############################################
        
        assert K.shape == (N, M), f"The shape of K appears wrong. Expected shape ({N}, {M}), but the actual shape was {K.shape}. Please check your code. "
        return K




#######################################################################################
# For plotting
########################################################################################

def plot_with_uncertainty(ax, Xp, mu, Sigma, sigma=0, color='g', color_samples='g', title="", num_samples=0, seed=0):
    
    mean, std = mu.ravel(), jnp.sqrt(jnp.diag(Sigma) + sigma**2)

    
    # plot distribution
    ax.plot(Xp, mean, color=color, label='GP')
    ax.plot(Xp, mean + 2*std, color=color, linestyle='--')
    ax.plot(Xp, mean - 2*std, color=color, linestyle='--')
    ax.fill_between(Xp.ravel(), mean - 2*std, mean + 2*std, color=color, alpha=0.25, label='95% interval')
    
    # generate samples
    if num_samples > 0:
        key = random.PRNGKey(seed)
        fs = generate_samples(key, mu[:, None], Sigma, num_samples, 1e-6)
        ax.plot(Xp, fs, color=color_samples, alpha=.25)
    
    ax.set_title(title)

    if num_samples > 0:
        return fs
    

def add_colorbar(im, fig, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
        
def eval_density_grid(density_fun, P=100, a=-5, b=5):
    x_grid = jnp.linspace(a, b, P)
    X1, X2 = jnp.meshgrid(x_grid, x_grid)
    XX = jnp.column_stack((X1.ravel(), X2.ravel()))
    return x_grid, density_fun(XX).reshape((P, P))


#######################################################################################
# compute classification error and std. error of the mean
########################################################################################


def compute_err(t, tpred):
    return jnp.mean(tpred.ravel() != t), jnp.std(tpred.ravel() != t)/jnp.sqrt(len(t))

#######################################################################################
# load subset of mnist data
########################################################################################

def load_MNIST_subset(filename, digits=[4,7], plot=True, subset=300, seed=0):

    data = jnp.load(filename)
    images = data['images']
    labels = data['labels']

    # we will only focus on binary classification using two digits
    idx = jnp.logical_or(labels == digits[0], labels == digits[1])
    
    # extract digits of interest
    X = images[idx, :]
    t = labels[idx].astype('float')
    
    # set labels to 0/1/2/...
    for i in range(len(digits)):
        t[t == digits[i]] = i
        
    # split into training/test
    N = len(X)
    Ntrain = int(0.5*N)
    Ntest = N - Ntrain
    key = random.PRNGKey(seed)
    train_idx = random.choice(key, jnp.arange(N), shape=(Ntrain,), replace=False)
    test_idx = jnp.setdiff1d(jnp.arange(N), train_idx)

    Xtrain = X[train_idx, :]
    Xtest = X[test_idx, :]
    ttrain = t[train_idx]
    ttest = t[test_idx]

    # standardize training set
    Xm = Xtrain.mean(0)
    Xs = Xtrain.std(0)
    Xs[Xs == 0] = 1 # avoid division by zero for "always black" pixels

    Xtrain_std = (Xtrain - Xm)/Xs
    Xtest_std = (Xtest - Xm)/Xs


    # reduce dimensionality to 2D using principal component analysis (PCA)
    U, s, V = jnp.linalg.svd(Xtrain_std)

    # get eigenvectors corresponding to the two largest eigenvalues
    eigen_vecs = V[:2, :]
    eigen_vals = s[:2]

    # set-up projection matrix
    Pmat = eigen_vecs.T*(jnp.sqrt(Ntrain)/eigen_vals)

    # project and standize
    Phi_train =Xtrain_std@Pmat
    Phi_test = Xtest_std@Pmat

    # Let's only use a small subset of the training data
    Phi_train = Phi_train[:subset]
    ttrain = ttrain[:subset]

    return Phi_train, Phi_test, ttrain, ttest,
