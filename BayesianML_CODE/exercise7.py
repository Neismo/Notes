import jax.numpy as jnp
from numpy import ndenumerate
import pylab as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable


def log_npdf(x, m, v):
    return -0.5*(x-m)**2/v - 0.5*jnp.log(2*jnp.pi*v)


def add_colorbar(im, fig, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
        






def PCA_dim_reduction(Xtrain, Xtest, num_components):

    
    N = len(Xtrain)
    
    # Center data
    Xm = Xtrain.mean(0)
    Xc_train = Xtrain - Xm
    Xc_test = Xtest - Xm

    # reduce dimensionality using principal component analysis (PCA) via SVD
    U, s, V = jnp.linalg.svd(Xc_train)

    # get eigenvectors corresponding to the two largest eigenvalues
    eigen_vecs = V[:num_components, :]
    eigen_vals = s[:num_components]

    # set-up projection matrix
    Pmat = eigen_vecs.T*(jnp.sqrt(N)/eigen_vals)

    # project and standize
    Ztrain = Xc_train@Pmat
    Ztest = Xc_test@Pmat 
    return Ztrain, Ztest

def visualize_utility(ax, U, labels=None):
    
    num_classes = len(U)
    
    ax.imshow(U, cmap=plt.cm.Greys_r, alpha=0.5)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_xticks(jnp.arange(num_classes))
    ax.set_yticks(jnp.arange(num_classes))
    
    if labels:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    
    ax.grid(False)
    
    for (j,i), val in ndenumerate(U):
        ax.text(i,j, val, ha='center', va='center', fontsize=16)
    ax.set_title('Utility matrix', fontweight='bold')
