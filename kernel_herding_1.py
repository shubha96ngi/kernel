import math
#from numpy.linalg import eig
#from scipy.linalg import eigvalsh
#from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.manifold import spectral_embedding
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_covariance, mean_euclid,mean_riemann
#from pyriemann.utils.base import sqrtm
from pyriemann.utils.kernel import kernel_logeuclid, kernel,kernel_riemann
from pyriemann.embedding import SpectralEmbedding, _check_dimensions, barycenter_weights
#from pyriemann.utils.geodesic import geodesic_riemann,geodesic_logeuclid,geodesic_euclid
from pyriemann.utils.distance import distance, distance_riemann, distance_logeuclid, pairwise_distance
from pyriemann.utils.test import is_sym, is_pos_semi_def,(is_sym_pos_def,_get_eigenvals
from pymanopt.manifolds.manifold import Manifold
from pymanopt.manifolds import PSDFixedRank,SymmetricPositiveDefinite
from pymanopt.optimizers.steepest_descent import SteepestDescent
from pymanopt.optimizers.nelder_mead import compute_centroid 
#from conftest import get_distances, get_means, get_metrics
import fnmatch

folder = '/home/shubhangi/Downloads/EEG/FullDatasetCBCI2020/' 
#preprocessed test data folder 
#test = '/Downloads/FullDatasetCBCI2020/' 
train = fnmatch.filter(os.listdir(folder),'parsed_*T.mat')
test = fnmatch.filter(os.listdir(folder),'parsed_*E.mat')
#test[0]
# appending train data and labels and resampling 
labels=[]
data = np.empty((0,12,4096))
for i in train:
    #print('i=',i)
    temp = sio.loadmat(folder+i)
    dat = temp['RawEEGData']
    #print(np.shape(dat))
    lab = np.squeeze(temp['Labels'])
    data = np.concatenate((data, dat), axis=0)
    labels = np.concatenate((labels, lab), axis =0)
    labels = [int(x) for x in labels]
#print(labels)
# appending test data 
tlabels=[]
tdata = np.empty((0,12,4096))
for i1 in test:
    #print('i=',i1)
    temp1 = sio.loadmat(folder+i1)
    tdat = temp['RawEEGData']  #[0:40,:,:]
    #print('tdat=', np.shape(tdat)) #, tdat[45:47, :, :])
    tlab = np.squeeze(temp1['Labels'])
    #print(lab)
    tdata = np.concatenate((tdata, tdat), axis=0)
    tlabels = np.concatenate((tlabels, tlab), axis =0)
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(signal, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    n_trials, n_chans, n_time = signal.shape
    filtered = np.zeros(signal.shape)
    print(filtered.shape)
    for i in range(n_trials):
        for j in range(n_chans):
            filtered[i][j] = lfilter(b, a, signal[i][j])
    return filtered


data_train = butter_bandpass_filter(data, 8,35,4096, order=5)
data_test = butter_bandpass_filter(tdata, 8,35,4096, order=5)
print('filtered train data shape', data_train.shape)
print('filtered test data shape', data_test.shape)


covmat = Covariances(estimator='lwf').fit(data, labels)
cov_train = covmat.transform(data) # covariance matrix 
select_train = TangentSpace().fit_transform(cov_train) #tangent vector 

#manifold = SymmetricPositiveDefinite(12)
#objective = compute_centroid(manifold,data)
# reference point 
Cref = mean_riemann(cov_train) # cov_mean = mean_covariance(cov_train, metric='riemann') 
Cref = np.array([Cref])
X = cov_train
pairwise_dists = pairwise_distance(Cref,X, metric=metric)
ind = np.array([np.argsort(dist) for dist in pairwise_dists])
B = barycenter_weights(Cref,X, neighbors, metric=metric, reg=reg)
# mean of data 
M = np.mean(data,axis=2)
mean_data = [M[i] for i in range(320)]

##################Sampling from gaussian distribution #####################
'''
import autograd.numpy as np
#np.set_printoptions(precision=2)
import matplotlib.pyplot as plt
#%matplotlib inline
# Number of data points
N = 100
# Dimension of each data point
D = 12
# Number of clusters
K = 320
pi = list(B[0])
pi = [np.abs(pi[i]) for i in range(320)]
mu =  mean_data #[np.array([-4, 1]), np.array([0, 0]), np.array([2, -1])]
Sigma = [np.array(cov_train[i]) for i in range(320)]
components = np.random.choice(K, size=N, p=pi)
samples = np.zeros((N, D))
# For each component, generate all needed samples
for k in range(K):
    # indices of current component in X
    indices = k == components
    # number of those occurrences
    n_k = indices.sum()
    if n_k > 0:
        samples[indices, :] = np.random.multivariate_normal(
            mu[k], Sigma[k], n_k
        )

for k in range(K):
    indices = k == components
    plt.scatter(
        samples[indices, 0],
        samples[indices, 1],
        alpha=0.4
    )
plt.axis("equal")
plt.show()

import sys
#sys.path.insert(0, "../..")
from autograd.scipy.special import logsumexp
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Euclidean, Product, SymmetricPositiveDefinite
from pymanopt.optimizers import SteepestDescent

# (1) Instantiate the manifold
manifold = Product([SymmetricPositiveDefinite(D + 1, k=K), Euclidean(K - 1)])

# (2) Define cost function
# The parameters must be contained in a list theta.
@pymanopt.function.autograd(manifold)
def cost(S, v):
    # Unpack parameters
    nu = np.append(v, 0)
    logdetS = np.expand_dims(np.linalg.slogdet(S)[1], 1)
    y = np.concatenate([samples.T, np.ones((1, N))], axis=0)
    # Calculate log_q
    y = np.expand_dims(y, 0)
    # 'Probability' of y belonging to each cluster
    log_q = -0.5 * (np.sum(y * np.linalg.solve(S, y), axis=1) + logdetS)

    alpha = np.exp(nu)
    alpha = alpha / np.sum(alpha)
    alpha = np.expand_dims(alpha, 1)

    loglikvec = logsumexp(np.log(alpha) + log_q, axis=0)
    return -np.sum(loglikvec)
problem = Problem(manifold, cost)

# (3) Instantiate a Pymanopt optimizer
optimizer = SteepestDescent(verbosity=1)

# let Pymanopt do the rest
Xopt = optimizer.run(problem).point
'''
################   abinesh #############################
############# kernel function ################
# exponential kernel  versus geodisic exponential kernel??
# find the difference 
# kernel should be a square matrix 
# I am not sure it should be 12x12 or 320x320 check with sir where 320 is nearest neighbors 
pairwise_dist = pairwise_distance(X,X, metric=metric)  # this is different from the one used above pairwise_dists 
eps = np.median(pairwise_dist)**2 / 2
kernel = np.exp(-pairwise_dist**2 / (4 * eps)) # 320x320    # kernel = kernel_riemann(X,Cref,reg=1e-10))
 # normalize the kernel matrix
q = np.dot(kernel, np.ones(len(kernel)))
kernel_n = np.divide(kernel, np.outer(q, q))
# check if it is positive semi definite matrix  # kernel deifne RKHS 
print(is_pos_semi_def(np.array([kernel_n]))) # its true 
# also check if it is SPD 
print(is_sym_pos_def(np.array([kernel_n])) # True

# empirical  mean embedding 
mu_cap = np.sum(B*kernel)
# I am not sure check once how to construct empirical mean embedding

# optimisation over riemannian manifolds 
TM = select_train()
# cross check definition of retraction map 
#taken from https://github.com/pymanopt/pymanopt/blob/master/src/pymanopt/manifolds/positive_definite.py
def retraction(X,TM):
  p_inv_tv = np.linalg.solve(X, TM)
  return multiherm(
            X + TM + TM @ p_inv_tv / 2
        )
def gradient():
#I will work on it next week. Currently busy with office works. till then you do check these 176-192 line steps 

    



