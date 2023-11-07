import math
#from numpy.linalg import eig
#from scipy.linalg import eigvalsh
#from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.manifold import spectral_embedding
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.tangentspace import transport,log_map_riemann
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

#objective = compute_centroid(manifold,data)
# reference point 
Cref = mean_riemann(cov_train) # cov_mean = mean_covariance(cov_train, metric='riemann') 
 # dont worry about code till here. Tharun sir will help you. its a basic processing of EEG data.
import copy
from copy import deepcopy
samples = deepcopy(cov_train)
# design the cost function 
def expKer(x,samples):
    #init vars
    numSamples =  320 #X.shape[0]
    k=np.zeros(numSamples)
    #calculate estimate of expectation value of kernel
    for i in range(numSamples):
        k[i] = np.exp(-1*distance_riemann(x,X[i,:],squared=True))  # riemannian distance eigenvalue based  # exponential gaussian kernel  # so squared  
    exp_est = sum(k)/numSamples;
    return exp_est

# sumKer is to check samples that have already been collected are not repeating so its is difference between supersamples and newly supersamples 
def sumKer(x,xss,numSSsoFar):
    #--calcuates the sum of k(x,x_ss) for the number of super samps so far
    #init vars
    total=0;
    k=np.zeros(numSSsoFar)
    #calculate sof of kernels
    for i in range(numSSsoFar):
        k[i] = np.exp(-1*distance_riemann(x,xss,squared=True))  # riemannian distance eigenvalue based
    total = np.sum(k)
    s = total/(numSSsoFar+1)
    return s
  
from geoopt.optim import RiemannianAdam #to do grad descent to find argmax
from geoopt.manifolds import Euclidean, ProductManifold, SymmetricPositiveDefinite
import autograd.numpy as anp
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
from pyriemann.utils.test import is_sym, is_sym_pos_def

D = 12; K = 320
# (1) Instantiate the manifold
manifold = ProductManifold([SymmetricPositiveDefinite(D + 1, k=K), Euclidean(K - 1)])  # I have doubt here should i choose product manifold or single SPD manifold 
manifold = SymmetricPositiveDefinite(D + 1, k=K)
'''
problem = pymanopt.Problem(manifold, cost)
optimizer = pymanopt.optimizers.NelderMead() #RiemannianAdam()
result = optimizer.run(problem).point
print(result)
'''
def herd(samples,totalSS,gamma):
    #-- calculate totalSS super samples from the distribution estimated by samples with kernel hyperparam gamma
    
    #init vars and extract useful info from inputs
    #get GMM dims and num samples
    numDim = samples.shape[1]
    numSamples = samples.shape[0]
    
    #init vars
    gradientFail = 0; #count when optimization fails, debugging
    xss = np.zeros((totalSS,numDim)) #open space in mem for array of super samples
    i=1
    #gradient descent can have some probems, so make bounds to terminate if goes too far away
    minBound = np.min(samples)
    maxBound = np.max(samples)
    #start our search at the origin, could be a random point
    bestSeed = np.zeros(numDim)
    
    while i<totalSS-1:
      
        #build function for  riemannian gradient descent to find best point
        f = lambda x: -expKer(point,samples)+sumKer(point,xss,i)  # cost function
        problem = pymanopt.Problem(manifold, f)
        optimizer = RiemannianAdam(params=[samples], lr=1e-2, stabilize=1)
        # let Pymanopt do the rest
        optimizer.step()
        results = optimizer.run(problem).point()  
        #pick next best start point to start minimization, this is how Chen, Welling, Smola do it
        #find best super sample that maximizes argmax and use that as a seed for the next search
        #init or clear seed array
        seed=np.array([])
        for j in range(i):
            seed = np.append(seed,-expKer(xss[j,:],samples)+sumKer(xss[j,:],xss,i))
        bestSeedIdx = np.argmin(seed)
        bestSeed=xss[bestSeedIdx,:]
        
        #grad descent succeeded (yay!), so assign new value to super samples
        xss[i,:]=results.x
        i=i+1
       
    return xss

totalSS=70
xss = herd(samples[:80],totalSS,gamma=1)

plt.plot(samples[:,0], samples[:,1], '.')
plt.plot(xss[:,0],xss[:,1],'o')



 
