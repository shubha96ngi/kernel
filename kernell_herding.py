import math
#from numpy.linalg import eig
#from scipy.linalg import eigvalsh
#from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.manifold import spectral_embedding
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_covariance, mean_euclid,mean_riemann
#from pyriemann.utils.base import sqrtm
#from pyriemann.utils.kernel import kernel_logeuclid, kernel,kernel_riemann
from pyriemann.embedding import SpectralEmbedding, _check_dimensions, barycenter_weights
#from pyriemann.utils.geodesic import geodesic_riemann,geodesic_logeuclid,geodesic_euclid
from pyriemann.utils.distance import distance, distance_riemann, distance_logeuclid, pairwise_distance
#from pyriemann.utils.test import is_sym, _get_eigenvals
from pymanopt.manifolds.manifold import Manifold
from pymanopt.manifolds import PSDFixedRank,SymmetricPositiveDefinite
#from pymanopt.optimizers.steepest_descent import SteepestDescent
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

#covmat = Covariances(estimator='lwf').fit(training_data, training_labels)
covmat = Covariances(estimator='lwf').fit(data, labels)
#cov_train = covmat.transform(training_data)
cov_train = covmat.transform(data)
select_train = TangentSpace().fit_transform(cov_train)

# weight for covariance matrix
data = cov_train[:80]
# lets try some random samples 
'''
p = 0.5
from scipy.stats import bernoulli
sample = bernoulli.rvs(p=p, size = 100)
num_trials = [1, 10, 15,  25, 50, 75, 100]
points = [sample[0:l].mean() for l in num_trials]
'''
manifold = SymmetricPositiveDefinite(12)
objective = compute_centroid(manifold,data)
#pairwise_dists = pairwise_distance(np.array([samples[5]]), np.array([samples[6]]), metric='riemann')
pairwise_dists = pairwise_distance(data, data)
ind = np.array([np.argsort(dist)[0:5 + 1] for dist in pairwise_dists])
w = (barycenter_weights(data,data,indices=ind,metric='riemann'))
# lets try some random weights
'''
from scipy.stats import bernoulli, binom
n = 10
p = 0.5
x = np.arange(0, n)
w = binom.pmf(k=x,n=n, p=p)
'''
np.sum(w) #it should be 1

#cov_mean = mean_covariance(cov_train, metric='riemann')
# generate gaussian sample from covariance matrix data
mu = []; md = []
for i in range(800):
    m = np.mean(cov_train[i], axis=0) #mean_covariance(cov_train[0])
   # m = mean_riemann(np.array([cov_train[i]]))
    d = np.dot(m, cov_train[i])
    md.append(d)
    mu.append(m)
mu = md 
Sigma = cov_train[:10]
components = np.random.choice(K, size=N, p=pi)
samples = np.zeros((N, D))
# For each component, generate all needed samples
for k in range(K):
    # indices of current component in X
    indices = k == components
    # number of those occurrences
    n_k = indices.sum()
    #print(n_k)
    if n_k > 0:
        samples[indices, :] = np.random.multivariate_normal(mu[k], Sigma[k], n_k)
        print(n_k)
        
colors = ["r", "g", "b", "c", "m"]
for k in range(10):
    indices = k == components
    print('indices=', list(indices).count(True))
    plt.scatter(
        samples[indices, 0],
         samples[indices, 1],
        alpha=0.4)
plt.axis("equal")
plt.show()




