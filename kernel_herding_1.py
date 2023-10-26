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


################   abinesh #############################
############# kernel function ################
# exponential kernel  versus geodisic exponential kernel??
# find the difference 
# kernel should be a square matrix 
# I am not sure it should be 12x12 or 320x320 check with sir where 320 is nearest neighbors 

# what we have to do here is take the centroid means Cref 
# and we have to update its location and this costfuction which is our exponetial kernel and minimise it 
# so problem is how do we update the position of Cref???
# expectation maximisation algorithm, proababilistic clustering and unsupervised learning 
#https://towardsdatascience.com/unsupervised-learning-and-data-clustering-eeecb78b422a

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
# should include a loss function and data line 214

# common tangent space of Cref
# parallel transport map 
T = transport(cov_train, Cref, metric='riemann') # Parallel transport of a set of SPD covariance matrices towards a reference matrix.
# this is for any reference point 
#why T and tmap are not equal ?
map_train = TangentSpace().fit_transform(T)  # changed tangent vector 
tangent_Cref = TangentSpace().fit_transform(Cref)
tmap = log_map_riemann(cov_train, Cref) # Project matrices in tangent space by Riemannian logarithmic map.
# this is used with respect to mean_reference matrix 

# but we have to transport tangent vectors not the matrices  
# now map the gradients 


def retraction(X,TM):
  p_inv_tv = np.linalg.solve(X, TM)
  return multiherm(
            X + TM + TM @ p_inv_tv / 2
        )

# applying riemannian gradient descent 
#compute loss function, then take a small step in the direction that will minimize loss.
# for euclidean space 
#W = W - alpha * gradient(x) and 
# for the non-euclidean/ riemannian space 

#loss: A function used to compute the loss over our current parameters W and input data.
#data: what is our training data here ?? # is it unsupervised in cased of ODEs?? Our training data where each training sample is represented by an image (or feature vector).
#W: Our actual weight matrix that we are optimizing over. Our goal is to apply gradient descent to find a W that yields minimal loss.
# activation function used here sigmoid or relu 
# but why do we calculate gradient of activation function 
#the gradient of a sigmoid activation function is used to update the weights & biases of a neural network. 
#If these gradients are tiny, the updates to the weights & biases are tiny and the network will not learn.
# To overcome this we use ReLU 
# so manually check this gradient of activation function 
preds = sigmoid_activation(trainX.dot(W))
error = preds - trainY  # how to calculate error in case of unsupervised learning 
loss = np.sum(error ** 2)
losses.append(loss)
d = error * sigmoid_deriv(preds)
Wgradient = trainX.T.dot(d)
Wgradient = evaluate_gradient(loss, data, W)  # retraction(-alpha*gradient(x))
W += -alpha * Wgradient # where alpha is our learning rate 

#The evaluate_gradient function returns a vector that is K-dimensional, where K is the number of dimensions in our image/feature vector.
#The Wgradient variable is the actual gradient, where we have a gradient entry for each dimension.



# if we take loss function from pymanopt 
#https://github.com/pymanopt/pymanopt/blob/master/examples/notebooks/mixture_of_gaussians.ipynb
def loss(S, v):
    # Unpack parameters
    nu = np.append(v, 0)  # random weights 
    # why we use slogdet 
    # same as in distance_kullback
    #https://github.com/pyRiemann/pyRiemann/blob/9ab58edf009bbcbdace83cadce9459d174a746af/pyriemann/utils/distance.py#L142
    logdetS = np.expand_dims(np.linalg.slogdet(S)[1], 1)  # take logdeterminent of S # something to do with eigenvalue and trace 
    y = np.concatenate([samples.T, np.ones((1, N))], axis=0) # is it adding bias 

    # Calculate log_q   # 'Probability' of y belonging to each cluster
    y = np.expand_dims(y, 0) 
    # y * np.linalg.solve(S, y) # this gives expected y as from tuorial https://aleksandarhaber.com/solve-optimization-problems-in-python-by-using-scipy-library/
    #then why sum ?
    log_q = -0.5 * (np.sum(y * np.linalg.solve(S, y), axis=1) + logdetS)

    alpha = np.exp(nu)
    alpha = alpha / np.sum(alpha)  # normalise the weights 
    alpha = np.expand_dims(alpha, 1)
    # three cvariance matrix so make it three or they just added one for the bias ?

    loglikvec = logsumexp(np.log(alpha) + log_q, axis=0)  # cross entropy loss of softmax function 
    #https://blog.feedly.com/tricks-of-the-trade-logsumexp/
    return -np.sum(loglikvec)


def gradient(x):
#I will work on it next week. Currently busy with office works. till then you do check these 176-192 line steps 


# cost function J
crit = np.linalg.norm(J, ord='fro')
if crit <= tol:
    break

#important site 
#https://agustinus.kristia.de/techblog/2019/02/22/optimization-riemannian-manifolds/

# riemannian gaussian distribution 
from pyriemann.datasets.sampling import sample_gaussian_spd




