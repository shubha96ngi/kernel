# after calculating sigma, mu and pi from previous code 
# kernel herding using gradient descent 
#https://github.com/rrphys/KernelHerding/blob/master/Herding.ipynb

def expKer(x,samples,gamma):
    #--calcuates the expectation value of the exponential kernel so argmax_x can be found
    #x = candidate super sample to optimize
    #samples = the GMM samples
    gamma =  1 #kernel hyperparameter, always 1 for my demo
    
    #init vars
    numSamples = 70 #X.shape[0]
    k=np.zeros(numSamples)
    #calculate estimate of expectation value of kernel
    for i in range(numSamples):
        # exponential gaussian kernel  # so squared  
        #k[i] = np.exp(-1*distance_euclid(x,samples[i,:],squared=True))  # euclid distance norm based
       # k[i] = np.exp(-1*distance_riemann(x,X[i,:],squared=True))  # riemannian distance eigenvalue based
        
        k[i] = np.exp(-np.linalg.norm(x-samples[i,:])/gamma**2)
    exp_est = sum(k)/numSamples;
    return exp_est

def sumKer(x,xss,numSSsoFar,gamma):
    #--calcuates the sum of k(x,x_ss) for the number of super samps so far
    #x = candidate super sample to optimize
    #samples = the GMM samples
    #numSSsoFar = number of super sampls so far
    #gamma = kernel hyperparameter, always 1 for my demo
    
    #init vars
    total=0;
    k=np.zeros(numSSsoFar)
    #calculate sof of kernels
    for i in range(numSSsoFar):
        k[i] = np.exp(-np.linalg.norm(x-xss[i,:])/gamma**2)
        #k[i] = np.exp(-1*distance_riemann(x,xss,squared=True))  # riemannian distance eigenvalue based
    total = np.sum(k)
    s = total/(numSSsoFar+1)
    return s


from scipy.optimize import minimize #to do grad descent to fing argmax
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
    
#     tick = time.clock()
    while i<totalSS-1:
        print(".")
        #debugging stuff
        #print "Working on SS num ber i=%d" % i
        #build function for gradient descent to find best point
        f = lambda x: -expKer(x,samples,gamma)+sumKer(x,xss,i,gamma)
        print(f) #.shape, f.ndim)
        results = minimize(f,
                           bestSeed,
                           method='nelder-mead',
                           options={'xtol': 1e-4, 'disp': False})
#         print "results.x"
#         print results.x
        
        #if grad descent failed, pick a random sample and try again
        if np.min(results.x) < minBound or np.max(results.x) > maxBound:
            bestSeed=samples[np.random.choice(numSamples)]
            gradientFail=gradientFail+1
#             print "Gradient descent failed.............."
            continue
        
        #pick next best start point to start minimization, this is how Chen, Welling, Smola do it
        #find best super sample that maximizes argmax and use that as a seed for the next search
        #init or clear seed array
        seed=np.array([])
        for j in range(i):
            seed = np.append(seed,-expKer(xss[j,:],samples,gamma)+sumKer(xss[j,:],xss,i,gamma))
        bestSeedIdx = np.argmin(seed)
        bestSeed=xss[bestSeedIdx,:]
        
        #grad descent succeeded (yay!), so assign new value to super samples
        xss[i,:]=results.x
        
        i=i+1
        #toc = time.clock()
#         print "Time elapsed %d" % (toc-tick)
    return xss

totalSS=70
xss = herd(samples[:80],totalSS,gamma=1)

plt.plot(samples[:,0], samples[:,1], '.')
plt.plot(xss[:,0],xss[:,1],'o')

# Calculate herding error
mu_p = np.mean(samples,axis=0)
err=np.zeros(totalSS)
for i in range(totalSS):
    err[i]    = np.linalg.norm(mu_p-np.sum(xss[1:i,:]        ,axis=0)/i)

idx = np.random.choice(70,totalSS)
samples_iid=samples[idx,:]
err_iid=np.zeros(totalSS)
for i in range(totalSS):
    err_iid[i]= np.linalg.norm(mu_p-np.sum(samples_iid[1:i,:],axis=0)/i)

plt.plot(err,'-o')
plt.plot(err_iid)
plt.plot([1,totalSS],[1,1./totalSS],'--')
plt.yscale('log')
plt.xscale('log')
plt.legend(['Herding error','iid error','~1/N'],'lower left')
