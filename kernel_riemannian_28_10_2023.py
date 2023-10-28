from pymanopt.manifolds import  SymmetricPositiveDefinite
def create_cost_and_derivates(manifold, matrix):
    euclidean_gradient = None
    @pymanopt.function.autograd(manifold)
    def cost(x):
        numSamples = len(matrix)
        return   np.sum([np.exp(distance_riemann(-x.T*x, matrix[i])**2) for i in range(numSamples)])/numSamples  
    return cost, euclidean_gradient
n = 12 ; k = 1
manifold =  SymmetricPositiveDefinite(n,k)
cost, euclidean_gradient = create_cost_and_derivates(manifold, matrix)
problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)

optimizer = SteepestDescent(verbosity=2 )
result = optimizer.run(problem).point


