from pymanopt.manifolds import  SymmetricPositiveDefinite
from tensorflow_riemopt import manifolds
E = manifolds.Euclidean()
SPD = manifolds.SPDAffineInvariant()
manifold = manifolds.Product((SPD, (3,3)), (E, (2,)))
from geoopt.manifolds import Euclidean, ProductManifold, SymmetricPositiveDefinite
SPD = SymmetricPositiveDefinite()
E = Euclidean()
manifold =  ProductManifold((SPD,(12,320)), (E,(12)))

def create_cost_and_derivates(manifold, matrix):
    euclidean_gradient = None
    @pymanopt.function.autograd(manifold)
    def cost(x):
        numSamples = len(matrix)
        return   np.sum([np.exp(distance_riemann(-x.T*x, matrix[i])**2) for i in range(numSamples)])/numSamples  
    return cost, euclidean_gradient
n = 12 
manifold =  SymmetricPositiveDefinite(n)
cost, euclidean_gradient = create_cost_and_derivates(manifold, matrix)
problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)

optimizer = SteepestDescent(verbosity=2 )
result = optimizer.run(problem).point


