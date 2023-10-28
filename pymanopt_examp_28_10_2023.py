import autograd.numpy as anp
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
from pyriemann.utils.test import is_sym, is_sym_pos_def

anp.random.seed(42)

dim = 3
manifold = pymanopt.manifolds.Sphere(dim)
matrix = anp.random.normal(size=(dim, dim))
print('check1 = ', is_sym(matrix))
matrix = 0.5 * (matrix + matrix.T) # to symmetrize
print('check2=', is_sym(matrix))
@pymanopt.function.autograd(manifold)
def cost(point):
    return -point @ matrix @ point
problem = pymanopt.Problem(manifold, cost)
optimizer = pymanopt.optimizers.NelderMead() #SteepestDescent()
result = optimizer.run(problem).point
print(result)
'''
eigenvalues, eigenvectors = anp.linalg.eig(matrix)
dominant_eigenvector = eigenvectors[:, eigenvalues.argmax()]
print("Dominant eigenvector:", dominant_eigenvector)
print("Pymanopt solution:", result.point)
'''
# form pymanopt.examples.dominant_eigenvector.py
def create_cost_and_derivates(manifold, matrix):
    euclidean_gradient = None
    @pymanopt.function.autograd(manifold)
    def cost(x):
        #print('x=', x.shape , 'matrix=', matrix.shape)
        #print('x=', x[:3] , 'matrix=', matrix[:3,:3])
        #print((-x.T @ matrix @ x))
        # it returns a scalar value 
        return -x.T @ matrix @ x
    return cost, euclidean_gradient
from pymanopt.manifolds import Sphere
n = 128
matrix = np.random.normal(size=(n, n))
matrix = 0.5 * (matrix + matrix.T)
#print('*********matrix=', matrix[:3,:3])
manifold = Sphere(n)
cost, euclidean_gradient = create_cost_and_derivates(manifold, matrix)
problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)
optimizer = SteepestDescent(verbosity=2 )
estimated_dominant_eigenvector = optimizer.run(problem).point
# so matrix is our covariance matrix samples , x is a weight vector which is generated by autograd 