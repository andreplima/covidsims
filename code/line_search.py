# Line search in gradient and Newton directions, from:
# https://people.duke.edu/~ccc14/sta-663-2018/notebooks/S09E_Optimization_Line_Search.html
# https://people.duke.edu/~ccc14/sta-663-2018/notebooks/S09G_Gradient_Descent_Optimization.html
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar

MAXITER=200
UB=20

#------------------------------------------------------------------------------------------
# Solvers
#------------------------------------------------------------------------------------------
def gd1(x, f, grad, hess=None, max_iter=MAXITER):
  """
  Gradient descent with step size found by numerical minimization (uses minimize_scalar)
  """
  orbit = np.zeros((max_iter+1, len(x)))
  orbit[0] = x.ravel()
  for i in range(max_iter):
    res = minimize_scalar(lambda alpha: f(x - alpha * grad(x)))
    alpha = res.x
    x = x - alpha * grad(x)
    #print('GD1', i, x.ravel())
    orbit[i+1] = x.ravel()
  return orbit

def gd2(x, f, grad, hess, max_iter=MAXITER):
  """
  Gradient descent with analytic step size for quadratic function
  """
  orbit = np.zeros((max_iter+1, len(x)))
  orbit[0] = x.ravel()
  for i in range(max_iter):
    p = -grad(x)
    alpha = (p.T @ p)/(p.T @ hess(x) @ p)
    x = x - alpha * grad(x)
    #print('GD2', i, x.ravel())
    orbit[i+1] = x.ravel()
  return orbit

def gdn(x, f, grad, hess, max_iter=MAXITER):
  """
  Line search in Newton direction with analytic step size
  """
  orbit = np.zeros((max_iter+1, len(x)))
  orbit[0] = x.ravel()
  for i in range(max_iter):
    x = x - np.linalg.inv(hess(x)) @ grad(x) #xxx hessian needs to be conditioned to SP?
    #print('GDN', i, x.ravel())
    orbit[i+1] = x.ravel()
  return orbit

def gdr(x, f, grad, hess=None, alpha=0.1, beta=0.9, eps=1E-8, max_iter=10):
  """
  Gradient descent with Root Mean Squared Propagation (RMSProp)
  """
  #xs = np.zeros((1 + max_iter, x.shape[0]))
  #xs[0, :] = x
  orbit = np.zeros((max_iter+1, len(x)))
  orbit[0] = x.ravel()
  v = 0
  for i in range(max_iter):
    v = (beta * v) + (1 - beta) * grad(x)**2
    x = x - alpha * grad(x) / (eps + np.sqrt(v))
    #print('GDP', i, x.ravel())
    orbit[i+1] = x.ravel()
  return orbit

def gdm(x, f, grad, hess=None, alpha=0.1, beta=0.9, max_iter=10):
    #xs = np.zeros((1 + max_iter, x.shape[0]))
    #xs[0, :] = x
    orbit = np.zeros((max_iter+1, len(x)))
    orbit[0] = x.ravel()
    v = 0
    for i in range(max_iter):
        v  = (beta * v) + (1 - beta) * grad(x)
        vc = v / (1 + beta**(i+1))
        x = x - alpha * vc
        orbit[i+1] = x.ravel()
    return orbit

def gda(x, f, grad, hess=None, alpha=0.1, beta1=0.9, beta2=0.9, eps=1E-8, max_iter=10):
  """
  Adaptive Moment Estimation (ADAM), combines ideas of momentum, RMSProp and bias correction
  """
  orbit = np.zeros((max_iter+1, len(x)))
  orbit[0] = x.ravel()
  m = 0
  v = 0
  for i in range(max_iter):
    m = (beta1 * m) + (1 - beta1) * grad(x)
    v = (beta2 * v) + (1 - beta2) * grad(x)**2
    mc = m / (1 + beta1**(i+1))
    vc = v / (1 + beta2**(i+1))
    x = x - alpha * mc / (eps + np.sqrt(vc))
    orbit[i+1] = x.ravel()
  return orbit

#------------------------------------------------------------------------------------------
# Problem P1
#------------------------------------------------------------------------------------------

def f1(x):
  return (x[0]-2)**2 + x[1]**2

def grad1(x):
  return np.array([2*x[0]-4, 2*x[1]]).reshape([-1,1])

def hess1(x):
  return np.array([
      [2, 0],
      [0, 2]
  ])

def solve1a(x0):
  return gd1(x0, f1,   grad1, None,  max_iter=MAXITER)

def solve1b(x0):
  return gd2(x0, f1,   grad1, hess1, max_iter=MAXITER)

def solve1c(x0):
  return gdn(x0, f1,   grad1, hess1, max_iter=MAXITER)

def solve1d(x0):
  return gdr(x0, None, grad1, None,  max_iter=MAXITER)

def solve1e(x0):
  return gdm(x0, None, grad1, None,  max_iter=MAXITER)

def solve1f(x0):
  return gda(x0, None, grad1, None,  max_iter=MAXITER)

def show1(plt, nrows, ncols, pos, title, orbit = None):

  x = np.linspace(-UB, UB, 100)
  y = np.linspace(-UB, UB, 100)
  X, Y = np.meshgrid(x, y)
  Z = (X-2)**2 + Y**2

  plt.subplot(nrows, ncols, pos)
  plt.gca().set_title(title)
  plt.contour(X, Y, Z, 10)
  if(orbit is not None):
    plt.plot(orbit[:, 0], orbit[:, 1], 'r-o', markersize=3, linewidth=.5)
  plt.plot(2, 0, marker='+', markersize=10)
  plt.axis('square')
  return None

#------------------------------------------------------------------------------------------
# Problem P2
#------------------------------------------------------------------------------------------

def f2(x):
  return (x[0]-2)**2 + 10*x[1]**2

def grad2(x):
  return np.array([2*x[0]-4, 20*x[1]]).reshape([-1,1])

def hess2(x):
  return np.array([
      [2, 0],
      [0, 20]
  ])

def solve2a(x0):
  return gd1(x0, f2,   grad2, None,  max_iter=MAXITER)

def solve2b(x0):
  return gd2(x0, f2,   grad2, hess2, max_iter=MAXITER)

def solve2c(x0):
  return gdn(x0, f2,   grad2, hess2, max_iter=MAXITER)

def solve2d(x0):
  return gdr(x0, None, grad2, None,  max_iter=MAXITER)

def solve2e(x0):
  return gdm(x0, None, grad2, None,  max_iter=MAXITER)

def solve2f(x0):
  return gda(x0, None, grad2, None,  max_iter=MAXITER)

def show2(plt, nrows, ncols, pos, title, orbit = None):
  x = np.linspace(-UB, UB, 100)
  y = np.linspace(-UB, UB, 100)
  X, Y = np.meshgrid(x, y)
  Z = (X-2)**2 + 10*Y**2

  plt.subplot(nrows, ncols, pos)
  plt.gca().set_title(title)
  plt.contour(X, Y, Z, 10)
  if(orbit is not None):
    plt.plot(orbit[:, 0], orbit[:, 1], 'r-o', markersize=3, linewidth=.5)
  plt.plot(2, 0, marker='+', markersize=10)
  plt.axis('square')
  return None

#------------------------------------------------------------------------------------------
# Problem P3
#------------------------------------------------------------------------------------------

def f3(x):
  return (x[0] + x[1])**2 / (x[0]**2 + x[1]**2)

def grad3(x):
  return np.array([2*(x[0] + x[1])*(x[0]**2 - x[0]*(x[0] + x[1]) + x[1]**2)/(x[0]**2 + x[1]**2)**2,
                   2*(x[0] + x[1])*(x[0]**2 + x[1]**2 - x[1]*(x[0] + x[1]))/(x[0]**2 + x[1]**2)**2]).reshape([-1,1])

def hess3(x):
  H = np.array([[         4*x[0]*x[1]*(x[0]**2 - 3*x[1]**2)/(x[0]**6 + 3*x[0]**4*x[1]**2 + 3*x[0]**2*x[1]**4 + x[1]**6), 2*(-x[0]**4 + 6*x[0]**2*x[1]**2 - x[1]**4)/(x[0]**6 + 3*x[0]**4*x[1]**2 + 3*x[0]**2*x[1]**4 + x[1]**6)],
                [2*(-x[0]**4 + 6*x[0]**2*x[1]**2 - x[1]**4)/(x[0]**6 + 3*x[0]**4*x[1]**2 + 3*x[0]**2*x[1]**4 + x[1]**6),         4*x[0]*x[1]*(-3*x[0]**2 + x[1]**2)/(x[0]**6 + 3*x[0]**4*x[1]**2 + 3*x[0]**2*x[1]**4 + x[1]**6)]
               ]).reshape(2,2)

  return H

def solve3a(x0):
  return gd1(x0, f3,   grad3, None,  max_iter=MAXITER)

def solve3b(x0):
  return gd2(x0, f3,   grad3, hess3, max_iter=MAXITER)

def solve3c(x0):
  return gdn(x0, f3,   grad3, hess3, max_iter=MAXITER)

def solve3d(x0):
  return gdr(x0, None, grad3, None,  max_iter=MAXITER)

def solve3e(x0):
  return gdm(x0, None, grad3, None,  max_iter=MAXITER)

def solve3f(x0):
  return gda(x0, None, grad3, None,  max_iter=MAXITER)

def show3(plt, nrows, ncols, pos, title, orbit = None):
  x = np.linspace(-UB, UB, 100)
  y = np.linspace(-UB, UB, 100)
  X, Y = np.meshgrid(x, y)
  Z = (X+Y)**2 / (X**2+Y**2)

  plt.subplot(nrows, ncols, pos)
  plt.gca().set_title(title)
  plt.contour(X, Y, Z, 10)
  if(orbit is not None):
    plt.plot(orbit[:, 0], orbit[:, 1], 'r-o', markersize=3, linewidth=.5)
  plt.plot(0, 0, marker='+', markersize=10)
  plt.axis('square')
  return None

#------------------------------------------------------------------------------------------
# Examples
#------------------------------------------------------------------------------------------


if(__name__ == '__main__'):

  x0 = np.array([10,10]).reshape([-1,1])

  (nrows, ncols) = (2, 6)
  (unitsizew, unitsizeh) = (4.0, 3.8)
  fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * unitsizew, nrows * unitsizeh))
  plt.subplots_adjust(top=0.95, bottom=0.05, left=0.03, right=0.97, hspace=0.2, wspace=0.2)

  show1(plt, nrows, ncols,  1, 'Problem 1, solver GD1', solve1a(x0))
  show1(plt, nrows, ncols,  2, 'Problem 1, solver GD2', solve1b(x0))
  show1(plt, nrows, ncols,  3, 'Problem 1, solver GDN', solve1c(x0))
  show1(plt, nrows, ncols,  4, 'Problem 1, solver GDR', solve1d(x0))
  show1(plt, nrows, ncols,  5, 'Problem 1, solver GDM', solve1e(x0))
  show1(plt, nrows, ncols,  6, 'Problem 1, solver GDA', solve1f(x0))

  show2(plt, nrows, ncols,  7, 'Problem 2, solver GD1', solve2a(x0))
  show2(plt, nrows, ncols,  8, 'Problem 2, solver GD2', solve2b(x0))
  show2(plt, nrows, ncols,  9, 'Problem 2, solver GDN', solve2c(x0))
  show2(plt, nrows, ncols, 10, 'Problem 2, solver GDR', solve2d(x0))
  show2(plt, nrows, ncols, 11, 'Problem 2, solver GDM', solve2e(x0))
  show2(plt, nrows, ncols, 12, 'Problem 2, solver GDA', solve2f(x0))

  #show3(plt, nrows, ncols,  7, 'Problem 3, solver GD1', solve3a(x0))
  #show3(plt, nrows, ncols,  8, 'Problem 3, solver GD2', solve3b(x0))
  #show3(plt, nrows, ncols,  9, 'Problem 3, solver GDN', solve3c(x0))
  #show3(plt, nrows, ncols, 10, 'Problem 3, solver GDR', solve3d(x0))
  #show3(plt, nrows, ncols, 11, 'Problem 3, solver GDM', solve3e(x0))
  #show3(plt, nrows, ncols, 12, 'Problem 3, solver GDA', solve3f(x0))

  plt.show()
