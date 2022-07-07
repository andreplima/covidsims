import re
import pandas as pd
import numpy  as np
from numpy.random   import seed, randn, random
from scipy.optimize import minimize

def headerfy(mask):
  res = re.sub('\:\d+\.\d+f', '', mask)
  res = re.sub('\:\d+d', '', res)
  return res

def dataset(normalise = False):

  # Anonymous (1978). Influenza in a boarding school. British Medical Journal, 1, 578. Retrieved from
  # http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1603269/pdf/brmedj00115-0064.pdf

  N  = 763 # the size of the population
  R0 = 0   # estimated number of recovered individuals at t = 0
  data = [(1, 3), (2, 8), (3, 28), (4, 75), (5, 221), (6, 291), (7, 255), (8, 235), (9, 190), (10, 126), (11, 70), (12, 28), (13, 12), (14, 5)]
  df = pd.DataFrame(data, columns =['t', 'It'])

  T  = df['t'].to_numpy()
  It = df['It'].to_numpy()

  if(normalise):
    # normalises data to population N = 1
    T  = (T - min(T))/ (max(T) - min(T))
    It = df['It'].to_numpy() / N
    N = 1.0

  return (N, R0, T, It)

def config():
  rndseed = 23
  max_iter  = 3000  # maximum number of epoches
  h = 32            # number of neuron in the first hidden layer of NC1
  sigma  = lambda x: np.tanh(x) # defines the activation function
  eta = 1E-3  # learning rate for online learning loop

  return (rndseed, max_iter, h, sigma, eta)

def cost(x, T, It):
  return np.linalg.norm(x - It)

def callback(xk):
  #print(xk)
  pass

def main():

  # recovers the network/training parameters
  (rndseed, max_iter, h, sigma, eta) = config()

  # recovers the dataset
  (N, R0, T, It) = dataset(normalise=False)

  # initialises the network weights using the Glorot method
  seed(rndseed)
  (lb, ub) = (-1.0, 1.0)
  (W1, b1) = ((ub-lb) * randn(h)   + lb, (ub-lb) * randn(h) + lb)
  (W2, b2) = ((ub-lb) * randn(3,h) + lb, (ub-lb) * randn(3) + lb)

  # initialises other network parameters
  (alpha, beta) = random(2)

  #
  x0 = random(len(T))
  #res = minimize(cost, x0, (T, It), method='BFGS', callback=callback)
  res = minimize(cost, x0, (T, It), method='L-BFGS-B', callback=callback)

  np.printoptions(suppress=True, precision=1)
  print(It)
  print(res.x)

  return None

if(__name__ == '__main__'):
  main()