import re
import pandas as pd
import numpy  as np
from numpy.random import seed, randn, random

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

def main():

  # recovers the network/training parameters
  (rndseed, max_iter, h, sigma, eta) = config()

  # recovers the dataset
  (N, R0, T, It) = dataset(normalise=True)

  # initialises the network weights using the Glorot method
  seed(rndseed)
  (lb, ub) = (-1.0, 1.0)
  (W1, b1) = ((ub-lb) * randn(h)   + lb, (ub-lb) * randn(h) + lb)
  (W2, b2) = ((ub-lb) * randn(3,h) + lb, (ub-lb) * randn(3) + lb)

  # initialises other network parameters
  (alpha, beta) = random(2)

  # defines the cost function
  cost = lambda E: 0.5 * (E*E).sum()

  # runs an online learning loop
  mask = '{0}\t{1:11.9f}\t{2:5.3f}\t{3:5.3f}\t{4:11.9f}\t{5:11.9f}\t{6:11.9f}\t{7:11.9f}\t{8:11.9f}\t{9:11.9f}'
  print(headerfy(mask).format('iter', 'cost', '\talpha', 'beta', 'E1', '\tE2', '\tE3', '\tE4', '\tE5', '\tE6'))

  last_iter = 0
  for iter in range(max_iter):

    costs  = []
    errors = []
    E = np.zeros(7)
    for i in range(len(T)):
      t = T[i]

      #---------------------------------------------------------------------------
      # performs the forward step
      #---------------------------------------------------------------------------

      # layer 1 of NC1 (hidden layer)
      z1 = sigma(W1*t + b1)

      # layer 2 of NC1
      z2 = W2.dot(z1) + b2
      (S,I,R) = z2

      # differentials (d./dt)
      dz2_dt = (W2).dot(((1 - z1 * z1) * W1))
      (dS, dI, dR) = dz2_dt

      # single layer of NC2 (errors)
      betaSI = beta  * I * S
      alphaI = alpha * I
      E[1] = dS + betaSI
      E[2] = dI - betaSI + alphaI
      E[3] = dR          - alphaI
      E[4] = N - (S + I + R)
      E[5] = (I - It[i])
      if(i == 0): E[6] = (R - R0)

      # accrues the total cost per epoch
      costs.append(cost(E))
      errors.append(E)

    #---------------------------------------------------------------------------
    # performs the backward step
    #---------------------------------------------------------------------------

    # computes reusable nodes
    E = np.mean(errors, axis=1)
    dL_dE  = (np.ones((3, E.shape[0])) * E).transpose()
    dE_dz2 = np.array([
                      [      0.0,               0.0,  0.0],
                      [ beta * I,          beta * S,  0.0],
                      [-beta * I,  alpha - beta * S,  0.0],
                      [      0.0, -alpha           ,  0.0],
                      [     -1.0,              -1.0, -1.0],
                      [      0.0,               1.0,  0.0],
                      [      0.0,               0.0,  1.0]
                     ])

    _W1_rows = W1.shape[0]
    _W2_rows = W2.shape[0]
    dz2_dW2  = np.einsum('ij, k', np.eye(_W2_rows, _W2_rows), z1)
    dz1_db1  = np.diag((np.ones(_W1_rows) - (z1 ** 2)))

    # updates the weights of the single layer of NC2
    #beta  = max(0.0, beta  - eta *     I * (E[2] - E[3]))
    #alpha = max(0.0, alpha - eta * S * I * (E[1] - E[2]))
    beta  -= eta * S * I * (E[1] - E[2])
    alpha -= eta *     I * (E[2] - E[3])

    # updates the weights in layer 2 of NC1
    dL_dW2 = sum([np.tensordot(dL_dE[i] * dE_dz2[i], dz2_dW2, axes=1) for i in range(7)])
    dL_db2 = sum([             dL_dE[i] * dE_dz2[i]                   for i in range(7)])
    W2 -= eta * dL_dW2
    b2 -= eta * dL_db2

    # updates the weights in layer 1 of NC1
    dL_dW1 = sum([(dL_dE[i] * dE_dz2[i]).dot((W2).dot(t*dz1_db1))     for i in range(7)])
    dL_db1 = sum([(dL_dE[i] * dE_dz2[i]).dot((W2).dot(  dz1_db1))     for i in range(7)])
    W1 -= eta * dL_dW1
    b1 -= eta * dL_db1

    if(iter % 1 == 0):
      print(mask.format(iter+last_iter, np.mean(costs), alpha, beta, E[1], E[2], E[3], E[4], E[5], E[6]))

  print(mask.format(iter+last_iter, np.mean(costs), alpha, beta, E[1], E[2], E[3], E[4], E[5], E[6]))
  last_iter += iter + 1

  return None

if(__name__ == '__main__'):
  main()