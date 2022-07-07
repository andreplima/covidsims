import re
import pandas as pd
import numpy  as np
from numpy.random   import seed, randn, random
from scipy.optimize import minimize

ECO_MAXDAYS     = 500
ECO_RESOLUTION  = 1E-3   # size of the simulation step (delta t)
ECO_GRANULARITY = 1E+3   # rate at which results are stored (e.g., 1 snapshot each 1K steps)
ECO_PRECISION   = 1E-6   # minimum difference between two floats required to consider them discernible

def headerfy(mask):
  res = re.sub('\:\d+\.\d+f', '', mask)
  res = re.sub('\:\d+d', '', res)
  return res

def dataset1(normalise = False):

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
    R = R/N
    N = 1.0

  return (N, R0, T, It)

def dataset2(normalise = False):

  # free parameters of the SIR model
  N  = 10000      # the size of the population
  I0 = 3         # the initial number of infective individuals
  R0 = 0          # the initial number of removed individuals (i.e., recovered and deceased)
  alpha = 0.07   # rate of transmission
  beta  = 0.03   # rate of removal

  # initialises the time series that describe the dynamics of the epidemic
  Ts = [ ]
  Ss = [ ]
  Is = [ ]
  Rs = [ ]

  # defines the diferentials for S (susceptibles) and I (infectives)
  dS = lambda S, I: -alpha * S * (I/N)
  dI = lambda S, I:  alpha * S * (I/N) - beta * I

  # simulates the dynamics of the epidemic (using Heun's variant of the Runge-Kutta method RK2)
  # see https://nm.mathforcollege.com/chapter-08-03-runge-kutta-2nd-order-method/
  # and https://autarkaw.org/2008/07/28/comparing-runge-kutta-2nd-order-methods/
  c  = 0
  t  = 0
  dt = ECO_RESOLUTION
  I  = I0
  R  = R0
  S  = N - I - R
  while (c // ECO_GRANULARITY) < ECO_MAXDAYS and round(I, 0) > 0:

    if(c % ECO_GRANULARITY == 0):
      Ts.append(t)
      Ss.append(int(S))
      Is.append(int(I))
      Rs.append(int(R))

      if((N - S - I - R) > ECO_PRECISION):
        print('-- Poor accounting! S + I + R = {0}'.format(S + I + R))

    # Step 1 - computes the differentials of the SIR variables at time t
    dS1 = dS(S, I)
    dI1 = dI(S, I)

    # Step 2 - computes the differentials of the SIR variables at time t + dt
    S2  = S + dS1 * dt
    I2  = I + dI1 * dt
    dS2 = dS(S2, I2)
    dI2 = dI(S2, I2)

    # Step 3 - updates the SIR variables using the average of the differentials
    S = S + (dS1 + dS2) / 2 * dt
    I = I + (dI1 + dI2) / 2 * dt
    R = N - S - I

    t = t + dt
    c += 1

  # formats the data to make it compatible with the interface of dataset1(.)
  T = np.array(Ts)
  It = np.array(Is)
  if(normalise):
    # normalises data to population N = 1
    T  = (T - min(T))/ (max(T) - min(T))
    It = df['It'].to_numpy() / N
    R0 = R0/N
    N = 1.0

  return (N, R0, T, It)

def config():
  rndseed = 23
  max_iter  = 3000  # maximum number of epoches
  h = 256            # number of neuron in the first hidden layer of NC1
  eta = 1E-3  # learning rate for online learning loop

  return (rndseed, max_iter, h, eta)

def cost(x, N, R0, T, It, h):
  (W1, b1, W2, b2, alpha, beta) = decode(x, h)
  sigma  = lambda x: np.tanh(x) # defines the activation function

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
    errors.append(E.dot(E))

  return np.mean(errors)

def encode(W1, b1, W2, b2, alpha, beta):
  return np.concatenate((W1, b1, W2.ravel(), b2, np.array([alpha, beta])))

def decode(x, h):
  W1 = x[0:h]
  b1 = x[h:2*h]
  W2 = x[2*h:5*h].reshape(3,h)
  b2 = x[5*h:5*h+3]
  (alpha, beta) = x[5*h+3:]
  return (W1, b1, W2, b2, alpha, beta)

def callback(xk):
  #print(xk)
  pass

def main():

  # recovers the network/training parameters
  (rndseed, max_iter, h, eta) = config()

  # recovers the dataset
  #(N, R0, T, It) = dataset1(normalise=False)
  (N, R0, T, It) = dataset2(normalise=False)
  print('Dataset has {0} observations'.format(len(T)))

  # initialises the network weights using the Glorot method
  seed(rndseed)
  (lb, ub) = (-1.0, 1.0)
  (W1, b1) = ((ub-lb) * randn(h)   + lb, (ub-lb) * randn(h) + lb)
  (W2, b2) = ((ub-lb) * randn(3,h) + lb, (ub-lb) * randn(3) + lb)

  # initialises other network parameters
  (alpha, beta) = (.1, .05) #random(2)

  # solving the underlying optimisation problem
  x0 = encode(W1, b1, W2, b2, alpha, beta)
  #res = minimize(cost, x0, (N, R0, T, It, h), method='BFGS', callback=callback)
  res = minimize(cost, x0, (N, R0, T, It, h), method='L-BFGS-B', callback=callback)

  np.printoptions(suppress=True, precision=3)
  print(cost(res.x, N, R0, T, It, h))
  print(alpha, beta)

  return None

if(__name__ == '__main__'):
  main()