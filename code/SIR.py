"""
Adatped from code made available in:
  Okabe, Yutaka, and Akira Shudo. A mathematical model of epidemics - A tutorial for students.
    Mathematics 8.7 (2020): 1174.
"""

import math
import matplotlib.pyplot as plt

ECO_MAXDAYS     = 356
ECO_RESOLUTION  = 1E-3   # size of the simulation step (delta t)
ECO_GRANULARITY = 1E+3   # rate at which results are stored (e.g., 1 snapshot each 1K steps)
ECO_PRECISION   = 1E-6   # minimum difference between two floats required to consider them discernible

def plot(N, Ts, Ss, Is, Rs):

  plt.title("SIR model of the epidemic")
  plt.plot(Ts, Ss, color=(0,1,0), linewidth=1.0, label='Susceptibles')
  plt.plot(Ts, Is, color=(1,0,0), linewidth=1.0, label='Infectives')
  plt.plot(Ts, Rs, color=(0,0,1), linewidth=1.0, label='Removed')

  plt.xlim(0, math.ceil(max(Ts)))
  plt.ylim(0, N)
  plt.legend()
  plt.xlabel('Days')
  plt.grid(True)
  plt.show()

def main():

  # free parameters of the SIR model
  N = 10000      # the size of the population
  I = 10         # the initial number of infective individuals
  R = 0          # the initial number of removed individuals (i.e., recovered and deceased)
  beta  = 0.07   # rate of transmission
  gamma = 0.03   # rate of removal

  # initialises the time series that describe the dynamics of the epidemic
  Ts = [ ]
  Ss = [ ]
  Is = [ ]
  Rs = [ ]

  # defines the diferentials for S (susceptibles) and I (infectives)
  dS = lambda S, I: -beta * S * (I/N)
  dI = lambda S, I:  beta * S * (I/N) - gamma * I

  # simulates the dynamics of the epidemic (using Heun's variant of the Runge-Kutta method RK2)
  # see https://nm.mathforcollege.com/chapter-08-03-runge-kutta-2nd-order-method/
  # and https://autarkaw.org/2008/07/28/comparing-runge-kutta-2nd-order-methods/
  c  = 0
  t  = 0
  dt = ECO_RESOLUTION
  S  = N - I - R
  while (c // ECO_GRANULARITY) < ECO_MAXDAYS and round(I, 0) > 0:

    if(c % ECO_GRANULARITY == 0):
      Ts.append(t)
      Ss.append(S)
      Is.append(I)
      Rs.append(R)

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

  print(S, I, R, t)
  plot(N, Ts, Ss, Is, Rs)

if __name__ == "__main__":

  main()
