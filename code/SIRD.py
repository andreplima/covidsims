"""
Adatped from code made available in:
  Okabe, Yutaka, and Akira Shudo. A Mathematical Model of Epidemics - A Tutorial for Students.
    Mathematics 8.7 (2020): 1174.
"""

import math
import matplotlib.pyplot as plt

from os.path    import join
from sharedDefs import getMountedOn, serialise

ECO_MAXDAYS     = 356
ECO_RESOLUTION  = 1E-3   # size of the simulation step (delta t)
ECO_GRANULARITY = 1E3    # rate at which results are stored (e.g., 1 snapshot each 1K steps)
ECO_PRECISION   = 1E-6

def plot(N, Ts, Ss, Is, Rs, Ds):

  plt.title("SIRD model of the epidemic")
  plt.plot(Ts, Ss, color=(0,1,0), linewidth=1.0, label='Susceptibles')
  plt.plot(Ts, Is, color=(1,0,0), linewidth=1.0, label='Infectives')
  plt.plot(Ts, Rs, color=(0,0,1), linewidth=1.0, label='Recovered')
  plt.plot(Ts, Ds, color=(0,1,1), linewidth=1.0, label='Deceased')

  plt.xlim(0, math.ceil(max(Ts)))
  plt.ylim(0, N)
  plt.legend()
  plt.xlabel('Days')
  plt.grid(True)
  plt.show()

def main():

  # free parameters of the SIRD model
  N  = 1E4                  # population size
  I0 = 10                   # initial number of infective individuals
  R0 = 0                    # initial number of recovered individuals
  D0 = 0                    # initial number of deceased  individuals
  beta = 0.07               # rate of infection
  gamma_d = 0.01            # rate of death
  gamma_r = 0.02            # rate of recovery

  # constrained parameters
  gamma = gamma_r + gamma_d # reciprocal of the expected duration (in days) of infection
  e = 1 / gamma             # average number of days an individual remains infective

  # initialises the time series that describe the dynamics of the epidemic
  (Ts, Ss, Is, Rs, Ds) = ([], [], [], [], [])

  # defines the diferentials for the SIRD variables
  dS = lambda S, I: -beta * S * (I/N)
  dI = lambda S, I:  beta * S * (I/N) - gamma   * I
  dR = lambda    I:                     gamma_r * I
  dD = lambda    I:                     gamma_d * I

  # simulates the dynamics of the epidemic using a Runge-Kutta method (RK2)
  c  = 0
  t  = 0
  dt = ECO_RESOLUTION   
  (S, I, R, D) = (N - I0 - R0 - D0, I0, R0, D0)
  while t < ECO_MAXDAYS and round(I, 0) > 0:

    if(c % ECO_GRANULARITY == 0):
      Ts.append(t)
      Ss.append(S)
      Is.append(I)
      Rs.append(R)
      Ds.append(D)

      if((N - S - I - R - D) > ECO_PRECISION):
        print('-- Poor accounting! S + I + R + D = {0} <> {1} = N'.format(S + I + R + D, N))

    # Step 1 - computes the differentials of the SIRD variables at time t
    dS1 = dS(S, I)
    dI1 = dI(S, I)
    dR1 = dR(I)
    dD1 = dD(I)

    # Step 2 - computes the differentials of the SIRD variables at time t + dt
    S2  = S + dS1 * dt
    I2  = I + dI1 * dt
    R2  = R + dR1 * dt
    D2  = D + dD1 * dt

    dS2 = dS(S2, I2)
    dI2 = dI(S2, I2)
    dR2 = dR(I2)
    dD2 = dD(I2)

    # Step 3 - updates the SIRD variables using the average of the differentials
    S = S + (dS1 + dS2) / 2 * dt
    I = I + (dI1 + dI2) / 2 * dt
    R = R + (dR1 + dR2) / 2 * dt
    D = D + (dD1 + dD2) / 2 * dt

    t = t + dt
    c += 1

  # plots the results
  print(S, I, R, D, t)
  plot(N, Ts, Ss, Is, Rs, Ds)

  # saves the results for inspection (and use as synthetic series in the inverse)
  T = len(Ts)
  timeline = Ts
  reports = {'S': Ss, 'I': Is, 'R': Rs, 'D': Ds}
  (dS, dI, dR, dD) = ([-I0], [I0 - R0 - D0], [R0], [D0])
  for t in range(1, T):
    dS.append(Ss[t] - Ss[t-1])
    dI.append(Is[t] - Is[t-1])
    dR.append(Rs[t] - Rs[t-1])
    dD.append(Ds[t] - Ds[t-1])
  changes = {'S': dS, 'I': dI, 'R': dR, 'D': Ds}

  sourcepath = [getMountedOn(), 'Task Stage', 'Task - covidsims', 'covidsims', 'results', 'P01', 'C1']
  serialise(timeline, join(*sourcepath, 'timeline'))
  serialise(reports,  join(*sourcepath, 'reports'))
  serialise(changes,  join(*sourcepath, 'changes'))
  print(len(timeline))
  print(len(reports['S']), len(changes['S']))
  print(len(reports['I']), len(changes['I']))
  print(len(reports['R']), len(changes['R']))
  print(len(reports['D']), len(changes['D']))

if __name__ == "__main__":

  main()
