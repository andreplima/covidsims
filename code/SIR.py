import math
import matplotlib.pyplot as plt

ECO_MAXDAYS     = 356 * 2
ECO_RESOLUTION  = 1E-3
ECO_GRANULARITY = 100
ECO_PRECISION   = 1E-6

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
  N = 10000
  I = 10
  R = 0
  beta  = 0.5
  gamma = 0.2

  # initialises the time series that describe the dynamics of the epidemic
  Ts = [ ]
  Ss = [ ]
  Is = [ ]
  Rs = [ ]

  # defines the diferentials for S (susceptibles) and I (infectives)
  dS = lambda S, I: -beta * S * (I/N)
  dI = lambda S, I:  beta * S * (I/N) - gamma * I

  # simulates the dynamics of the epidemic
  c  = 0
  t  = 0
  dt = ECO_RESOLUTION
  S  = N - I - R
  while t < ECO_MAXDAYS and round(I, 0) > 0:

    if(c % ECO_GRANULARITY == 0):
      Ts.append(t)
      Ss.append(S)
      Is.append(I)
      Rs.append(R)
      #xxx add Ds (separate recovered from deceased)

      if((N - S - I - R) > ECO_PRECISION):
        print('-- Bad accounting! S + I + R = {0}'.format(S + I + R))

    # Step 1 - computes the differentials of S and I in time t
    dS1 = dS(S, I)
    dI1 = dI(S, I)

    # Step 1 - computes the differentials of S and I in time t + dt
    S2  = S + dS1 * dt
    I2  = I + dI1 * dt
    dS2 = dS(S2, I2)
    dI2 = dI(S2, I2)

    # update
    S = S + (dS1 + dS2) / 2 * dt
    I = I + (dI1 + dI2) / 2 * dt
    R = N - S - I
    t = t + dt

    c += 1

  print(S, I, R, t)
  plot(N, Ts, Ss, Is, Rs)

if __name__ == "__main__":

  main()
