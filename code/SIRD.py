import math
import matplotlib.pyplot as plt

ECO_MAXDAYS     = 356 * 2
ECO_RESOLUTION  = 1E-3
ECO_GRANULARITY = 100
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
  N = 10000                 # population size
  I = 500                   # initial number of infective individuals
  R = 2000                  # initial number of recovered individuals
  D = 0                     # initial number of deceased  individuals
  m = 20                    # average number of persons contacted by an infective individual (per day)
  p = 0.025                 # probability of transmission during contact with an infective individual
  e = 14                    # average number of days an individual remains infective
  gamma_d = 0.02            # rate of death

  # implied parameters
  gamma_r = 1/e - gamma_d   # rate of recovery
  gamma = gamma_r + gamma_d # reciprocal of the expected duration (in days) of infection
  beta = m * p              # rate of infection

  # initialises the time series that describe the dynamics of the epidemic
  (Ts, Ss, Is, Rs, Ds) = ([], [], [], [], [])

  # defines the diferentials for the SIRD variables
  dS = lambda S, I: -beta * S * (I/N)
  dI = lambda S, I:  beta * S * (I/N) - gamma * I
  dR = lambda I: gamma_r * I
  dD = lambda I: gamma_d * I

  # simulates the dynamics of the epidemic
  c  = 0
  t  = 0
  dt = ECO_RESOLUTION   # size of the simulation step (delta t)
  S  = N - I - R        # initial population of susceptibles
  while t < ECO_MAXDAYS and round(I, 0) > 0:

    if(c % ECO_GRANULARITY == 0):
      Ts.append(t)
      Ss.append(S)
      Is.append(I)
      Rs.append(R)
      Ds.append(D)

      if((N - S - I - R - D) > ECO_PRECISION):
        print('-- Bad accounting! S + I + R + D = {0}'.format(S + I + R + D))

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

    # Step 3 - updates the SIRD variables with the average of the differentials
    S = S + (dS1 + dS2) / 2 * dt
    I = I + (dI1 + dI2) / 2 * dt
    R = R + (dR1 + dR2) / 2 * dt
    D = D + (dD1 + dD2) / 2 * dt

    t = t + dt
    c += 1

  print(S, I, R, D, t)
  plot(N, Ts, Ss, Is, Rs, Ds)

if __name__ == "__main__":

  main()
