"""
Adatped from code made available in:
  Okabe, Yutaka, and Akira Shudo. A mathematical model of epidemics - A tutorial for students.
    Mathematics 8.7 (2020): 1174.
    
The fundamental premise, that CKD can be characterised as an epidemic, is based on 
https://www.pennmedicine.org/news/news-blog/2019/june/the-underrecognized-epidemic-of-chronic-kidney-disease

The premise (and rates) of progression of micro to macroalbuminuria is based on
End-Stage Renal Failure in the Diabetic Patient
https://www.sciencedirect.com/topics/medicine-and-dentistry/macroalbuminuria

"""

import math
import matplotlib.pyplot as plt

ECO_MAXDAYS     = 100 * 365
ECO_RESOLUTION  = 1E-2   # size of the simulation step (delta t)
ECO_GRANULARITY = 1E+2   # rate at which results are stored (e.g., 1 day = 100 steps)
ECO_PRECISION   = 1E-6   # minimum required difference between two floats for discernibility

def plot(N, Ts, Ss, ms, Ms):

  plt.title("Micro/Macroalbuminuria as an Epidemic")
  plt.plot(Ts, Ss, color=(0,1,0), linewidth=1.0, label='Susceptibles')
  plt.plot(Ts, ms, color=(1,0,0), linewidth=1.0, label='Microalbuminuria')
  plt.plot(Ts, Ms, color=(0,0,1), linewidth=1.0, label='Macroalbuminuria')

  plt.xlim(0, math.ceil(max(Ts)))
  plt.ylim(0, N)
  plt.legend()
  plt.xlabel('Days')
  plt.grid(True)
  plt.show()

def main():

  # free parameters of the model
  N = 100        # the number of individuals in the population at time t=0
  m = 0          # the initial number of individuals with microalbuminuria
  M = 0          # the initial number of individuals with Macroalbuminuria
  S  = N - m - M

  alpha = 2.0E-4  # rate of susceptible-to-micro (2.0E-4 to burn up in 70 yrs)
  beta  = 1.5E-3  # rate of micro-to-Macro       (1.5E-3 to burn up in 10 yrs)
  gamma = 8.0E-4  # rate of Macro-to-Susceptible (8.0E-4 to burn up in 20 yrs)

  # initialises the time series that describe the dynamics of the epidemic
  Ts = [ ]
  Ss = [ ]
  ms = [ ]
  Ms = [ ]

  # defines the diferentials
  dS = lambda S, m, M: -alpha * S + gamma * M
  dm = lambda S, m, M:  -beta * m + alpha * S
  dM = lambda S, m, M: -gamma * M +  beta * m

  # simulates the model
  c  = 0
  t  = 0
  dt = ECO_RESOLUTION
  while (c // ECO_GRANULARITY) < ECO_MAXDAYS:

    if(c % ECO_GRANULARITY == 0):
      Ts.append(t)
      Ss.append(S)
      ms.append(m)
      Ms.append(M)

      if((N - S - m - M) > ECO_PRECISION):
        print('-- Poor accounting! S + m + M = {0}'.format(S + m + M))

    # Step 1 - computes the differentials at time t
    dS1 = dS(S, m, M)
    dm1 = dm(S, m, M)
    dM1 = dM(S, m, M)

    # Step 2 - computes the differentials at time t + dt
    S2  = S + dS1 * dt
    m2  = m + dm1 * dt
    M2  = M + dM1 * dt

    dS2 = dS(S2, m2, M2)
    dm2 = dm(S2, m2, M2)
    dM2 = dM(S2, m2, M2)

    # Step 3 - updates the variables using the average of the differentials
    S = S + (dS1 + dS2) / 2 * dt
    m = m + (dm1 + dm2) / 2 * dt
    M = M + (dM1 + dM2) / 2 * dt
    #S = N - m - M

    t += dt
    c += 1

  print(S, m, M, t)
  plot(N, Ts, Ss, ms, Ms)

if __name__ == "__main__":

  main()
