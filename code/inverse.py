import numpy as np

from os.path     import join
from collections import defaultdict
from sharedDefs  import getMountedOn, deserialise
from sharedDefs  import ECO_SUSCEPTIBLE, ECO_INFECTIOUS, ECO_RECOVERED, ECO_DECEASED, ECO_CONFIRMED
from sharedDefs  import mse, tu, pocid, mcpm

ECO_SERIESTYPES = [ECO_SUSCEPTIBLE, ECO_INFECTIOUS, ECO_RECOVERED, ECO_DECEASED, ECO_CONFIRMED]

def reformat(D, timeline, start = 0):
  # reformats a dictionary D[date][seriesType] = val into D[seriesType] = [val, ...]
  series = defaultdict(list)
  for date in timeline[start:]:
    for seriesType in ECO_SERIESTYPES:
      series[seriesType].append(D[date][seriesType])
  return series

def main():

  # assumes the expected duration of infection (in days) -- see logbook, reference [17]
  e  = 17.86

  # recovers the preprocessed time series
  sourcepath = [getMountedOn(), 'Task Stage', 'Task - covidsims', 'covidsims', 'results', 'T01', 'BR']
  bol  = deserialise(join(*sourcepath, 'bol'))  # bol[date][seriesType]  = int (changes)
  data = deserialise(join(*sourcepath, 'data')) # data[date][seriesType] = int (accumulated)
  timeline = sorted(data)
  T = len(timeline)

  # reformats the recovered data to a more suitable format
  changes = reformat(bol,  timeline, 1) # daily changes of the SIRD+C variables
  reports = reformat(data, timeline, 0) # daily reported values of the SIRD+C variables

  # obtains estimates for rate parameters of the SIRD model
  g = 1/e    # corresponds to the value of gamma
  N = sum([reports[seriesType][1] for seriesType in [ECO_SUSCEPTIBLE, ECO_INFECTIOUS, ECO_RECOVERED, ECO_DECEASED]])
  gamma   = []
  gamma_d = []
  gamma_r = []
  beta    = []
  for t in range(T - 1):
    print(t)
    t_ = t - 1
    aux = changes[ECO_DECEASED][t] / reports[ECO_INFECTIOUS][t_]
    gamma.append(g)
    gamma_d.append(aux)
    gamma_r.append(g - aux)
    aux = (    (changes[ECO_INFECTIOUS][t] + g * reports[ECO_INFECTIOUS][t_]) *
           N / (reports[ECO_SUSCEPTIBLE][t_]   * reports[ECO_INFECTIOUS][t_]) )
    beta.append(aux)

  # run some sanity checks
  aux = sum([1 if len(series) == T - 1 else 0 for series in [gamma, gamma_r, gamma_d, beta]])
  print('SC1. Right length: ', aux == 4)

  aux = sum([gamma[t] - gamma_r[t] - gamma_d[t] for t in range(T - 1)])
  print('SC2. Consistent : ', aux < ECO_PRECISION)

  # reconstructs the original series from the estimated rate parameters
  I = [reports[ECO_INFECTIOUS][0]]
  R = [reports[ECO_RECOVERED][0]]
  D = [reports[ECO_DECEASED][0]]
  S = [N - I[0] - R[0] - D[0]]
  for t in range(len(timeline) - 2):
    dS = -beta[t] * S[t] * I[t]/N
    dI = dS - gamma[t] * I[t]
    dR = gamma_r[t] * I[t]
    dD = gamma_d[t] * I[t]
    S.append(S[t] + dS)
    I.append(I[t] + dI)
    R.append(R[t] + dR)
    D.append(D[t] + dD)


  # run some more sanity checks
  aux = sum([1 if len(series) == T else 0 for series in [S, I, R, D]])
  print('SC3. Right lenght: ', aux == 4)

  aux = sum([1 if N == S[t] + I[t] + R[t] + D[t] else 0 for t in range(1, T)])
  print('SC4. Consistency : ', aux == 4)

  # assesses the quality of the reconstructed time series -- using Parmezan's framework
  score_mse = mse(reports[ECO_INFECTIOUS], I)
  print(score_mse)

if __name__ == "__main__":

  main()
