import numpy as np

from os.path     import join
from collections import defaultdict
from sharedDefs  import getMountedOn, deserialise, saveAsText
from sharedDefs  import ECO_PRECISION
from sharedDefs  import ECO_SUSCEPTIBLE, ECO_INFECTIOUS, ECO_RECOVERED, ECO_DECEASED, ECO_CONFIRMED
from sharedDefs  import mse

ECO_SERIESTYPES = [ECO_SUSCEPTIBLE, ECO_INFECTIOUS, ECO_RECOVERED, ECO_DECEASED]

def reformat(D, timeline, start = 0):
  # reformats a dictionary D[date][seriesType] = val into D[seriesType] = [val, ...]
  series = defaultdict(list)
  for date in timeline[start:]:
    for seriesType in ECO_SERIESTYPES:
      series[seriesType].append(D[date][seriesType])
  return series

def saveSeries(timeline, reports, changes, S, I, R, D, dS, dI, dR, dD, N, gamma, gamma_r, gamma_d, beta):

  buffer_mask = '\t'.join(['{{{0}}}'.format(i) for i in range(22)])
  header = buffer_mask.format('t',
                              'gamma',
                              'gamma_r',
                              'gamma_d',
                              'beta',
                              'N',
                              'S', 'S_',
                              'I', 'I_',
                              'R', 'R_',
                              'D', 'D_',
                              '∆S(t)', '∆S_(t)',
                              '∆I(t)', '∆I_(t)',
                              '∆R(t)', '∆R_(t)',
                              '∆D(t)', '∆D_(t)'
                             )

  content = [header]

  T = len(timeline)
  for t in range(T):
    print(t)
    buffer = buffer_mask.format(t,
                                gamma[t],
                                gamma_r[t],
                                gamma_d[t],
                                beta[t],
                                N,
                                reports[ECO_SUSCEPTIBLE][t],  S[t],
                                reports[ECO_INFECTIOUS][t],   I[t],
                                reports[ECO_RECOVERED][t],    R[t],
                                reports[ECO_DECEASED][t],     D[t],
                                changes[ECO_SUSCEPTIBLE][t], dS[t],
                                changes[ECO_INFECTIOUS][t],  dI[t],
                                changes[ECO_RECOVERED][t],   dR[t],
                                changes[ECO_DECEASED][t],    dD[t]
                               )
    content.append(buffer)

  return '\n'.join(content)

def loadRealData():

  # recovers the preprocessed time series for COVID-19 download from Brasil-io
  sourcepath = [getMountedOn(), 'Task Stage', 'Task - covidsims', 'covidsims', 'results', 'T01', 'BR']
  data = deserialise(join(*sourcepath, 'data')) # data[date][seriesType] = int (accumulated)
  bol  = deserialise(join(*sourcepath, 'bol'))  # bol[date][seriesType]  = int (changes)
  timeline = deserialise(join(*sourcepath, 'timeline')) # timeline ~ [date, ...] sorted

  # reformats the recovered data to a more suitable format
  reports = reformat(data, timeline) # daily reported values of the SIRD variables (reports[seriesType] = [val, ...])
  changes = reformat(bol,  timeline) # daily changes of the SIRD variables         (changes[seriesType] = [val, ...])

  return (timeline, reports, changes, sourcepath)

def loadSimulatedData():

  sourcepath = [getMountedOn(), 'Task Stage', 'Task - covidsims', 'covidsims', 'results', 'P01', 'C1']
  timeline = deserialise(join(*sourcepath, 'timeline')) # timeline ~ [float, ...] sorted
  reports  = deserialise(join(*sourcepath, 'reports'))  # daily reported values of the SIRD variables (reports[seriesType] = [val, ...])
  changes  = deserialise(join(*sourcepath, 'changes'))  # daily changes of the SIRD variables         (changes[seriesType] = [val, ...])

  return (timeline, reports, changes, sourcepath)

def main():

  # recovers preprocessed time series
  (timeline, reports, changes, sourcepath) =  loadSimulatedData()
  T = len(timeline)

  # obtains estimates for rate parameters of the SIRD model
  print('Estimating the parameters over time')
  N = sum([reports[seriesType][0] for seriesType in ECO_SERIESTYPES])
  (gamma, gamma_d, gamma_r, beta) = ([], [], [], [])
  for t in range(T - 1):
    print(t)
    t_ = t + 1 # uses look-ahead to justify changes from "t to t+1" based on current report
    gr = changes[ECO_RECOVERED][t_] / reports[ECO_INFECTIOUS][t]
    gd = changes[ECO_DECEASED][t_]  / reports[ECO_INFECTIOUS][t]
    g  = gr + gd
    gamma_r.append(gr)
    gamma_d.append(gd)
    gamma.append(g)
    b = (    (changes[ECO_INFECTIOUS][t_] + g * reports[ECO_INFECTIOUS][t]) *
         N / (reports[ECO_SUSCEPTIBLE][t]     * reports[ECO_INFECTIOUS][t]) )
    beta.append(b)

    # xxx A mistery to me: why beta estimated this way does not work?
    #b = N / (reports[ECO_SUSCEPTIBLE][t] * reports[ECO_INFECTIOUS][t])
    #beta.append(b)

  # adds another set of parameters for the last time point (null because there is no look-ahead)
  gamma.append(0.0)
  gamma_r.append(0.0)
  gamma_d.append(0.0)
  beta.append(0.0)

  print('Reconstructing the original series')
  # initialises the 'report' series with their initial values (i.e., first day of surveillance)
  I  = [reports[ECO_INFECTIOUS][0]]
  R  = [reports[ECO_RECOVERED][0]]
  D  = [reports[ECO_DECEASED][0]]
  S  = [reports[ECO_SUSCEPTIBLE][0]]
  (dS, dI, dR, dD) = ([-I[0]], [I[0]], [0], [0]) # 'Star' premise operating here!

  # reconstructs the original series from the initial surveillance data and
  # the series of estimated rate parameters
  for t in range(T - 1):
    print(t)
    dR.append(gamma_r[t] * I[t]) # note that this corresponds to dR[t+1] = gamma_r[t] * I[t]
    dD.append(gamma_d[t] * I[t])
    dS.append(-beta[t] * S[t] * I[t]/N)
    dI.append(-dS[-1] - (dR[-1] + dD[-1]))  # solved by degree of freedom
    S.append(S[t] + dS[-1])
    I.append(I[t] + dI[-1])
    R.append(R[t] + dR[-1])
    D.append(D[t] + dD[-1])

  # saves results for inspection
  saveAsText(saveSeries(timeline, reports, changes,
                        S, I, R, D, dS, dI, dR, dD, N,
                        gamma, gamma_r, gamma_d, beta),
                        join(*sourcepath, 'inverse.csv'))

  # run some sanity checks
  print()

  aux = sum([1 if len(series) == T else 0 for series in [gamma, gamma_r, gamma_d, beta]])
  print('SC1. Right length: ', aux == 4)

  aux = sum([gamma[t] - gamma_r[t] - gamma_d[t] for t in range(T - 1)])
  print('SC2. Consistent .: ', aux < ECO_PRECISION)

  aux = sum([1 if len(series) == T else 0 for series in [S, I, R, D, dS, dI, dR, dD]])
  print('SC3. Right lenght: ', aux == 8)

  aux = sum([1 if (N - S[t] - I[t] - R[t] - D[t]) < 1 else 0 for t in range(T)])
  print('SC4. Consistent .: ', aux == T)

  # assesses the quality of the reconstructed time series -- using Parmezan's framework
  score_S = mse(reports[ECO_SUSCEPTIBLE], S)
  score_I = mse(reports[ECO_INFECTIOUS],  I)
  score_R = mse(reports[ECO_RECOVERED],   R)
  score_D = mse(reports[ECO_DECEASED],    D)
  print('SC5. MSE for S ..: {0:e}'.format(score_S))
  print('SC6. MSE for I ..: {0:e}'.format(score_I))
  print('SC7. MSE for R ..: {0:e}'.format(score_R))
  print('SC8. MSE for D ..: {0:e}'.format(score_D))

if __name__ == "__main__":

  main()
