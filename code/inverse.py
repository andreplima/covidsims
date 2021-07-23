import math
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

from os          import makedirs
from os.path     import join, exists
from collections import defaultdict
from sharedDefs  import getMountedOn, deserialise, saveAsText
from sharedDefs  import ECO_PRECISION
from sharedDefs  import ECO_SUSCEPTIBLE, ECO_INFECTIOUS, ECO_RECOVERED, ECO_DECEASED, ECO_CONFIRMED
from sharedDefs  import mse
from SIRD        import plotSeries

ECO_SERIESTYPES = [ECO_SUSCEPTIBLE, ECO_INFECTIOUS, ECO_RECOVERED, ECO_DECEASED]

def reformat(D, timeline, start = 0):
  # reformats a dictionary D[date][seriesType] = val into D[seriesType] = [val, ...] sorted by date
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


  plt.title("SIRD model of the epidemic - reconstructed")
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

def plotParams(Ts, beta, gamma_r, gamma_d):

  plt.title("SIRD parameters estimated from surveillance data")
  plt.plot(Ts, beta,     color=(1.0, 0.0, 0.5), linewidth=1.0, label=r'$\hat{\beta}(t)$')
  plt.plot(Ts, gamma_r,  color=(1.0, 0.5, 0.0), linewidth=1.0, label=r'$\hat{\gamma}_r(t)$')
  plt.plot(Ts, gamma_d,  color=(1.0, 0.5, 0.5), linewidth=1.0, label=r'$\hat{\gamma}_d(t)$')

  plt.xlim(0, math.ceil(max(Ts)))
  plt.ylim(0, 1.05 * max(beta + gamma_r + gamma_d))
  plt.legend()
  plt.xlabel('Days')
  plt.grid(True)
  plt.show()

  return None

def loadRealData():

  print('Loading real surveillance data of COVID-19')
  # recovers the preprocessed time series for COVID-19 download from Brasil-io
  sourcepath = [getMountedOn(), 'Task Stage', 'Task - covidsims', 'covidsims', 'results', 'T01', 'SP']
  data = deserialise(join(*sourcepath, 'data')) # data[date][seriesType] = int (accumulated)
  bol  = deserialise(join(*sourcepath, 'bol'))  # bol[date][seriesType]  = int (changes)
  timeline = deserialise(join(*sourcepath, 'timeline')) # timeline ~ [date, ...] sorted

  # reformats the recovered data to a more suitable format
  reports = reformat(data, timeline) # daily reported values of the SIRD variables (reports[seriesType] = [val, ...])
  changes = reformat(bol,  timeline) # daily changes of the SIRD variables         (changes[seriesType] = [val, ...])

  return (timeline, reports, changes, sourcepath)

def loadSimulatedData():

  print('Loading simulated data.')
  sourcepath = [getMountedOn(), 'Task Stage', 'Task - covidsims', 'covidsims', 'results', 'P01', 'C1']
  timeline = deserialise(join(*sourcepath, 'timeline')) # timeline ~ [float, ...] sorted
  reports  = deserialise(join(*sourcepath, 'reports'))  # daily reported values of the SIRD variables (reports[seriesType] = [val, ...])
  changes  = deserialise(join(*sourcepath, 'changes'))  # daily changes of the SIRD variables         (changes[seriesType] = [val, ...])

  return (timeline, reports, changes, sourcepath)

def main():

  # recovers preprocessed time series
  (timeline, reports, changes, sourcepath) =  loadRealData()
  T = len(timeline)

  # obtains estimates for rate parameters of the SIRD model
  print()
  print('Estimating the parameters from surveillance data')
  N = sum([reports[seriesType][0] for seriesType in ECO_SERIESTYPES])
  (gamma, gamma_d, gamma_r, beta) = ([], [], [], [])
  for t in range(T - 1):
    t_ = t + 1 # uses look-ahead to justify changes from "t to t+1" based on current report
    b  = -(changes[ECO_SUSCEPTIBLE][t_] * N) / (reports[ECO_SUSCEPTIBLE][t] * reports[ECO_INFECTIOUS][t])
    gr = changes[ECO_RECOVERED][t_] / reports[ECO_INFECTIOUS][t]
    gd = changes[ECO_DECEASED][t_]  / reports[ECO_INFECTIOUS][t]
    g  = gr + gd
    beta.append(b)
    gamma_r.append(gr)
    gamma_d.append(gd)
    gamma.append(g)

  # adds another set of parameters for the last time point (null because there is no look-ahead)
  beta.append(beta[-1])
  gamma_r.append(gamma_r[-1])
  gamma_d.append(gamma_d[-1])
  gamma.append(gamma[-1])

  print('Reconstructing the original series from inferred parameters')
  # initialises the 'report' series with their initial values (i.e., first day of surveillance)
  I0  = reports[ECO_INFECTIOUS][0]
  R0  = reports[ECO_RECOVERED][0]
  D0  = reports[ECO_DECEASED][0]
  S0  = N - I0 - R0 - D0
  (S, I, R, D) = [[S0], [I0], [R0], [D0]]
  (dS, dI, dR, dD) = ([-I0], [I0], [R0], [D0]) # evoking the 'star' premise

  # reconstructs the original series from the initial surveillance data and
  # the series of estimated rate parameters
  for t in range(T - 1):
    dS.append(-beta[t] * S[t] * I[t]/N)
    dR.append(gamma_r[t] * I[t]) # note that this corresponds to dR[t+1] <- gamma_r[t] * I[t]
    dD.append(gamma_d[t] * I[t])
    dI.append(-dS[-1] - (dR[-1] + dD[-1]))  # no more degrees of freedom, so ...
    S.append(S[t] + dS[-1])
    I.append(I[t] + dI[-1])
    R.append(R[t] + dR[-1])
    D.append(D[t] + dD[-1])

  # saves results for inspection
  filename = 'inverse.csv'
  print('-- Results saved in {0}'.format(join(*sourcepath, filename)))
  if(not exists(join(*sourcepath))): makedirs(join(*sourcepath))
  saveAsText(saveSeries(timeline, reports, changes,
                        S, I, R, D, dS, dI, dR, dD, N,
                        gamma, gamma_r, gamma_d, beta),
                        join(*sourcepath, filename))

  # plots the results
  plotParams(range(T), beta, gamma_r, gamma_d)
  plotSeries(range(T), S, I, R, D, N, "SIRD model of the epidemic - reconstructed", ylog=True)

  # run some sanity/quality checks
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