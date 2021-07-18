"""

  Many ideas implemented here follow definitions and insights from these sources:

  [1] Funk, S., Camacho, A., Kucharski, A. J., Lowe, R., Eggo, R. M., & Edmunds, W. J. (2019).
      Assessing the performance of real-time epidemic forecasts: A case study of Ebola in the
      Western Area region of Sierra Leone, 2014-15. PLoS computational biology, 15(2).

  [2] http://www.legislacao.sp.gov.br/legislacao/dg280202.nsf/5fb5269ed17b47ab83256cfb00501469/35ea1f3341ab9b9c83258577004cd65e

  [3] Parmezan, A. R. S., Souza, V. M., & Batista, G. E. (2019). Evaluation of statistical and machine
      learning models for time series prediction: Identifying the state-of-the-art and the best conditions
      for the use of each model. Information Sciences, 484, 302-337.

  [4] Younhee Lee and Woong Lim. 2017. Shoelace formula: Connecting the area of a polygon with vector
      cross product. Mathematics Teacher 110, 8 (2017), 631–636.

  [5] Parmezan, A. R. S., Batista, G. E. (2015, December). A study of the use of complexity measures in the
      similarity search process adopted by knn algorithm for time series prediction. In 2015 IEEE 14th
      International Conference on Machine Learning and Applications (ICMLA) (pp. 45-51). IEEE.

"""

import os
import sys
import numpy as np
import sharedDefs as ud

from os          import listdir, makedirs, remove
from os.path     import join, isfile, isdir, exists
from random      import seed
from collections import namedtuple, defaultdict

from sharedDefs import ECO_SEED, ECO_PRECISION
from sharedDefs import setupEssayConfig, getEssayParameter, setEssayParameter, overrideEssayParameter
from sharedDefs import getMountedOn, serialise, saveAsText, stimestamp, tsprint, saveLog
from sharedDefs import file2List
from sharedDefs import mse, tu, pocid, mcpm

Metrics = namedtuple('Metrics', ['mse', 'tu', 'pocid', 'er', 'mcpm'])

class TimeSeries:

  def __init__(self, dataset, modelType, modelParams, plength = 14):

    # properties defined during instantiation (and unmodified during runtime)
    self.dataset     = dataset    # dataset ~ {id: {S: ts, I: ts, R: ts, D: ts, C: ts}, ...}
    self.modelType   = modelType  # str: 'MA', 'SIRD', etc.
    self.modelParams = modelParams
    self.plength     = plength    # int: length of the predictions to be produced
                                  # deciding for a default plength is 2 weeks is based on:
                                  # - predictions beyond this timeframe are no good to guide public policy [1]
                                  # - it has been tested to update/review emergency plans (e.g., Plano São Paulo [2])

    # properties modified during runtime
    self.fitted      = []         # fitted ~ [(fitted model, model assessment), ...]

  def fit(self, seriesType):

    for id in self.dataset:

      # splits the time series into training and test series
      ts = self.dataset[id][seriesType]
      cut = len(ts) - self.plength
      (ts_tr, ts_te) = (ts[0:cut], ts[cut:])

      # instantiates and fits the model to the training series,
      # and obtains a prediction (i.e., a series of length plength that continues the training series)
      model = self.instantiateModel(self.modelType, self.modelParams)
      model.fit(ts_tr)
      ts_pr = model.predict()

      # assesses the quality of the prediction using the evaluation framework from [3]
      metrics = self.assess(ts_te, ts_pr, ts_tr[-1])

      # stores the model and the obtained assessment
      self.fitted.append((model, metrics))

    return None

  def assess(self, ts_te, ts_pr, last):

    # assesses the quality of the prediction using the metrics in [3]
    score_mse   = mse(ts_te, ts_pr)
    score_tu    = tu(ts_te, ts_pr, last)
    score_pocid = pocid(ts_te, ts_pr, last)
    score_er    = 100 - score_pocid
    score_mcpm  = mcpm([score_mse, score_tu, score_er])
    metrics     = Metrics(score_mse, score_tu, score_pocid, score_er, score_mcpm)

    return metrics

  def instantiateModel(self, modelType, modelParams):

    if(modelType == 'kNN-TSPi'):
      model = kNN_TSPi(modelParams)
    else:
      raise ValueError('Model {0} has not been implemented.'.format(modelType))

    return model

class kNN_TSPi:
  """
  performs time series prediction using the knn-TSPi algorithm [5]
  """

  def __init__(self, modelParams):

    # unpacks the parameters
    (max_p, h) = modelParams

    # properties defined during instantiation (and unmodified during runtime)
    self.max_p  = max_p # upper bound for the number of observations in a season
    self.h      = h     # the length of the predicted series

    # properties modified during runtime
    self.l      = None  # the length of the query subsequence, in number of observations
    self.k      = None  # the number of similar subsequences
    self.Z      = None  # the base time series with m observations
    self.m      = None

  def fit(self, ts_tr):
    """
    implements the holdout validation procedure in [3]
    """

    # splits the time series into training and validation series
    n = len(ts_tr) - self.h
    (S, ts_val) = (ts_tr[0:n], np.array(ts_tr[n:]))
    self.Z = S

    # seeks for suitable values for parameters l and k
    min_error = np.inf
    for self.l in range(3, self.max_p + 1, 2):
      for self.k in range(1, 9 + 1, 2):
        ts_pr = self.predict()
        error = np.linalg.norm(ts_val - ts_pr)
        print(self.l, self.k, error)
        if(error < min_error):
          min_error = error
          (l_best, k_best) = (self.l, self.k)

    self.l = l_best
    self.k = k_best
    self.Z = ts_tr
    print(self.l, self.k)

    # xxx check if l and h are compatible with m

    return None

  def predict(self):
    """
    implements the kNN-TSPi procedure in [5]
    """

    # 1. S contains all subsequences of length l that makes up the search space
    #    q is the query subsequence
    (S, q) = self.generate_subsequences(self.Z, self.l)

    # 2. S_ contains the subsequences in S, normalised (using z-core)
    # 3. q_ corresponds to normalised q
    (S_, S_stats) = zip(*[self.normalise(subseq) for subseq in S])
    (q_, q_stats) = self.normalise(q)

    # 4. D[j] contains the complexity-invariant distance between q_ and each subsequence S_[j]
    D = [(j, self.CID(q_, S_[j])) for j in range(len(S_))]

    # 5. selects the k subsequences in S_ most similar to q_
    P = self.search_nearest_neighbours(D, self.k)

    # 6. recovers the next h observations of each of the k most similar subsequences in P
    ts_pr_ = self.recover_samples(P, S_stats, self.Z, self.l, self.h)

    # 7. denormalises the obtained prediction
    ts_pr = self.denormalise(ts_pr_, q_stats)

    return ts_pr

  def generate_subsequences(self, Z, l):
    m  = len(Z)
    n  = m - l + 1                                 # the number of subsequences of Z with length l
    ss = [np.array(Z[j: j + l]) for j in range(n)] # the list of all such subsequences
    S  = ss[0: n - 1]                              # the list of subsequences making up the search space
    q  = ss[-1]                                    # the query subsequence

    # xxx delete: block inserted for inspection
    t = lambda l: ';'.join([str(e) for e in l])
    d = defaultdict(int)
    X = [Z[j: j + l] for j in range(n)]
    for L in X: d[t(L)] += 1

    return (S, q)

  def normalise(self, subseq):
    mu  = np.mean(subseq)
    sd  = np.std(subseq, ddof=1)
    stats   = (mu, sd)
    subseq_ = (subseq - mu) / sd
    return (subseq_, stats)

  def CE(self, subseq):
    return sum([(subseq[i] - subseq[i+1])**2 for i in range(len(subseq) - 1)]) ** .5

  def CID(self, q_, s_):
    ed  = np.linalg.norm(q_ - s_)
    cq_ = self.CE(q_)
    cs_ = self.CE(s_)
    cf  = max(cq_, cs_)/min(cq_, cs_)
    res = ed * cf
    return res

  def search_nearest_neighbours(self, D, k):
    L = [j for (j, dist) in sorted(D, key = lambda e:e[1])]
    return L[0:k]

  def recover_samples(self, P, S_stats, Z, l, h):
    ss_ = []
    for j in P:
      (mu, sd) = S_stats[j]
      subseq = (np.array(Z[j: j + l + h]) - mu) / sd
      ss_.append(subseq)
    ss_ = [subseq for subseq in ss_ if len(subseq) == (l + h)]
    ts_pr_ = np.mean(ss_, 0)[-h:]
    return ts_pr_

  def denormalise(self, ts_pr_, stats):
    (mu, sd) = stats
    ts_pr = (ts_pr_ * sd) + mu
    return ts_pr


def at_knn_TSPi():

  # assembly test for knn-TSPi class
  sourcepath = [getMountedOn(), 'Task Stage', 'Task - covidsims', 'covidsims', 'results', 'P01', 'C0']
  filename   = 'fa_constant_level.data'
  #filename   = 'fa_increasing_trend.data'
  #filename   = 'fc_constant_level.data'
  ts = [float(l[0]) for l in file2List(join(*sourcepath, filename))]

  max_p = 25
  h     = 14
  modelParams = (max_p, h)
  model = kNN_TSPi(modelParams)
  cut   = len(ts) - h
  (ts_tr, ts_te) = (ts[0:cut], np.array(ts[cut:]))
  model.fit(ts_tr)
  ts_pr = model.predict()

  print()
  print(ts_pr)
  print(ts_te)
  print(np.linalg.norm(ts_te - ts_pr))

def at_TimeSeries():

  # assembly test for knn-TSPi class
  sourcepath = [getMountedOn(), 'Task Stage', 'Task - covidsims', 'covidsims', 'results', 'P01', 'C0']
  filenames  = ['fa_constant_level.data', 'fa_increasing_trend.data', 'fc_constant_level.data']
  loader = lambda filename: [float(l[0]) for l in file2List(join(*sourcepath, filename))]
  dataset = {filename: {'Test': loader(filename)} for filename in filenames}

  o = TimeSeries(dataset, 'kNN-TSPi', (25, 14))
  o.fit('Test')
  serialise(o, 'o')

def main(configFile):

  ud.LogBuffer = []

  #at_knn_TSPi()
  at_TimeSeries()

  tsprint('Process completed')

if __name__ == "__main__":

  main(sys.argv[1])
