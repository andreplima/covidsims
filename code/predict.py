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
from collections import namedtuple

from sharedDefs import ECO_SEED, ECO_PRECISION
from sharedDefs import setupEssayConfig, getEssayParameter, setEssayParameter, overrideEssayParameter
from sharedDefs import getMountedOn, serialise, saveAsText, stimestamp, tsprint, saveLog
from sharedDefs import file2List

Metrics = namedtuple('Metrics', ['mse', 'tu', 'pocid', 'er', 'mcpm'])

class TimeSeries:

  def __init__(self, dataset, modelType, modelParams, pmethod, plength = 14):

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

    for id in dataset:

      # splits the time series into training and test series
      ts = dataset[id][seriesType]
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

    # defines the metrics to evaluate the quality of predictions used in [3]
    def mse(ts_te, ts_pr):         # mean squared error
      score_mse = np.mean([(ts_te[t] - ts_pr[t]) ** 2 for t in range(self.plength)])
      return score_mse

    def tu(ts_te, ts_pr, last):    # Theil's U
      num = 0.0
      den = 0.0
      for t in range(self.plength):
        num += (ts_te[t] - ts_pr[t]) ** 2
        den += (ts_te[t] - last)     ** 2
        last =  ts_te[t]
      if(den == 0.0): den = ECO_PRECISION
      score_tu = num/den
      return score_tu

    def pocid(ts_te, ts_pr, last): # prediction of change in direction (POCID)
      (last_te, last_pr) = (last, last)
      acc = 0.0
      for t in range(self.plength):
        acc += 1 if (ts_te[t] - last_te) * (ts_pr[t] - last_pr) > 0 else 0
        (last_te, last_pr) = (ts_te[t], ts_pr[t])
      score_pocid = 100 * acc/self.plength
      return score_pocid

    def mcpm(scores):              # multi-criteria performance measure
                                   # computes the area of a polygon defined by the scores

      # projects the scores to points over equiangular axes in R^2
      nd = len(scores)
      ra = 2 * np.pi / nd
      axes = [i * ra for i in range(nd)]
      (X, Y) = ([], [])
      for i in range(nd):
        (r, theta) = (scores[i], axes[i])
        X.append(r * np.cos(theta))
        Y.append(r * np.sin(theta))

      # applies the Shoelace theorem to compute the area of the polygon [4]
      acc = 0
      for k in range(nd):
        next_k = (k+1) % nd
        acc += X[k] * Y[next_k] - X[next_k] * Y[k]
      score_mcpm = acc/2
      return score_mcpm

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
    #    Q is the query subsequence
    (S, Q) = self.generate_subsequences(self.Z, self.l)

    # 2. S_ contains the normalised subsequences in S
    # 3. Q_ corresponds to normalised Q
    (S_,  _)    = zip(*[self.normalise(subseq) for subseq in S])
    (Q_, stats) = self.normalise(Q)

    # 4. D[j] contains the complexity-invariant distance between Q and each subsequence S[j]
    D = self.CID(Q_, S_)

    # 5. selects k subsequences in S_ most similar to Q
    P = self.search_nearest_neighbours(D, self.k)

    # 6. recovers the next h observations of each of the k most similar subsequences in P
    ts_pr_ = self.recover_samples(P, self.Z, self.l, self.h)

    # 7. averages the observations in R_ and denormalises the result
    ts_pr = self.denormalise(ts_pr_, stats)

    return ts_pr

  def generate_subsequences(self, Z, l):
    m  = len(Z)
    n  = m - l + 1                                 # the number of subsequences of Z with length l
    ss = [np.array(Z[j: j + l]) for j in range(n)] # the list of all such subsequences
    S  = ss[0: n - 1]                              # the list of subsequences making up the search space
    Q  = ss[-1]                                    # the query subsequence
    return (S, Q)

  def normalise(self, subseq):
    mu  = np.mean(subseq)
    sd  = np.std(subseq, ddof=1)
    stats = (mu, sd)
    subseq_ = (subseq - mu) / sd
    return (subseq_, stats)

  def CID(self, Q_, S_):

    def CE(subseq):
      return sum([(subseq[i] - subseq[i+1])**2 for i in range(len(subseq) - 1)]) ** .5

    cQ_ = CE(Q_)
    D = []
    for j in range(len(S_)):
      s_ = S_[j]
      ed  = np.linalg.norm(Q_ - s_)
      cs_ = CE(s_)
      cf  = max(cQ_, cs_)/min(cQ_, cs_)
      D.append((j, ed * cf))
    return D

  def search_nearest_neighbours(self, D, k):
    L = [j for (j, dist) in sorted(D, key = lambda e:e[1])]
    return L[0:k]

  def recover_samples(self, P, Z, l, h):
    ss = [np.array(Z[j: j + l + h]) for j in P] # the list of all nearest neighbours
    ss = [subseq for subseq in ss if len(subseq) == (l + h)]
    (ss_, _) = zip(*[self.normalise(subseq) for subseq in ss])
    ts_pr_ = np.mean(ss_, 0)[-h:]
    return ts_pr_

  def denormalise(self, ts_pr_, stats):
    (mu, sd) = stats
    ts_pr = (ts_pr_ * sd) + mu
    return ts_pr


def main(configFile):

  ud.LogBuffer = []

  # recovers an exemplar of time series
  sourcepath = [getMountedOn(), 'Task Stage', 'Task - covidsims', 'covidsims', 'results', 'P01', 'C0']
  filename   = 'fa_constant_level.data'
  #filename   = 'fa_increasing_trend.data'
  #filename   = 'fc_constant_level.data'
  ts = [float(l[0]) for l in file2List(join(*sourcepath, filename))]

  # instantiates the model
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

  tsprint('Process completed')

if __name__ == "__main__":

  main(sys.argv[1])
