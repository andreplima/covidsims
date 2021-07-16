"""

  Ideas implemented here follow some ideas from these sources:
  [1] Funk, S., Camacho, A., Kucharski, A. J., Lowe, R., Eggo, R. M., & Edmunds, W. J. (2019).
      Assessing the performance of real-time epidemic forecasts: A case study of Ebola in the
      Western Area region of Sierra Leone, 2014-15. PLoS computational biology, 15(2).

  [2] http://www.legislacao.sp.gov.br/legislacao/dg280202.nsf/5fb5269ed17b47ab83256cfb00501469/35ea1f3341ab9b9c83258577004cd65e

  [3] Parmezan, A. R. S., Souza, V. M., & Batista, G. E. (2019). Evaluation of statistical and machine
      learning models for time series prediction: Identifying the state-of-the-art and the best conditions
      for the use of each model. Information Sciences, 484, 302-337.
"""

import numpy as np

ECO_PRECISION = 1E-6

ECO_ITERATIVE_METHOD = 0
ECO_UPDATE_METHOD    = 1

class TimeSeries:

  def __init__(self, dataset, modelType, modelParams, pmethod, plength = 14):

    # properties defined during instantiation (and unmodified during runtime)
    self.dataset     = dataset    # dataset ~ {id: {S: ts, I: ts, R: ts, D: ts, C: ts}, ...}
    self.modelType   = modelType  # str: 'MA', 'SIRD', etc.
    self.modelParams = modelParams
    self.pmethod     = pmethod    # ECO_ITERATIVE_METHOD or ECO_UPDATE_METHOD
    self.plength     = plength    # int: length of the predictions to be produced
                                  # deciding for a default plength is 2 weeks is based on:
                                  # - predictions beyond this timeframe are no good to guide public policy [1]
                                  # - it has been tested to update/review emergency plans (e.g., Plano São Paulo [2])

    # properties modified during runtime
    self.fitted      = []         # fitted ~ [(fitted model, model assessment), ...]

  def fit(self, seriesType = 'O'):

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
        self.fitted.append(model, metrics)

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
      nd = len(scores)
      ra = 2 * np.pi / nd
      axes = [i * ra for i in range(nd)]
      (X, Y) = ([], [])
      for i in range(nd):
        (r, theta) = (scores[i], axes[i])
        X.append(r * np.cos(theta))
        Y.append(r * np.sin(theta))
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

    return (score_mse, score_tu, score_pocid, score_mcpm)

  def instantiateModel(self, modelType, modelParams):

    if(modelType == 'kNN-TSPi'):
      model = kNN_TSPi(modelParams)
    else:
      raise ValueError('Model {0} has not been implemented.'.format(modelType))

		return model

class kNN_TSPi:

  def __init__(self, modelParams):

    # unpacks the parameters
    (max_p, h) = modelParams

    # properties defined during instantiation (and unmodified during runtime)
    self.max_p  = max_p # upper bound for the number of observations in a season
    self.h      = h     # the length of the predicted series

    # properties modified during runtime
    self.l      = None  # xxx
    self.k      = None  # xxx
    self.series = None  # xxx

  def fit(self, ts_tr):


  def predict(self):


