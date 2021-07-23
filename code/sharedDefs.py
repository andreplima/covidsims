import os
import pickle
import codecs
import numpy as np

from copy         import copy
from random       import seed, random
from datetime     import datetime, timedelta
from collections  import OrderedDict, defaultdict
from configparser import RawConfigParser

ECO_SEED = 23
ECO_PRECISION = 1E-9
ECO_DATETIME_FMT = '%Y%m%d%H%M%S' # used in logging
ECO_RAWDATEFMT   = '%Y-%m-%d'     # used in file/memory operations
ECO_FIELDSEP     = ','

ECO_SUSCEPTIBLE = 'S'
ECO_INFECTIOUS  = 'I'
ECO_RECOVERED   = 'R'
ECO_DECEASED    = 'D'
ECO_CONFIRMED   = 'C'

ECO_ROULETTE_SIZE = 100

#-----------------------------------------------------------------------------------------------------------
# General purpose definitions - I/O helper functions
#-----------------------------------------------------------------------------------------------------------

LogBuffer = [] # buffer where all tsprint messages are stored
ECO_DATETIME_FMT = '%Y%m%d%H%M%S'

def stimestamp():
  return(datetime.now().strftime(ECO_DATETIME_FMT))

def stimediff(finishTs, startTs):
  return str(datetime.strptime(finishTs, ECO_DATETIME_FMT) - datetime.strptime(startTs, ECO_DATETIME_FMT))

def tsprint(msg, verbose=True):
  buffer = '[{0}] {1}'.format(stimestamp(), msg)
  if(verbose):
    print(buffer)
  LogBuffer.append(buffer)

def resetLog():
  LogBuffer = []

def saveLog(filename):
  saveAsText('\n'.join(LogBuffer), filename)

def serialise(obj, name):
  f = open(name + '.pkl', 'wb')
  p = pickle.Pickler(f)
  p.fast = True
  p.dump(obj)
  f.close()
  p.clear_memo()

def deserialise(name):
  f = open(name + '.pkl', 'rb')
  p = pickle.Unpickler(f)
  obj = p.load()
  f.close()
  return obj

def file2List(filename, separator = ',', erase = '"', _encoding = 'utf-8'):

  contents = []
  f = codecs.open(filename, 'r', encoding=_encoding)
  if(len(erase) > 0):
    for buffer in f:
      contents.append(buffer.replace(erase, '').strip().split(separator))
  else:
    for buffer in f:
      contents.append(buffer.strip().split(separator))
  f.close()

  return(contents)

def dict2text(d, header, mask = '{0}\t{1}'):

  content = [mask.format(*header)]

  for key in sorted(d):
    content.append(mask.format(key, d[key]))

  return '\n'.join(content)

def saveAsText(content, filename, _encoding='utf-8'):
  f = codecs.open(filename, 'w', encoding=_encoding)
  f.write(content)
  f.close()

def getMountedOn():

  if('PARAM_MOUNTEDON' in os.environ):
    res = os.environ['PARAM_MOUNTEDON'] + os.sep
  else:
    res = os.getcwd().split(os.sep)[-0] + os.sep

  return res

#-------------------------------------------------------------------------------------------------------------------------------------------
# General purpose definitions - interface to handle parameter files
#-------------------------------------------------------------------------------------------------------------------------------------------

# Essay Parameters hashtable
EssayParameters = {}

def setupEssayConfig(configFile = ''):

  # defines default values for some configuration parameters
  setEssayParameter('ESSAY_ESSAYID',  'None')
  setEssayParameter('ESSAY_CONFIGID', 'None')
  setEssayParameter('ESSAY_SCENARIO', 'None')
  setEssayParameter('ESSAY_RUNS',     '1')

  # overrides default values with user-defined configuration
  loadEssayConfig(configFile)

  return listEssayConfig()

def setEssayParameter(param, value):
  """
  Purpose: sets the value of a specific parameter
  Arguments:
  - param: string that identifies the parameter
  - value: its new value
    Premises:
    1) When using inside python code, declare value as string, independently of its true type.
       Example: 'True', '0.32', 'Rastrigin, normalised'
    2) When using parameters in Config files, declare value as if it was a string, but without the enclosing ''.
       Example: True, 0.32, Rastrigin, only Reproduction
  Returns: None
  """

  so_param = param.upper()

  # boolean-valued parameters
  if(so_param in ['PARAM_MASK_ERRORS']):

    so_value = eval(value[0]) if isinstance(value, list) else bool(value)

  # integer-valued parameters
  elif(so_param in ['ESSAY_RUNS', 'PARAM_MAXCORES', 'PARAM_MA_WINDOW']):

    so_value = eval(value[0])

  # floating-point-valued parameters
  elif(so_param in ['PARAM_new']):

    so_value = float(eval(value[0]))

  # parameters that requires eval expansion
  elif(so_param in ['PARAM_SOURCEPATH', 'PARAM_TARGETPATH', 'PARAM_TERRITORY', 'PARAM_POPSIZES',
                    'PARAM_OUTCOMES', 'PARAM_DATAFIELDS']):

    so_value = value

  # parameters that represent text
  else:

    so_value = value[0]

  EssayParameters[so_param] = so_value

def getEssayParameter(param):
  return EssayParameters[param.upper()]

def overrideEssayParameter(param):

  if(param in os.environ):
    param_value = os.environ[param]
    tsprint('-- option {0} replaced from {1} to {2} (environment variable setting)'.format(param,
                                                                                           getEssayParameter(param),
                                                                                           param_value))
    setEssayParameter(param, [str(param_value)])

  return getEssayParameter(param)

class OrderedMultisetDict(OrderedDict):

  def __setitem__(self, key, value):

    try:
      item = self.__getitem__(key)
    except KeyError:
      super(OrderedMultisetDict, self).__setitem__(key, value)
      return

    if isinstance(value, list):
      item.extend(value)
    else:
      item.append(value)

    super(OrderedMultisetDict, self).__setitem__(key, item)

def loadEssayConfig(configFile):

  """
  Purpose: loads essay configuration coded in a essay parameters file
  Arguments:
  - configFile: name and path of the configuration file
  Returns: None, but EssayParameters dictionary is updated
  """

  if(len(configFile) > 0):

    if(os.path.exists(configFile)):

      # initialises the config parser and set a custom dictionary in order to allow multiple entries
      # of a same key (example: several instances of GA_ESSAY_ALLELE
      config = RawConfigParser(dict_type = OrderedMultisetDict)
      config.read(configFile)

      # loads parameters codified in the ESSAY section
      for param in config.options('ESSAY'):
        setEssayParameter(param, config.get('ESSAY', param))

      # loads parameters codified in the PROBLEM section
      for param in config.options('PROBLEM'):
        setEssayParameter(param, config.get('PROBLEM', param))

      # expands parameter values that requires evaluation
      # parameters that may occur once, and hold lists or tuples
      if('PARAM_SOURCEPATH' in EssayParameters):
        EssayParameters['PARAM_SOURCEPATH']  = eval(EssayParameters['PARAM_SOURCEPATH'][0])

      if('PARAM_TARGETPATH' in EssayParameters):
        EssayParameters['PARAM_TARGETPATH']  = eval(EssayParameters['PARAM_TARGETPATH'][0])

      if('PARAM_DATAFIELDS' in EssayParameters):
        EssayParameters['PARAM_DATAFIELDS']  = eval(EssayParameters['PARAM_DATAFIELDS'][0])

      if('PARAM_TERRITORY' in EssayParameters):
        EssayParameters['PARAM_TERRITORY']  = eval(EssayParameters['PARAM_TERRITORY'][0])

      if('PARAM_POPSIZES' in EssayParameters):
        EssayParameters['PARAM_POPSIZES']  = eval(EssayParameters['PARAM_POPSIZES'][0])

      if('PARAM_OUTCOMES' in EssayParameters):
        EssayParameters['PARAM_OUTCOMES']  = eval(EssayParameters['PARAM_OUTCOMES'][0])

      # checks if configuration is ok
      (check, errors) = checkEssayConfig(configFile)
      if(not check):
        print(errors)
        exit(1)

    else:

      print('*** Warning: Configuration file [{1}] was not found'.format(configFile))

def checkEssayConfig(configFile):

  check = True
  errors = []
  errorMsg = ""

  # insert criteria below
  if(EssayParameters['ESSAY_ESSAYID'] not in EssayParameters['ESSAY_SCENARIO']):
    check = False
    param_name = 'ESSAY_ESSAYID'
    restriction = 'be part of the ESSAY_SCENARIO identification'
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if(EssayParameters['ESSAY_CONFIGID'] not in EssayParameters['ESSAY_SCENARIO']):
    check = False
    param_name = 'ESSAY_CONFIGID'
    restriction = 'be part of the ESSAY_SCENARIO identification'
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if(EssayParameters['ESSAY_CONFIGID'].lower() not in configFile.lower()):
    check = False
    param_name = 'ESSAY_CONFIGID'
    restriction = 'be part of the config filename'
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if(EssayParameters['PARAM_MA_WINDOW'] < 1):
    check = False
    param_name = 'PARAM_MA_WINDOW'
    restriction = 'be larger than zero'
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  opts = ['Peddireddy', 'IB-forward']
  if(EssayParameters['PARAM_CORE_MODEL'] not in opts):
    check = False
    param_name = 'PARAM_CORE_MODEL'
    restriction = 'be one of {0}'.format(opts)
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  # summarises errors found
  if(len(errors) > 0):
    separator = "=============================================================================================================================\n"
    errorMsg = separator
    for i in range(0, len(errors)):
      errorMsg = errorMsg + errors[i]
    errorMsg = errorMsg + separator

  return(check, errorMsg)

# recovers the current essay configuration
def listEssayConfig():

  res = ''
  for e in sorted(EssayParameters.items()):
    res = res + "{0} : {1} (as {2})\n".format(e[0], e[1], type(e[1]))

  return res

#-------------------------------------------------------------------------------------------------------------------------------------------
# Problem specific definitions - preprocessing surveillance data
#-------------------------------------------------------------------------------------------------------------------------------------------

def rint(val):
  return int(round(val, 0))

def loadSourceData(sourcepath, filename, fields, territory, popsizes): #xxx use the header to locate the fields

  # maps the fields in the raw data that will compose the source data
  content = file2List(os.path.join(*sourcepath, filename))
  header  = content[0]
  sourceFields  = [header.index(field) for field in fields]

  #- order of the sourceFields:
  #  0  regiao
  #  1  estado
  #  2  municipio
  #  3  data
  #  4  casosNovos
  #  5  obitosNovos

  # determines the data types into which source data fields must be cast
  fieldTypes    = {3: ECO_RAWDATEFMT, 4: 'int', 5: 'int'}

  # parses the territory data into its component areas (i.e., territorial units)
  areas = [[s.strip() for s in area.split(ECO_FIELDSEP)] for area in territory]

  # converts the raw data into the source data (intermediary format)
  relevantAreas = []
  sourceDataByArea = []
  for e in content[1:]:

    buffer = [e[i] for i in sourceFields]
    for (level0, level1, level2) in areas:

      if((buffer[0] == level0 or level0 == '*') and
         (buffer[1] == level1 or level1 == '*') and
         (buffer[2] == level2 or level2 == '*')):

          for k in range(3):
            if(buffer[k] == ''): buffer[k] = '*'
          #newArea = '{0}, {1}, {2}'.format(buffer[0], buffer[1], buffer[2])
          newArea = '{0}, {1}, {2}'.format(level0, level1, level2)
          relevantAreas.append(newArea)
          buffer[2] = newArea

          for i in fieldTypes:
            if(fieldTypes[i] == 'int'):
              buffer[i] = int(float(buffer[i]))
            elif(fieldTypes[i] == ECO_RAWDATEFMT):
              buffer[i] = datetime.strptime(buffer[i], fieldTypes[i])

          sourceDataByArea.append(buffer[2:])

  # aggregates data from different areas by date of report
  tmpData = defaultdict(lambda: defaultdict(int))
  for (area, date, newCases, newDeaths) in sourceDataByArea:
    tmpData[date][ECO_CONFIRMED] += newCases
    tmpData[date][ECO_DECEASED]  += newDeaths

  (timeline, date2t) = createTimeline(sourceDataByArea)
  sourceData = []
  for date in timeline:
    sourceData.append((territory, date, tmpData[date][ECO_CONFIRMED], tmpData[date][ECO_DECEASED]))

  tsprint('-- {0:8d} cases reported in total.'.format(sum([tmpData[date][ECO_CONFIRMED] for date in timeline])))
  tsprint('-- {0:8d} death reported in total.'.format(sum([tmpData[date][ECO_DECEASED]  for date in timeline])))

  # determines the population residing in the relevant areas of the territory
  N = sum([popsizes[area] for area in set(relevantAreas)])

  return sourceData, N, timeline, date2t

def createTimeline(sourceData):

  # creates the timeline and the reverse dictionary
  timeline = sorted(set([date for (territory, date, newCases, newDeaths) in sourceData]))
  date2t   = {date: t for (t, date) in enumerate(timeline)}

  return (timeline, date2t)

def createBoL(sourceData, timeline, date2t, outcomes, ma_window = 1, coreModel = 'Peddireddy', maskErrors = True):

  # initialises the book of life
  bol = defaultdict(lambda: defaultdict(int))
  for (territory, date, newCases, newDeaths) in sourceData:
    # records cases and deaths reported in the surveillance system
    # (i.e., these variables are measured)
    bol[date][ECO_CONFIRMED] = newCases
    bol[date][ECO_DECEASED]  = newDeaths

  # applies moving average on the reported data, if required
  if(ma_window > 1):
    confirmedAtDate = [bol[date][ECO_CONFIRMED] for date in timeline]
    deceasedAtDate  = [bol[date][ECO_DECEASED]  for date in timeline]
    for date in timeline:
      t = date2t[date] + 1
      bol[date][ECO_CONFIRMED] = rint(np.mean(confirmedAtDate[max(0, t - ma_window):t]))
      bol[date][ECO_DECEASED]  = rint(np.mean( deceasedAtDate[max(0, t - ma_window):t]))

  roulette = {}
  if(coreModel == 'Peddireddy'):

    # completes the book with estimates of recovered cases, using the methodology described in:
    #   A. S. Peddireddy et al., "From 5Vs to 6Cs: Operationalizing Epidemic Data Management
    #   with COVID-19 Surveillance," 2020 IEEE International Conference on Big Data (Big Data),
    #   2020, pp. 1380-1387, doi: 10.1109/BigData50022.2020.9378435. (see Equation 1)

    # It seems relevant to make explicit a number of premises adopted in this reasoning:
    # P0: the adopted model describes the disease dynamics at the population level (macro-model)
    # P1: newly reported cases and deaths have been measured (meaning they are not estimated)
    # P2: new cases are timely reported, meaning that the onset of the disease coincides with
    #     the date of report of a new case. Thus, the actual mean recovery time (i.e., the period
    #     between the onset of the disease and its resolution) coincides with the median reported
    #     recovery time (i.e., the period between case confirmation and its resolution)
    # P3: recovered individuals are not infective (i.e., they stop spreading the disease)
    # P4: all deceased individuals were previously reported as confirmed cases
    # P5: although the available heuristics are probabilistic, the estimation is deterministic
    # P6: estimate of new recovered cases at time t uses measured data prior to t (backward)

    for (territory, date, newCases, newDeaths) in sourceData:

      bol[date][ECO_SUSCEPTIBLE] = -bol[date][ECO_CONFIRMED]

      acc = 0.0
      for (proportionOfCases, meanRecoveryTime, _) in outcomes:
        dt = timedelta(days = meanRecoveryTime)
        acc += proportionOfCases * bol[date - dt][ECO_CONFIRMED]

      bol[date][ECO_RECOVERED]  = rint(acc) - bol[date][ECO_DECEASED]
      bol[date][ECO_INFECTIOUS] = bol[date][ECO_CONFIRMED] - bol[date][ECO_RECOVERED] - bol[date][ECO_DECEASED]

      if(maskErrors and bol[date][ECO_RECOVERED] < 0):
        bol[date][ECO_INFECTIOUS] += bol[date][ECO_RECOVERED]
        bol[date][ECO_RECOVERED]   = 0

  elif(coreModel == 'IB-forward'):

    # completes the book with estimates of recovered cases using an individual-based model
    # this approach is based on the same premises above, except:
    # P0: the adopted model describes the disease dynamics at the individual level (micro-model)
    # P5: estimation follows a probabilistic approach, in-line with the available heuristics
    # P6: new confirmed cases at time t will become recovered cases at later time (forward)

    # builds a roulette from the case outcome stats
    (proportionOfCases, meanRecoveryTime, rsdRecoveryTime) = zip(*outcomes)
    nlcs = len(proportionOfCases) # the number of levels of case severity
    thresholds = [int(sum(proportionOfCases[:k+1] * ECO_ROULETTE_SIZE)) for k in range(nlcs)]
    for pocket in range(ECO_ROULETTE_SIZE + 1):
      k = 0
      while pocket > thresholds[k]: k += 1
      mu = meanRecoveryTime[k]
      sd = mu * rsdRecoveryTime[k]
      roulette[pocket] = int(np.random.normal(mu, sd, 1)[0])

    def spinWheel():
      pocket = np.random.randint(0, ECO_ROULETTE_SIZE)
      return roulette[pocket]

    for (territory, date, newCases, newDeaths) in sourceData:

      bol[date][ECO_SUSCEPTIBLE] = -bol[date][ECO_CONFIRMED]

      # (over)estimates the number of recovered cases (i.e., everyone eventually recovers)
      for _ in range(newCases):
        dt = timedelta(days = spinWheel())
        bol[date + dt][ECO_RECOVERED] += 1

    # adjusts the estimate of recovered cases to account for terminal outcomes (deceased)
    for date in timeline:

      bol[date][ECO_RECOVERED] -= bol[date][ECO_DECEASED]
      bol[date][ECO_INFECTIOUS] = bol[date][ECO_CONFIRMED] - bol[date][ECO_RECOVERED] - bol[date][ECO_DECEASED]

      if(maskErrors and bol[date][ECO_RECOVERED] < 0):
        bol[date][ECO_INFECTIOUS] += bol[date][ECO_RECOVERED]
        bol[date][ECO_RECOVERED]   = 0

  else:
    raise ValueError

  return bol, roulette

def playBoL(bol, N, timeline):

  template = '{0}\t{1}\t{2}\t{3}\t{4}'
  header = template.format('File', 'Date', 'Variable', 'Value', 'Description')
  violations = [header]
  d1 = timedelta(days = 1)

  # creates the accumulated curves
  data = defaultdict(lambda: defaultdict(int))
  accs = {ECO_SUSCEPTIBLE: 0, ECO_INFECTIOUS: 0, ECO_RECOVERED: 0, ECO_DECEASED: 0, ECO_CONFIRMED: 0}
  for date in timeline:

    accs[ECO_DECEASED]   += bol[date][ECO_DECEASED]
    accs[ECO_CONFIRMED]  += bol[date][ECO_CONFIRMED]
    accs[ECO_RECOVERED]  += bol[date][ECO_RECOVERED]

    accs[ECO_INFECTIOUS]  = accs[ECO_CONFIRMED] - accs[ECO_RECOVERED] - accs[ECO_DECEASED]
    accs[ECO_SUSCEPTIBLE] = N - accs[ECO_CONFIRMED]

    data[date] = copy(accs)

    # checks the presence of violations in the generated data
    if(bol[date][ECO_DECEASED] < 0):
      violations.append(template.format('daily_changes',
                                        date.strftime(ECO_RAWDATEFMT),
                                        '∆D(t)',
                                        bol[date][ECO_DECEASED],
                                        'must be larger than zero'))

    if(bol[date][ECO_CONFIRMED] < 0):
      violations.append(template.format('daily_changes',
                                        date.strftime(ECO_RAWDATEFMT),
                                        '∆C(t)',
                                        bol[date][ECO_CONFIRMED],
                                        'must be larger than zero'))

    if(bol[date][ECO_RECOVERED] < 0):
      violations.append(template.format('daily_changes',
                                        date.strftime(ECO_RAWDATEFMT),
                                        '∆R(t)',
                                        bol[date][ECO_RECOVERED],
                                        'must be larger than zero'))

    if(bol[date][ECO_INFECTIOUS] != bol[date][ECO_CONFIRMED] - bol[date][ECO_RECOVERED] - bol[date][ECO_DECEASED]):
      violations.append(template.format('daily_changes',
                                        date.strftime(ECO_RAWDATEFMT),
                                        '∆I(t)',
                                        bol[date][ECO_INFECTIOUS],
                                        'must be equal to ∆C(t) - ∆R(t) - ∆D(t)'))

    if(bol[date][ECO_RECOVERED] != data[date][ECO_RECOVERED] - data[date - d1][ECO_RECOVERED]):
      violations.append(template.format('daily_change',
                                        date.strftime(ECO_RAWDATEFMT),
                                        '∆R(t)',
                                        bol[date][ECO_RECOVERED],
                                        'must be equal to R(t) - R(t-1)'))

    if(bol[date][ECO_INFECTIOUS] != data[date][ECO_INFECTIOUS] - data[date - d1][ECO_INFECTIOUS]):
      violations.append(template.format('daily_change',
                                        date.strftime(ECO_RAWDATEFMT),
                                        '∆I(t)',
                                        data[date][ECO_INFECTIOUS],
                                        'must be equal to I(t) - I(t-1)'))

    if(data[date][ECO_RECOVERED] < 0):
      violations.append(template.format('surveillance',
                                        date.strftime(ECO_RAWDATEFMT),
                                        'R(t)',
                                        data[date][ECO_RECOVERED],
                                        'must be larger than zero'))

    if(data[date][ECO_INFECTIOUS] < 0):
      violations.append(template.format('surveillance',
                                        date.strftime(ECO_RAWDATEFMT),
                                        'I(t)',
                                        data[date][ECO_INFECTIOUS],
                                        'must be larger than zero'))

  return (data, violations)

def bol2content(territory, bol, N, timeline, date2t, derivatives = False):

  # converts content to textual format
  buffer_mask = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}'
  if(derivatives):
    header = buffer_mask.format('Territory', 'Date', 't', 'N', '∆S(t)', '∆C(t)', '∆I(t)', '∆R(t)', '∆D(t)')
  else:
    header = buffer_mask.format('Territory', 'Date', 't', 'N', 'S(t)', 'C(t)', 'I(t)', 'R(t)', 'D(t)')

  content = [header]
  for date in timeline:

    buffer = buffer_mask.format(territory,
                                date.strftime(ECO_RAWDATEFMT),
                                date2t[date],
                                N,
                                bol[date][ECO_SUSCEPTIBLE],
                                bol[date][ECO_CONFIRMED],
                                bol[date][ECO_INFECTIOUS],
                                bol[date][ECO_RECOVERED],
                                bol[date][ECO_DECEASED])
    content.append(buffer)

  return '\n'.join(content)

#-------------------------------------------------------------------------------------------------------------------------------------------
# Problem specific definitions - metrics to evaluate the quality of predictions used in:
#  Parmezan, A. R. S., Souza, V. M., & Batista, G. E. (2019). Evaluation of statistical and machine
#  learning models for time series prediction: Identifying the state-of-the-art and the best conditions
#  for the use of each model. Information Sciences, 484, 302-337.
#-------------------------------------------------------------------------------------------------------------------------------------------

def mse(ts_te, ts_pr):         # mean squared error
  score_mse = np.mean([(ts_te[t] - ts_pr[t]) ** 2 for t in range(len(ts_pr))])
  return score_mse

def tu(ts_te, ts_pr, last):    # Theil's U
  num = 0.0
  den = 0.0
  for t in range(len(ts_pr)):
    num += (ts_te[t] - ts_pr[t]) ** 2
    den += (ts_te[t] - last)     ** 2
    last =  ts_te[t]
  if(den == 0.0): den = ECO_PRECISION
  score_tu = num/den
  return score_tu

def pocid(ts_te, ts_pr, last): # prediction of change in direction (POCID)
  (last_te, last_pr) = (last, last)
  acc = 0.0
  for t in range(len(ts_pr)):
    acc += 1 if (ts_te[t] - last_te) * (ts_pr[t] - last_pr) > 0 else 0
    (last_te, last_pr) = (ts_te[t], ts_pr[t])
  score_pocid = 100 * acc/len(ts_pr)
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


