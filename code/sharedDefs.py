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

ECO_ROULETTE_SIZE = 100

ECO_SUSCEPTIBLE = 'S'
ECO_INFECTIOUS  = 'I'
ECO_RECOVERED   = 'R'
ECO_DECEASED    = 'D'

ECO_CONFIRMED   = 'C'

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

def file2List(filename, separator = ',', erase = '"', _encoding = 'iso-8859-1'):

  contents = []
  f = codecs.open(filename, 'r', encoding=_encoding)
  if(len(erase) > 0):
    for buffer in f:
      contents.append(buffer.replace(erase, '').rstrip().split(separator))
  else:
    for buffer in f:
      contents.append(buffer.rstrip().split(separator))
  f.close()

  return(contents)

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
  if(so_param in ['PARAM_new']):

    so_value = eval(value[0]) if isinstance(value, list) else bool(value)

  # integer-valued parameters
  elif(so_param in ['ESSAY_RUNS', 'PARAM_MAXCORES', 'PARAM_MA_WINDOW']):

    so_value = eval(value[0])

  # floating-point-valued parameters
  elif(so_param in ['PARAM_new']):

    so_value = float(eval(value[0]))

  # parameters that requires eval expansion
  elif(so_param in ['PARAM_SOURCEPATH', 'PARAM_TARGETPATH', 'PARAM_TERRITORY', 'PARAM_POPSIZES',
                    'PARAM_OUTCOMES']):

    so_value = value

  # parameters that represent text
  else:

    so_value = value[0]

  EssayParameters[so_param] = so_value

def getEssayParameter(param):
  return EssayParameters[param.upper()]

def overrideEssayParameter(param):

  if(param in os.environ):
    param_value = int(os.environ[param])
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
    #errors.append("Parameter {0} must respect restriction: {1}\n".format('ESSAY_ESSAYID', 'be part of the ESSAY_SCENARIO identification'))
    param_name = 'ESSAY_ESSAYID'
    restriction = 'be part of the ESSAY_SCENARIO identification'
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if(EssayParameters['ESSAY_CONFIGID'] not in EssayParameters['ESSAY_SCENARIO']):
    check = False
    #errors.append("Parameter {0} must respect restriction: {1}\n".format('ESSAY_CONFIGID', 'be part of the ESSAY_SCENARIO identification'))
    param_name = 'ESSAY_CONFIGID'
    restriction = 'be part of the ESSAY_SCENARIO identification'
    errors.append("Parameter {0} (set as {2}) must respect restriction: {1}\n".format(param_name, restriction, EssayParameters[param_name]))

  if(EssayParameters['ESSAY_CONFIGID'].lower() not in configFile.lower()):
    check = False
    #errors.append("Parameter {0} must respect restriction: {1}\n".format('ESSAY_CONFIGID', 'be part of the configuration filename'))
    param_name = 'ESSAY_CONFIGID'
    restriction = 'be part of the config filename'
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

def loadSourceData(sourcepath, filename, territory, popsizes):

  #- field order in the raw data
  #  0  regiao
  #  1  estado
  #  2  municipio
  #  3  data
  #  4  populacao
  #  5  casosAcumulado
  #  6  casosNovos
  #  7  obitosAcumulado
  #  8  obitosNovos
  #  9  Recuperadosnovos
  # 10  emAcompanhamentoNovos


  # determines the fields in the raw data that will compose the source data
  sourceFields  = [0, 1, 2, 3, 6, 8]

  #- field order in the source data
  #  0  regiao
  #  1  estado
  #  2  municipio
  #  3  data
  #  4  casosNovos
  #  5  obitosNovos

  # determines the data types into which source data fields must be converted
  fieldTypes    = {3: ECO_RAWDATEFMT, 4: 'int', 5: 'int'}

  # parses the territory data into its component areas (i.e., territorial units)
  areas = [[s.strip() for s in area.split(ECO_FIELDSEP)] for area in territory]

  # converts the raw data into the source data
  allAreas = []
  sourceData = []
  for e in file2List(os.path.join(*sourcepath, filename)):

    buffer = [e[i] for i in sourceFields]
    for (level0, level1, level2) in areas:

      if((buffer[0] == level0 or level0 == '*') and
         (buffer[1] == level1 or level1 == '*') and
         (buffer[2] == level2 or level2 == '*')):

          for k in range(3):
            if(buffer[k] == ''): buffer[k] = '*'
          newArea = '{0}, {1}, {2}'.format(buffer[0], buffer[1], buffer[2])
          allAreas.append(newArea)
          buffer[2] = newArea

          for i in fieldTypes:
            if(fieldTypes[i] == 'int'):
              buffer[i] = int(float(buffer[i]))
            elif(fieldTypes[i] == ECO_RAWDATEFMT):
              buffer[i] = datetime.strptime(buffer[i], fieldTypes[i])

          sourceData.append(buffer[2:])

  # determines the population residing in the territory
  N = sum([popsizes[area] for area in set(allAreas)])

  return sourceData, N

def createTimeline(sourceData):

  # creates the timeline and the reverse dictionary
  timeline = sorted(set([date for (territory, date, newCases, newDeaths) in sourceData]))
  date2t   = {date: t for (t, date) in enumerate(timeline)}

  return (timeline, date2t)

def createBoL(sourceData, timeline, date2t, outcomes, ma_window = 1):

  # builds the book of life
  bol = defaultdict(lambda: defaultdict(int))
  for (territory, date, newCases, newDeaths) in sourceData:

    # records cases and deaths reported in the surveillance system
    # (i.e., these variables are measured)
    bol[date][ECO_CONFIRMED] = newCases
    bol[date][ECO_DECEASED]  = newDeaths

    # estimates the number of recovered cases, using the methodology described in:
    #   A. S. Peddireddy et al., "From 5Vs to 6Cs: Operationalizing Epidemic Data Management
    #   with COVID-19 Surveillance," 2020 IEEE International Conference on Big Data (Big Data),
    #   2020, pp. 1380-1387, doi: 10.1109/BigData50022.2020.9378435. (see Equation 1)

    # It seems relevant to make explicit a number of premises adopted in this reasoning:
    # P0: cases and deaths ensured by the authority in charge of epidemiological surveillance
    # P1: new cases are timely reported, meaning that the onset of the disease coincides with
    #     the date of report of a new case
    # P2: recovered individuals are assumed to be non-infective (they stop spreading the disease)
    # xxx should be probabilistic, but it is not; will introduce need to replicate and assess

    bol[date][ECO_SUSCEPTIBLE] = -newCases
    bol[date][ECO_INFECTIOUS]  =  newCases

    acc = 0 #xxx reverse time
    for (proportionOfCases, recoveryTime) in outcomes:
      dt = timedelta(days = recoveryTime)
      try:
        acc += proportionOfCases * bol[date - dt][ECO_CONFIRMED]
      except:
        None

    bol[date][ECO_RECOVERED]  = int(acc)

  return bol

def playBoL(bol, timeline, N):

  # creates the accumulated curves
  data = {}
  accs = {ECO_SUSCEPTIBLE: 0, ECO_INFECTIOUS: 0, ECO_RECOVERED: 0, ECO_DECEASED: 0, ECO_CONFIRMED: 0}
  for date in timeline:

    accs[ECO_DECEASED]    += bol[date][ECO_DECEASED]
    accs[ECO_CONFIRMED]   += bol[date][ECO_CONFIRMED]
    accs[ECO_RECOVERED]   += bol[date][ECO_RECOVERED]

    accs[ECO_INFECTIOUS]  = accs[ECO_CONFIRMED] - accs[ECO_RECOVERED] - accs[ECO_DECEASED]
    accs[ECO_SUSCEPTIBLE] = N - accs[ECO_CONFIRMED]

    data[date] = copy(accs)

    # computes additional epidemiological stats
    stats = {}

  return (data, stats)

def bol2content(territory, bol, N, timeline, date2t, derivatives = False):

  # converts content to textual format
  buffer_mask = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}'
  if(derivatives):
    header = buffer_mask.format('Territory', 'Date', 't', 'N', '∆S(t)', '∆I(t)', '∆R(t)', '∆D(t)')
  else:
    header = buffer_mask.format('Territory', 'Date', 't', 'N', 'S(t)', 'I(t)', 'R(t)', 'D(t)')

  content = [header]
  for date in timeline:

    buffer = buffer_mask.format(territory,
                                date.strftime(ECO_RAWDATEFMT),
                                date2t[date],
                                N,
                                bol[date][ECO_SUSCEPTIBLE],
                                bol[date][ECO_INFECTIOUS],
                                bol[date][ECO_RECOVERED],
                                bol[date][ECO_DECEASED])
    content.append(buffer)

  return '\n'.join(content)
