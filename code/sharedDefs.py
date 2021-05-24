import os
import pickle
import codecs
import numpy as np

from random       import seed, random
from datetime     import datetime, timedelta
from collections  import OrderedDict, defaultdict
from configparser import RawConfigParser

ECO_SEED = 23
ECO_PRECISION = 1E-9
ECO_DATETIME_FMT = '%Y%m%d%H%M%S'
ECO_FIELDSEP = ','

ECO_SUSCEPTIBLE = 'S'
ECO_INFECTIOUS  = 'I'
ECO_RECOVERED   = 'R'
ECO_DECEASED    = 'D'

#-----------------------------------------------------------------------------------------------------------
# Timing and I/O helper functions
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
# Parameter files helper functions
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
  elif(so_param in ['ESSAY_RUNS', 'PARAM_MAXCORES']):

    so_value = eval(value[0])

  # floating-point-valued parameters
  elif(so_param in ['PARAM_new']):

    so_value = float(eval(value[0]))

  # parameters that requires eval expansion
  elif(so_param in ['PARAM_SOURCEPATH', 'PARAM_TARGETPATH', 'PARAM_POPSIZES',
                    'PARAM_CASES_P1', 'PARAM_CASES_P2', 'PARAM_CASES_P3']):

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

      if('PARAM_POPSIZES' in EssayParameters):
        EssayParameters['PARAM_POPSIZES']  = eval(EssayParameters['PARAM_POPSIZES'][0])

      if('PARAM_CASES_P1' in EssayParameters):
        EssayParameters['PARAM_CASES_P1']  = eval(EssayParameters['PARAM_CASES_P1'][0])

      if('PARAM_CASES_P2' in EssayParameters):
        EssayParameters['PARAM_CASES_P2']  = eval(EssayParameters['PARAM_CASES_P2'][0])

      if('PARAM_CASES_P3' in EssayParameters):
        EssayParameters['PARAM_CASES_P3']  = eval(EssayParameters['PARAM_CASES_P3'][0])

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
# Problem specific functions
#-------------------------------------------------------------------------------------------------------------------------------------------

def loadSourceData(sourcepath, filename, territory, popsizes):

  (level0, level1, level2) = [s.strip() for s in territory.split(ECO_FIELDSEP)]

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

  sourceFields  = [0, 1, 2, 3, 4, 6, 8]

  #- field order in the source data
  #  0  regiao
  #  1  estado
  #  2  municipio
  #  3  data
  #  4  populacao
  #  5  casosNovos
  #  6  obitosNovos

  fieldTypes    = {3: '%Y-%m-%d', 4: 'int', 5: 'int', 6: 'int'}

  sourceData = []
  for e in file2List(os.path.join(*sourcepath, filename)):
    buffer = [e[i] for i in sourceFields]
    if(buffer[0] == level0 and buffer[1] == level1):
      if(buffer[2] == ''):
         buffer[2] = '*'
      if(buffer[2] == level2):
        buffer[2] = territory
        if(popsizes[territory] > 0):
          buffer[4] = popsizes[territory]
        for i in fieldTypes:
          if(fieldTypes[i] == 'int'):
            buffer[i] = int(float(buffer[i]))
          elif(fieldTypes[i] == '%Y-%m-%d'):
            buffer[i] = datetime.strptime(buffer[i], fieldTypes[i])
        sourceData.append(buffer[2:])

  return sourceData

def createBoL(sourceData, params):

  # builds the roulette from the case severity stats
  (probs, recovery) = zip(*params)
  roulette = [sum(probs[:k+1]) for k in range(len(probs))]

  def playRoulette():
    p = random()
    k = 0
    while p > roulette[k]:
      k += 1
    return recovery[k]

  # builds the book of life
  bol = defaultdict(lambda: defaultdict(int))
  for (territory, date, N, newCases, newDeaths) in sourceData:

    # processes the new cases
    for _ in range(newCases):

      # (1) a new active case is accounted, and ...
      bol[date][ECO_INFECTIOUS] += 1

      # (2) ... after some time, this is resolved, for the better ...
      dt = timedelta(days = playRoulette()) # xxx add normal noise?
      bol[date + dt][ECO_INFECTIOUS] -= 1
      bol[date + dt][ECO_RECOVERED]  += 1 # P1: assumes everyone recovers

    # processes the new deaths
    for _ in range(newDeaths):

      # (3) ... or for the worse.
      bol[date][ECO_RECOVERED] -= 1 # retracts the assumption P1
      bol[date][ECO_DECEASED]  += 1

    # xxx issues with balance

  return bol