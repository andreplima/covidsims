"""
  This script uses the methodology described in [1] to derive the disease surveillance data from a dataset
  of new cases and new deaths, aiming to feed a SIRD model.

  [1] https://www.viser.com.br/covid-19/sp-covid-info-tracker
  See note "para aferir A(t) e Rec(t) tal como apresentada na plataforma online, foi considerada a Metodologia 2".

  Raw data:
    regiao, estado, municipio, data,
    populacao, casosAcumulado, casosNovos, obitosAcumulado, obitosNovos,
    Recuperadosnovos, emAcompanhamentoNovos

  Source data (i.e., data that is actually used to derive the surveillance data):
    territory, date,
    N, newCases, newDeaths

  Resulting data
    territory, date,
    N    :- number of individuals living in the territory
    S(t) :- number of susceptible individuals    -- estimated as N - I(t) - R(t) - D(t)
    I(t) :- number of active cases at time t     -- estimated as proposed in [1]
    R(t) :- number of recovered cases at time t  -- estimated as proposed in [1]
    D(t) :- number of deaths occurred at time t  -- measured
"""

import os
import sys
import numpy as np
import sharedDefs as ud

from os         import listdir, makedirs, remove
from os.path    import join, isfile, isdir, exists
from random     import seed

from sharedDefs import getMountedOn, ECO_SEED
from sharedDefs import setupEssayConfig, getEssayParameter, setEssayParameter, overrideEssayParameter
from sharedDefs import serialise, saveAsText, stimestamp, tsprint, saveLog
from sharedDefs import loadSourceData, createBoL

def main(configFile):

  ud.LogBuffer = []

  # recovers the of the config file
  tsprint('Running essay with specs recovered from [{0}]\n'.format(configFile))
  if(not isfile(configFile)):
    print('Command line parameter is not a file: {0}'.format(configFile))
    exit(1)

  tsprint('Processing essay configuration file [{0}]\n{1}'.format(configFile, setupEssayConfig(configFile)))

  # recovers attributes that identify the essay
  essayid  = getEssayParameter('ESSAY_ESSAYID')
  configid = getEssayParameter('ESSAY_CONFIGID')
  scenario = getEssayParameter('ESSAY_SCENARIO')
  label    = getEssayParameter('ESSAY_LABEL')
  replicas = getEssayParameter('ESSAY_RUNS')

  # ensures the essay slot (where some log files will be created) is available
  essay_beginning_ts = stimestamp()
  slot  = join('..', 'essays', essayid, configid, essay_beginning_ts)
  if(not exists(slot)): makedirs(slot)

  # recovers parameters related to the problem instance
  param_sourcepath = getEssayParameter('PARAM_SOURCEPATH')
  param_targetpath = getEssayParameter('PARAM_TARGETPATH')
  param_rawdata    = getEssayParameter('PARAM_RAWDATA')
  param_territory  = getEssayParameter('PARAM_TERRITORY')
  param_popsizes   = getEssayParameter('PARAM_POPSIZES')
  param_cases_p1   = getEssayParameter('PARAM_CASES_P1')
  param_cases_p2   = getEssayParameter('PARAM_CASES_P2')
  param_cases_p3   = getEssayParameter('PARAM_CASES_P3')

  # overrides parameters recovered from the config file with environment variables
  param_cases_p1   = overrideEssayParameter('PARAM_CASES_P1')
  param_cases_p2   = overrideEssayParameter('PARAM_CASES_P2')
  param_cases_p3   = overrideEssayParameter('PARAM_CASES_P3')

  # adjusts the output directory to account for essay and config IDs
  param_targetpath += [essayid, configid]

  # ensures the folder where results will be saved is available and empty
  if(exists(join(*param_targetpath))):
    for f in listdir(join(*param_targetpath)):
      remove(join(*param_targetpath, f))
  else:
    makedirs(join(*param_targetpath))

  # initialises the random seed
  seed(ECO_SEED)

  #---------------------------------------------------------------------------------------------
  # This is where the job is done; the rest is boilerpate
  #---------------------------------------------------------------------------------------------

  # loads the dataset
  tsprint('Loading raw data')
  sourceData = loadSourceData(param_sourcepath, param_rawdata, param_territory, param_popsizes)
  tsprint('-- {0} records have been loaded.'.format(len(sourceData)))
  print(sourceData[0])
  print(sourceData[-1])

  #xxx window average?

  # creates the "book of life"
  print()
  tsprint('Creating the Book of Life')
  params = (param_cases_p1, param_cases_p2, param_cases_p3)
  bol  = createBoL(sourceData, params)

  # simulates the disease dynamics based on the book of life
  #data = simulateDiseaseDynamics(bol)

  # saves the results
  #saveAsText(list2content(data), join(*param_targetpath, 'surveillance_data.csv'))

  #---------------------------------------------------------------------------------------------
  # That's it!
  #---------------------------------------------------------------------------------------------

  tsprint('Results and inspection logs were saved in [{0}]'.format(join(*param_targetpath)))
  tsprint('Finished essay with specs recovered from [{0}]\n'.format(configFile))
  saveLog(join(slot, 'config.log'))
  saveLog(join(*param_targetpath, 'config.log'))

  print()
  tsprint('Essay completed.')

if __name__ == "__main__":

  main(sys.argv[1])
