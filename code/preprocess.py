"""
  This script uses the methodology described in [1] to derive the disease surveillance data from a dataset
  of new cases and new deaths, aiming to feed a SIRD model.

  [1] https://www.viser.com.br/covid-19/sp-covid-info-tracker
  See note:
  "para aferir A(t) e Rec(t) tal como apresentada na plataforma online, foi considerada a Metodologia 2".
  translation:
  "The online platform assesses A(t) and Rec(t) according to Methodology 2", where A(t) corresponds to the
  number of active cases (infective individuals) at time t, whereas Rec(t) represents the total number of
  recoreved cases at time t.

  Raw data:
    regiao, estado, municipio, data,
    populacao, casosAcumulado, casosNovos, obitosAcumulado, obitosNovos,
    Recuperadosnovos, emAcompanhamentoNovos

  Source data (i.e., data that is actually used to derive the surveillance data):
    territory, date,
    N, newCases, newDeaths

  Resulting data
    territory, date,
    N    :- number of individuals residing at the territory
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
from sharedDefs import loadSourceData, createBoL, bol2content

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
  param_outcomes   = getEssayParameter('PARAM_OUTCOMES')

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
  sourceData = loadSourceData(param_sourcepath, param_rawdata, param_territory, param_popsizes) #xxx moving average?
  tsprint('-- {0} records have been loaded.'.format(len(sourceData)))
  print(sourceData[0])
  print(sourceData[-1])

  # creates the "book of life" #xxx bol is not a good name
  print()
  tsprint('Creating the Book of Life')
  (bol, sbol, N, timeline, date2t) = createBoL(sourceData, param_outcomes)

  # simulates the disease dynamics based on the book of life
  #data = simulateDiseaseDynamics(bol) # also computes general epidemiological stats, se [1]

  # saves the results
  print()
  tsprint('Saving the results')
  serialise(sourceData, join(*param_targetpath, 'sourceData'))
  serialise(dict(bol), join(*param_targetpath, 'bol'))
  serialise(sbol, join(*param_targetpath, 'bol'))
  saveAsText(bol2content(param_territory, sbol, N, timeline, date2t), join(*param_targetpath, 'book_of_life.csv'))

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
