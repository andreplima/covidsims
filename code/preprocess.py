"""
  This script uses a methodology described in [1] to derive SIRD-like epidemiological surveillance data from a
  dataset of daily records of new cases and new deaths. The methodology is further detailed in [2].

  [1] https://www.viser.com.br/covid-19/sp-covid-info-tracker
  [2] A. S. Peddireddy et al., "From 5Vs to 6Cs: Operationalizing Epidemic Data Management with COVID-19 Surveillance,"
      IEEE International Conference on Big Data (Big Data), 2020, pp. 1380-1387,
      doi: 10.1109/BigData50022.2020.9378435. (see Equation 1)

  See note: "para aferir A(t) e Rec(t) tal como apresentada na plataforma online, foi considerada a Metodologia 2".
  translation: "The online platform assesses A(t) and Rec(t) according to Methodology 2",
  with A(t)   corresponding to the number of active cases (infective individuals) at time t,
   and Rec(t) corresponding to the total number of recovered cases at time t.

  Input data (the format of the input data file):
    regiao, estado, municipio, date,
    populacao, casosAcumulado, casosNovos, obitosAcumulado, obitosNovos, Recuperadosnovos, emAcompanhamentoNovos

  Source data (i.e., data that is actually used to derive the surveillance data):
    territory, date,
    newCases, newDeaths

  Processed data:
    territory, date,
    S(t) :- number of susceptible individuals    -- estimated as N - I(t) - R(t) - D(t)
    I(t) :- number of active cases at time t     -- estimated as proposed in [1]
    R(t) :- number of recovered cases at time t  -- estimated as proposed in [1]
    D(t) :- number of deaths occurred at time t  -- measured by agents in charge of surveillance
"""

import os
import sys
import numpy as np
import sharedDefs as ud

from os         import listdir, makedirs, remove
from os.path    import join, isfile, isdir, exists
from random     import seed

from sharedDefs import ECO_SEED
from sharedDefs import setupEssayConfig, getEssayParameter, setEssayParameter, overrideEssayParameter
from sharedDefs import getMountedOn, serialise, saveAsText, stimestamp, tsprint, saveLog
from sharedDefs import loadSourceData, createTimeline, createBoL, bol2content, playBoL

def main(configFile):

  ud.LogBuffer = []

  # parses the config file
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

  # recovers parameters related to the problem instance
  param_sourcepath = getEssayParameter('PARAM_SOURCEPATH')
  param_targetpath = getEssayParameter('PARAM_TARGETPATH')
  param_datafile   = getEssayParameter('PARAM_DATAFILE')
  param_territory  = getEssayParameter('PARAM_TERRITORY')
  param_popsizes   = getEssayParameter('PARAM_POPSIZES')
  param_outcomes   = getEssayParameter('PARAM_OUTCOMES')
  param_ma_window  = getEssayParameter('PARAM_MA_WINDOW')

  # ensures the essay slot (where some log files will be created) is available
  essay_beginning_ts = stimestamp()
  slot  = join('..', 'essays', essayid, configid, essay_beginning_ts)
  if(not exists(slot)): makedirs(slot)

  # adjusts the output directory to account for essay and config IDs
  param_targetpath += [essayid, configid]

  # ensures the folder where results will be saved is available and empty
  if(exists(join(*param_targetpath))):
    for f in listdir(join(*param_targetpath)):
      remove(join(*param_targetpath, f))
  else:
    makedirs(join(*param_targetpath))

  # initialises the random number generator
  seed(ECO_SEED)

  #---------------------------------------------------------------------------------------------
  # This is where the job is done; the rest is boilerpate
  #---------------------------------------------------------------------------------------------

  # loads the dataset
  tsprint('Loading raw data')
  (sourceData, N, timeline, date2t) = loadSourceData(param_sourcepath, param_datafile, param_territory, param_popsizes)
  tsprint('-- {0} records have been loaded.'.format(len(sourceData)))
  print(sourceData[0])
  print(sourceData[-1])

  # creates a "book of life", which is to be understood as a daily record of who will be sick,
  # and recover or die, from a reference point in time that preceeds the initial reported event
  print()
  tsprint('Creating the Book of Life')
  bol = createBoL(sourceData, timeline, date2t, param_outcomes, param_ma_window)
  tsprint('-- {0} records have been created.'.format(len(bol)))

  # simulates the disease dynamics based on the book of life
  # (also computes general epidemiological stats, see [1])
  print()
  tsprint('Playing the Book of Life forward')
  (data, stats) = playBoL(bol, timeline, N)
  tsprint('-- {0} records have been created.'.format(len(data)))

  # saves the results
  print()
  tsprint('Saving the results')
  serialise(sourceData, join(*param_targetpath, 'sourceData'))
  serialise(dict(bol),  join(*param_targetpath, 'bol'))
  serialise(data, join(*param_targetpath, 'data'))
  saveAsText(bol2content(param_territory, bol,  N, timeline, date2t, True), join(*param_targetpath, 'daily_changes.csv'))
  saveAsText(bol2content(param_territory, data, N, timeline, date2t),       join(*param_targetpath, 'surveillance.csv'))

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
