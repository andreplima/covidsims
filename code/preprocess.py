"""
  This script uses a methodology described in [1] to derive SIRD-like surveillance data from a
  dataset of daily records of new cases and new deaths.

  [1] A. S. Peddireddy et al., "From 5Vs to 6Cs: Operationalizing Epidemic Data Management with COVID-19 Surveillance,"
      IEEE International Conference on Big Data (Big Data), 2020, pp. 1380-1387,
      doi: 10.1109/BigData50022.2020.9378435. (see Equation 1)

  Input data (the fields of the input data file):
    regiao, estado, municipio, date,
    populacao, casosAcumulado, casosNovos, obitosAcumulado, obitosNovos, Recuperadosnovos, emAcompanhamentoNovos

  Source data (i.e., the input data that is actually used to derive the surveillance data):
    territory (= regiao + estado + municipio), date,
    newCases, newDeaths

  Processed data:
    territory, date,
    S(t) :- number of susceptible individuals    -- estimated as N - I(t) - R(t) - D(t)
    C(t) :- number of confirmed casesd at time t -- measured by agents in charge of surveillance
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
from sharedDefs import ECO_SUSCEPTIBLE, ECO_INFECTIOUS, ECO_RECOVERED, ECO_DECEASED, ECO_CONFIRMED
from sharedDefs import setupEssayConfig, getEssayParameter, setEssayParameter, overrideEssayParameter
from sharedDefs import getMountedOn, serialise, saveAsText, stimestamp, tsprint, saveLog
from sharedDefs import loadSourceData, createTimeline, createBoL, bol2content, dict2text, playBoL
from inverse    import reformat
from SIRD       import plotSeries

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
  param_sourcepath  = getEssayParameter('PARAM_SOURCEPATH')
  param_targetpath  = getEssayParameter('PARAM_TARGETPATH')
  param_datafile    = getEssayParameter('PARAM_DATAFILE')
  param_datafields  = getEssayParameter('PARAM_DATAFIELDS')
  param_territory   = getEssayParameter('PARAM_TERRITORY')
  param_popsizes    = getEssayParameter('PARAM_POPSIZES')
  param_outcomes    = getEssayParameter('PARAM_OUTCOMES')
  param_ma_window   = getEssayParameter('PARAM_MA_WINDOW')
  param_core_model  = getEssayParameter('PARAM_CORE_MODEL')
  param_mask_errors = getEssayParameter('PARAM_MASK_ERRORS')

  # overrides parameters recovered from the config file with environment variables
  param_ma_window   = overrideEssayParameter('PARAM_MA_WINDOW')
  param_core_model  = overrideEssayParameter('PARAM_CORE_MODEL')
  param_mask_errors = overrideEssayParameter('PARAM_MASK_ERRORS')

  # ensures the journal slot (where all executions are recorded) is available
  essay_beginning_ts = stimestamp()
  slot  = join('..', 'journal', essayid, configid, essay_beginning_ts)
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
  # This is where the job is actually done; the rest is boilerplate
  #---------------------------------------------------------------------------------------------

  # loads the raw dataset
  tsprint('Loading raw data')
  (sourceData, N, timeline, date2t) = loadSourceData(param_sourcepath, param_datafile,
                                                     param_datafields, param_territory, param_popsizes)
  tsprint('-- {0} records have been loaded.'.format(len(sourceData)))
  tsprint('-- samples:')
  tsprint('   {0}'.format(sourceData[0]))
  tsprint('   {0}'.format(sourceData[-1]))

  # creates a "book of life", which is to be understood as a daily record of who will get sick,
  # and the disease outcome (recover or die), from a reference point in time that preceeds
  # the first reported event
  print()
  tsprint('Creating the Book of Life')
  bol, roulette = createBoL(sourceData, timeline, date2t,
                            param_outcomes, param_ma_window, param_core_model, param_mask_errors)
  tsprint('-- {0} records have been created.'.format(len(bol)))

  # simulates the dynamics of the disease in the territory, based on the book of life
  # (also computes common epidemiological stats, see [1])
  print()
  tsprint('Playing the Book of Life forward')
  (data, violations) = playBoL(bol, N, timeline)
  tsprint('-- {0} records have been created.'.format(len(data)))

  # saves the results
  print()
  tsprint('Saving the results')

  serialise(sourceData, join(*param_targetpath, 'sourceData'))
  serialise(timeline,   join(*param_targetpath, 'timeline'))
  serialise(date2t,     join(*param_targetpath, 'date2t'))
  serialise(dict(bol),  join(*param_targetpath, 'bol'))
  serialise(roulette,   join(*param_targetpath, 'roulette'))
  serialise(dict(data), join(*param_targetpath, 'data'))

  saveAsText('\n'.join(violations),                                         join(*param_targetpath, 'violations.csv'))
  saveAsText(dict2text(roulette, ['Pocket', 'Days']),                       join(*param_targetpath, 'roulette.csv'))
  saveAsText(bol2content(param_territory, bol,  N, timeline, date2t, True), join(*param_targetpath, 'daily_changes.csv'))
  saveAsText(bol2content(param_territory, data, N, timeline, date2t),       join(*param_targetpath, 'surveillance.csv'))

  # plots the results
  reports = reformat(data, timeline) # daily reported values of the SIRD variables (reports[seriesType] = [val, ...])
  T = len(timeline)
  S = reports[ECO_SUSCEPTIBLE]
  I = reports[ECO_INFECTIOUS]
  R = reports[ECO_RECOVERED]
  D = reports[ECO_DECEASED]
  plotSeries(range(T), S, I, R, D, N, "SIRD series from surveillance data", ylog = True)

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
