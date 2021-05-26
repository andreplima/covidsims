# covidsims
Simulations of COVID-19 dynamics in Brazil

To run the preprocessing pipeline:

python preprocess.py ..\configs\T01\SP\preprocess_T01_SP.cfg

PARAM_SOURCEPATH : [getMountedOn(), 'Task Stage', 'Task - covidsims', 'covidsims', 'datasets']
PARAM_TARGETPATH : [getMountedOn(), 'Task Stage', 'Task - covidsims', 'covidsims', 'results']
PARAM_DATAFILE   : dados_diarios_estado_SP.csv
PARAM_TERRITORY  : ['Sudeste, SP, *']
PARAM_POPSIZES   : {'Sudeste, SP, *': 46289333} # https://cidades.ibge.gov.br/brasil/sp/panorama
PARAM_OUTCOMES   : [(0.81, 14, 0.2), (0.14, 28, 0.2), (0.05, 42, 0.2)] # see logbook, reference [17]
PARAM_MA_WINDOW  : 1


The resulting data is saved in ..\results\T01\SP, and can be visualised with Excel; see worksheet in ..\assets\Helper_SP.xlsx
