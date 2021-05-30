# covidsims
Simulations of COVID-19 dynamics in Brazil

To run the preprocessing pipeline:

python preprocess.py ..\configs\T01\SP\preprocess_T01_SP.cfg

Config file encodes:

PARAM_SOURCEPATH  : directory where the raw datafile is stored

PARAM_TARGETPATH  : directory where results will be stored

PARAM_DATAFILE    : the name of the raw datafile

PARAM_TERRITORY   : list of areas that compose the territory under analysis (i.e., territorial units)

PARAM_POPSIZES    : dictionary of population sizes per territorial unit

PARAM_OUTCOMES    : list of (p, mu, rsd) tuples, with 
                             p as the share of cases with some outcome category (e.g., cases with mild to moderate progression)
                             mu is the mean recovery time of cases (in days)
                             rsd is the relative standard deviation of mu (see https://en.wikipedia.org/wiki/Coefficient_of_variation)

PARAM_MA_WINDOW   : the number of days to consider in the moving average

PARAM_CORE_MODEL  : Peddireddy of IB-forward. See comments in sharedDefs.py/createBoL

PARAM_MASK_ERRORS : True will allow adjustments in the R(t) estimator to avoid negative values

The resulting data can be visualised with Excel; a template is available at ..\assets\Helper_SP.xlsx
