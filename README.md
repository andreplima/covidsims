# covidsims - Simulating COVID-19 dynamics in Brazil

To run the preprocessing pipeline:

python preprocess.py ../configs/T01/SP/preprocess_T01_SP.cfg

Config file must encode:
| Parameter | Description |
| ----------- | ----------- |
|`PARAM_SOURCEPATH`|directory where the raw datafile is stored, as a list|
|`PARAM_TARGETPATH`|directory where results will be stored, as a list|
|`PARAM_DATAFILE`|the name of the raw datafile|
|`PARAM_TERRITORY`|list of areas that compose the territory under analysis (i.e., territorial units)|
|`PARAM_POPSIZES`|dictionary of population sizes per territorial unit|
|`PARAM_OUTCOMES`|list of (`p`, `mu`, `rsd`) tuples, with `p` as the share of cases with some outcome category (e.g., cases with mild to moderate progression), `mu` is the mean recovery time of cases (in days), and `rsd` is the [relative standard deviation](https://en.wikipedia.org/wiki/Coefficient_of_variation) of recovery time|
|`PARAM_MA_WINDOW`|the number of days to consider in the moving average of new cases and deaths|
|`PARAM_CORE_MODEL`|`Peddireddy` of `IB-forward`. See comments in ./code/sharedDefs.py/createBoL(...)|
|`PARAM_MASK_ERRORS`|If True, adjustments in the R(t) estimator to avoid negative values will be performed|


Several UTF-8, flat files with tab-separated fields are created:
| File | Description |
| ----------- | ----------- |
|`daily_changes.csv`|File with daily changes of the variables of interest, namely ∆S(t), ∆C(t), ∆I(t), ∆R(t), and ∆D(t)|
|`surveillance.csv`|File with daily status of the variables of interest, namely S(t), C(t), I(t), R(t), and D(t)|
|`violations.csv`|File listing the violations of quality criteria for the generated data. See criteria in ./code/sharedDefs.py/playBoL(...)|
|`roulette.csv`|File describing the roulette employed to assign the recovery time to an arbitrary case|


The resulting data can be visualised with Excel; a template is available at ./assets/Helper_SP.xlsx
