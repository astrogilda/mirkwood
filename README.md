# mirkwood

Tested in Python3.8.
X_simba, y_simba: fluxes and galaxy properties for samples in the Simba simulation.
X_eagle, y_eagle: fluxes and galaxy properties for samples in the Eagle simulation.
X_tng, y_tng: fluxes and galaxy properties for samples in the Illustris-TNG simulation.

1. Create an Anaconda3 environment with the packages (along with their respective versions) listed in conda_environment.txt.
2. Say the above environment is called 'environ'.
  (a) In `~/anaconda3/envs/tpot/lib/python3.8/site-packages/ngboost/`, replace `ngboost.py` with the one provided.
  (b) In `~/anaconda3/envs/tpot/lib/python3.8/site-packages/ngboost/distns/`, replace `normal.py` with the one provided.
3. Run `main.py` to run mirkwood.
4. Run `make_plots.py` to recreate the figures in the paper.


1. `bootstrap_handler.py`
  1. Contains the BootstrapHandler class for resampling and performing bootstrap training and prediction.
  2. It also includes helper functions for calculating weights and applying inverse transforms.

2. `model_handler.py`
  1. Defines the ModelHandler class for handling various tasks related to the model, such as data transformation, fitting/loading the estimator, and computing prediction bounds and SHAP values.
  2. It utilizes the BootstrapHandler for bootstrapping.

3. `trainpredict_handler.py`
  1. Introduces the TrainPredictHandler class for training and predicting a model using cross-validation, bootstrapping, and parallel computing.
  2. It utilizes both ModelHandler and BootstrapHandler for their respective functionalities.
