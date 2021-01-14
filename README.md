# mirkwood

Tested in Python3.8.
X_simba, y_simba: fluxes and galaxy properties for samples in the Simba simulation.
X_eagle, y_eagle: fluxes and galaxy properties for samples in the Eagle simulation.
X_tng, y_tng: fluxes and galaxy properties for samples in the Illustris-TNG simulation.

1) Create an Anaconda3 environment with the packages (along with their respective versions) listed in conda_environment.txt.
2) Say the above environment is called 'environ'.
  (a) In `~/anaconda3/envs/tpot/lib/python3.8/site-packages/ngboost/', replace `ngboost.py' with the one provided.
  (b) In `~/anaconda3/envs/tpot/lib/python3.8/site-packages/ngboost/distns/', replace `normal.py' with the one provided.
3) Run `main.py' to run mirkwood.
4) Run `make_plots.py' to recreate the figures in the paper.
