# QALD-KRLS (combination of ALD-KRLS and QKRLS algorithms)

The QALD-KRLS is a model proposed by Guo et al. [1].

- [QALD-KRLS](https://github.com/kaikerochaalves/QALD-KRLS/blob/fbee73b7bb0cf656134f8b5df1f40c1e9e126584/Model/QALD_KRLS.py) is the QALD-KRLS model.

- [GridSearch_AllDatasets](https://github.com/kaikerochaalves/QALD-KRLS/blob/fbee73b7bb0cf656134f8b5df1f40c1e9e126584/GridSearch_AllDatasets.py) is the file to perform a grid search for all datasets and store the best hyper-parameters.

- [Runtime_AllDatasets](https://github.com/kaikerochaalves/QALD-KRLS/blob/fbee73b7bb0cf656134f8b5df1f40c1e9e126584/Runtime_AllDatasets.py) perform 30 simulations for each dataset and compute the mean runtime and the standard deviation.

- [MackeyGlass](https://github.com/kaikerochaalves/QALD-KRLS/blob/fbee73b7bb0cf656134f8b5df1f40c1e9e126584/MackeyGlass.py) is the script to prepare the Mackey-Glass time series, perform simulations, compute the results and plot the graphics. 

- [Nonlinear](https://github.com/kaikerochaalves/QALD-KRLS/blob/fbee73b7bb0cf656134f8b5df1f40c1e9e126584/Nonlinear.py) is the script to prepare the nonlinear dynamic system identification time series, perform simulations, compute the results and plot the graphics.

- [LorenzAttractor](https://github.com/kaikerochaalves/QALD-KRLS/blob/fbee73b7bb0cf656134f8b5df1f40c1e9e126584/LorenzAttractor.py) is the script to prepare the Lorenz Attractor time series, perform simulations, compute the results and plot the graphics. 

[1] J. Guo, H. Chen, S. Chen, Improved kernel recursive least squares algorithm based online prediction for nonstationary time series, IEEE Signal Processing Letters 27 (2020) 1365–1369. doi:https://doi.org/10.1109/LSP.2020.3011892
