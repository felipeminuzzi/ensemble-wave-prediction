# Ensemble Ocean Waves Prediction
A package to use deep learning to predict oceanography variables from historical data.

## Explanation of the reports folders:
The simulations are heald for four different buoys locations in the coast of brazil, namely, Rio Grande, Itajai, Santos and Vitoria.
the five different neural networks are applied and an ensemble of the results is developed. 

- v8: first simulation, for testing the code, only buoy location of Santos, 6 days of predictiion.
- full_v0: simulations with 6 hidden layers, predictions of 16 days, activation - tanh
- full_v1: simulation with 3 hidden layers, prediction of 16 days, activation - tanh
- full_v2: simulation with 6 hidden layers, prediction of 16 days, activation - sigmoid
- full_v3: simulations with 6 hidden layers, predictions of 10 days, activation - tanh
- full_v4: simulations with 6 hidden layers, predictions of 10 days, activation - tanh with lstm_future
- full_v5: simulations with 6 hidden layers, predictions of 16 days, activation - tanh with lstm_future
- full_v6: simulations with 6 hidden layers, predictions of 6 days, activation - tanh 
