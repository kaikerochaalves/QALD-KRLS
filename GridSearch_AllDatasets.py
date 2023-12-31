# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 16:46:28 2022

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""

# Importing libraries
import math
import numpy as np
import pandas as pd
import statistics as st
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 

# Feature scaling
#from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Including to the path another fold
import sys

# Including to the path another fold
sys.path.append(r'Model')
sys.path.append(r'Functions')

# Importing the model
from QALD_KRLS import QALD_KRLS

# Importing the library to generate the Mackey Glass time series
from MackeyGlassGenerator import MackeyGlass

# Importing the library to generate the Mackey Glass time series
from NonlinearGenerator import Nonlinear

# Importing the library to generate the Mackey Glass time series
from LorenzAttractorGenerator import Lorenz


def Create_Leg(data, ncols, leg, leg_output = None):
    X = np.array(data[leg*(ncols-1):].reshape(-1,1))
    for i in range(ncols-2,-1,-1):
        X = np.append(X, data[leg*i:leg*i+X.shape[0]].reshape(-1,1), axis = 1)
    X_new = np.array(X[:,-1].reshape(-1,1))
    for col in range(ncols-2,-1,-1):
        X_new = np.append(X_new, X[:,col].reshape(-1,1), axis=1)
    if leg_output == None:
        return X_new
    else:
        y = np.array(data[leg*(ncols-1)+leg_output:].reshape(-1,1))
        return X_new[:y.shape[0],:], y
    

#-----------------------------------------------------------------------------
# Define the search space for the hyperparameters
#-----------------------------------------------------------------------------


# Setting the range of hyperparameters
l_sigma = [0.01, 0.1, 0.5, 1, 10]
l_nu = [0.01, 0.1]
l_epsilon1 = [0.1, 0.3, 0.5]
l_epsilon2 = [0.001, 0.01, 0.1, 0.5]


#-----------------------------------------------------------------------------
# Generating the Mackey-Glass time series
#-----------------------------------------------------------------------------

# The theory
# Mackey-Glass time series refers to the following, delayed differential equation:
    
# dx(t)/dt = ax(t-\tau)/(1 + x(t-\tau)^10) - bx(t)


# Input parameters
a        = 0.2;     # value for a in eq (1)
b        = 0.1;     # value for b in eq (1)
tau      = 17;		# delay constant in eq (1)
x0       = 1.2;		# initial condition: x(t=0)=x0
sample_n = 6000;	# total no. of samples, excluding the given initial condition

# MG = mackey_glass(N, a = a, b = b, c = c, d = d, e = e, initial = initial)
MG = MackeyGlass(a = a, b = b, tau = tau, x0 = x0, sample_n = sample_n)

# Defining the atributes and the target value
X, y = Create_Leg(MG, ncols = 4, leg = 6, leg_output = 85)

# Spliting the data into train and test
X_train, X_test = X[201:3201,:], X[5001:5501,:]
y_train, y_test = y[201:3201,:], y[5001:5501,:]


# Min-max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train, y_train)
#print(scaler.data_max_)
X_test = scaler.transform(X_test)

#-----------------------------------------------------------------------------
# Executing the Grid-Search for the Mackey-Glass time series
#-----------------------------------------------------------------------------

simul = 0
# Creating the DataFrame to store results
columns = ['sigma', 'nu', 'epsilon', 'epsilon2', 'RMSE', 'NDEI', 'MAE']
result = pd.DataFrame(columns = columns)
for sigma in l_sigma:
    for nu in l_nu:
        for epsilon1 in l_epsilon1:
            for epsilon2 in l_epsilon2:
                            
                print(f"sigma = {sigma} and nu = {nu} and epsilon1 = {epsilon1} and epsilon2 = {epsilon2}")
            
                # Initializing the model
                model = QALD_KRLS(sigma = sigma, nu = nu, epsilon1 = epsilon1, epsilon2 = epsilon2)
                # Train the model
                OutputTraining = model.fit(X_train, y_train)
                # Test the model
                OutputTest = model.predict(X_test)
                
                # Calculating the error metrics
                # Compute the Root Mean Square Error
                RMSE = math.sqrt(mean_squared_error(y_test, OutputTest))
                # Compute the Non-Dimensional Error Index
                NDEI= RMSE/st.stdev(y_test.flatten())
                # Compute the Mean Absolute Error
                MAE = mean_absolute_error(y_test, OutputTest)
                
                simul = simul + 1
                print(f'Simulação: {simul}')
                
                NewRow = pd.DataFrame([[sigma, nu, epsilon1, epsilon2, RMSE, NDEI, MAE]], columns = columns)
                result = pd.concat([result, NewRow], ignore_index=True)
        
name = f"GridSearchResults\Hyperparameters Optimization_MackeyGlass.xlsx"
result.to_excel(name)


#-----------------------------------------------------------------------------
# Generating the Nonlinear time series
#-----------------------------------------------------------------------------
    
sample_n = 6000
NTS, u = Nonlinear(sample_n)       

# Defining the atributes and the target value
X, y = Create_Leg(NTS, ncols = 2, leg = 1, leg_output = 1)
X = np.append(X, u[:X.shape[0]].reshape(-1,1), axis = 1)

# Spliting the data into train and test
X_train, X_test = X[2:5002,:], X[5002:5202,:]
y_train, y_test = y[2:5002,:], y[5002:5202,:]

# Min-max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train, y_train)
#print(scaler.data_max_)
X_test = scaler.transform(X_test)

#-----------------------------------------------------------------------------
# Executing the Grid-Search for the Nonlinear time series
#-----------------------------------------------------------------------------

simul = 0
# Creating the DataFrame to store results
columns = ['sigma', 'nu', 'epsilon', 'epsilon2', 'RMSE', 'NDEI', 'MAE']
result = pd.DataFrame(columns = columns)
for sigma in l_sigma:
    for nu in l_nu:
        for epsilon1 in l_epsilon1:
            for epsilon2 in l_epsilon2:
                            
                print(f"sigma = {sigma} and nu = {nu} and epsilon1 = {epsilon1} and epsilon2 = {epsilon2}")
            
                # Initializing the model
                model = QALD_KRLS(sigma = sigma, nu = nu, epsilon1 = epsilon1, epsilon2 = epsilon2)
                # Train the model
                OutputTraining = model.fit(X_train, y_train)
                # Test the model
                OutputTest = model.predict(X_test)
                
                # Calculating the error metrics
                # Compute the Root Mean Square Error
                RMSE = math.sqrt(mean_squared_error(y_test, OutputTest))
                # Compute the Non-Dimensional Error Index
                NDEI= RMSE/st.stdev(y_test.flatten())
                # Compute the Mean Absolute Error
                MAE = mean_absolute_error(y_test, OutputTest)
                
                simul = simul + 1
                print(f'Simulação: {simul}')
                
                NewRow = pd.DataFrame([[sigma, nu, epsilon1, epsilon2, RMSE, NDEI, MAE]], columns = columns)
                result = pd.concat([result, NewRow], ignore_index=True)
        
name = f"GridSearchResults\Hyperparameters Optimization_Nonlinear.xlsx"
result.to_excel(name)


#-----------------------------------------------------------------------------
# Generating the Lorenz Attractor time series
#-----------------------------------------------------------------------------


# Input parameters
x0 = 0.
y0 = 1.
z0 = 1.05
sigma = 10
beta = 2.667
rho=28
num_steps = 10000

# Creating the Lorenz Time Series
x, y, z = Lorenz(x0 = x0, y0 = y0, z0 = z0, sigma = sigma, beta = beta, rho = rho, num_steps = num_steps)

# Defining the atributes and the target value
X = np.concatenate([x[:-1].reshape(-1,1), y[:-1].reshape(-1,1), z[:-1].reshape(-1,1)], axis = 1)
y = x[1:].reshape(-1,1)

# Spliting the data into train and test
X_train, X_test = X[:8000,:], X[8000:,:]
y_train, y_test = y[:8000,:], y[8000:,:]

# Min-max scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train, y_train)
#print(scaler.data_max_)
X_test = scaler.transform(X_test)

#-----------------------------------------------------------------------------
# Executing the Grid-Search for the Lorenz Attractor time series
#-----------------------------------------------------------------------------

simul = 0
# Creating the DataFrame to store results
columns = ['sigma', 'nu', 'epsilon', 'epsilon2', 'RMSE', 'NDEI', 'MAE']
result = pd.DataFrame(columns = columns)
for sigma in l_sigma:
    for nu in l_nu:
        for epsilon1 in l_epsilon1:
            for epsilon2 in l_epsilon2:
                            
                print(f"sigma = {sigma} and nu = {nu} and epsilon1 = {epsilon1} and epsilon2 = {epsilon2}")
            
                # Initializing the model
                model = QALD_KRLS(sigma = sigma, nu = nu, epsilon1 = epsilon1, epsilon2 = epsilon2)
                # Train the model
                OutputTraining = model.fit(X_train, y_train)
                # Test the model
                OutputTest = model.predict(X_test)
                
                # Calculating the error metrics
                # Compute the Root Mean Square Error
                RMSE = math.sqrt(mean_squared_error(y_test, OutputTest))
                # Compute the Non-Dimensional Error Index
                NDEI= RMSE/st.stdev(y_test.flatten())
                # Compute the Mean Absolute Error
                MAE = mean_absolute_error(y_test, OutputTest)
                
                simul = simul + 1
                print(f'Simulação: {simul}')
                
                NewRow = pd.DataFrame([[sigma, nu, epsilon1, epsilon2, RMSE, NDEI, MAE]], columns = columns)
                result = pd.concat([result, NewRow], ignore_index=True)
        
name = f"GridSearchResults\Hyperparameters Optimization_Lorenz.xlsx"
result.to_excel(name)