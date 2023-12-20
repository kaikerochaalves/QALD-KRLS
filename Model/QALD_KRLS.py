# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:27:20 2021

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""
# Importing libraries
import pandas as pd
import numpy as np

class QALD_KRLS:
    def __init__(self, sigma = 0.1, nu = 0.1, epsilon1 = 0.1, epsilon2 = 0.1):
        # Define the dictionary of model parameters
        self.parameters = pd.DataFrame(columns = ['Kinv', 'alpha', 'P', 'm', 'Dict'])
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
        # Computing the residual square in the testing phase
        self.ResidualTestPhase = np.array([])
        # Hyperparameters and parameters
        self.sigma = sigma
        # nu, epsilon1, and epsilon2 are thresholds for the sparsification techinique
        self.nu = nu
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
         
    def fit(self, X, y):

        # Compute the number of samples
        n = X.shape[0]
        
        # Initialize the first input-output pair
        x0 = X[0,].reshape(-1,1)
        y0 = y[0]
        
        # Initialize QALD-KRLS
        self.Initialize_QALD_KRLS(x0, y0)

        for k in range(1, n):

            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
                      
            # Update QALD-KRLS
            k_til = self.QALD_KRLS(x, y[k])
            
            # Compute output
            Output = self.parameters.loc[0, 'alpha'].T @ k_til
            
            # Store results
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output )
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(y[k]) - Output )
        return self.OutputTrainingPhase
            
    def predict(self, X):

        for k in range(X.shape[0]):
            
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T

            # Compute k
            k_til = np.array(())
            for ni in range(self.parameters.loc[0, 'Dict'].shape[1]):
                k_til = np.append(k_til, [self.Kernel(self.parameters.loc[0, 'Dict'][:,ni].reshape(-1,1), x)])
            k_til = k_til.reshape(k_til.shape[0],1)
            
            # Compute the output
            Output = self.parameters.loc[0, 'alpha'].T @ k_til
            
            # Store the output
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output )

        return self.OutputTestPhase

    def Kernel(self, x1, x2):
        k = np.exp( - ( 1/2 ) * ( (np.linalg.norm( x1 - x2 ))**2 ) / ( self.sigma**2 ) )
        return k
    
    def Initialize_QALD_KRLS(self, x, y):
        k11 = self.Kernel(x, x)
        Kinv = np.ones((1,1)) / ( k11 )
        alpha = np.ones((1,1)) * y / k11
        NewRow = pd.DataFrame([[Kinv, alpha, np.ones((1,1)), 1., x]], columns = ['Kinv', 'alpha', 'P', 'm', 'Dict'])
        self.parameters = pd.concat([self.parameters, NewRow], ignore_index=True)
        # Initialize first output and residual
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, y)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, 0.)
        
    def QALD_KRLS(self, x, y):
        i = 0
        # Compute k
        k = np.array(())
        for ni in range(self.parameters.loc[i, 'Dict'].shape[1]):
            k = np.append(k, [self.Kernel(self.parameters.loc[i, 'Dict'][:,ni].reshape(-1,1), x)])
        k_til = k.reshape(-1,1)
        # Compute a
        a = np.matmul(self.parameters.loc[i, 'Kinv'], k)
        A = a.reshape(-1,1)
        delta = self.Kernel(x, x) - ( k_til.T @ A ).item()
        if delta == 0:
            delta = 1.
        # Searching for the lowest distance between the input and the dictionary inputs
        distance = []
        for ni in range(self.parameters.loc[i, 'Dict'].shape[1]):
            distance.append(np.linalg.norm(self.parameters.loc[i, 'Dict'][:,ni].reshape(-1,1) - x))
        # Find the index of minimum distance
        j = np.argmin(distance)
        # Estimating the error
        EstimatedError = ( y - np.matmul(k_til.T, self.parameters.loc[i, 'alpha']) ).item()
        # Novelty criterion
        if delta > self.nu and distance[j] > self.epsilon1:
            self.parameters.at[i, 'Dict'] = np.hstack([self.parameters.loc[i, 'Dict'], x])
            self.parameters.at[i, 'm'] = self.parameters.loc[i, 'm'] + 1
            # Updating Kinv                      
            self.parameters.at[i, 'Kinv'] = (1/delta)*(self.parameters.loc[i, 'Kinv'] * delta + np.matmul(A, A.T))
            self.parameters.at[i, 'Kinv'] = np.lib.pad(self.parameters.loc[i, 'Kinv'], ((0,1),(0,1)), 'constant', constant_values=(0))
            sizeKinv = self.parameters.loc[i,  'Kinv'].shape[0] - 1
            self.parameters.at[i, 'Kinv'][sizeKinv,sizeKinv] = (1/delta)
            self.parameters.at[i, 'Kinv'][0:sizeKinv,sizeKinv] = (1/delta)*(-a)
            self.parameters.at[i, 'Kinv'][sizeKinv,0:sizeKinv] = (1/delta)*(-a)
            # Updating P
            self.parameters.at[i, 'P'] = np.lib.pad(self.parameters.loc[i, 'P'], ((0,1),(0,1)), 'constant', constant_values=(0))
            sizeP = self.parameters.loc[i,  'P'].shape[0] - 1
            self.parameters.at[i, 'P'][sizeP,sizeP] = 1.
            # Updating alpha
            self.parameters.at[i, 'alpha'] = self.parameters.loc[i, 'alpha'] - ( ( A / delta ) * EstimatedError )
            self.parameters.at[i, 'alpha'] = np.vstack([self.parameters.loc[i, 'alpha'], ( 1 / delta ) * EstimatedError ])
            k_til = np.append(k_til, self.Kernel(x, x).reshape(1,1), axis=0)
        else:
            if distance[j] <= self.epsilon2:
                xi = np.zeros(self.parameters.at[i, 'alpha'].shape)
                xi[j] = 1.
                A = xi
            # Calculating q
            q = np.matmul( self.parameters.loc[i, 'P'], A) / ( 1 + np.matmul(np.matmul(A.T, self.parameters.loc[i, 'P']), A ) )
            # Updating P
            self.parameters.at[i, 'P'] = self.parameters.loc[i, 'P'] - (np.matmul(np.matmul(np.matmul(self.parameters.loc[i, 'P'], A), A.T), self.parameters.loc[i, 'P'])) / ( 1 + np.matmul(np.matmul(A.T, self.parameters.loc[i, 'P']), A))
            # Updating alpha
            self.parameters.at[i, 'alpha'] = self.parameters.loc[i, 'alpha'] + np.matmul(self.parameters.loc[i, 'Kinv'], q) * EstimatedError
        return k_til