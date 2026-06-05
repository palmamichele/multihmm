import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler





def perform_PCA(D, v_t=0.9):
    """
    Given Data matrix, and explained variance threshold, compute k-approximation of eigendecomposition
    """
    pca = PCA()
    scaler = StandardScaler()  ##standardize the data
    standardized_data = scaler.fit_transform(D) #column-wise standardization
    pca.fit(standardized_data)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)  # cumulative sum

    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.title('Cumulative Explained Variance by Principal Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.savefig("kPCA_cumulative.png")
    #plt.show()
    #print("Eigenvalues (Explained Variance):", pca.explained_variance_) 
    #print("Cumulative Explained Variance:", cumulative_variance)
    k = np.argmax(cumulative_variance >= v_t) + 1  # +1 because index starts from 0
    #minimum number of components that first reaches the thresh of  variance
    print("PCA k=", k)
    return k


def state_discretization(log_returns, delta, z_min, z_max):
    """
    Discretizes the log-return process according to Changepointdynamicsforfinancialdata: anindexedMarkovchainapproach by 
    DAmico, Lika, Petroni.
    
    Parameters:
    log_returns: Sequence of log-returns.
    delta (float): Grid amplitude of the discrete state space.
    z_min (int): Minimum index for discretization.
    z_max (int): Maximum index for discretization.
    
    Returns:
    np.ndarray: Discretized state sequence J_n.
    """
    def M(R):
        i = np.floor(R / delta + 0.5) 
        return np.clip(i, z_min, z_max) * delta 
    
    discrete_returns = np.array([M(R) for R in log_returns])
    return discrete_returns

def M(r, delta, z_min, z_max):
    """
    Discretizes the log-return process according to Change point dynamics for financial data by 
    D'Amico, Lika, Petroni.
    
    Parameters:
    r: Sequence of inputs to be discretized (R_n).
    delta (float): Grid amplitude of the discrete state space.
    z_min (int): Minimum index for discretization.
    z_max (int): Maximum index for discretization.
    
    Returns:
    np.ndarray: Discretized state sequence (J_n).

    """
    z = np.floor(r / delta + 0.5)
    z = np.clip(z, z_min, z_max)
    return delta * z



def apply_global_mapping(column, mapping):
    return np.array([mapping[v] for v in column]).astype(int)

def get_symbols_for_discretized(stock_discretized_returns):
    unique_vals = np.unique(stock_discretized_returns) #map stock i discretized returns to 0..M-1 symbols (required by hmm implementation)
    mapping = {v:i for i,v in enumerate(unique_vals)}
    inverse_mapping = {i:v for v,i in mapping.items()}
    obs_int = np.array([mapping[v] for v in stock_discretized_returns])
    obs_int=obs_int.astype(int)
    n_obs = np.max(obs_int) + 1  # number of distinct emission symbols
    X_counts = np.zeros((len(obs_int), n_obs), dtype=int)
    X_counts[np.arange(len(obs_int)), obs_int] = 1
    return mapping, inverse_mapping, X_counts



def percentage_rsmd(matrix1, matrix2):
    """
    Calculate percentage Root Mean Square Deviation (%RSMD) between two vectors of same size
    """
    n = matrix1.shape
    a1 = np.sqrt(np.sum((matrix1 - matrix2) ** 2)/(n))
    a2 = (n)/np.sum(matrix2)
    return (a1 * a2) * 100


def percentage_mad(matrix1, matrix2):
    """
    Calculate percentage Mean Absolute Deviation (%MAD)
    """
    numerator = np.sum(np.abs(matrix1 - matrix2))
    denominator = np.sum(matrix2)
    return (numerator / denominator) * 100


def hidden_similarities(hmm):
    #more robust algo for state removal (to do )

    res = ""
    states_tbd = set()  # collect states to delete
    emission_matrix = hmm.emissionprob_
    K =len(emission_matrix)  #number of hidden states 
    rsmd_matrix = np.zeros((K, K))
    pmad_matrix = np.zeros((K, K))
    

    for i in range(K):
        for j in range(i, K):
            rsmd_matrix[i, j] =percentage_rsmd(emission_matrix[i, :], emission_matrix[j, :])
            rsmd_matrix[j, i] = percentage_rsmd(emission_matrix[j, :], emission_matrix[i, :])
            
            pmad_matrix[i, j] = percentage_mad(emission_matrix[i, :], emission_matrix[j, :])
            pmad_matrix[j, i] = percentage_mad(emission_matrix[j, :], emission_matrix[i, :])

            if j!=i and (rsmd_matrix[i, j]<50 or pmad_matrix[i, j]<50):
                res += f"state {i} and {j} are similar rsmd={rsmd_matrix[i, j]}, pmad={pmad_matrix[i, j]}"
                states_tbd.add(j)

    return res, sorted(list(states_tbd)), rsmd_matrix, pmad_matrix