import numpy as np
import pandas as pd
import os
from math import gcd
from functools import reduce
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


seed=0
np.random.seed(seed)


def save_acf_csv(filename, acf_series, confint, lags, folder_name):
    data = pd.DataFrame({
        "lag": lags,
        "acf": acf_series[1:],
        "lower": confint[1:, 0],
        "upper": confint[1:, 1]
    })

    data.to_csv(
        os.path.join(folder_name, filename),
        index=False
    )

def perform_PCA(D, v_t=0.9, asset_names=None):
    """
    Given Data matrix, and explained variance threshold, compute k-approximation of eigendecomposition
    """
    pca = PCA()
    scaler = StandardScaler()  ##standardize the data
    standardized_data = scaler.fit_transform(D) #column-wise standardization
    
    scores = pca.fit_transform(standardized_data)   

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance) 
    eigenvalues = pca.explained_variance_


    # plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    # plt.title('Cumulative Explained Variance by Principal Components')
    # plt.xlabel('Number of Principal Components')
    # plt.ylabel('Cumulative Explained Variance')
    # plt.grid(True)
    # plt.savefig("kPCA_cumulative.png")
    #plt.show()
    #print("Eigenvalues (Explained Variance):", pca.explained_variance_) 
    #print("Cumulative Explained Variance:", cumulative_variance)
    k = np.argmax(cumulative_variance >= v_t) + 1  # +1 because index starts from 0
    #minimum number of components that first reaches the thresh of  variance
    if asset_names is None:
        asset_names = [f"Asset{i+1}" for i in range(D.shape[1])]
    loadings = pd.DataFrame(
        pca.components_.T,
        index=[f"stock{i+1}" for i in range(len(explained_variance))],
        columns=[f"PC{i+1}" for i in range(len(explained_variance))]
    )

    scores = pd.DataFrame(
        scores,
        columns=[f"PC{i+1}" for i in range(len(explained_variance))]
    )

    return (
        k,
        loadings,
        scores,
        explained_variance,
        cumulative_variance,
        eigenvalues,
    )










def state_discretization(log_returns, delta, z_min, z_max):
    """
    Discretizes the log-return process according to Change point dynamics for financial data: an indexed Markov chain approach by 
    D'Amico, Lika, Petroni.
    
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


def fpt_from_log_returns(log_returns, rho=1.005):
    threshold = np.log(rho)

    n = len(log_returns)
    fpts = []

    for start in range(n): #investment starting time t
        cum = 0.0

        for tau in range(1, n-start):
            #cum = sum(log_returns[start : start + tau])
            cum += log_returns[start + tau - 1]

            if cum >= threshold: #ignore times for which start never hits the target before the sample ends
                fpts.append(tau)
                break

    return np.array(fpts)



def apply_global_mapping(column, mapping):
    return np.array([mapping[v] for v in column])

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



def check_stochastic(P, atol=1e-10):
    P = np.asarray(P, dtype=float)

    if np.any(P < -atol):
        raise ValueError("Matrix contains negative entries.")

    if not np.allclose(P.sum(axis=1), 1.0, atol=atol):
        raise ValueError("Rows do not sum to 1.")

    return P




def js_distance(P, Q, eps=1e-15):

    # add tiny pseudocounts
    P = P + eps
    Q = Q + eps

    # renormalize rows
    P = P / P.sum(axis=1, keepdims=True)
    Q = Q / Q.sum(axis=1, keepdims=True)

    M = 0.5 * (P + Q)

  
    kl_pm = np.sum(P * np.log2(P / M), axis=1)
    kl_qm = np.sum(Q * np.log2(Q / M), axis=1)

    return np.sqrt(0.5 * (kl_pm + kl_qm))


def total_variation(P, Q):
    P = check_stochastic(P)
    Q = check_stochastic(Q)

    # distance for each row
    return 0.5 * np.abs(P - Q).sum(axis=1)

def regime_durations(P):
    return 1 / (1 - np.diag(P))

def is_irreducible(P):
    graph = csr_matrix(P > 0)
    n_components, _ = connected_components(
        graph,
        directed=True,
        connection='strong'
    )
    return n_components == 1



def period(P, max_power=100):
    n = P.shape[0]
    powers = np.eye(n)
    periods = []

    for k in range(1, max_power + 1):
        powers = powers @ P
        if powers[0, 0] > 1e-12:
            periods.append(k)

    return reduce(gcd, periods)


def stationary_distribution(P):
    n = P.shape[0]

    A = P.T - np.eye(n)

    # replace one equation with normalization
    A[-1] = np.ones(n)

    b = np.zeros(n)
    b[-1] = 1

    return np.linalg.solve(A, b)