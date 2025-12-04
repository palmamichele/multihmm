import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from IPython.display import display
seed=0
np.random.seed(seed)
data_file = "data.xlsx"
stock_prices = pd.read_excel(data_file, index_col=0, parse_dates=True)
stock_prices.index = pd.to_datetime(stock_prices.index) #use the first column as time index 
N = len(stock_prices.iloc[0]) 
stock_prices = stock_prices.iloc[:,:N] #filter number of stocks (in case we set it manually)
days = sorted(set(stock_prices.index.date))
num_days = len(days)

stocks_returns = {}  #hashmap indexed by day, containing an ordered list of intra-day returns of the day for all n-stocks
for day in days:
    day_stock_prices = stock_prices[stock_prices.index.date == day]
    r_day = []
    for (index, row) in enumerate(day_stock_prices.iterrows()):
        if index==0:
           continue #skip to next iteration, as there is no previous stock price
        r_day.append(np.log(stock_prices.iloc[index]/stock_prices.iloc[index-1]))
    formatted_date = day.strftime('%Y-%m-%d')
    stocks_returns[formatted_date]=r_day

all_stock_returns = []  #D
for day in stocks_returns.keys(): 
    for el in stocks_returns[day]:
        all_stock_returns.append(el)

all_stock_returns = np.array(all_stock_returns).reshape(-1, N)  #first column is referred to first stock, second column to second stock, ...


def perform_PCA(D, v_t=0.8):
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
    print("k=", k)
    return k


def state_discretization(log_returns, delta, z_min, z_max):
    """
    Discretizes the log-return process according to Changepointdynamicsforfinancialdata: anindexedMarkovchainapproach by 
    D’Amico, Lika, Petroni.
    
    Parameters:
    log_returns: Sequence of log-returns.
    delta (float): Grid amplitude of the discrete state space.
    z_min (int): Minimum index for discretization.
    z_max (int): Maximum index for discretization.
    
    Returns:
    np.ndarray: Discretized state sequence J_n.
    """
    def M(R):
        i = np.round(R / delta) 
        return np.clip(i, z_min, z_max) * delta 
    
    discrete_returns = np.array([M(R) for R in log_returns])
    return discrete_returns

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

def autocorr(x, max_lag=100):
    """
    Compute autocorrelation of a 1D array up to max_lag
    """
    n = len(x)
    x_mean = np.mean(x)
    x_var = np.var(x)
    autocorrs = np.zeros(max_lag+1)
    for lag in range(max_lag+1):
        autocorrs[lag] = np.sum((x[:n-lag]-x_mean)*(x[lag:]-x_mean)) / ((n-lag)*x_var)
    return autocorrs



K = perform_PCA(all_stock_returns)
all_stock_returns = np.array(all_stock_returns)
delta = 0.0001
z_min = -5
z_max = 5

discretized_returns = np.zeros_like(all_stock_returns)
for col in range(N): 
    column = all_stock_returns[:, col] 
    discretized_returns[:,col]=state_discretization(column, delta, z_min, z_max) 


all_discretized = discretized_returns.flatten()
unique_vals_global = np.unique(all_discretized)
global_mapping = {v:i for i,v in enumerate(unique_vals_global)}
global_inverse_mapping = {i:v for v,i in global_mapping.items()}
n_symbols = len(global_mapping) 
print(n_symbols)

n_trials = 1 
hmm_models = []
X_counts = np.zeros((len(discretized_returns[:,0]), n_symbols), dtype=int)
obs_int = apply_global_mapping(discretized_returns[:,0], global_mapping) #apply (state) index mapping to column 0 of discretized returns
X_counts[np.arange(len(obs_int)), obs_int] = 1 #one hot encoding for hmm: put 1 in the position of index mapping, 0 elsewhere 
#can map back using np.argmax(X_counts, axis=1)
model = hmm.MultinomialHMM(n_components=K, n_iter=100, random_state=seed, n_trials=n_trials)
model.fit(X_counts)
hmm_models.append(model)
common_transition = model.transmat_.copy()
common_initial_distr = model.startprob_.copy()



for i in range(1,N):
    print(f"processing model {i+1}")
    X_counts = np.zeros((len(discretized_returns[:,i]), n_symbols), dtype=int)
    obs_int = apply_global_mapping(discretized_returns[:,i], global_mapping)
    X_counts[np.arange(len(obs_int)), obs_int] = 1
    model = hmm.MultinomialHMM(n_components=K, n_iter=100, random_state=seed, n_trials=n_trials) #model for stock i
    model.startprob_ = common_initial_distr
    model.transmat_ = common_transition
    model.params = 'e'
    model.init_params = 'e'
    model.fit(X_counts)
    hmm_models.append(model)
    X_next, Z = model.sample(n_samples=1)
    hidden_states = model.predict(X_counts)  #uses viterbi algorithm to find most likely sequence of hidden states for obs
    #print("Generated Hidden States:", Z)
    #print("Generated Observations:", X_next[0], "that is emission ", inverse_mapping[np.argmax(X_next[0])])
    #log_likelihood = model.score(X_counts)
    #print("Log-likelihood of the sequence:", log_likelihood)


n_sim = 50000  # number of returns to simulate (adjust based on max lag)
simulated_returns = np.zeros((n_sim, N))  # store simulated returns for each stock
max_lag = 100 
autocorr_matrix = np.zeros((max_lag+1, N))

for I in range(N):
    model = hmm_models[I] 
    emission_matrix = model.emissionprob_  #(K x n_symbols)
    rsmd_matrix = np.zeros((K, K))
    pmad_matrix = np.zeros((K, K))
    
    o,z=model.sample(n_samples=n_sim)
    index_returns = np.argmax(o,axis=1)
    discretized_pred_returns = [global_inverse_mapping[index] for index in index_returns]
    simulated_returns[:, i] = discretized_pred_returns

    for i in range(K):
        for j in range(i, K):
            rsmd_matrix[i, j] =percentage_rsmd(emission_matrix[i, :], emission_matrix[j, :])
            rsmd_matrix[j, i] = percentage_rsmd(emission_matrix[j, :], emission_matrix[i, :])
            
            pmad_matrix[i, j] = percentage_mad(emission_matrix[i, :], emission_matrix[j, :])
            pmad_matrix[j, i] = percentage_mad(emission_matrix[j, :], emission_matrix[i, :])

            if j!=i:
                if rsmd_matrix[i, j]<=50 or pmad_matrix[i, j]<=50:
                    print(f"model {I} has not too different probabilities" )


    state_names = [f"P_{k}" for k in range(K)]
    rsmd_df = pd.DataFrame(rsmd_matrix, index=state_names, columns=state_names)
    pmad_df = pd.DataFrame(pmad_matrix, index=state_names, columns=state_names)
    rsmd_df.to_csv("model_"+str(I)+"rsmd_matrix.csv")
    pmad_df.to_csv("model_"+str(I)+"mad_matrix.csv")



squared_returns = simulated_returns ** 2


for i in range(N):
    vals = squared_returns[:, i]
    print(f"Stock {i}: unique values = {np.unique(vals)}, variance = {np.var(vals):.6g}")

#squared_returns = squared_returns+1 #to avoid zero div
for i in range(N):
    try:
        autocorr_matrix[:, i] = autocorr(squared_returns[:, i], max_lag=max_lag)
        autocorr_df = pd.DataFrame(autocorr_matrix[:,i])
        autocorr_df.to_csv("model_"+str(I)+"autocorr.csv")
    except:
        pass
    plt.figure(figsize=(10,6))
    plt.plot(range(0, max_lag+1), autocorr_matrix[:, i], marker='o')
    plt.title("Autocorrelation of Squared Returns for Stock "+str(i))
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.grid(True)
    plt.savefig(f"autocorrelation_plot_{i}.png")
    plt.close()