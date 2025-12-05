
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

