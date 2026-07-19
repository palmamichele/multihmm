import numpy as np
import pandas as pd 
import os
import itertools
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import total_variation, js_distance, regime_durations, is_irreducible, period, stationary_distribution

seed=0
np.random.seed(seed)

data_path = "data/data.xlsx"
out_path = "out"
prices = pd.read_excel(data_path, index_col=0, parse_dates=True)
prices.index = pd.to_datetime(prices.index) #use the first column as time index 
N = len(prices.iloc[0]) 

transitions = []
emissions=[]

for i in range(N):
    folder_name = os.path.join(out_path, f"model_{i}")

    if os.path.isdir(folder_name):
        filename = f"model_{i}trans.csv"
        path = os.path.join(folder_name, filename)

        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            de = pd.read_csv(os.path.join(folder_name, f"model_{i}emissions.csv"), index_col=0)
            #print(df)
            transitions.append((i, df))
            emissions.append((i,de))


#compute the stationary distr per each transition matrix.
stationary_distrs = {}
for model_id, df in transitions:
    P = df.to_numpy(dtype=float)
   
    #regime_durations(P)
    irreducible= is_irreducible(P)

    

    if irreducible:
        print(f"the {model_id}-th chain is irreducible")
        #look for self loop
        if (np.any(np.diag(P) > 0)):
            print(f"via a self loop,  {model_id}-th chain is aperiodic")

            #for ergodic 
            #stationary distr. exact solving 
            stationary_distrs[f"model_{model_id}"] = stationary_distribution(P)

            #approx of stationary distr. 
            P_power = np.linalg.matrix_power(P, 1000)
            #print(P_power)

    else:
        print(f"the {model_id}-th chain is not irreducible")
        print(period(P, max_power=100)) #approximated period
    

stationary_distrs = pd.DataFrame.from_dict(
    stationary_distrs,
    orient="index"
)
stationary_distrs.to_csv(os.path.join(out_path,"trans_stationary_distrs.csv"))


exit()

#investigate the hidden states wrt returns
features=[]
for model_id, df in emissions:
    P = df.to_numpy(dtype=float)

    symbols = df.columns.astype(float).to_numpy()
    state_means = P @ symbols

    #second moment 
    state_second_moments = P @ (symbols ** 2)

    #variance per hidden state
    state_vars = state_second_moments - state_means**2
    volatility = np.sqrt(state_vars)

    state_third_moments = P @ (symbols ** 3)


    mu3 = (
        state_third_moments
        - 3 * state_means * state_second_moments
        + 2 * state_means**3
    )

    eps = 1e-12
    skewness = mu3 / np.maximum(volatility, eps)**3

    state_fourth_moments = P @ (symbols ** 4)

    mu4 = (
        state_fourth_moments
        - 4 * state_means * state_third_moments
        + 6 * state_means**2 * state_second_moments
        - 3 * state_means**4
    )

    eps = 1e-12
    kurtosis = mu4 / np.maximum(volatility, eps)**4

    print(model_id)

    for s in range(len(state_means)):
        features.append(
            [state_means[s], volatility[s], model_id, s, skewness[s], kurtosis[s]]
        )


symbols = df.columns.astype(float).to_numpy()
features = np.array(features, dtype=object)
mu = features[:,5].astype(float) #features[:,5] #features[:,0].astype(float) #or even features[:,0]
vol = features[:,1].astype(float)
X = np.column_stack([mu, vol])
kmeans = KMeans(n_clusters=4, random_state=seed)
clusters = kmeans.fit_predict(X)

df = pd.DataFrame({
    "mu": mu,
    "vol": vol,
    "model": features[:,2].astype(int),
    "state": features[:,3].astype(int) + 1,
    "cluster": clusters
})

df["label"] = "S" + df["state"].astype(str)

df.to_csv(
    os.path.join(out_path, "state_kmeans.csv"),
    index=False
)

# centroids = pd.DataFrame({
#     "mu": kmeans.cluster_centers_[:,0],
#     "vol": kmeans.cluster_centers_[:,1],
#     "cluster": np.arange(kmeans.n_clusters)
# })

# centroids.to_csv(
#     os.path.join(out_path, "kmeans_centroids.csv"),
#     index=False
# )



    




#we need to find first the permutaiton
distance_matrix = np.zeros((N, N))
for (model_i, P_df), (model_j, Q_df) in itertools.combinations(transitions, 2):


    P = P_df.to_numpy(dtype=float, copy=True)
    Q = Q_df.to_numpy(dtype=float, copy=True)

    tv_mean = total_variation(P, Q).mean()
    js_mean = js_distance(P, Q).mean()

    # upper triangular = JS
    distance_matrix[model_i, model_j] = js_mean

    # lower triangular = TV
    distance_matrix[model_j, model_i] = tv_mean


#np.fill_diagonal(distance_matrix, 0)

distance_df = pd.DataFrame(
    distance_matrix,
    index=[f"model_{i}" for i in range(N)],
    columns=[f"model_{i}" for i in range(N)]
)

distance_df.to_csv(os.path.join(out_path, "js_tv_for_transitions_matrix.csv"))



