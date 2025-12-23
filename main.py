import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
import warnings
import os
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from utils import perform_PCA, state_discretization, apply_global_mapping, percentage_rsmd, percentage_mad, hidden_similarities
from statsmodels.tsa.stattools import acf
from scipy.spatial.distance import jensenshannon


seed=0
np.random.seed(seed)
warnings.filterwarnings("ignore")

#--PARS
max_iterations=1000
max_lag = 100 #maximum number of time steps for which autocorrelation is computed
frequency_step = 10 # re-sampling minute frequency for returns
M_target = 900000 #upper bound on number of bins
merge=False
#--
saving_name = str(M_target)
data_file = "data/data.xlsx"
logs = []
stock_prices = pd.read_excel(data_file, index_col=0, parse_dates=True)
stock_prices.index = pd.to_datetime(stock_prices.index) #use the first column as time index 
N = len(stock_prices.iloc[0]) 
savepoint_filename = str(merge)+str(frequency_step)
stock_prices = stock_prices.iloc[:,:N] #filter number of stocks (in case we set it manually)
days = sorted(set(stock_prices.index.date))
num_days = len(days)
original_T = len(stock_prices)
logs.append(f"{original_T} observations, {num_days} days \n")
all_stocks_returns = []  #ordered list of returns at desired freq for all n-stocks

for i in range(frequency_step, original_T, frequency_step):
    all_stocks_returns.append(np.log(stock_prices.iloc[i]/stock_prices.iloc[i-frequency_step]))

T = len(all_stocks_returns)
all_stocks_returns = np.vstack(all_stocks_returns)   #first column is referred to first stock, second column to second stock, ...
gt_autocorr_continuous = np.zeros((max_lag, N))
K = perform_PCA(all_stocks_returns)
z_min = np.min(all_stocks_returns) - 3*np.std(all_stocks_returns)
z_max = np.max(all_stocks_returns) + 3*np.std(all_stocks_returns)
delta = (z_max - z_min) / M_target
z_min_idx = int(np.floor(z_min / delta))
z_max_idx = int(np.ceil(z_max / delta))
discretized_returns = np.zeros_like(all_stocks_returns)
for col in range(N): 
    folder_name = f"model_{col}"
    os.makedirs(folder_name, exist_ok=True)
    column = all_stocks_returns[:, col] 
    gt_autocorr_continuous[:,col] = acf(column**2, nlags=max_lag, fft=False)[1:]

    
    logs.append(f"continuous) Stock {col}: mean = {np.mean(column)}, variance = {np.var(column):.6g} \n")
    discretized_returns[:,col]=state_discretization(column, delta, z_min_idx, z_max_idx) 
    #print(f"discretized) Stock {col}: mean = {np.mean(discretized_returns[:,col])}, variance = {np.var(discretized_returns[:, col]):.6g}")


all_discretized = discretized_returns.flatten()
unique_vals_global = np.unique(all_discretized)
global_mapping = {v:i for i,v in enumerate(unique_vals_global)}
global_inverse_mapping = {i:v for v,i in global_mapping.items()}
n_symbols = len(global_mapping) 
logs.append(f"\n z_min={z_min}, z_max={z_max}, delta={delta}, number of distinct_emission_symbols:{n_symbols} \n")


n_trials = 1 #each obs is a single symbol from a multinomial distr.
hmm_models = []
X_counts = np.zeros((len(discretized_returns[:,0]), n_symbols), dtype=int)
obs_int = apply_global_mapping(discretized_returns[:,0], global_mapping) #apply (state) index mapping to column 0 of discretized returns
X_counts[np.arange(len(obs_int)), obs_int] = 1 #one hot encoding matrix of size T x n_symbols for hmm: put 1 in the position of index mapping, 0 elsewhere 
#can map back using np.argmax(X_counts, axis=1)
model = hmm.MultinomialHMM(n_components=K, n_iter=max_iterations, random_state=seed, n_trials=n_trials)
model.fit(X_counts)
#hmm_models.append(model)
common_transition = model.transmat_.copy()
common_initial_distr = model.startprob_.copy()


for i in range(0,N):
    print(f"processing model {i+1}")

    try:
        X_counts = np.zeros((len(discretized_returns[:,i]), n_symbols), dtype=int)
        obs_int = apply_global_mapping(discretized_returns[:,i], global_mapping)
        X_counts[np.arange(len(obs_int)), obs_int] = 1
        model = hmm.MultinomialHMM(n_components=K, n_iter=max_iterations, random_state=seed, n_trials=n_trials) #model for stock i
        model.startprob_ = common_initial_distr
        model.transmat_ = common_transition
        model.params = 'e' #only emission probs shall be updated
        model.init_params = 'e'#only emission probs are randomly initialized
        model.fit(X_counts)
        sim, states_tbd, rsmd_matrix, pmad_matrix = hidden_similarities(model) #return string and list of states to be deleted
        print(states_tbd)
        logs.append(f"\n model{i}"+sim+"\n")
        new_K = K
        K_current = len(model.startprob_)
        while len(states_tbd) > 0 and merge and K_current>2:
            keep_mask = np.ones(K_current, dtype=bool)
            keep_mask[states_tbd] = False
            new_K = keep_mask.sum()
            
            new_startprob = model.startprob_[keep_mask]
            new_startprob /= new_startprob.sum()  #normalize
            new_transmat = model.transmat_[keep_mask, :][:, keep_mask]
            new_transmat /= new_transmat.sum(axis=1, keepdims=True)  #row-normalize
            
            pruned_model = hmm.MultinomialHMM(n_components=new_K, n_iter=max_iterations, random_state=seed, n_trials=n_trials)
            pruned_model.startprob_ = new_startprob
            pruned_model.transmat_ = new_transmat
            pruned_model.params = 'e'
            pruned_model.init_params = 'e'
            pruned_model.fit(X_counts)
            model = pruned_model 
            sim, states_tbd, rsmd_matrix, pmad_matrix = hidden_similarities(model) #return string and list of states to be deleted
            logs.append(f"\n model{i}"+sim+"\n")
            K_current = len(model.startprob_)
        hmm_models.append(model)
        state_names = [f"S_{k}" for k in range(new_K)]
        rsmd_df = pd.DataFrame(rsmd_matrix,index=state_names, columns=state_names)
        pmad_df = pd.DataFrame(pmad_matrix,index=state_names, columns=state_names)
        rsmd_df.to_csv(os.path.join(folder_name, savepoint_filename+"model_"+str(i)+"rsmd_matrix.csv"))
        pmad_df.to_csv(os.path.join(folder_name, savepoint_filename+"model_"+str(i)+"mad_matrix.csv"))

    except Exception as e:
        logs.append(f"\n Error while fitting model{i}: {e}")
        print(logs[-1])



    #X_next, Z = model.sample(n_samples=1)
    #hidden_states = model.predict(X_counts)  #uses viterbi algorithm to find most likely sequence of hidden states for obs
    #print("Generated Hidden States:", Z)
    #print("Generated Observations:", X_next[0], "that is emission ", inverse_mapping[np.argmax(X_next[0])])
    #log_likelihood = model.score(X_counts)
    #print("Log-likelihood of the sequence:", log_likelihood)

n_sim = len(discretized_returns[:,0])  #number of returns to simulate (adjust based on max lag) (say at least as original sample obs for fair variance comparison
sim_returns = np.zeros((n_sim, N))  #store simulated returns for each stock
autocorr_matrix = np.zeros((max_lag, N))
gt_autocorr_discrete = np.zeros((max_lag, N))


for q in range(N): #final models
    model = hmm_models[q] 
    folder_name = f"model_{q}"
    emission_matrix = model.emissionprob_  #(K x n_symbols)
    state_names = [f"S_{k}" for k in range(len(emission_matrix))]
    pd.DataFrame(emission_matrix, index=state_names, columns= [f"O_{k}" for k in range(n_symbols)]).to_csv(os.path.join(folder_name, savepoint_filename+"model_"+str(q)+"emissions-hiddens_matrix.csv"))
    pd.DataFrame(model.startprob_,index=state_names).to_csv(os.path.join(folder_name, savepoint_filename+"model_"+str(q)+"startprob.csv"))
    pd.DataFrame(model.transmat_,index=state_names, columns=state_names).to_csv(os.path.join(folder_name, savepoint_filename+"model_"+str(q)+"transmat_matrix.csv"))
 
    o,z=model.sample(n_samples=n_sim)
    index_returns = np.argmax(o,axis=1)
    discretized_pred_returns = [global_inverse_mapping[index] for index in index_returns]
    sim_returns[:, q] = discretized_pred_returns



for i in range(N):
    folder_name = f"model_{i}"
    logs.append(f"\n Stock {i}: model squared returns mean = {np.mean(sim_returns[:,i] **2)}, variance = {np.var(sim_returns[:,i] **2):.6g},  \n gt squared returns mean = {np.mean(discretized_returns[:,i]**2)} variance = {np.var(discretized_returns[:,i]**2)} \n ")
    logs.append(f"\n Stock {i}: model returns mean = {np.mean(sim_returns[:,i])}, variance = {np.var(sim_returns[:,i]):.6g},  \n gt returns mean = {np.mean(discretized_returns[:,i])} variance = {np.var(discretized_returns[:,i])} \n ")

    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    gt_autocorr_discrete[:,i] = acf(discretized_returns[:, i]**2, nlags=max_lag, fft=False)[1:]
    #ymin, ymax = gt_autocorr_discrete[:, i].min(),gt_autocorr_discrete[:, i].max()
    x = range(1,max_lag+1)
    axes[0].plot(x, gt_autocorr_continuous[:,i], color="blue")
    axes[0].set_title("ACF for returns**2 (continuous)")
    axes[1].plot(x, gt_autocorr_discrete[:, i], color='blue', label='obs')

    try:
        autocorr_matrix[:, i] = acf(sim_returns[:, i]**2, nlags=max_lag, fft=False)[1:]
        #autocorr_df = pd.DataFrame(autocorr_matrix[:,i])
        #autocorr_df.to_csv(os.path.join(folder_name, savepoint_filename+"model_"+str(i)+"autocorr.csv"))
        axes[1].plot(x, autocorr_matrix[:,i], color='red',label='model')

    except Exception as e:
        print(f"Error computing autocorr for stock {i}: {e}")
        traceback.print_exc()

    axes[1].legend()
    axes[1].set_title("ACF for returns**2 (discretized returns)")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, savepoint_filename+f"acf_sqreturns_plot_{i}.png"))
    plt.close()


    ##...
    # plt.figure()
    # plt.plot(range(1,max_lag+1), acf(discretized_returns[:, i], nlags=max_lag, fft=False)[1:], marker='o')
    # plt.title(f"gt ACF of (discretized) returns - Stock {i}")
    # plt.axhline(0, color='black', linestyle='--')
    # plt.grid(True)
    # plt.savefig(os.path.join(folder_name, savepoint_filename+f"gt_returnautocorrelation_plot_{i}.png"))
    # plt.close()

    # plt.figure()
    # plt.plot(range(1,max_lag+1), acf(sim_returns[:, i], nlags=max_lag, fft=False)[1:], marker='o')
    # plt.title(f"model ACF of (discretized) returns - Stock {i}")
    # plt.axhline(0, color='black', linestyle='--')
    # plt.grid(True)
    # plt.savefig(os.path.join(folder_name, savepoint_filename+f"returnautocorrelation_plot_{i}.png"))
    # plt.close()


    #js_div = jensenshannon(hist_empirical, hist_simulated)
    #print(f"Jensen-Shannon divergence for stock {i}: {js_div}")

with open(savepoint_filename+"log.txt", "w") as file:
    for l in logs:
        file.write(l)

print(f"log has been written to disk.")