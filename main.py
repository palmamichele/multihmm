import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback
import warnings
import os
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
from hmmlearn import hmm
from utils import perform_PCA, state_discretization, apply_global_mapping, percentage_rsmd, percentage_mad, hidden_similarities
from statsmodels.tsa.stattools import acf
from scipy.stats import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf
#from scipy.spatial.distance import jensenshannon

seed=0
np.random.seed(seed)
warnings.filterwarnings("ignore")

#--PARS
max_iterations=1000
max_lag = 100 #maximum number of time steps for which autocorrelation is computed
frequency_step = 1 # re-sampling minute frequency for returns
M_target = 10 #upper bound on number of bins
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

returns_dates = []
for i in range(frequency_step, original_T, frequency_step):
    #if they are both related to the same day (e.g. exclude overnight returns computation)
    if stock_prices.index[i].date()==stock_prices.index[i-frequency_step].date():
        all_stocks_returns.append(np.log(stock_prices.iloc[i]/stock_prices.iloc[i-frequency_step]))
        returns_dates.append(stock_prices.index[i].date())

returns_dates = pd.Series(returns_dates)     

counts_per_day = returns_dates.value_counts().sort_index()
print(counts_per_day)

# Check if all days have same number of observations
if counts_per_day.nunique() == 1:
    print(f"All days have the same number of observations: {counts_per_day.iloc[0]}")
else:
    print("Days have varying number of observations.")

T = len(all_stocks_returns)
print("number of returns observations", T)

all_stocks_returns = np.vstack(all_stocks_returns)   #first column is referred to first stock, second column to second stock, ...
gt_autocorr_continuous = np.zeros((max_lag, N))
K = perform_PCA(all_stocks_returns)
z_min = np.min(all_stocks_returns) - 3*np.std(all_stocks_returns) #not per column, but globally 
z_max = np.max(all_stocks_returns) + 3*np.std(all_stocks_returns)
delta = (z_max - z_min) / M_target
z_min_idx = int(np.floor(z_min / delta))
z_max_idx = int(np.ceil(z_max / delta))
discretized_returns = np.zeros_like(all_stocks_returns)

for col in range(N): 
    folder_name = f"model_{col}"
    os.makedirs(folder_name, exist_ok=True)
    column = all_stocks_returns[:, col] 
    acf_vals, confint = acf(column**2, nlags=max_lag, fft=False, alpha=0.05)
    gt_autocorr_continuous[:,col] = acf_vals[1:]
    gt_autocorr_continuous_confint = confint[1:]
    jb_stat, p_val = jarque_bera(column)
    logs.append(f"continuous) Stock {col}: mean = {np.mean(column)}, median = {np.median(column)}, stdev = {np.std(column)}, skewness = {skew(column)}, kurtosis (Pearson) ={kurtosis(column, fisher=False)}, jarque-bera {jb_stat}, jarque-bera p-value {p_val} \n")
    discretized_returns[:,col]=state_discretization(column, delta, z_min_idx, z_max_idx) 
    #print(f"discretized) Stock {col}: mean = {np.mean(discretized_returns[:,col])}, variance = {np.var(discretized_returns[:, col]):.6g}")
    print(logs[-1])
    discr_column = discretized_returns[:,col]
    jb_stat, p_val = jarque_bera(discr_column)
    logs.append(f"discretized) Stock {col}: mean = {np.mean(discr_column)}, median = {np.median(discr_column)}, stdev = {np.std(discr_column)}, skewness = {skew(discr_column)}, kurtosis (Pearson) ={kurtosis(discr_column, fisher=False)}, jarque-bera {jb_stat}, jarque-bera p-value {p_val} \n")

    print(logs[-1])

all_discretized = discretized_returns.flatten()
unique_vals_global = np.unique(all_discretized)
global_mapping = {v:i for i,v in enumerate(unique_vals_global)} #given symbol v map it in {0,1,...,n_symbols-1}
global_inverse_mapping = {i:v for v,i in global_mapping.items()}
n_symbols = len(global_mapping) 
logs.append(f"\n z_min={z_min}, z_max={z_max}, delta={delta}, number of distinct_emission_symbols:{n_symbols} \n")

n_trials = 1 #categorical multihmm, as each row must sum to n_trials (fixed a model, we have one emission per time step)
hmm_models = []
X_counts = np.zeros((len(discretized_returns[:,0]) , n_symbols), dtype=int)
obs_int = apply_global_mapping(discretized_returns[:,0], global_mapping) #apply (state) index mapping to column 0 of discretized returns
X_counts[np.arange(len(obs_int)), obs_int] = 1 #one hot encoding matrix of size T x n_symbols for hmm: put 1 in the position of index  
#can map back using np.argmax(X_counts, axis=1)
model = hmm.MultinomialHMM(n_components=K, n_iter=max_iterations, random_state=seed, n_trials=n_trials) #At each time step (n_samples), a emits a multinomial draw with n_trials trials over n_features (alphabet size)
model.fit(X_counts)
#assert model.n_features == n_symbols
#hmm_models.append(model)
common_transition = model.transmat_.copy()
common_initial_distr = model.startprob_.copy()


for i in range(0,N):
    print(f"processing model {i}")
    folder_name = f"model_{i}"
    try:
        X_counts = np.zeros((len(discretized_returns[:,0]), n_symbols), dtype=int)
        obs_int = apply_global_mapping(discretized_returns[:,i], global_mapping)
        X_counts[np.arange(len(obs_int)), obs_int] = 1
        model = hmm.MultinomialHMM(n_components=K, n_iter=max_iterations, random_state=seed, n_trials=n_trials) #model for stock i
        model.startprob_ = common_initial_distr
        #model.transmat_ = common_transition
        model.params = 'te' 
        model.init_params = 'te' 
        model.fit(X_counts)
        #sim, states_tbd, rsmd_matrix, pmad_matrix = hidden_similarities(model) #return string and list of states to be deleted
        logs.append(f"\n model{i}"+"\n")
        new_K = K
        K_current = len(model.startprob_)
        hmm_models.append(model)
        state_names = [f"S_{k}" for k in range(new_K)]
        #rsmd_df = pd.DataFrame(rsmd_matrix,index=state_names, columns=state_names)
        #pmad_df = pd.DataFrame(pmad_matrix,index=state_names, columns=state_names)
        #rsmd_df.to_csv(os.path.join(folder_name, savepoint_filename+"model_"+str(i)+"rsmd_matrix.csv"))
        #pmad_df.to_csv(os.path.join(folder_name, savepoint_filename+"model_"+str(i)+"mad_matrix.csv"))

    except Exception as e:
        logs.append(f"\n Error while fitting model{i}: {e}")
        print(logs[-1])
        exit()



    #X_next, Z = model.sample(n_samples=1)
    #hidden_states = model.predict(X_counts)  #uses viterbi algorithm to find most likely sequence of hidden states for obs
    #print("Generated Hidden States:", Z)
    #print("Generated Observations:", X_next[0], "that is emission ", inverse_mapping[np.argmax(X_next[0])])
    #log_likelihood = model.score(X_counts)
    #print("Log-likelihood of the sequence:", log_likelihood)

n_sim = len(discretized_returns[:,0]) #number of returns to simulate (adjust based on max lag) (say at least as original sample obs for fair variance comparison
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

    jb_stat, p_val = jarque_bera(sim_returns[:,i]**2)
    logs.append(f"model) Stock {i} on squared returns: mean = {np.mean(sim_returns[:,i]**2)}, median = {np.median(sim_returns[:,i]**2)}, stdev = {np.std(sim_returns[:,i]**2)}, skewness = {skew(sim_returns[:,i]**2)}, kurtosis (Pearson) ={kurtosis(sim_returns[:,i]**2, fisher=False)}, jarque-bera {jb_stat}, jarque-bera p-value {p_val} \n")

  
    jb_stat, p_val = jarque_bera(sim_returns[:,i])
    logs.append(f"model) Stock {i} returns: mean = {np.mean(sim_returns[:,i])}, median = {np.median(sim_returns[:,i])}, stdev = {np.std(sim_returns[:,i])}, skewness = {skew(sim_returns[:,i])}, kurtosis (Pearson) ={kurtosis(sim_returns[:,i], fisher=False)}, jarque-bera {jb_stat}, jarque-bera p-value {p_val} \n")
    print(logs[-1])

    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    acf_vals, confint = acf(discretized_returns[:, i]**2, nlags=max_lag, fft=False, alpha=0.05)
    gt_autocorr_discrete[:,i] = acf_vals[1:]
    gt_autocorr_discrete_confint = confint[1:]
    #ymin, ymax = gt_autocorr_discrete[:, i].min(),gt_autocorr_discrete[:, i].max()
    x = range(1,max_lag+1)
    axes[0].plot(x, gt_autocorr_continuous[:,i], color="blue")
    axes[0].fill_between(x, gt_autocorr_continuous_confint[:,0], gt_autocorr_continuous_confint[:,1], color='blue', alpha=0.2)
    axes[0].set_title("ACF for returns**2 (continuous)")
    axes[1].plot(x, gt_autocorr_discrete[:, i], color='blue', label='obs')
    axes[1].fill_between(x, gt_autocorr_discrete_confint[:,0], gt_autocorr_discrete_confint[:,1], color='blue', alpha=0.1)
    try:
        acf_vals, confint= acf(sim_returns[:, i]**2, nlags=max_lag, fft=False, alpha=0.05)
        autocorr_matrix[:, i] = acf_vals[1:] 
        confint = confint[1:]
        #autocorr_df = pd.DataFrame(autocorr_matrix[:,i])
        #autocorr_df.to_csv(os.path.join(folder_name, savepoint_filename+"model_"+str(i)+"autocorr.csv"))
        axes[1].plot(x, autocorr_matrix[:,i], color='red',label='model')
        axes[1].fill_between(x, confint[:,0], confint[:,1], color='red', alpha=0.2)

    except Exception as e:
        print(f"Error computing autocorr for stock {i}: {e}")
        traceback.print_exc()

    axes[1].legend()
    axes[1].set_title("ACF for returns**2 (discretized returns)")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, savepoint_filename+f"acf_sqreturns_plot_{i}.png"))
    plt.close()


    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    axes[0].plot(x, acf(all_stocks_returns[:, i],nlags=max_lag, fft=False)[1:], color="blue")
    axes[0].set_title(f"gt ACF of (continuous) returns - Stock {i}")
    axes[1].plot(x, acf(discretized_returns[:, i], nlags=max_lag, fft=False)[1:], color="blue", label='obs')
    axes[1].plot(x, acf(sim_returns[:, i], nlags=max_lag, fft=False)[1:], color="red", label='model')
    axes[1].set_title(f"ACF of (discretized) returns - Stock {i}")
    #axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, savepoint_filename+f"acf_returns_plot_{i}.png"))
    plt.close()

    # plt.figure()
    # plt.plot(range(1,max_lag+1), acf(sim_returns[:, i], nlags=max_lag, fft=False)[1:], marker='o')
    # plt.title(f"model ACF of (discretized) returns - Stock {i}")
    # plt.axhline(0, color='black', linestyle='--')
    # plt.grid(True)
    # plt.savefig(os.path.join(folder_name, savepoint_filename+f"returnautocorrelation_plot_{i}.png"))
    # plt.close()


    #js_div = jensenshannon(hist_empirical, hist_simulated)
    #print(f"Jensen-Shannon divergence for stock {i}: {js_div}")

with open(savepoint_filename+str(M_target)+"log.txt", "w") as file:
    for l in logs:
        file.write(l)

print(f"log has been written to disk.")