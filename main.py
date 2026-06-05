import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
from hmmlearn import hmm
from statsmodels.tsa.stattools import acf
from scipy.stats import jarque_bera, skew, kurtosis
from utils import perform_PCA, state_discretization, apply_global_mapping


seed=0
np.random.seed(seed)

#PARS
max_iterations=1000
max_lags = 100 #for acf 
M_target=10
frequency_step = 1 # re-sampling minute frequency for returns
variance_thr=0.9
data_path = "data/data.xlsx"
###

prices = pd.read_excel(data_path, index_col=0, parse_dates=True)
prices.index = pd.to_datetime(prices.index) #use the first column as time index 
N = len(prices.iloc[0]) 
T_original = len(prices)


print(T_original, N)

days = sorted(set(prices.index.date))
num_days = len(days)
print(num_days)

R_obs = []

for i in range(frequency_step, T_original, frequency_step):
    #if they are both related to the same day (e.g. exclude overnight returns computation)
    if prices.index[i].date()==prices.index[i-frequency_step].date():
        R_obs.append(np.log(prices.iloc[i]/prices.iloc[i-frequency_step]))
        

T = len(R_obs)
R_obs = np.vstack(R_obs)   #first column is referred to first stock, second column to second stock, ...


K = perform_PCA(R_obs, variance_thr)

J_obs = np.zeros_like(R_obs)
Jsim = np.zeros_like(R_obs)

acf_R = np.zeros((max_lags, N))
acf_J = np.zeros((max_lags, N))
acf_Jsim = np.zeros((max_lags, N))

acf_Rsq = np.zeros((max_lags, N))
acf_Jsq = np.zeros((max_lags, N))
acf_Jsimsq = np.zeros((max_lags, N))


R_stats = pd.DataFrame(columns=['Stock','Mean','Median','Stdev','Skewness','Kurtosis','JB Stat','JB p-value'])
J_stats = pd.DataFrame(columns=['Stock','Mean','Median','Stdev','Skewness','Kurtosis','JB Stat', 'JB p-value'])
Jsim_stats = pd.DataFrame(columns=['Stock','Mean','Median','Stdev','Skewness','Kurtosis','JB Stat', 'JB p-value'])

stats = [R_stats, J_stats, Jsim_stats]
hmm_models=[]

for i in range(N):
    folder_name = f"model_{i}"
    os.makedirs(folder_name, exist_ok=True)

    R = R_obs[:, i]

    #we make saturation coincides with the extremas
    # r_min = np.min(column)
    # r_max = np.max(column)
    # delta= 0.05 * np.std(column)
    # z_min = np.floor(r_min/delta -0.5).astype(int)
    # z_max = np.floor(r_max/delta +0.5).astype(int)
    #discretized_returns[:,col]=M(column, delta, z_min, z_max) 

    #what if we use this discretization but per stock?
    z_min = np.min(R) - 3*np.std(R) 
    z_max = np.max(R) + 3*np.std(R)
    delta = (z_max - z_min) / M_target
    z_min_idx = int(np.floor(z_min / delta))
    z_max_idx = int(np.ceil(z_max / delta))

   

    J = state_discretization(R, delta, z_min_idx, z_max_idx) 
    J_obs[:,i] = J

    unique_vals_global = np.unique(J)
    global_mapping = {v:i for i,v in enumerate(unique_vals_global)} #given symbol v map it in {0,1,...,n_symbols-1}
    global_inverse_mapping = {i:v for v,i in global_mapping.items()}
    n_symbols = len(global_mapping) 

    print(n_symbols)

    print(f"processing model {i}")

    obs_int = apply_global_mapping(J, global_mapping) #apply (state) index mapping 
    X_counts = np.asarray(obs_int, dtype=int).reshape(-1, 1)

    if i==0:
        model = hmm.CategoricalHMM(n_components=K, n_iter=max_iterations, random_state=seed, n_features=n_symbols)
        model.fit(X_counts)
        common_transition = model.transmat_.copy()
        common_initial_distr = model.startprob_.copy()

    else:
        model =hmm.CategoricalHMM(n_components=K, n_iter=max_iterations, random_state=seed, n_features=n_symbols, init_params='e',  params='te')
        model.transmat_ = common_transition
        model.startprob_ = common_initial_distr
        model.fit(X_counts)

    assert model.n_features==n_symbols

 
    hmm_models.append(model)
    
    state_names = [f"S_{k}" for k in range(K)] #hidden
    #rsmd_df = pd.DataFrame(rsmd_matrix,index=state_names, columns=state_names)
    #pmad_df = pd.DataFrame(pmad_matrix,index=state_names, columns=state_names)
    #rsmd_df.to_csv(os.path.join(folder_name, savepoint_filename+"model_"+str(i)+"rsmd_matrix.csv"))
    #pmad_df.to_csv(os.path.join(folder_name, savepoint_filename+"model_"+str(i)+"mad_matrix.csv"))

    emission_matrix = model.emissionprob_  #(K x n_symbols)
    state_names = [f"S_{k}" for k in range(len(emission_matrix))]

    pd.DataFrame(emission_matrix, index=state_names, columns= [f"O_{k}" for k in range(n_symbols)]).to_csv(os.path.join(folder_name, "model_"+str(i)+"emissions-hiddens_matrix.csv"))
    pd.DataFrame(model.startprob_,index=state_names).to_csv(os.path.join(folder_name, "model_"+str(i)+"startprob.csv"))
    pd.DataFrame(model.transmat_,index=state_names, columns=state_names).to_csv(os.path.join(folder_name, "model_"+str(i)+"transmat_matrix.csv"))

    o,z=model.sample(n_samples=T)
    J_hat = np.array([global_inverse_mapping[index] for index in o.flatten()])
    Jsim [:, i] = J_hat




    # plt.figure()

    # plt.hist(
    #     R,
    #     density=True,
    #     alpha=0.6,
    #     label="R", bins=30
    # )

    # plt.hist(
    #     J,
    #     density=True,
    #     alpha=0.4,
    #     label="J", bins=30
    # )

    # plt.xlabel("returns")
    # plt.ylabel("Density")
    # plt.title(f"Returns histogram {i+1}")
    # plt.legend()

    # plt.savefig(
    #     f"{i}_hist.png",
    #     dpi=300,
    #     bbox_inches="tight"
    # )

    # plt.close()



    acf_vals, confint = acf(R, nlags=max_lags, fft=False, alpha=0.05)
    acf_R[:, i] = acf_vals[1:]   # drop lag 0

   
    acf_vals, confint = acf(J, nlags=max_lags, fft=False, alpha=0.05)
    acf_J[:, i] = acf_vals[1:]


    acf_vals, confint  = acf(R**2, nlags=max_lags, fft=False, alpha=0.05)
    acf_Rsq[:, i] = acf_vals[1:]


    acf_vals, confint = acf(J**2, nlags=max_lags, fft=False, alpha=0.05)
    acf_Jsq[:, i] = acf_vals[1:]

    acf_vals, confint = acf(J_hat, nlags=max_lags, fft=False, alpha=0.05)
    acf_Jsim[:, i] = acf_vals[1:]

    acf_vals, confint = acf(J_hat**2, nlags=max_lags, fft=False, alpha=0.05)
    acf_Jsimsq[:, i] = acf_vals[1:]

    #gt_autocorr_continuous_confint = confint[1:]



    for idx, var in enumerate([R, J, J_hat]):
        jb_stat, p_val = jarque_bera(var)

        stats[idx] = pd.concat([stats[idx], pd.DataFrame({
        'Stock': [prices.columns[i]],
        'Mean': [np.mean(var)],
        'Median': [np.median(var)],
        'Stdev': [np.std(var)],
        'Skewness': [skew(var)],
        'Kurtosis': [kurtosis(var, fisher=False)],
        'JB Stat': [jb_stat],
        'JB p-value': [p_val]
    })], ignore_index=True)
        

        fig, axes = plt.subplots(2, 1, figsize=(6, 8))

        gt_autocorr_discrete_confint = confint[1:]

        x = range(1,max_lags+1)
        axes[0].plot(x, acf_Rsq[:, i], color="blue")
        #axes[0].fill_between(x, gt_autocorr_continuous_confint[:,0], gt_autocorr_continuous_confint[:,1], color='blue', alpha=0.2)
        axes[0].set_title("ACF for returns**2 (continuous)")
        axes[1].plot(x, acf_Jsq[:, i], color='blue', label='obs')
        #axes[1].fill_between(x, gt_autocorr_discrete_confint[:,0], gt_autocorr_discrete_confint[:,1], color='blue', alpha=0.1)
      
       
        axes[1].plot(x, acf_Jsimsq[:,i], color='red',label='model')
        #axes[1].fill_between(x, confint[:,0], confint[:,1], color='red', alpha=0.2)

        axes[1].legend()
        axes[1].set_title("ACF for returns**2 (discretized returns)")
        plt.tight_layout()
        plt.savefig(os.path.join(folder_name, f"acf_sqreturns_plot_{i}.png"))
        plt.close()


        
        # plt.hist(
        #     J,
        #     density=True,
        #     alpha=0.6,
        #     label="J", bins=30
        # )

        # plt.hist(
        #     J_hat,
        #     density=True,
        #     alpha=0.4,
        #     label="J_hat", bins=30
        # )

        # plt.xlabel("returns")
        # plt.ylabel("Density")
        # plt.title(f"Returns histogram {i+1}")
        # plt.legend()

        # plt.savefig(
        #     f"{i}_hissim.png",
        #     dpi=300,
        #     bbox_inches="tight"
        # )

        # plt.close()


pd.DataFrame(R_obs, columns=prices.columns).to_csv("R_obs.csv", index=False)
pd.DataFrame(J_obs, columns=prices.columns).to_csv("J_obs.csv", index=False)
pd.DataFrame(Jsim, columns=prices.columns).to_csv("Jsim.csv", index=False)
stats[0].to_csv("R_stats.csv", index=False)
stats[1].to_csv("J_stats.csv", index=False)
stats[2].to_csv("Jsim_stats.csv", index=False)