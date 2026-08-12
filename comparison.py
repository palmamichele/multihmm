import numpy as np 
import pandas as pd 
import os
import sys
import time
import matplotlib.pyplot as plt
from hmmlearn import hmm
from statsmodels.tsa.stattools import acf
from scipy.stats import jarque_bera, skew, kurtosis
from utils import perform_PCA, state_discretization, apply_global_mapping, fpt_from_log_returns, save_acf_csv
from scipy.stats import entropy, wasserstein_distance


def alignment_score(w, w_prime):
    """
    Relative L1 alignment and alignment score from beyond lipschitz paper.
    """
    w = np.asarray(w)
    w_prime = np.asarray(w_prime)

    denominator = np.sum(np.abs(w))

    if denominator == 0:
        return np.nan, np.nan

    alignment = np.sum(np.abs(w - w_prime)) / denominator
    score = 1.0 - alignment

    return alignment, score

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


seed=0
np.random.seed(seed)

#PARS
max_iterations=1000
max_lags = 100 #for acf 
frequency_step = 1 # re-sampling minute frequency for returns
data_path = "data/data.xlsx"
out_path = "out"
ablation_path = "out_abl"
log_path = "log.txt"
###




prices = pd.read_excel(data_path, index_col=0, parse_dates=True)
prices.index = pd.to_datetime(prices.index) #use the first column as time index 
N = len(prices.iloc[0]) 
hidden_candidates = [i for i in range(1,N+1)] #for simplicity let's assume <= N
T_original = len(prices)
days = sorted(set(prices.index.date))
num_days = len(days)
R_obs = []
for i in range(frequency_step, T_original, frequency_step):
    #if they are both related to the same day (e.g. exclude overnight returns computation)
    if prices.index[i].date()==prices.index[i-frequency_step].date():
        R_obs.append(np.log(prices.iloc[i]/prices.iloc[i-frequency_step]))
        

T = len(R_obs)
R_obs = np.vstack(R_obs)   #first column is referred to first stock, second column to second stock, ...



aic_results = pd.DataFrame(
    index=prices.columns,
    columns=hidden_candidates,
    dtype=float
)

bic_results = pd.DataFrame(
    index=prices.columns,
    columns=hidden_candidates,
    dtype=float
)

acf_alignment_results = pd.DataFrame(
    index=prices.columns,
    columns=hidden_candidates,
    dtype=float
)

acf_score_results = pd.DataFrame(
    index=prices.columns,
    columns=hidden_candidates,
    dtype=float
)

acf_squared_alignment_results = pd.DataFrame(
    index=prices.columns,
    columns=hidden_candidates,
    dtype=float
)

acf_squared_score_results = pd.DataFrame(
    index=prices.columns,
    columns=hidden_candidates,
    dtype=float
)

for K in hidden_candidates:

    J_obs = np.zeros_like(R_obs)
    Jsim = np.zeros_like(R_obs)
    Jsim_abl = np.zeros_like(R_obs)
    R_stats = pd.DataFrame(columns=['Stock','Mean','Stdev','Skewness','Kurtosis','JB Stat','JB p-value'])
    J_stats = pd.DataFrame(columns=['Stock','Mean','Stdev','Skewness','Kurtosis','JB Stat', 'JB p-value'])
    Jsim_stats = pd.DataFrame(columns=['Stock','Mean','Stdev','Skewness','Kurtosis','JB Stat', 'JB p-value'])
    Jsim_abl_stats = pd.DataFrame(columns=['Stock','Mean','Stdev','Skewness','Kurtosis','JB Stat', 'JB p-value'])

    stats = [R_stats, J_stats, Jsim_stats, Jsim_abl_stats]
    hmm_models=[]
    comparisons_ftp =[]
    lags = np.arange(1, max_lags + 1)

    nbins=30 #for R

    for i in range(N):

        R = R_obs[:, i]

        R_edges = np.linspace(
            R.min(),
            R.max(),
            nbins + 1
        )

        R_density, _ = np.histogram(
            R,
            bins=R_edges,
            density=True
        )


        R_centers = (
            R_edges[:-1] + R_edges[1:]
        ) / 2


        R_hist = pd.DataFrame({
            "center": R_centers,
            "density": R_density
        })



        #we make saturation coincides with the extremas
        r_min = np.min(R)
        r_max = np.max(R)

        
        delta= 0.2 * np.std(R, ddof=1)
        z_min = np.floor(r_min/delta -0.5).astype(int)
        z_max = np.floor(r_max/delta +0.5).astype(int)

        J = state_discretization(R, delta, z_min, z_max) 
        J_obs[:,i] = J


        states = np.arange(
        z_min,
        z_max + 1
        ) * delta


        # bin edges centered on states
        J_edges = np.concatenate((
            [states[0] - delta/2],
            states + delta/2
        ))


        J_density, _ = np.histogram(
            J,
            bins=J_edges,
            density=True
        )

        
        unique_vals_global = np.unique(J)
        global_mapping = {v:i for i,v in enumerate(unique_vals_global)} #given symbol v map it in {0,1,...,n_symbols-1}
        global_inverse_mapping = {i:v for v,i in global_mapping.items()}
        n_symbols = len(global_mapping) 
        obs_int = apply_global_mapping(J, global_mapping).astype(int) #apply (state) index mapping 


        X_counts = np.asarray(obs_int, dtype=int).reshape(-1, 1)
        if i==0:
            model = hmm.CategoricalHMM(n_components=K, n_iter=max_iterations, random_state=seed, n_features=n_symbols)
            start = time.perf_counter()
            model.fit(X_counts)
            elapsed = time.perf_counter() - start
            common_transition = model.transmat_.copy()
            common_initial_distr = model.startprob_.copy()
            #the first model is already randomly initialized 
            model_abl = model
            elapsed_abl = elapsed


        else:
            model =hmm.CategoricalHMM(n_components=K, n_iter=max_iterations, random_state=seed, n_features=n_symbols, init_params='e',  params='e')
            #change params to 'e' for fixing the transition matrix.
            model.transmat_ = common_transition
            model.startprob_ = common_initial_distr

            start = time.perf_counter()
            model.fit(X_counts)
            elapsed = time.perf_counter() - start

            #ablation
            model_abl =hmm.CategoricalHMM(n_components=K, n_iter=max_iterations, random_state=seed, n_features=n_symbols)
            start_abl = time.perf_counter()
            model_abl.fit(X_counts)
            elapsed_abl = time.perf_counter() - start_abl

       
        log_likelihood = model.score(X_counts)
        n_obs = len(X_counts)
        M = model.n_features


        if i == 0:
            n_params = (
                (K - 1) +
                K * (K - 1) +
                K * (M - 1)
            )
        else:
            n_params = (
                (K - 1) +
                K * (M - 1)
            )

        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n_obs) - 2 * log_likelihood

        aic_results.loc[prices.columns[i], K] = aic
        bic_results.loc[prices.columns[i], K] = bic



        assert model.n_features==n_symbols
        hmm_models.append(model)
        print(f"Model {i} ({prices.columns[i]}) estimated in {elapsed:.3f} seconds")
        print(f"Model {i} converged? {model.monitor_.converged}")
        print(f"ablation Model {i} ({prices.columns[i]}) estimated in {elapsed_abl:.3f} seconds")
        print(f"ablation Model {i} converged? {model_abl.monitor_.converged}")


        emission_matrix = model.emissionprob_  #(K x n_symbols)
        state_names = [f"S_{k}" for k in range(len(emission_matrix))]
        emission_names = [str(global_inverse_mapping[k]) for k in range(n_symbols)]
        print(f"model_{i} has {len(emission_names)} emissions")

    
        o,z=model.sample(n_samples=T)
        J_hat = np.array([global_inverse_mapping[index] for index in o.flatten()])
        Jsim [:, i] = J_hat


        Jsim_density, _ = np.histogram(
            J_hat,
            bins=J_edges,
            density=True
        )

        # save difference
        # pd.DataFrame({
        #     "center": states,
        #     "difference": J_density - Jsim_density,
        #     "J": J_density,
        #     "Jsim": Jsim_density
        # }).to_csv(
        #     os.path.join(
        #         folder_name,
        #         f"hist_difference_{i}.csv"
        #     ),
        #     index=False
        # )

        
        o_abl,z_abl=model_abl.sample(n_samples=T)
        J_hat_abl = np.array([global_inverse_mapping[index] for index in o_abl.flatten()])
        Jsim_abl [:, i] = J_hat_abl

        Jsim_density_abl, _ = np.histogram(
            J_hat_abl,
            bins=J_edges,
            density=True
        )

        # pd.DataFrame({
        #     "center": states,
        #     "difference": J_density - Jsim_density_abl,
        #     "J": J_density,
        #     "Jsim": Jsim_density_abl
        # }).to_csv(
        #     os.path.join(
        #         abl_folder_name,
        #         f"hist_difference_{i}.csv"
        #     ),
        #     index=False
        # )

        acf_J = acf(
            J,
            nlags=max_lags,
            fft=False
        )
        
        acf_Jsim = acf(
            J_hat,
            nlags=max_lags,
            fft=False
        )

        acf_J_squared = acf(
            J**2,
            nlags=max_lags,
            fft=False
        )

        acf_Jsim_squared = acf(
            J_hat**2,
            nlags=max_lags,
            fft=False
        )
    
        acf_vals, confint = acf(R, nlags=max_lags, fft=False, alpha=0.05)
      


        acf_vals, confint = acf(J, nlags=max_lags, fft=False, alpha=0.05)
       


        acf_vals, confint = acf(R**2, nlags=max_lags, fft=False, alpha=0.05)
        


        acf_vals, confint = acf(J**2, nlags=max_lags, fft=False, alpha=0.05)
      

        acf_vals, confint = acf(J_hat, nlags=max_lags, fft=False, alpha=0.05)
       
        acf_vals, confint = acf(J_hat**2, nlags=max_lags, fft=False, alpha=0.05)
     

        acf_vals, confint = acf(J_hat_abl, nlags=max_lags, fft=False, alpha=0.05)
       


        acf_vals, confint = acf(J_hat_abl**2, nlags=max_lags, fft=False, alpha=0.05)


        # Remove lag 0.
        # We compare lags 1,...,100 => vectors in R^100
        acf_J_vec = acf_J[1:max_lags + 1]
        acf_Jsim_vec = acf_Jsim[1:max_lags + 1]

        acf_J_squared_vec = acf_J_squared[1:max_lags + 1]
        acf_Jsim_squared_vec = acf_Jsim_squared[1:max_lags + 1]


        # Alignment and score
        acf_alignment, acf_score = alignment_score(
            acf_J_vec,
            acf_Jsim_vec
        )

        acf_squared_alignment, acf_squared_score = alignment_score(
            acf_J_squared_vec,
            acf_Jsim_squared_vec
        )

        acf_alignment_results.loc[
            prices.columns[i], K
        ] = acf_alignment

        acf_score_results.loc[
            prices.columns[i], K
        ] = acf_score

        acf_squared_alignment_results.loc[
            prices.columns[i], K
        ] = acf_squared_alignment

        acf_squared_score_results.loc[
            prices.columns[i], K
        ] = acf_squared_score


        
        rho_values = [1.0025, 1.005, 1.01, 1.02]
        for rho in rho_values:

            #fpts_R = fpt_from_log_returns(R, rho=rho)

            fpts_J= fpt_from_log_returns(J, rho=rho)
            
            fpts_Jsim= fpt_from_log_returns(J_hat, rho=rho)

            #fpts_Jsim_abl= fpt_from_log_returns(J_hat_abl, rho=rho)
            
            
            taus = np.union1d(fpts_J, fpts_Jsim)
            #r = np.array([(fpts_R == t).mean() for t in taus])
            p = np.array([(fpts_J == t).mean() for t in taus])
            q = np.array([(fpts_Jsim == t).mean() for t in taus])

            # pd.DataFrame({
            #     "tau": taus,
            #     #"R": r,
            #     "J": p,
            #     "Jsim": q
            # }).to_csv(
            #     os.path.join(folder_name, f"fpts_{i}_{rho}.csv"),
            #     index=False
            # )


            eps = 1e-12
            p += eps
            q += eps
            p /= p.sum()
            q /= q.sum()

            kl = entropy(p, q)

            wd = wasserstein_distance(fpts_J, fpts_Jsim)

            comparisons_ftp.append({
                "Stock": prices.columns[i],
                "rho": rho,
                "KL": kl,
                "Wasserstein": wd
            })

        
        for idx, var in enumerate([R, J, J_hat, J_hat_abl]):
            jb_stat, p_val = jarque_bera(var)

            stats[idx] = pd.concat([stats[idx], pd.DataFrame({
            'Stock': [prices.columns[i]],
            'Mean': [np.mean(var)],
            #'Median': [np.median(var)], it is zero for all our returns
            'Stdev': [np.std(var)],
            'Skewness': [skew(var)],
            'Kurtosis': [kurtosis(var, fisher=False)],
            'JB Stat': [jb_stat],
            'JB p-value': [p_val]
            })], ignore_index=True)
            


    # stats[0].to_csv(
    #     os.path.join(out_path, "R_stats.csv"), index=False
    # )

    # stats[1].to_csv(
    #     os.path.join(out_path, "J_stats.csv"), index=False
    # )

    # stats[2].to_csv(
    #     os.path.join(out_path, "Jsim_stats.csv"), index=False
    # )

    # stats[3].to_csv(
    #     os.path.join(ablation_path, "Jsim_abl_stats.csv"), index=False
    # )


aic_results.index.name = "Model"
bic_results.index.name = "Model"

aic_results.to_csv(
    os.path.join(out_path, "AIC_results.csv")
)

bic_results.to_csv(
    os.path.join(out_path, "BIC_results.csv")
)

aic_results.index.name = "Model"
bic_results.index.name = "Model"

acf_alignment_results.index.name = "Model"
acf_score_results.index.name = "Model"

acf_squared_alignment_results.index.name = "Model"
acf_squared_score_results.index.name = "Model"


acf_alignment_results.to_csv(
    os.path.join(out_path, "ACF_alignment.csv")
)

acf_score_results.to_csv(
    os.path.join(out_path, "ACF_alignment_score.csv")
)

acf_squared_alignment_results.to_csv(
    os.path.join(out_path, "ACF_squared_alignment.csv")
)

acf_squared_score_results.to_csv(
    os.path.join(out_path, "ACF_squared_alignment_score.csv")
)

print("end")