rng(0)

%--PARS
maxiter=1000;
max_lags = 100; 
M_target = 10;
frequency_step = 1; 
variance_thr = 0.9; 
data_path = "data/data.xlsx";
%--

output_path = strcat(pwd,'/','experiments');
if not(isfolder(output_path))
    mkdir experiments 
end


data = readtable(data_path);
dates = datetime(data{:,1}); %time stamps
prices = data{:,2:end};

[T_original, N] = size(prices);

fprintf('%d observations, %d stocks\n', T_original, N);

R_obs = [];
returns_dates = [];

for i = (1 + frequency_step):T_original
    %check if they are both related to the same day (exclude overnight returns)
    if dateshift(dates(i), 'start', 'day') == ...
            dateshift(dates(i-frequency_step), 'start', 'day')

        r = log(prices(i,:) ./ prices(i-frequency_step,:));

        R_obs = [R_obs; r];
        returns_dates = [returns_dates; dates(i)];
    end
end

T = size(R_obs,1);
K = perform_PCA(R_obs, variance_thr);

%fprintf('Number of %d-min return observations: %d\n', frequency_step,T);

J_obs = [];
J_sim = [];

acf_R = zeros(max_lags, N);
acf_J = zeros(max_lags, N);
acf_Jsim = zeros(max_lags, N);

acf_Rsq = zeros(max_lags, N);
acf_Jsq = zeros(max_lags, N);
acf_Jsimsq = zeros(max_lags, N);


R_stats = table();
J_stats = table();

for i=1:N
    R = R_obs(:, i);
    
    z_min = min(R) - 3*std(R);
    z_max = max(R) + 3*std(R);
    delta = (z_max - z_min) / M_target;
    zmin = floor(z_min / delta);
    zmax = ceil(z_max / delta);


    disp([z_min, z_max, delta, zmin, zmax])

    %delta = 0.05 * std(R);
    %zmin = floor(min(R)/delta);
    %zmax = ceil(max(R)/delta);
    % 
    % delta = (max(R) - min(R)) / M_target;

    J = discretize_returns(R,delta,zmin,zmax);

    fig = figure;
    histogram(R, 'Normalization', 'probability'); 
    hold on; 
    histogram(J, 'Normalization', 'probability', ... 
        'FaceAlpha', 0.3);
    xlabel('returns');
    ylabel('Density');
    title(strcat('Returns histogram ', num2str(i)));
    legend('R', 'J');
    hold off; 
    exportgraphics(fig, strcat(num2str(i), '_hist.png'), 'Resolution', 300);
    close(fig);


    J_obs = [J_obs, J];
    model_path = strcat(output_path, '/', "model_", num2str(i));
    if ~isfolder(model_path)
        mkdir(model_path);
    end


    
    acf_vals = autocorr(R, NumLags=max_lags);
    acf_R(:,i) = acf_vals(2:end);
    
    acf_vals = autocorr(J, NumLags=max_lags);
    acf_J(:,i) = acf_vals(2:end);
     

    acf_vals = autocorr(R.^2, NumLags=max_lags);
    acf_Rsq(:,i) = acf_vals(2:end);

    acf_vals = autocorr(J.^2, NumLags=max_lags);
    acf_Jsq(:,i) = acf_vals(2:end);

    figure; hold on; grid on;
    plot(1:max_lags, acf_Rsq(:,i), '-ob', ... 
        'DisplayName', 'R');
    %plot(1:max_lags, acf_J(:,i), '-or', ...
    %     'DisplayName', 'J');
    %yline(0, 'k-');
     
    xlabel('Lag');
    ylabel('Autocorrelation');
    title(strcat('ACF Comparison', num2str(i)));
    legend;
     

    [h,p,jbstat] = jbtest(R);
    R_stats = [R_stats; table(data.Properties.VariableNames(i+1),mean(R), std(R), skewness(R), kurtosis(R), jbstat, 'VariableNames', {'Stock','Mean', 'Stdev', 'Skewness', 'Kurtosis', 'JB Stat'})];
    
    [h,p,jbstat] = jbtest(J);
    J_stats = [J_stats; table(data.Properties.VariableNames(i+1),mean(J), std(J), skewness(J), kurtosis(J), jbstat, 'VariableNames', {'Stock','Mean', 'Stdev', 'Skewness', 'Kurtosis', 'JB Stat'})];


end

hmm_models = cell(N,1);

%initialize the first HMM
%distr p 
n_symbols = length(unique(J_obs(:,1)));

disp(n_symbols);

[C, ~, J_int] = unique(J_obs(:,1)); %mapping each discretized return to integer index

%every row sums up to 1
%p = ones(1,K) / K;
%TRANS_GUESS = ones(K,K) / K;
%EMIS_GUESS  = ones(K,n_symbols) / n_symbols;
%[TRANS, EMIS] = hmmtrain(J_int,TRANS_GUESS,EMIS_GUESS,'maxiterations',maxiter);

TRANS_GUESS = rand(K,K);
TRANS_GUESS = TRANS_GUESS ./ sum(TRANS_GUESS, 2);

EMIS_GUESS = rand(K,n_symbols);
EMIS_GUESS = EMIS_GUESS ./ sum(EMIS_GUESS, 2);
[TRANS, EMIS] = hmmtrain(J_int,TRANS_GUESS,EMIS_GUESS,'maxiterations',maxiter);
common_T = TRANS;

fprintf('common T');
disp(common_T)



for i = 1:N
    fprintf('\nProcessing model %d\n', i);
    n_symbols = length(unique(J_obs(:,i)));
    [C, ~, J_int] = unique(J_obs(:,i)); 
    
    [TRANS, EMIS] = hmmtrain(J_int,...
        TRANS_GUESS,...
        EMIS_GUESS,...
        'maxiterations',maxiter);

    %we shall fix TRANS to be always common_T
    disp(TRANS)
    hmm_models{i}.transmat = TRANS;
    hmm_models{i}.emission = EMIS;

    [JsimInt, hiddenstates]=hmmgenerate(T, TRANS, EMIS);
    Jsim = C(JsimInt);
    J_sim = [J_sim,Jsim];



    
    acf_vals = autocorr(Jsim.^2, NumLags=max_lags);
    acf_Jsimsq(:,i) = acf_vals(2:end);

    figure; hold on; grid on;
    plot(1:max_lags, acf_Jsq(:,i), '-ob', ... 
        'DisplayName', 'J sqd');
    plot(1:max_lags, acf_Jsimsq(:,i), '-or', ...
         'DisplayName', 'Jsim sqd');
    yline(0, 'k-');

    xlabel('Lag');
    ylabel('Autocorrelation');
    title(strcat('ACF Comparison (discretized returns)', num2str(i)));
    legend;

end


%TRANS_HAT = [0 p; zeros(size(TRANS,1),1) TRANS];
%EMIS_HAT = [zeros(1,size(EMIS,2)); EMIS];

writetable(R_stats, 'R_stats.csv');
writetable(J_stats, 'J_stats.csv');
writematrix(acf_R, 'acf_R.csv');
writematrix(acf_J, 'acf_J.csv');
writematrix(acf_Rsq, 'acf_Rsq.csv');
writematrix(acf_Jsq, 'acf_Jsq.csv');
writematrix(J_sim, 'J_sim.csv');
writematrix(J_obs, 'J_obs.csv');
writematrix(R_obs, 'R_obs.csv');


function k = perform_PCA(D, variance_threshold)

if nargin < 2
    variance_threshold = 0.9;
end

D_std = zscore(D);

[coeff, score, latent, tsquared, explained] = pca(D_std);

cumulative_variance = cumsum(explained)/100;

figure;
plot(cumulative_variance,'-o');
grid on;

xlabel('Number of Components');
ylabel('Cumulative Explained Variance');

k = find(cumulative_variance >= variance_threshold,1);
fprintf('PCA k = %d\n', k);
end

function J = discretize_returns(R, delta, zmin, zmax)
    i = floor(R/delta + 0.5);
    i = min(zmax, max(zmin, i));
    J = i * delta;
end

