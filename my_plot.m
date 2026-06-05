
J_obs = readtable("J_obs.csv");
R_obs = readtable("R_obs.csv");
Jsim_obs = readtable("Jsim.csv");

N=10;

for i=1:N
    
    R =  R_obs{2:end, i};
    J =  J_obs{2:end, i};
    Jsim =  Jsim_obs{2:end, i};

    fig = figure;
    histogram(R, 'Normalization', 'probability', ... 
        'FaceAlpha', 0.5); 
    hold on; 
    histogram(J, 'Normalization', 'probability', ... 
        'FaceAlpha', 0.5);
    xlabel('returns');
    ylabel('Density');
    title(strcat('Returns histogram ', num2str(i)));
    legend('R', 'J');
    hold off; 
    exportgraphics(fig, strcat(num2str(i), '_hist.png'), 'Resolution', 300);
    close(fig);

    fig = figure;
    histogram(J, 'Normalization', 'probability', ... 
        'FaceAlpha', 0.5); 
    hold on; 
    histogram(Jsim, 'Normalization', 'probability', ... 
        'FaceAlpha', 0.5);
    xlabel('returns');
    ylabel('Density');
    title(strcat('Returns histogram ', num2str(i)));
    legend('J', 'Jsim');
    hold off; 
    exportgraphics(fig, strcat(num2str(i), '_histsim.png'), 'Resolution', 300);
    close(fig);
    
end