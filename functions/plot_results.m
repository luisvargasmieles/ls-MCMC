function plot_results(Y,X,nStagesSKROCK,firstMoment,logPiTrace,mseValues,secondMoment)

close all

% Dock the generated figures
set(0,'DefaultFigureWindowStyle','docked');

% Initial configurations for the plots
set(0, 'DefaultAxesFontSize',20);
set(0, 'DefaultLineLineWidth', 3);
set(0, 'defaultTextInterpreter', 'latex');

% 1. PLOT ORIGINAL, OBSERVATIONS AND ESTIMATES

    % Plot the original image
    figure(1);
    imagesc(X,[0 255]);
    title('Original image');
    axis equal; axis off;colormap('gray');

    % Plot the noisy observation
    figure(2);
    imagesc(Y, [0 255]); hold on
    axis equal; axis off;colormap('gray');
    title('Observation $y$', 'Interpreter', 'latex');
 
    % Plot the MMSE of x
    figure(3);
    imagesc(firstMoment,[0 255]);
    axis equal; axis off;colormap('gray');
    title('First moment estimate');
    
    % Plot the \log\pi trace of the samples
    figure(4);
    semilogx(1:nStagesSKROCK:nStagesSKROCK*length(logPiTrace),logPiTrace);
    xlabel('number of gradient evaluations');
    ylabel('$\log\pi(X_n)$','interpreter','latex');
    title('$\log\pi$ trace of $X_n$','Interpreter','latex');
    
    % Plot the evolution of the MSE in stationarity
    figure(5);
    semilogx(1:nStagesSKROCK:nStagesSKROCK*length(mseValues),mseValues);
    xlabel('number of gradient evaluations');
    ylabel('MSE','interpreter','latex');
    title('Evolution of MSE in stationarity','Interpreter','latex');

    % Plot the pixel-wise standard deviation of the generated samples
    figure(6);
    imagesc(sqrt(secondMoment - firstMoment.^2));
    axis equal; axis off; colormap('gray'); colorbar;
    title('Pixel-wise standard deviation');

end