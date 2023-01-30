function [alpha, rho2] = SAPG_inp_TV(varargin)
%
% 
% function to run the stochastic approximation proximal gradient
% (SAPG) algorithm to estimate the hyperparameter of the total-variation
% prior \alpha, and the augmentation parameter \rho for the cameraman
% inpainting experiment (See Algorithm 3.1 in paper for details)
%
%%  =========== inputs ====================
%
% 'true_img'                    : true image
%
% 'obs_vector'                  : observation vector
%           
% 'operator'                    : linear operator A
%           
% 'transpose_operator'          : transpose of the linear operator A
%
% 'inverse_precision_matrix'    : inverse of the precision matrix
%                                 Q = 1/sigma^2 * H^T *H + 1/rho^2 * I_N
%                    
% 'number_samples_burn_in'      : number of samples to consider in the
%                                 burn-in stage
%
% 'max_number_samples'          : maximum number of samples to consider in
%                                 the SAPG optimisation algorithm
%
% 'sigma'                       : noise level of the likelihood
%
%
%% =========== Outputs ====================
%
% alpha  : estimated hyperparameter of the total-variation prior
%
% rho    : estimated augmentation parameter
% 
% ===================================================

%% ------------------------------------------------------------
% Read the parameters
%--------------------------------------------------------------
for i = 1:2:(length(varargin)-1)
    switch upper(varargin{i})
        case 'TRUE_IMG'
            X          = varargin{i+1};
        case 'OBS_VECTOR'
            Y          = varargin{i+1};
        case 'INVERSE_PRECISION_MATRIX'
            invQ      = varargin{i+1};
        case 'TRANSPOSE_OPERATOR'
            At         = varargin{i+1};
        case 'NUMBER_SAMPLES_BURN_IN'
            nBurnIn    = varargin{i+1};
        case 'MAX_NUMBER_SAMPLES'
            maxSamples = varargin{i+1};
        case 'SIGMA'
            sigma      = varargin{i+1};
        case 'TOL_TV_PARAM'
            tol_tv     = varargin{i+1};
        case 'TOL_RHO2_PARAM'
            tol_rho2   = varargin{i+1};
    end
end

%--------------------------------------------------------------
% Main body
%--------------------------------------------------------------
N = size(X,1); % Dimension of the image

% setting initial min/max value for the hyperparameters to estimate

% total variation prior hyperparameter \alpha
thtv_init = 0.5;
min_thtv = 1e-3;
max_thtv = 0.5;

% augmentation parameter \rho^2
thrho2_init = 0.5 * sigma^2;
min_thrho2 = 1e-2 * thrho2_init;
max_thrho2 = 100 * thrho2_init;

% to avoid numerical errors, it is strongly recommended
% to work in logarithmic scale, so we define an axiliary variable 
% etatv/etarho2 such that theta=exp{eta}. 
etatv_init = log(thtv_init);
min_etatv = log(min_thtv);
max_etatv = log(max_thtv);

etarho2_init = log(thrho2_init);
min_etarho2 = log(min_thrho2);
max_etarho2 = log(max_thrho2);

% variable step-sizes for the SAPG algorithm
deltaTV = @(i) 10*(i^(-0.8)) / (N*N);
deltarho2 = @(i) 10*(i^(-0.8)) / (N*N);

% Algorithm parameters
AtY = At(Y);
lambda = sigma^2; % Moreau-Yosida approximation parameter
dt = 1/(1/lambda + 1/(thrho2_init + sigma^2)); % step-size for the MCMC method

% proximal operator of the total-variation
proxG = @(x,alpha) chambolle_prox_TV_stop(x, 'lambda',alpha*lambda,...
                                             'maxiter',20);

% initial conditions
fix_thtv = thtv_init; % fix initial parameter for the burn-in stage
fix_thrho2 = thrho2_init; % fix initial parameter for the burn-in stage
Zk = AtY; % initial condition

%%% Warm-up
perc = round(nBurnIn/10); % to show the progress percentage of Warm-up
fprintf("Running Warm-up SAPG stage     \n");
for i=2:nBurnIn
    %%% MCMC update
    Xk_grad = invQ(AtY/sigma^2 + Zk/fix_thrho2,fix_thrho2);
    
    Zk = Zk -dt*(Zk - Xk_grad)/fix_thrho2 ...
        -dt*(Zk -proxG(Zk,fix_thtv))/lambda ...
        +sqrt(2*dt)*randn(N);

    %%% to show progress of Warm-up stage
    if rem(i,perc)==0
        fprintf('\b\b\b\b%2d%%\n', 10*round(i/perc));
    end
end
disp("Warm-up stage finished");

%%% sampling stage variables
% to save the progress of the convergence of the hyperparameters
% to estimate
thTV = zeros(1,maxSamples);
thTV(1)=fix_thtv;
etaTV = zeros(1,maxSamples);
etaTV(1)=etatv_init;

thrho2 = zeros(1,maxSamples);
thrho2(1)=fix_thrho2;
etarho2 = zeros(1,maxSamples);
etarho2(1)=etarho2_init;

% stopping criteria initialisation
stopCriTV = 1;
stopCriRho2 = 1;
i=2;

TVavg = zeros(1,maxSamples);
rho2avg = zeros(1,maxSamples);

disp("Starting estimation of hyperparameters (SAPG algorithm)");
while (stopCriTV >= tol_tv || stopCriRho2 >= tol_rho2) && i < maxSamples
    
    % update step-size with new value of \rho
    dt = 1/(1/lambda + 1/(thrho2(i-1) + sigma^2));
    
    % Markov chain update
    Xk_grad = invQ(AtY/sigma^2 + Zk/thrho2(i-1),thrho2(i-1));
    
    Zk = Zk -dt*(Zk - Xk_grad)/thrho2(i-1) ...
                -dt*(Zk -proxG(Zk,thTV(i-1)))/lambda ...
                +sqrt(2*dt)*randn(N);
    
    % Update parameters to estimate
    % \alpha
    etaTVii = etaTV(i-1) + deltaTV(i)*((N*N)/thTV(i-1)- ...
        TVnorm(Zk))*exp(etaTV(i-1));    
    etaTV(i) = min(max(etaTVii,min_etatv),max_etatv);
    thTV(i)=exp(etaTV(i)); 
    
    % \rho^2
    etarho2ii = etarho2(i-1) + deltarho2(i)*(...
        (norm(Xk_grad-Zk,'fro')^2)/(2*thrho2(i-1)^2) ...
        - (N*N)/(2*thrho2(i-1)))*exp(etarho2(i-1));
    etarho2(i) = min(max(etarho2ii,min_etarho2),max_etarho2);
    thrho2(i)=exp(etarho2(i));
    
    % to compute \bar(\theta_TV) and \bar(\theta_rho2) as stopping criteria
    % for SAPG algorithm (we start at some threshold iteration to avoid
    % initial oscilations)
    if i > 5e2
        TVavg(i) = ((i-5e2-1)/(i-5e2))*TVavg(i-1) + (1/(i-5e2))*thTV(i);
        rho2avg(i) = ((i-5e2-1)/(i-5e2))*rho2avg(i-1) + (1/(i-5e2))*thrho2(i);
        
        stopCriRho2 = abs(rho2avg(i) - rho2avg(i-1))/rho2avg(i-1);
        stopCriTV = abs(TVavg(i) - TVavg(i-1))/TVavg(i-1);
    end
    
    i=i+1;
end

alpha = TVavg(i-1);
rho2 = rho2avg(i-1);

disp("Hyperparameters estimation (SAPG algorithm) finished");

end