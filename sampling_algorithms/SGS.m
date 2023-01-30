function [logPiTrace,mse_from_burnIn,mse_stationarity,...
          firstMoment,secondMoment] = SGS(varargin)
%
% function to sample using MCMC
% algorithm called Split Gibbs Sampler (see Vono, Dobigeon & Chainais, 2019)
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
% 'operator_transpose_operator' : A^T * A
%
% 'inverse_precision_matrix'    : inverse of the precision matrix
%                                 Q = 1/sigma^2 * H^T *H + 1/rho^2 * I_N
%
% 'number_samples_burn_in'      : number of samples to consider in the
%                                 burn-in stage
%
% 'number_samples_stationarity' : number of samples to generate in the
%                                 sampling stage (stationarity)
%
% 'hyperparam_tv'               : value for the total-variation
%                                 hyperparameter
%
% 'moreau_yosida_param'         : Moreau-Yosida approximation parameter
%
% 'sigma'                       : noise level of the likelihood
%
% 'rho_2'                       : augmentation parameter
%
% =========== Outputs ====================
%
% logPiTrace       : log(\pi) scalar value of each generated sample
%
% mse_from_burnIn  : mse between the posterior mean and the true image
%                    (from burn-in)
%
% mse_stationarity : mse between the posterior mean and the true image
%                    (from stationarity)
% 
% firstMoment      : posterior mean estimated from the samples generated
%                    by the MCMC method (in stationarity).
%
% secondMoment     : posterior second moment estimated from the samples
%                    generated by the MCMC method (in stationarity).
% 
% ===================================================

%--------------------------------------------------------------
% Read the input parameters
%--------------------------------------------------------------
for i = 1:2:(length(varargin)-1)
    switch upper(varargin{i})
        case 'TRUE_IMG'
            X          = varargin{i+1};
        case 'OBS_VECTOR'
            Y          = varargin{i+1};
        case 'OPERATOR'
            A          = varargin{i+1};
        case 'INVERSE_PRECISION_MATRIX'
            invQ      = varargin{i+1};
        case 'TRANSPOSE_OPERATOR'
            At         = varargin{i+1};
        case 'NUMBER_SAMPLES_BURN_IN'
            nBurnIn    = varargin{i+1};
        case 'NUMBER_SAMPLES_STATIONARITY'
            nSamples   = varargin{i+1};
        case 'HYPERPARAM_TV'
            alpha      = varargin{i+1};
        case 'MOREAU_YOSIDA_PARAM'
            lambda = varargin{i+1};
        case 'SIGMA'
            sigma      = varargin{i+1};
        case 'RHO_2'
            rho2       = varargin{i+1};
    end
end

%--------------------------------------------------------------
% Main body
%--------------------------------------------------------------
%%% Inverse of the precision matrix 
%%% Q = 1/sigma^2 * H^T *H + 1/rho^2 * I_N
%%% only depending on the image x and not on the parameter rho2
invQ = @(x) invQ(x,rho2);

% Gradients functions and Log(Pi) generator function
proxG = @(x) chambolle_prox_TV_stop(x, 'lambda',alpha*lambda,...
                                       'maxiter',20);
logPi = @(x) -(norm(Y-A(x),'fro')^2)/(2*(sigma^2)) -alpha*TVnorm(x);

% MYULA initial conditions
dim_Samples = size(X);
dt = 1/(1/lambda + 1/(rho2 + sigma^2));
Zk = At(Y);
logPiTrace = zeros(1,nSamples + nBurnIn);
logPiTrace(1) = logPi(Zk);

%%% burn-in variables
meanSamplesSGS_burnIn = Zk; % mean from burn-in stage
mse_from_burnIn = zeros(1,nSamples + nBurnIn);
perc = round(nBurnIn/10); % to show the progress percentage of burn-in

disp("Starting the execution of the MCMC method")
fprintf("Running burn-in stage     \n");
tic;
for i=2:nBurnIn
    % 1 Sampling the variable X using EPO algorithm
    Xk = EPO(Y,At,sigma,Zk,rho2,invQ);
    
    % 2 Sampling the variable Z using MYULA
    Zk = Zk - dt*(Zk - Xk)/rho2 ...
                  - dt*(Zk - proxG(Zk))/lambda ...
                  + sqrt(2*dt)*randn(dim_Samples);

    % save results
    logPiTrace(i)=logPi(Xk);
    
    %%% mean samples burn-in
    meanSamplesSGS_burnIn = ((i-1)/i)*meanSamplesSGS_burnIn + (1/i)*Xk;
    
    %%% mse burn-in
    mse_from_burnIn(i)=immse(meanSamplesSGS_burnIn,X);

    %%% to show progress of burn-in stage
    if rem(i,perc)==0
        fprintf('\b\b\b\b%2d%%\n', 10*round(i/perc));
    end
end
disp("Burn-in stage finished");
toc

%%% sampling stage

% to save the progress of the mse between the mean and the true image
mse_stationarity(nSamples)=0;
% to save the mean (first moment) of each iter
firstMoment = zeros(dim_Samples);
% to save the second moment of each iter
secondMoment = zeros(dim_Samples);
perc = round(nSamples/100); % to show the progress (percentage) of sampling

fprintf("Running sampling stage     \n");
tic;
for i = 1:nSamples
    % 1 Sampling the variable X using EPO algorithm
    Xk = EPO(Y,At,sigma,Zk,rho2,invQ);
    
    % 2 Sampling the variable Z using MYULA
    Zk = Zk - dt*(Zk - Xk)/rho2 ...
                  - dt*(Zk - proxG(Zk))/lambda ...
                  + sqrt(2*dt)*randn(dim_Samples);
    
    %%% Record value logPi(Xk) as a scalar summary
    logPiTrace(i+nBurnIn) = logPi(Xk);
              
    %%% mean samples burn-in
    meanSamplesSGS_burnIn =... 
        ((i + nBurnIn -1)/(i + nBurnIn))...
        *meanSamplesSGS_burnIn + (1/(i + nBurnIn))*Xk;
    
    %%% mse burn-in
    mse_from_burnIn(i+nBurnIn)=immse(meanSamplesSGS_burnIn,X);
    
    %%% mean of the samples in stationarity
    firstMoment = ((i-1)/i)*firstMoment + (1/i)*Xk;
    %%% second moment of the samples in stationarity
    secondMoment = ((i-1)/i)*secondMoment + (1/i)*(Xk.^2);
    %%% mean square error MSE in stationarity
    mse_stationarity(i)=immse(firstMoment,X);
    %%% to save progress of the work
    if rem(i,perc)==0
        fprintf('\b\b\b\b%2d%%\n', round(i/perc));
    end
end
disp("Sampling stage finished");
toc

end