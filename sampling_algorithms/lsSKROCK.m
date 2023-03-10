function [logPiTrace,mse_from_burnIn,mse_stationarity,...
          firstMoment,secondMoment] = lsSKROCK(varargin)
%
% function to sample using proximal MCMC
% algorithm called ls-SK-ROCK (see Algorithm 4.2)
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
% 'approximation_param'         : Moreau-Yosida approximation parameter
%
% 'sigma'                       : noise level of the likelihood
%
% 'rho_2'                       : augmentation parameter
%
% 'n_stages'                    : number of internal stages of SKROCK
%
% 'perc_dt'                     : fraction of the maximum step-size to use
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
        case 'N_STAGES'
            nStages    = varargin{i+1};
        case 'PERC_DT'
            percDt     = varargin{i+1};
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

%%% Lipschitz constant for ls-SK-ROCK
LipschitzC = 1/lambda + 1/(rho2 + sigma^2);

% Gradients functions and Log(Pi) generator function
proxG = @(x) chambolle_prox_TV_stop(x, 'lambda',alpha*lambda,...
                                       'maxiter',20);
logPi = @(x) -(norm(Y-A(x),'fro')^2)/(2*(sigma^2)) -alpha*TVnorm(x);

%%% SK-ROCK parameters
%%% First kind Chebyshev function
T_s = @(s,x) cosh(s*acosh(x));

%%% First derivative Chebyshev polynomial first kind
T_prime_s = @(s,x) s*sinh(s*acosh(x))/sqrt(x^2 -1);

%%% computing SK-ROCK stepsize given a number of stages
eta=0.05;
denNStag=(2-(4/3)*eta);
rhoSKROCK = ((nStages - 0.5)^2)*denNStag -1.5; % stiffness ratio
dt = percDt*rhoSKROCK/LipschitzC; %%% step-size
w0=1 + eta/(nStages^2);
w1=T_s(nStages,w0)/T_prime_s(nStages,w0);

%%% precompute the SK-ROCK parameters mu, nu and kappa
mu = zeros(1,nStages);
nu = zeros(1,nStages);
kappa = zeros(1,nStages);
for jj = 1:nStages
    if jj == 1
        mu(jj) = w1/w0;
        nu(jj) = nStages*w1/2;
        kappa(jj) = nStages*(w1/w0);
    else
        mu(jj) = 2*w1*T_s(jj-1,w0)/T_s(jj,w0);
        nu(jj) = 2*w0*T_s(jj-1,w0)/T_s(jj,w0);
        kappa(jj) = 1-nu(jj);
    end
end

% SKROCK initial conditions
AtY = At(Y);
dim_Samples = size(X);
Zk = AtY;
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
    Q=sqrt(2*dt)*randn(dim_Samples); %%% diffusion term
    % Z_MC + nu_1*Q term
    Zk_plus_noise = Zk + nu(1)*Q;
    % First step: computing the exact mean of p(x|z,u,y)
    Xk_grad = invQ(AtY/sigma^2 + Zk_plus_noise/rho2);
    % compute Grad_f in p(z|x,u)
    grad_f = (1 / rho2) * (Zk_plus_noise - Xk_grad);
    % 1. Sample z from p(z|x) using SKROCK
    gradU = grad_f + (Zk_plus_noise - proxG(Zk_plus_noise))/lambda;
    %%% Sampling the auxiliary variable z1 (SKROCK)
    %%% SKROCK first iteration
    Xts= Zk - mu(1)*dt*gradU +kappa(1)*Q;
    XtsMinus2 = Zk;
    for js = 2:nStages % deterministic stages
        XprevSMinus2 = Xts;
        % 1. computing the exact mean of p(x|z,u,y) 
        Xk_grad = invQ(AtY/sigma^2 + Xts/rho2);
        % compute Grad_f in p(z|x,u)
        grad_f = (1 / rho2) * (Xts - Xk_grad);
        gradU = grad_f + (Xts - proxG(Xts))/lambda;
        %%% updating z
        Xts=-mu(js)*dt*gradU + nu(js)*Xts + kappa(js)*XtsMinus2;        
        XtsMinus2=XprevSMinus2;
    end
    Zk=Xts;
    
    % sample from the marginal of X
    Xk = EPO(Y,At,sigma,Zk,rho2,invQ);

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

%%% sampling stage variables
% to save the progress of the mse between the mean and the true image
mse_stationarity(nSamples)=0;
% to save the mean (first moment) on each iter
firstMoment = zeros(dim_Samples);
% to save the second moment on each iter
secondMoment = zeros(dim_Samples);
perc = round(nSamples/100); % to show the progress percentage of sampling

fprintf("Running sampling stage     \n");
tic;
for i = 1:nSamples
    Q=sqrt(2*dt)*randn(dim_Samples); %%% diffusion term
    % Z_MC + nu_1*Q term
    Zk_plus_noise = Zk + nu(1)*Q;
    % First step: computing the exact mean of p(x|z,u,y)
    Xk_grad = invQ(AtY/sigma^2 + Zk_plus_noise/rho2);
    % compute Grad_f in p(z|x,u)
    grad_f = (1 / rho2) * (Zk_plus_noise - Xk_grad);
    % 1. Sample z from p(z|x) using SKROCK
    gradU = grad_f + (Zk_plus_noise - proxG(Zk_plus_noise))/lambda;
    %%% Sampling the auxiliary variable z1 (SKROCK)
    %%% SKROCK first iteration
    Xts= Zk - mu(1)*dt*gradU +kappa(1)*Q;
    XtsMinus2 = Zk;
    for js = 2:nStages % deterministic stages
        XprevSMinus2 = Xts;
        % 1. computing the exact mean of p(x|z,u,y) 
        Xk_grad = invQ(AtY/sigma^2 + Xts/rho2);
        % compute Grad_f in p(z|x,u)
        grad_f = (1 / rho2) * (Xts - Xk_grad);
        gradU = grad_f + (Xts - proxG(Xts))/lambda;
        %%% updating z
        Xts=-mu(js)*dt*gradU + nu(js)*Xts + kappa(js)*XtsMinus2;        
        XtsMinus2=XprevSMinus2;
    end
    Zk=Xts;
    
    % sample from the marginal of X
    Xk = EPO(Y,At,sigma,Zk,rho2,invQ);

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