
%-------------------------------------------------------------------------%
%             E-PO ALGORITHM TO SAMPLE THE VARIABLE X                     %
%-------------------------------------------------------------------------%

function x = EPO(y,At,sigma,Z,rho2,invQ)

%-------------------------------------------------------------------------%
% This function computes the E-PO algorithm as described in the paper of C.
% Gilavert et al., 2015. This algorithm deals with the exact resolution 
% case of the linear system Q*x = eta and with a guaranteed convergence to 
% the target distribution.

    % INPUTS:
        % y: noisy observation (1D array).
        % At: transpose of the operator in the inverse problem y = A*x + n.
        % sigma: user-defined standard deviation of the noise.
        % Z: current MCMC iterates of the latent variable.
        % rho2: augmentation parameter.
        % invQ: pre-computed covariance matrix involved in the posterior
        %       distribution of the variable of interest x.
        
    % OUTPUT:
        % x: sample from the posterior distribution of x (2D-array).
%-------------------------------------------------------------------------%

%-------------------------------------------------------------------------%
% 1. Sample eta from N(Q*mu,Q)
    % 1.1. Sample eta_y from N(y,sigma^2*I_M)
    if isvector(y)
        diagY = diag((sigma^2)*speye(size(y,1)));
        eta_y = mvnrnd(y, diagY')';
        clear y diagY;
    else
        eta_y = y + sigma*randn(size(y,1));
    end

    % 1.2. Sample eta_x from N(z,rho^2*I_N)
    eta_x = Z + sqrt(rho2)*randn(size(Z));

    % 3. Set eta
    eta_aux = At(eta_y)/(sigma^2) + eta_x/rho2;
    clear eta_y eta_x;
%-------------------------------------------------------------------------%

%-------------------------------------------------------------------------%
% 2. Compute an exact solution x_new of Q*x = eta <=> x = invQ*eta
x = invQ(eta_aux);
%-------------------------------------------------------------------------%

end