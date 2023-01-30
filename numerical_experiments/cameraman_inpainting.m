function [X, Y, sigma, A, At, AtA, invQ] = cameraman_inpainting()
%
% function to produce the necessary data to perform the cameraman
% inpainting experiment
%
%% =========== Outputs ====================
%
% X      : true image
%
% Y      : observation vector
%
% sigma  : noise level 
% 
% A      : Linear operator
%
% H_FFT  : Linear operator (in the fourier domain)
%
% At     : Transpose of the linear operator
%
% AtA    : At*A operator
%
% invQ   : Inverse of the precision matrix
%          Q = 1/sigma^2 * H^T *H + 1/rho^2 * I_N
%          to compute X_grad on latent-space methods (ls-MYULA, ls-SK-ROCK)
% 
% ===================================================

%--------------------------------------------------------------
% Main body
%--------------------------------------------------------------
X = double(imread('cameraman.tif')); % Cameraman img.
N = size(X,1); % Dimension of the image

%%%% function handle for inpainting operator
load('data/inpainting_operator.mat','H');
% H = rand(size)> 0.4; % 40% of the pixels missing
Y = X.*H;

% set BSNR, noise level (sigma) and observation Y
BSNRdb = 40; % SNR expressed in decibels
P_signal = var(X(:)); % signal power
sigma = sqrt((P_signal/10^(BSNRdb/10))); % standard deviation of the noise
Y = H.*(Y + sigma*randn(N)); % observation vector
Y = Y(Y~=0); % recover only observed pixels

%%%% function handle
A = @(x) x(H>0);
At = @(x) Transp(H,x);
AtA = @(x) H.*x;

%%% precompute the inverse of the precision matrix 
%%% Q = 1/sigma^2 * H^T *H + 1/rho^2 * I_N
%%% cf. Sherman-Morrison-Woodbury formula
invQ = @(x,rho2) (rho2*(1 - (rho2/(rho2 + sigma^2))*H)).*x;

end

function result = Transp(H,x)
result = zeros(size(H));
result(H) = x;
end