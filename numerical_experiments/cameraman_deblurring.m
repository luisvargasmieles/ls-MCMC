function [X, Y, sigma, A, At, AtA, invQ] = cameraman_deconvolution()
%
% function to produce the necessary data to perform the cameraman
% deconvolution experiment
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

%%%% function handle for uniform blur operator
h = [1 1 1 1 1];
lh = length(h);
h = h/sum(h);
h = [h zeros(1,N-length(h))];
h = cshift(h,-(lh-1)/2);
h = h'*h;

H_FFT = fft2(h);
HC_FFT = conj(H_FFT);

A = @(x) real(ifft2(H_FFT.*fft2(x))); % A operator
At = @(x) real(ifft2(HC_FFT.*fft2((x)))); % A transpose operator
AtA = @(x) real(ifft2((HC_FFT.*H_FFT).*fft2((x)))); % AtA operator

% set BSNR, noise level (sigma) and observation Y
BSNRdb = 40; % SNR expressed in decibels
Ax = A(X);
sigma = norm(Ax-mean(mean(Ax)),'fro')/sqrt(N*N*10^(BSNRdb/10));
Y = Ax + sigma*randn(N);

%%% precompute the inverse of the precision matrix 
%%% Q = 1/sigma^2 * H^T *H + 1/rho^2 * I_N
%%% cf. Sherman-Morrison-Woodbury formula
invQ_FFT = @(rho2) 1./((abs(H_FFT).^2)./(sigma^2) + 1/rho2);
invQ = @(x,rho2) real(ifft2(invQ_FFT(rho2).*fft2(x)));
end