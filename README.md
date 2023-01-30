# ls-MCMC
The following code reproduce some of the numerical results available in the paper "The split Gibbs sampler revisited: improvements to its algorithmic structure and augmented target distribution"

## Usage
Download the code and run the `MCMC_sampler_code.m` script in MATLAB.

You can change the imaging experiment by changing line 31 with `cameraman_deblurring()` or `cameraman_inpainting()`, and line 35 with `SAPG_deb_TV` or `SAPG_inp_TV`. Remember to change both lines for a correct implementation.

You can also change the sampler set in line 58 with one of the following: `MYULA`, `SKROCK`, `SGS`, and the proposed latent-space MCMC methods: `lsMYULA` and `lsSKROCK`.