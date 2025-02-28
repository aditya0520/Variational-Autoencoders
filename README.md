# Variational Autoencoders (VAE) 

This repository showcases the development and experimentation with various Variational Autoencoder (VAE) models for generative modeling and semi-supervised learning tasks. The project covers several VAE variants, including standard VAE, Mixture of Gaussians VAE (GMVAE), Importance Weighted Autoencoder (IWAE) and Semi-Supervised VAE (SSVAE)

## Project Overview
The project focuses on building and experimenting with different VAE architectures to learn probabilistic models of high-dimensional data, primarily using the MNIST and SVHN datasets. The models developed in this project include:

1. **Variational Autoencoder (VAE):**
   - Implemented a latent variable model to learn the distribution of the MNIST dataset.
   - Utilized the reparameterization trick to sample latent variables and optimized the Evidence Lower Bound (ELBO).

2. **Mixture of Gaussians VAE (GMVAE):**
   - Extended the VAE with a mixture of Gaussians as the prior distribution to improve expressivity.
   - Implemented a numerically stable negative ELBO bound using Monte Carlo sampling.

3. **Importance Weighted Autoencoder (IWAE):**
   - Developed the IWAE model to tighten the lower bound on the marginal log-likelihood using multiple samples from the approximate posterior.
   - Evaluated the model for different sample sizes (m = 1, 10, 100, 1000).

4. **Semi-Supervised VAE (SSVAE):**
   - Implemented a semi-supervised learning approach leveraging both labeled and unlabeled data.
   - Achieved >90% classification accuracy on the MNIST dataset by combining generative modeling with a classification objective.

## Setup
1. Clone the repository:
```bash
git clone https://github.com/aditya0520/Variational-Autoencoders.git
cd variational-autoencoders-project
```

2. Create and activate the conda environment:
```bash
conda env create -f src/environment.yml
conda activate vae-env
```

3. Run model training and evaluations:
```bash
# Train the VAE model
python main.py --model vae --train

# Train the GMVAE model
python main.py --model gmvae --train

# Train the SSVAE model
python main.py --model ssvae --train

```

## Results
The project generated promising results with all VAE variants, showcasing their effectiveness in both generative modeling and semi-supervised learning contexts. The SSVAE model achieved over 90% classification accuracy on the MNIST dataset.

## Visualization
To visualize generated samples:
```bash
# Generate samples from the VAE model
python main.py --model vae

# Generate samples from the GMVAE model
python main.py --model gmvae
```

---
Thank you for exploring this project! 
