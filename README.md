# GMM and VAE Repository

This repository contains implementations of two popular machine learning models: Gaussian Mixture Model (GMM) and Variational Autoencoder (VAE). The GMM model focuses on clustering non-circular distributed data and provides an alternative approach to traditional clustering algorithms like K-Means. The VAE model, on the other hand, is designed for generating and reconstructing MNIST digits.

## Gaussian Mixture Model (GMM)

The GMM implementation in this repository offers a powerful tool for clustering data with non-circular distributions. It uses a probabilistic model that represents the data as a mixture of Gaussian components. The GMM algorithm iteratively learns the parameters of the Gaussian components, allowing it to capture complex and overlapping clusters in the data.

Key features of the GMM implementation:

Handles non-circular distributions: GMM is particularly useful when dealing with data that doesn't conform to circular clusters, as it can model arbitrary shapes and capture the underlying distributions more accurately.

Flexible cluster assignment: Unlike K-Means, which assigns each point to a single cluster, GMM assigns probabilities to each point belonging to multiple clusters. This soft assignment is beneficial when dealing with data points that lie in the overlapping regions of multiple clusters.

## Variational Autoencoder (VAE)

The VAE implementation in this repository showcases its ability to generate and reconstruct MNIST digits. VAE is a generative model that combines deep learning techniques with variational inference. It learns a low-dimensional representation (latent space) of the input data -  as implemented in the repo, latent_dim just equals to 2, allowing it to generate new samples from this learned distribution.

Key features of the VAE implementation:

Generating MNIST digits: The VAE model can generate new MNIST digits by sampling from the latent space and decoding them into realistic digit images.
Reconstructing MNIST digits: Given an input MNIST digit, the VAE can reconstruct it by encoding it into the latent space and then decoding it back into an image. This reconstruction showcases the model's ability to capture the essential features of the input data.
Training process included: The provided code includes the training process for the VAE model using the MNIST dataset. You can easily train the model on your own or use the pre-trained weights for generating and reconstructing digits.

## Acknowledgments

The GMM and VAE implementations in this repository were inspired by various research papers, online tutorials, and community contributions. We express our gratitude to the researchers, authors, and developers who have made their work available to the public.

## References
https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
https://www.oreilly.com/library/view/python-data-science/9781491912126/
https://www.oranlooney.com/post/ml-from-scratch-part-5-gmm/
https://theaisummer.com/latent-variable-models/#variational-autoencoders
https://theaisummer.com/jax-tensorflow-pytorch/
https://www.jeremyjordan.me/variational-autoencoders/
