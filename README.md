# pytorch-ddpm

Simple [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2010.02502) implementation in PyTorch.

The base implementation is taken from: https://github.com/cloneofsimo/minDiffusion.

Modifications that I have made on top includes
* class bias
* positional encoding
* residual connection to the model

In order to generate diffusion model for mnist, simply run
```
$ python train_mnist.py
```

sample MNIST outputs are available [here](./Oct-00-29-21)

Please note that `train_celelba.py` and `train_cifar10.py` are not yet working