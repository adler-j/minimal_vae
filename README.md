Minimal Variational Auto-Encoder
================================

This is a minimal implementation of an Variational Auto-Encoder in Tensorflow applied to MNIST.

Some example generated numbers:

![VAE results](https://user-images.githubusercontent.com/2202312/41727363-75961a06-7574-11e8-92b6-849efaa7f9c4.png)

How to run
----------

Simply clone the directory and run the file [`vae_mnist.py`](vae_mnist.py). Results will be displayed in real time, while full training takes a few minutes.

Implementation details
----------------------

The implementation follows [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). Both the generator and discriminator uses 3 convolutional layers with 5x5 convolutions, with obvious room for improvements.
