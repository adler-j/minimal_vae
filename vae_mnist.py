"""Minimal implementation of an Variational Auto-Encoder for MNIST."""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.examples.tutorials.mnist import input_data


session = tf.InteractiveSession()


with tf.name_scope('placeholders'):
    x_true = tf.placeholder(tf.float32, [None, 28, 28, 1])
    z = tf.placeholder(tf.float32, [None, 128])

with tf.variable_scope('encoder'):
    x = layers.conv2d(x_true, num_outputs=64, kernel_size=5, stride=2)
    x = layers.conv2d(x, num_outputs=128, kernel_size=5, stride=2)
    x = layers.conv2d(x, num_outputs=256, kernel_size=5, stride=2)

    x = layers.flatten(x)
    mu = layers.fully_connected(x, num_outputs=128, activation_fn=None)
    logsigma = layers.fully_connected(x, num_outputs=128, activation_fn=None)
    sigma = tf.exp(logsigma)

with tf.variable_scope('latent_variable'):
    z = mu + tf.random_normal(tf.shape(sigma)) * sigma

with tf.variable_scope('decoder'):
    x = layers.fully_connected(z, num_outputs=4096)
    x = tf.reshape(x, [-1, 4, 4, 256])

    x = layers.conv2d_transpose(x, num_outputs=128, kernel_size=5, stride=2)
    x = layers.conv2d_transpose(x, num_outputs=64, kernel_size=5, stride=2)
    x = layers.conv2d_transpose(x, num_outputs=1, kernel_size=5, stride=2,
                                activation_fn=tf.nn.sigmoid)
    reconstruction = x[:, 2:-2, 2:-2, :]

with tf.name_scope('loss'):
    latent_losses = 0.5 * tf.reduce_sum(tf.square(mu) +
                                        tf.square(sigma) -
                                        tf.log(tf.square(sigma)) - 1,
                                        axis=1)

    reconstruction_losses = tf.reduce_sum(tf.square(x_true - reconstruction),
                                          axis=[1, 2, 3])
    loss = tf.reduce_mean(reconstruction_losses + latent_losses)

with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train = optimizer.minimize(loss)

tf.global_variables_initializer().run()

mnist = input_data.read_data_sets('MNIST_data')

for i in range(10000):
    batch = mnist.train.next_batch(100)
    images = batch[0].reshape([-1, 28, 28, 1])

    session.run(train, feed_dict={x_true: images})

    if i % 100 == 0:
        print('iter={}/10000'.format(i))
        z_validate = np.random.randn(1, 128)
        generated = reconstruction.eval(feed_dict={z: z_validate}).squeeze()

        plt.figure('results')
        plt.imshow(generated, clim=[0, 1], cmap='bone')
        plt.pause(0.001)
