import os
import time
import argparse
import importlib
import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import scipy

from scipy import optimize as opt
from scipy.optimize import minimize_scalar
from scipy.misc import logsumexp

from visualize import *


class WassersteinGAN(object):
    def __init__(self, g_net, d_net, x_sampler, z_sampler, data, model, batch_size, epsilon):
        self.model = model
        self.data = data
        self.g_net = g_net
        self.d_net = d_net
        self.x_sampler = x_sampler
        self.z_sampler = z_sampler
        self.x_dim = self.d_net.x_dim
        self.z_dim = self.g_net.z_dim
        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')      # External inputs: so use of placeholders
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.lambda_hat = tf.placeholder(tf.float32, name='lambda_hat')
        self.mu_hat = tf.placeholder(tf.float32, name='mu_hat')

        self.batch_size = batch_size
        self.epsilon = epsilon
        self.x_ = self.g_net(self.z)

        self.d = self.d_net(self.x, reuse=False)
        self.d_ = -self.d_net(self.x_)         # I guess this loss function should be -self.d_net(self.x_)

        self.g_loss = tf.reduce_logsumexp(tf.divide(self.d_,self.lambda_hat + self.epsilon))
        self.d_loss = self.lambda_hat*tf.reduce_logsumexp(tf.divide(self.d,self.lambda_hat + self.epsilon)) + self.mu_hat*tf.reduce_logsumexp(tf.divide(self.d_, self.mu_hat + self.epsilon))  # This loss function looks fine

        self.reg = tc.layers.apply_regularization(
            tc.layers.l1_regularizer(2.5e-5),
            weights_list=[var for var in tf.global_variables() if 'weights' in var.name]
        )
        self.g_loss_reg = self.g_loss + self.reg
        self.d_loss_reg = self.d_loss + self.reg

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_rmsprop = tf.train.RMSPropOptimizer(learning_rate=2e-5)\
                .minimize(self.d_loss_reg, var_list=self.d_net.vars)
            self.g_rmsprop = tf.train.RMSPropOptimizer(learning_rate=2e-5)\
                .minimize(self.g_loss_reg, var_list=self.g_net.vars)        

        self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.d_net.vars]
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    # Function for the optimization problem

    def f(self, x, dta):
        max_val = np.amax(dta)
        epsilon = 1.e-10
        rho = np.log(self.batch_size)
        kappa = rho - np.log(self.batch_size)
        return x*kappa + (x+epsilon)*( max_val/(x+epsilon) + logsumexp( (dta - max_val)/(x+epsilon) )  )


    def train(self, num_batches=100000):
        plt.ion()
        self.sess.run(tf.global_variables_initializer())
        start_time = time.time()
        for t in range(0, num_batches):
            d_iters = 10
            if t % 500 == 0 or t < 25:
                 d_iters = 100

            for _ in range(0, d_iters):
                bx = self.x_sampler(self.batch_size)
                bz = self.z_sampler(self.batch_size, self.z_dim)
                self.sess.run(self.d_clip)
                values_disc = self.sess.run(self.d, feed_dict={self.x: bx, self.z: bz})  # These do not need lambda_hat and mu_hat and henc enot passe
                values_gen = self.sess.run(self.d_, feed_dict={self.x: bx, self.z: bz})  # lambda_hat and mu_hat only neede for the modified loss function
                # First solving the one-dimensional optimization problem:
                lambda_hat = minimize_scalar(self.f, bounds=(0,1000), args=values_disc, method='bounded')
                lambda_hat = lambda_hat.x
                mu_hat = minimize_scalar(self.f, bounds=(0,1000), args=-values_gen, method='bounded')
                mu_hat = mu_hat.x

                #print("lambda_hat=", lambda_hat)
                #print("mu_hat", mu_hat)

                # Pass both lambda_hat and mu_hat in the optimization objective
                self.sess.run(self.d_rmsprop, feed_dict={self.x: bx, self.z: bz, self.lambda_hat: lambda_hat, self.mu_hat: mu_hat})

            # New optimization problem for the generator
            bz = self.z_sampler(self.batch_size, self.z_dim)
            values_gen_outer = self.sess.run(self.d_, feed_dict={self.x: bx, self.z: bz})
            mu_hat_outer = minimize_scalar(self.f, bounds=(0,1000), args=-values_gen_outer, method='bounded')
            mu_hat_outer = mu_hat_outer.x
            # Pass it in the optimization objective
            self.sess.run(self.g_rmsprop, feed_dict={self.z: bz, self.x: bx, self.lambda_hat: lambda_hat, self.mu_hat: mu_hat_outer})

            if t % 100 == 0:
                bx = self.x_sampler(self.batch_size)
                bz = self.z_sampler(self.batch_size, self.z_dim)

                d_loss = self.sess.run(
                    self.d_loss, feed_dict={self.x: bx, self.z: bz, self.lambda_hat: lambda_hat, self.mu_hat: mu_hat}
                )
                g_loss = self.sess.run(
                    self.g_loss, feed_dict={self.z: bz, self.x: bx, self.lambda_hat: lambda_hat, self.mu_hat: mu_hat}
                )
                print('Iter [%8d] Time [%5.4f] d_loss [%.4f] g_loss [%.4f]' %
                        (t, time.time() - start_time, d_loss - g_loss, g_loss))

            if t % 100 == 0:
                bz = self.z_sampler(self.batch_size, self.z_dim)
                bx = self.sess.run(self.x_, feed_dict={self.z: bz})
                bx = xs.data2img(bx)
                fig = plt.figure(self.data + '.' + self.model)
                grid_show(fig, bx, xs.shape)
                fig.savefig('logs/{}/{}.pdf'.format(self.data, t/100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--model', type=str, default='r_dcgan')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--batch_size', type=float, default=64)
    parser.add_argument('--epsilon', type=float, default=1.e-10)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    data = importlib.import_module(args.data)
    model = importlib.import_module(args.data + '.' + args.model)
    xs = data.DataSampler()
    zs = data.NoiseSampler()
    d_net = model.Discriminator()
    g_net = model.Generator()
    wgan = WassersteinGAN(g_net, d_net, xs, zs, args.data, args.model, args.batch_size, args.epsilon)
    wgan.train()



