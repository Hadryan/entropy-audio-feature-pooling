# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 02:52:14 2021
@author: George
"""
import tensorflow as tf

def KL_div(mu, sigma):
    '''KL divergence between N(mu,sigma**2) and N(0,1)'''
    return .5 * (mu**2 + sigma**2 - 2 * tf.math.log(sigma) - 1)

def KL_div2(mu, sigma, mu1, sigma1):
    '''KL divergence between N(mu,sigma**2) and N(mu1,sigma1**2)'''
    return 0.5 * ((sigma/sigma1)**2 + (mu - mu1)**2/sigma1**2 - 1 + 2*(tf.math.log(sigma1) - tf.math.log(sigma)))

def sample_lognormal(mean, sigma=None, sigma0=1.):
    '''Samples a log-normal using the reparametrization trick'''
    e = tf.keras.backend.random_normal(tf.shape(mean), mean=0., stddev=1.)
    return tf.exp(mean + sigma * sigma0 * e)

def batch_average(x):
    '''Sum over all dimensions and averages over the first'''
    return tf.math.reduce_mean(tf.math.reduce_sum(tf.reshape( x , shape=( tf.shape( x )[0] , -1 )),1))