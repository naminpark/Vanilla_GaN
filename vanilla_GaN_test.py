from vanilla_GaN import *
from tensorflow.examples.tutorials.mnist import input_data


GaN=vanilla_GaN()
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
GaN.RUN(mnist)

