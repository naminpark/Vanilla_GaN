import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os



class vanilla_GaN():

    def __init__(self, weight='he', bias='zero'):

        self.n_input = 784
        self.n_hidden1 = 128
        self.n_hidden2 = 1
        self.z_num=100

        self.setVariable()

        self.weights= self.weight_initializer[weight]
        self.biases = self.bias_initializer[bias]

        self.theta_D = [self.weights['D_W1'], self.weights['D_W2'], self.biases['D_b1'], self.biases['D_b2']]
        self.theta_G = [self.weights['G_W1'], self.weights['G_W2'], self.biases['G_b1'], self.biases['G_b2']]

        self.Model()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def setVariable(self):
        # CNN:input: 28*28
        # random_noise: 10*10

        self.X = tf.placeholder(tf.float32, shape=[None, self.n_input])
        self.Z = tf.placeholder(tf.float32, shape=[None, self.z_num])

        self.normal_weights = {
            'D_W1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden1])),
            'D_W2': tf.Variable(tf.random_normal([self.n_hidden1, self.n_hidden2])),

            'G_W1': tf.Variable(tf.random_normal([self.z_num, self.n_hidden1])),
            'G_W2': tf.Variable(tf.random_normal([self.n_hidden1, self.n_input]))
        }

        self.he_weights = {
            'D_W1': tf.get_variable('D_W1_he',[self.n_input, self.n_hidden1],initializer=tf.contrib.layers.variance_scaling_initializer()),
            'D_W2': tf.get_variable('D_W2_he',[self.n_hidden1, self.n_hidden2],initializer=tf.contrib.layers.variance_scaling_initializer()),

            'G_W1': tf.get_variable('G_W1_he',[self.z_num, self.n_hidden1],initializer=tf.contrib.layers.variance_scaling_initializer()),
            'G_W2': tf.get_variable('G_W2_he',[self.n_hidden1, self.n_input],initializer=tf.contrib.layers.variance_scaling_initializer())
        }


        self.normal_biased = {

            'D_b1': tf.Variable(tf.random_normal([self.n_hidden1])),
            'D_b2': tf.Variable(tf.random_normal([self.n_hidden2])),

            'G_b1': tf.Variable(tf.random_normal([self.n_hidden1])),
            'G_b2': tf.Variable(tf.random_normal([self.n_input]))

        }

        self.zero_biased = {
            'D_b1': tf.Variable(tf.zeros([self.n_hidden1])),
            'D_b2': tf.Variable(tf.zeros([self.n_hidden2])),

            'G_b1': tf.Variable(tf.zeros([self.n_hidden1])),
            'G_b2': tf.Variable(tf.zeros([self.n_input]))


        }

        self.weight_initializer = {'normal':self.normal_weights, 'he':self.he_weights}
        self.bias_initializer = {'normal':self.normal_biased, 'zero':self.zero_biased}


    def sample_Z(self,m, n):
        return np.random.uniform(-1., 1., size=[m, n])


    def generator(self,z):
        G_h1 = tf.nn.relu(tf.matmul(z, self.weights['G_W1']) + self.biases['G_b1'])
        G_log_prob = tf.matmul(G_h1, self.weights['G_W2']) + self.biases['G_b2']
        G_prob = tf.nn.sigmoid(G_log_prob)

        return G_prob


    def discriminator(self,x):
        D_h1 = tf.nn.relu(tf.matmul(x, self.weights['D_W1']) +self.biases['D_b1'])
        D_logit = tf.matmul(D_h1, self.weights['D_W2']) + self.biases['D_b2']
        D_prob = tf.nn.sigmoid(D_logit)

        return D_prob, D_logit


    def Model(self):

        self.G_sample = self.generator(self.Z)
        _, D_logit_real = self.discriminator(self.X)
        _, D_logit_fake = self.discriminator(self.G_sample)

        # D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
        # G_loss = -tf.reduce_mean(tf.log(D_fake))

        # Alternative losses:
        # -------------------
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        self.D_loss = D_loss_real + D_loss_fake
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

        self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.theta_D)
        self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.theta_G)




    def RUN(self,mnist):



        mb_size = self.n_hidden1 #128
        Z_dim = self.z_num #100

        if not os.path.exists('out/'):
            os.makedirs('out/')

        i = 0

        for it in range(1000000):
            if it % 1000 == 0:
                samples = self.sess.run(self.G_sample, feed_dict={self.Z: self.sample_Z(16, Z_dim)})

                fig = self.plot(samples)
                plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                i += 1
                plt.close(fig)

            X_mb, _ = mnist.train.next_batch(mb_size)

            _, D_loss_curr = self.sess.run([self.D_solver, self.D_loss], feed_dict={self.X: X_mb, self.Z: self.sample_Z(mb_size, Z_dim)})
            _, G_loss_curr = self.sess.run([self.G_solver, self.G_loss], feed_dict={self.Z: self.sample_Z(mb_size, Z_dim)})

            if it % 1000 == 0:
                print('Iter: {}'.format(it))
                print('D loss: {:.4}'. format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()



    def plot(self,samples):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        return fig


if __name__ == "__main__":

     GaN=vanilla_GaN()
     mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
     GaN.RUN(mnist)

