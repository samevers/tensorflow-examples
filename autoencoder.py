#! /usr/bin/python 
#encoding:gb18030
############################################
#
# Author: 
# E-Mail:@sogou-inc.com
# Create time: 2017 4月 21 11时17分50秒
# version 1.0
#
############################################


import tensorflow as tf
import sys,os,re,time
import numpy as np

# Import minst data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

# Netword Parameters
n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784

# tf graph input
X = tf.placeholder("float", [None, n_input], name="input_data")

weights = {
    'encoder_h1':tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2':tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1':tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2':tf.Variable(tf.random_normal([n_hidden_1, n_input]))
    }

biases = {
    'encoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2':tf.Variable(tf.random_normal([n_input])),
}


# Buiding the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights["encoder_h1"]), biases["encoder_b1"]))

    # Decoder Hidder layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights["encoder_h2"]), biases["encoder_b2"]))
    
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights["decoder_h1"]), biases["decoder_b1"]))
    
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights["decoder_h2"]), biases["decoder_b2"]), name= "decoder_result")

    return layer_2



# Construct the model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Predict
y_pred = decoder_op

# Targets (Labels) are the input data
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_pred - y_true, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initialize the variables
init = tf.global_variables_initializer()

# Run the graph
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver(tf.all_variables())
    
    total_batch = int(mnist.train.num_examples/batch_size)

    for epoch in range(training_epochs):

        # Loop all the batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
   
            # Run optimization op (backprop) and cost op(to get loss value)
            _,c = sess.run([optimizer, cost], feed_dict={X:batch_xs})

            # Display logs per epoch
            if epoch % display_step == 0:
                print("Epoch:","%04d" %(epoch+1), " cost = ","{:.9f}".format(c)) 
                checkpoint_path = os.path.join("save_autoencoder/", 'model.ckpt') 
                saver.save(sess,checkpoint_path, global_step=training_epochs * total_batch + i)

    print("Optimization Finished!")

