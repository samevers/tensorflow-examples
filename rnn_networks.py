#! /usr/bin/python 
#encoding:gb18030
############################################
#
# Author: 
# E-Mail:@sogou-inc.com
# Create time: 2017 4月 24 16时56分25秒
# version 1.0
#
############################################


import tensorflow as tf
import numpy as np

# Import minst data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters 
learning_rate = 0.001
training_iters = 10000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

# tf Graph input
weights = {
    "out":tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    "out":tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):
    # Prepare data shape to match "rnn" function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstach to get a list of "n_steps" tensors of shape (batch_size, n_input)
    x = tf.unstach(x,n_steps,1)
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.add(tf.matmul(outputs[-1], weights["out"]), biases["out"], name = "rnnOutput")


pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Evaluate the model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initializer
init = tr.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    step = 1
    # Keep training util reach max iterations
    while step * batch_size < training_iters:
        batchx, batchy = mnist.train.next_batch(batch_size)
        # Reshape data to 28 seq of 28 elements
        batchx = batchx.reshape((batch_size,n_steps,n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x:batchx, y:batchy})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x:batchx, y:batchy})
            # Calculate the loss
            loss = sess.run(cost, feed_dict={x:batchx, y:batchy})
            print ("Iter:" + str(step*batch_size) + ", Minibatch loss = " + \
                    "{:.6f}".format(loss) + ",Training Accuracy = " + \
                    "{:.5f}".format(acc))
        step += 1
    print ("Optimizer Finished!")
    

    # Test, size of one batch_size.
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1,n_steps,n_input))
    test_label = mnist.test.labels[:test_len]
    print ("Testing Accuracy : ", \
        sess.run(accuracy, feed_dict={x:test_data, y:test_label}))
