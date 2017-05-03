#! /usr/bin/python 
#encoding:gb18030
############################################
#
# Author: 
# E-Mail:@sogou-inc.com
# Create time: 2017 4月 24 16时07分38秒
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
training_epochs = 15
batch_size = 128
display_step = 1

# Network Parmeters
n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden Layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights["h1"]), biases["b1"])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights["h2"]), biases["b2"])
    layer_2 = tf.nn.relu(layer_2)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights["out"]) + biases["out"]
    return out_layer

# Store layers weight & bias
weights = {
    "h1":tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    "h2":tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    "out":tf.Variable(tf.random_normal([n_hidden_2, n_classes])),
}

biases = {
    "b1":tf.Variable(tf.random_normal([n_hidden_1])),
    "b2":tf.Variable(tf.random_normal([n_hidden_2])),
    "out":tf.Variable(tf.random_normal([n_classes])),
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Initialize the variables
init = tf.global_variables_initializer()

# Launch the model
with tf.Session() as sess:
    sess.run(init)
    
    # Training loop
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for e in range(total_batch):
            batchx, batchy = mnist.train.next_batch(batch_size)

            # Run Optimization op(backprop) and cost op(to get loss value)
            _,c = sess.run([optimizer, cost], feed_dict={x:batchx, y:batchy})

            # Compute average loss
            avg_cost += c/total_batch
        
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", "%04d" % (epoch+1), "cost = ", "{:.9f}".format(avg_cost))
    print("Optimization finished!")


    # Test model
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.train.labels}))
