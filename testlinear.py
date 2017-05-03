#! /usr/bin/python 
#encoding:gb18030
############################################
#
# Author: 
# E-Mail:@sogou-inc.com
# Create time: 2017 4月 20 11时37分15秒
# version 1.0
#
############################################


import tensorflow as tf
import sys,os,re
import numpy as np
#import matplotlib.pyplot as plt
learning_rate = 0.001
training_epochs = 1000
display_step = 50

# Training data
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
num_samples = train_X.shape[0]
# tf graph input
X = tf.placeholder("float", name = "x_input")
Y = tf.placeholder("float", name = "y_input")
# set model weights
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name = "bias")
# construct a linear model
pred = tf.add(tf.multiply(X, W), b)
# mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*num_samples)
# gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# initialize the variables
init = tf.global_variables_initializer()

# launch the graph
with tf.Session() as sess:
    sess.run(init)
    # fit all training data
    for epoch in range(training_epochs):
        for (x,y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X:x,Y:y})
        # display logs per epoch step
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

            # graphic display
            #   plt.plot(train_X, train_Y, 'ro', label = "Original Data")
            #   plt.plot(train_X, sess_run(W)*train_X + sess.run(b), label = "Fitted Line")
            #   plt.legend()
            #   plt.show()
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')
