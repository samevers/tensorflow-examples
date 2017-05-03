#! /usr/bin/python 
#encoding:gb18030
############################################
#
# Author: 
# E-Mail:@sogou-inc.com
# Create time: 2017 4月 20 14时37分32秒
# version 1.0
#
############################################


import tensorflow as tf
import sys,os,re
import time

# Import minst data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# Parameters
learning_rate = 0.001
training_epochs = 25
batch_size = 100
display_step = 1

# tf graph input
x = tf.placeholder(tf.float32, [None,784],name = "digitInputs")# minst data image os shape 28 * 28
y = tf.placeholder(tf.float32, [None,10], name = "digitLabels") # 0-9 digits => 10 classes.

# Set model weights
W = tf.Variable(tf.zeros([784,10]), name = "weights")
b = tf.Variable(tf.zeros([10]), name =  "bias")

# construct model
pred = tf.nn.softmax(tf.matmul(x,W) + b, name = "predProbs") # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices = 1))
# Gradient Descent 
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the parameters
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver(tf.all_variables())
    # training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op(backprop) and cost op(to get loss value)
            _,c = sess.run([optimizer,cost], feed_dict={x:batch_xs, y:batch_ys})
            # Compute average loss
            avg_cost += c/total_batch

        # Display logs per epoch
        if (epoch + 1) % display_step == 0:
            print ("Epoch:","%04d" % (epoch+1), "cost: ", "{:.9f}".format(avg_cost))
            checkpoint_path = os.path.join("save/", 'model.ckpt') 
            saver.save(sess,checkpoint_path, global_step=epoch * total_batch + i)

    print("Optimizer finished.")
    
    # Test
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

