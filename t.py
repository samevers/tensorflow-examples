#! /usr/bin/python 
#encoding:gb18030
############################################
#
# Author: 
# E-Mail:@sogou-inc.com
# Create time: 2017 4月 21 10时50分56秒
# version 1.0
#
############################################

#import tensorflow as tf
import numpy as np

a = []
a.append([1,2])
a.append([2,3])
a.append([3,4])
a.append([4,5])
batch_size = 2
num_batches = len(a)/batch_size
print "num_batches = ",num_batches

import numpy as np

print "ori array:",a
xbatches = np.array_split(a, num_batches)
print "After split:"
for x in xbatches:
    print x
    #mn = tf.reduce_mean(x)
    #print "mean = ",mn
    #print "-----"
x = np.zeros((2,2))
print x
