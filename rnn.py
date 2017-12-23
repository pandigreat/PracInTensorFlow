from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.constant([[1.0, 0.0], [2.0, 1.0], [1.0, 0.0], [0.0, 1.0]],dtype=tf.float32)
y = tf.unstack(x,axis=0)
 
x_one_hot = tf.one_hot([[1,3],[1,2]], 2)
init = tf.global_variables_initializer()

with tf.Session() as sess:
	print(sess.run(y))
	
