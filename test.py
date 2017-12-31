from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import struct

dic = {}
dic[1] = 'hello'
for (i, v) in dic.items():
	print (type(i))

x = tf.random_normal([3,4,2], dtype=tf.float32)
y = tf.reshape(x, [-1, 2])

with tf.Session() as s:
	s.run(tf.global_variables_initializer())
	print(s.run([x,y]))
