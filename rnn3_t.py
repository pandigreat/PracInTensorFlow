from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import random
import os
import struct
import time
import sys
'''
	practice tensorflow using mnist dataset 

'''

def loadMNISTdata():
	filenames = ['train-images-idx3-ubyte','train-labels-idx1-ubyte', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']
	listname = ['traig_img', 'train_lb', 'test_img', 'test_lb'];

 
	with open(filenames[1], 'rb') as p:
		magic, n = struct.unpack('>II', p.read(8))
		train_lb = np.fromfile(p, dtype=np.uint8)

	with open(filenames[0], 'rb') as p1:
		magic, n, row, col = struct.unpack('>IIII', p1.read(16))
		train_img = np.fromfile(p1, dtype=np.uint8).reshape(len(train_lb), 28*28);

	with open(filenames[3], 'rb') as p:
		magic, n = struct.unpack('>II', p.read(8))
		test_lb = np.fromfile(p, dtype=np.uint8)

	with open(filenames[2], 'rb') as p1:
		magic, n, row, col = struct.unpack('>IIII', p1.read(16))
		test_img = np.fromfile(p1, dtype=np.uint8).reshape(len(test_lb), 28*28);

	with open('check.txt', 'w') as p:
		train_img.tofile(p)

	return [np.array(train_img), train_lb, np.array(test_img), test_lb]


[train_img, train_lb, test_img, test_lb] = loadMNISTdata()
train_lb = np.eye(10)[train_lb]
test_lb = np.eye(10)[test_lb]
step = 0.5
train_loop = 1000

x = tf.placeholder('float', shape=[None, 784])
y_ = tf.placeholder('float', shape=[None, 10])
W = tf.get_variable('W', [784, 10], initializer=tf.constant_initializer(30))
b = tf.get_variable('b', [10], initializer=tf.constant_initializer(30))
y = tf.nn.softmax(tf.matmul(x,W) + b) 
#cross_entropy = -tf.reduce_mean(y_*tf.log(y))
#cross_entropy = tf.reduce_mean( -tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))

#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))

train_step = tf.train.GradientDescentOptimizer(step).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
batch = 2000

for i in range(100001):
	n = random.randint(0, 30)
	batch_lb = train_lb[n*batch: (n+1)*batch,:]
	batch_img = train_img[n*batch:(n+1)*batch, :] 
	#batch_lb = train_lb
	#batch_img = train_img
	#print (sess.run(y, feed_dict={x:test_img, y_:test_lb}))
	if i % 10000 == 0:
		print (sess.run(correct_prediction, feed_dict={x:test_img,y_:test_lb}))
		print (i, ": ", sess.run(accuracy, feed_dict={x:test_img, y_:test_lb}))

		sess.run(train_step, feed_dict={x:batch_img, y_:batch_lb})


 


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

#print (sess.run(correct_prediction, feed_dict={x:test_img, y_:test_lb}))
#print (sess.run(accuracy, feed_dict={y_:test_lb}))

