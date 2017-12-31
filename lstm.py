# -*- coding:utf-8 -*-  
from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import struct


'''
	This is the first version to use lstm and crf to sovle ner problem
'''

batch_size = 0
max_sequence_length = 0
word_embedding = 0
hidden_size = 0
num_layers = 0
num_class = 5
epoch_size = 1000
check_size = 100
learning_rate = 0.1



tf_input = tf.placeholder([batch_size, max_sequence_length], dtype=tf.float32)
tf_target = tf.placeholder([batch_size, max_sequence_length], dtype=tf.float32)


cell_input = tf.nn.embedding_lookup(word_embedding, tf_input)

cell_fw = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
cell_bk = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True)
cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob = 1.0, output_keep_prob = 1.0)
cell_bk = tf.nn.rnn_cell.DropoutWrapper(cell_bk, input_keep_prob = 1.0, output_keep_prob = 1.0)

cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw]*num_layers, dtype=tf.float32)
cell_bk = tf.nn.rnn_cell.MultiRNNCell([cell_bk]*num_layers, dtype=tf.float32)

initial_state_fw = cell_fw.zero_state(batch_size, dtype=tf.float32)
initial_state_bk = cell_bk.zero_state(batch_size, dtype=tf.float32)
inputs_list = [tf.squeeze(s, squeeze_dims=[1]) for s in tf.split(1, max_sequence_length, cell_input)]  

outputs, state_fw, state_bw = \
			tf.nn.bidirectional_rnn(cell_fw, cell_bw, inputs_list, initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bw)  

output = tf.reshape((1, outputs), [-1, hidden_size])
W = tf.get_variable('W', [hidden_size, num_class], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
b = tf.get_variable('b', [num_class], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
logits = tf.matmul(output, W) + b
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.reshape(tf_target, [-1, num_class]))

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(tf_target,1), tf.argmax(logits,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, max))

init = tf.global_variables_initializer()
sess = tf.Session()

if __name__ is '__main__':
	sess.run(init)
	for i in range(epoch_size):
		sess.run(train_op, feed_dict={tf_input: , tf_target: })
		pass
		if i % check_size == 0:
			print (i, ": ", sess.run([accuracy], feed_dict={tf_input: , tf_target: }))


