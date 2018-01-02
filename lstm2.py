# -*- coding:utf-8 -*-  
from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import struct
import util

'''
	This is the first version to use lstm and crf to sovle ner problem
'''


#-----------------------------hyper parameters------------------------
batch_size = 99
max_sequence_length = util.max_sequence_length #default 110
word_embedding = []
word_embedding_size = 0
hidden_size = 120    #the steps of nn, means the number of sequences in an article
num_layers = 10
num_class = 5
epoch_size = 1000
check_size = 100
learning_rate = 0.1
forget_bias = 0.0
input_keep_prob = 1.0
output_keep_prob = 1.0
#--------------------------------------------------
init = 0
sess = []
correct_prediction = []
train_op = 0
accuracy = 0
tf_input = 0
tf_target = 0
#--------------------------------------------------
(data, data_ner, word_embedding) = util.loadChineseData()
(batch_size, train_set, train_lb, test_set, test_lb) = util.DivideDataSet(data, data_ner)
word_embedding.insert(0, [0 for i in range(len(word_embedding[0]))])
word_embedding_size = len(word_embedding )
word_embedding_dim = len(word_embedding[0])

#def mynet ():
tf_input = tf.placeholder(dtype=tf.int32, shape=[None, max_sequence_length])
tf_target = tf.placeholder(dtype=tf.int32, shape=[None, max_sequence_length])
print('fuck')
#class_output = tf.nn.embedding_lookup(np.eye(5,5), tf_target)
map_nertype = [[0 if j != i else 1 for j in range(5)] for i in range(5)]
word_embedding = np.array(word_embedding, dtype=np.float64)
map_nertype = np.array(map_nertype, dtype=np.float64)
cell_input = tf.nn.embedding_lookup(word_embedding, tf_input)
class_output = tf.nn.embedding_lookup(map_nertype, tf_target)

cell_fw = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=forget_bias, state_is_tuple=True)
cell_bk = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=forget_bias, state_is_tuple=True, reuse=True)
#cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob)
#cell_bk = tf.nn.rnn_cell.DropoutWrapper(cell_bk, input_keep_prob = input_keep_prob, output_keep_prob=output_keep_prob)

#cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw]*num_layers, state_is_tuple=True)
#cell_bk = tf.nn.rnn_cell.MultiRNNCell([cell_bk]*num_layers, state_is_tuple=True)


initial_state_fw = cell_fw.zero_state(batch_size, dtype=tf.float64)
initial_state_bk = cell_bk.zero_state(batch_size, dtype=tf.float64)
#inputs_list = [tf.squeeze(s, squeeze_dims=1) for s in tf.split(cell_input, num_or_size_splits=max_sequence_length, axis=1)]  
tf_tmp = 0
s = tf.split(cell_input, num_or_size_splits=max_sequence_length, axis=1)
for i in range(len(s)):
	if i == 0:
		tf_tmp = s[i]
	else:
		tf_tmp = tf.concat([tf_tmp, s[i]], 0)
inputs_list = tf.reshape(tf_tmp, [-1, batch_size, word_embedding_dim])
print (inputs_list)

inputs_list = cell_input
print(inputs_list)
# st = tf.split(cell_input, num_or_size_splits=max_sequence_length, axis=1)
# for s in st:
# 	inputs_list.append(s)

#outputs, state_fw, state_bw = \
#			tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bk, inputs_list, initial_state_fw = initial_state_fw, initial_state_bw = initial_state_bk)  


outputs, state_ = tf.nn.dynamic_rnn (cell_fw, inputs=inputs_list, initial_state=initial_state_fw)
print(outputs)
output = tf.reshape( outputs, [-1, hidden_size])
W = tf.get_variable('W', [hidden_size, num_class], dtype=tf.float64, initializer=tf.constant_initializer(1.0))
b = tf.get_variable('b', [num_class], dtype=tf.float64, initializer=tf.constant_initializer(1.0))
logits = tf.matmul(output, W) + b
labels = tf.reshape(class_output, [-1, num_class])
print ('logits', logits)
print ('labels', labels)
#loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(labels, 1))

#train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#correct_prediction = tf.equal(tf.argmax(class_output, 2), tf.argmax(logits,2))

cross_entropy = -tf.reduce_mean(labels*tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(labels,1), tf.argmax(logits, 1))
print (correct_prediction)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

init = tf.global_variables_initializer()
sess = tf.Session()

#if __name__ == '__main__':

#--------------------Preservation----------------------- 
# (data, data_ner, word_embedding) = util.loadChineseData()
# (batch_size, train_set, train_lb, test_set, test_lb) = util.DivideDataSet(data, data_ner)
# word_embedding.insert(0, [0 for i in range(len(word_embedding[0]))])
word_embedding_size = len(word_embedding )
word_embedding_dim = len(word_embedding[0])
print(word_embedding_size, word_embedding_dim)



#----------------------Run session--------------------------

print(len(train_set), len(train_set[0]))
sess.run(init)
for i in range(epoch_size):
	sess.run(train_op, feed_dict={tf_input: train_set, tf_target: train_lb})
	pass
	if i % check_size == 0:
		print ("epoch_size:", i, sess.run([accuracy], feed_dict={tf_input: test_set, tf_target: test_lb}))


