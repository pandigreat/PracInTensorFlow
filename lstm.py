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

def loadDataSet():

	return input_data, word_embedding


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



