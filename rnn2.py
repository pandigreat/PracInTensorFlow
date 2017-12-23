from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
为简单起见，本篇就以简单的二进制序列作为训练数据，而不实现具体的论文仿真，主要目的是理解RNN的原理和如何在TensorFlow中构造一个简单基础的模型架构。
首先我们看一下实验数据的构造：
输入数据X：在时间t，Xt的值有50%的概率为1，50%的概率为0；
输出数据Y：在实践t，Yt的值有50%的概率为1，50%的概率为0，除此之外，如果Xt-3 == 1，Yt为1的概率增加50%， 如果Xt-8 == 1，则Yt为1的概率减少25%， 如果上述两个条件同时满足，则Yt为1的概率为75%。

如果RNN没有学习到任何一条依赖，那么Yt为1的概率就是0.625（0.5+0.5*0.5-0.5*0.25），所以所获得的交叉熵应该是0.66。
如果RNN学习到第一条依赖关系，即Xt-3为1时Yt一定为1。那么，所以最终的交叉熵应该是0.52（-0.5* (0.875 * np.log(0.875) + 0.125 * np.log(0.125)) -0.5 * (0.625* np.log(0.625) + 0.375* np.log(0.375))）。
如果RNN学习到了两条依赖， 那么有0.25的概率全对，0.5的概率正确率是75%，还有0.25的概率正确率是0.5。所以其交叉熵为0.45（-0.50 * (0.75* np.log(0.75) + 0.25* np.log(0.25)) - 0.25 * (2 * 0.50 * np.log (0.50)) - 0.25 * (0)）。


'''

#生成实验用数据
def gen_data(size=100000):
	"""生成数据
	输入数据X：在时间t，Xt的值有50%的概率为1，50%的概率为0；
	输出数据Y：在实践t，Yt的值有50%的概率为1，50%的概率为0，除此之外，如果`Xt-3 == 1`，Yt为1的概率增加50%， 如果`Xt-8 == 1`，则Yt为1的概率减少25%， 如果上述两个条件同时满足，则Yt为1的概率为75%。
	"""
	X = np.array(np.random.choice(2, size=(size,)))
	Y = []
	for i in range(size):
		threshold = 0.5
		#判断X[i-3]和X[i-8]是否为1，修改阈值
		if X[i-3] == 1:
			threshold += 0.5
		if X[i-8] == 1:
			threshold -= 0.25
		#生成随机数，以threshold为阈值给Yi赋值
		if np.random.rand() > threshold:
			Y.append(0)
		else:
			Y.append(1)
	return X, np.array(Y)
	
def gen_batch(raw_data, batch_size, num_steps):
	#raw_data是使用gen_data()函数生成的数据，分别是X和Y
	raw_x, raw_y = raw_data 
	data_length = len(raw_x)
	#print('gen_batch_' + %d)(data_length)
	
	# 首先将数据切分成batch_size份，0-batch_size，batch_size-2*batch_size
	batch_partition_length = data_length / batch_size
	data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
	data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
	for i in range(batch_size):
		data_x[i] = raw_x[batch_partition_length*i : batch_partition_length*(i+1)]
		data_y[i] = raw_y[batch_partition_length*i : batch_partition_length*(i+1)]
	
	#因为RNN模型一次只处理num_steps个数据，所以将每个batch_size在进行切分成epoch_size份，每份num_steps个数据。注意这里的epoch_size和模型训练过程中的epoch不同
	epoch_size = batch_partition_length / num_steps
	
	 #x是0-num_steps， batch_partition_length -batch_partition_length +num_steps。。。共batch_size个
	 for i in range(epoch_size):
		x = data_x[:, i*num_steps:(i+1)*num_steps]
		y = data_y[:, i*num_steps:(i+1)*num_steps]
		yield (x, y)
		

#这里的n就是训练过程中用的epoch，即在样本规模上循环的次数
def gen_epochs(n, num_steps):
	for i in range(n):
		yield gen_batch(gen_data(), batch_size, num_steps)
		
		
batch_size = 3
num_classes = 2
state_size = 4
num_steps = 10
learning_rate = 0.2

x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholder')
#RNN的初始化状态，全设为零。注意state是与input保持一致，接下来会有concat操作，所以这里要有batch的维度。即每个样本都要有隐层状态

init_state = tf.zeros([batch_size, state_size])

#将输入转化为one-hot编码，两个类别。[batch_size, num_steps, num_classes]
x_one_hot = tf.one_hot(x, num_classes)
#将输入unstack，即在num_steps上解绑，方便给每个循环单元输入。这里可以看出RNN每个cell都处理一个batch的输入（即batch个二进制样本输入）
rnn_inputs = tf.unstack(x_one_hot, axis=1)

#定义rnn_cell的权重参数，
with tf.variable_scope('rnn_cell'):
"""由于tf.Variable() 每次都在创建新对象，所有reuse=True 和它并没有什么关系。对于get_variable()，来说，如果已经创建的变量对象，就把那个对象返回，如果没有创建变量对象的话，就创建一个新的。"""
	W = tf.get_variable('W', [num_classes+state_size, state_size])
	b = tf.get_variable('b', [state_size], initializer=tf.constant_initialize(0.0))
#使之定义为reuse模式，循环使用，保持参数相同
def rnn_cell(rnn_input, state):
	with tf.variable_scope('rnn_cell', reuse=True):
		W = tf.get_variable('W', [num_classes+state_size, state_size])
		b = tf.get_variable('b', [state_size], initializer=tf.constant_initialize(0.0))
	#定义rnn_cell具体的操作，这里使用的是最简单的rnn，不是LSTM
	return tf.tanh(tf.matmul(tf.concat([rnn_input,state], 1), W) + b)

state = init_state
rnn_outputs = []
#循环num_steps次，即将一个序列输入RNN模型
for rnn_input in rnn_inputs:
	state = rnn_cell(rnn_input, state)
	rnn_outputs.append(state)
final_state = rnn_outputs[-1]

#define softmax layer 
with tf.variable_scope('softmax'):
	W = tf.get_variable('W', [state_szie,num_classes])
	b = tf.get_variable('b', [num_classes], initialize=tf.constant_initialize(0.0))

#注意，这里要将num_steps个输出全部分别进行计算其输出，然后使用softmax预测
logits = [tf.nn.softmax(rnn_output, W) + b for rnn_output in rnn_outputs]
predictions = [tf.nn.softmax(logit) for logit in logits]
# Turn our y placeholder into a list of labels
y_as_list = tf.unstack(y, num=num_steps, axis=1)

#losses and train_step
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logit) for logit, label in zip(logits, y_as_list)]
total_loss = tf.reduce_mean(losses)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def train_network(num_epochs, num_steps, state_size=4, verbose=True):
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		training_losses = []
		 #得到数据，因为num_epochs==5，所以外循环只执行五次
		 for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
			training_loss = 0
			 #保存每次执行后的最后状态，然后赋给下一次执行
			 training_state = np.zeros((batch_size, state_size))
			 if verbose:
				print("\nEpoch", idx)
			#这是具体获得数据的部分
			for step, (X, Y) in enumerate(epoch):
				tr_losses, training_loss_, traing_state, _ = sess.run([loss,total_loss,final_state,train_step], feed_dict={x:X, y:Y, init_state:training_state})
				training_loss += training_loss_
				if step%100 == 0 and step > 0:
					if verbose:
						print("Average loss at step", step, "for last 100 steps ", traing_loss/100)
					training_losses.append(training_loss/100)
					training_loss = 0
	return traing_losses
	
training_losses = train_network(5, num_steps)
plt.plot(training_losses)
plt.show()


















