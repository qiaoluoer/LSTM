# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v1 as tf
from pdb import set_trace
tf.disable_v2_behavior()

#——————————————————导入数据——————————————————————
f = open('人民币汇率中间价2015.9-2022.8.csv')
df = pd.read_csv(f)         #读入汇率数据
data = np.array(df['美元'])  #获取美元序列
train_size = int(len(data) * 0.85)  #定义训练集比例
test_size = len(data) - train_size  #定义测试集比例
data_train = data[:train_size]    #取前5800个做训练
data_test = data[train_size:]     #后X个做测试 
#data = data[::-1]#反转，使数据按照日期先后顺序排列
#以折线图展示data
# plt.figure()
# plt.plot(data)
# plt.show()
normalize_data = (data_train - np.mean(data_train)) / np.std(data_train)#标准化
normalize_data = normalize_data[:, np.newaxis]#增加1个维度

#———————————————————形成训练集—————————————————————
time_step = 20      #时间步
rnn_unit = 10       #hidden layer units
lstm_layers = 2     #每一批次训练多少个样例
batch_size = 60     #输入层维度  #每一批次训练多少个样例
input_size = 1      #输入层维度
output_size = 1     #输出层维度
lr = 0.0006         #学习率
train_x, train_y = [], []#训练集
for i in range(len(normalize_data) - time_step - 1):
    x = normalize_data[i:i + time_step]
    y = normalize_data[i + 1:i + time_step + 1]
    train_x.append(x.tolist())
    train_y.append(y.tolist())
	
test_x, test_y = [],[] #测试集
for i in range(test_size):
	x = 

# 定义每个X sample的形状(?, time_step, input_size)
X = tf.placeholder(tf.float32, [None, time_step, input_size])
# 定义每个Y sample的形状(?, time_step, output_size)
Y = tf.placeholder(tf.float32, [None, time_step, output_size])
#——————————————————定义神经网络变量——————————————————
#输入层、输出层权重、偏置
weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
print(weights)
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}
print(biases)

#参数：输入网络批次数目
def lstm(batch):#参数：输入网络批次数目
    w_in = weights['in']
    b_in = biases['in']
    print(X)
    input = tf.reshape(X, [-1, input_size])#需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    print(input)
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])#将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(rnn_unit) for i in range(lstm_layers)])
    init_state = cell.zero_state(batch, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])#作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


def train_lstm():
    global batch_size
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(batch_size)
    # 损失函数
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 重复训练10000次
        for i in range(100):  # We can increase the number of iterations to gain better result.
            step = 0
            start = 0
            end = start + batch_size
            print("i = ",i)
            while (end < len(train_x)):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[start:end], Y: train_y[start:end]})
                start += batch_size
                end = start + batch_size

                if step % 100 == 0:  #每10步保存一次参数
                    print("Number of iterations:", i, " loss:", loss_)
                    print("model_save", saver.save(sess, 'model_save1\\modle.ckpt'))
                    # I run the code in windows 10,so use  'model_save1\\modle.ckpt'
                    # if you run it in Linux,please use  'model_save1/modle.ckpt'
                step += 1
        print("The train has finished")


train_lstm()


def prediction():
    with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
        pred, _ = lstm(1)#预测时只输入[1,time_step,input_size]的测试数据
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        saver.restore(sess, 'model_save1\\modle.ckpt')#参数恢复
        # I run the code in windows 10,so use  'model_save1\\modle.ckpt'
        # if you run it in Linux,please use  'model_save1/modle.ckpt'
        # 取训练集最后一行为测试样本。shape=[1,time_step,input_size]
        prev_seq = train_x[-1]
        predict = []
        # 得到之后100个预测结果
        for i in range(100):
            next_seq = sess.run(pred, feed_dict={X: [prev_seq]})
            predict.append(next_seq[-1])
            # 每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
        
        predict_data = []
        for i in range(len(predict)):
            predict_data.append(np.array(predict[i])*np.std(data_train)+np.mean(data_train))
        # 以折线图表示结果
        plt.figure()
        #plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
        #plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r')
        plt.plot(list(range(len(data))), data, color='b')
        plt.plot(list(range(len(data), len(data) + len(predict_data))), predict_data, color='r')
        plt.show()
    input()


prediction()
