# coding=utf-8

import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import tf_slim as slim
tf.disable_v2_behavior()
from pdb import set_trace

rnn_unit = 50  # 隐层神经元的个数
lstm_layers = 2  # 隐层层数
#input_size = 7  参数个数？
input_size = 1  #因变量个数
output_size = 1
lr = 0.001  # 学习率
epoch = 500
# ——————————————————导入数据——————————————————————

#f = open('dataset_2.csv')
#data = df.iloc[:, 2:10].values  # 取第3-10列
filename = '汇率-单变量.csv'
f = open(filename)
df = pd.read_csv(f)         #读入汇率数据
data = df.iloc[:, 1:input_size+2].values
data = pd.DataFrame(data).dropna().values    #丢弃含nan值的行

train_size = int(len(data) * 0.85)  #定义训练集比例
test_size = len(data) - train_size  #定义测试集比例

#set_trace()
def datanormalization(data, normal_method):
    if normal_method == 'z-score':
        normalized_train_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)  # z-score标准化
    return data


# 获取训练集
def get_train_data(batch_size=60, time_step=20, train_begin=0, train_end=train_size):
    batch_index = []
    data_train = data[train_begin:train_end]
    normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # z-score标准化
    #normalized_train_data = (data_train - np.min(data_train))/(np.max(data_train) - np.min(data_train))  #最大最小值归一化
    train_x, train_y = [], []  # 训练集x和y初定义
    #set_trace()
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step, :input_size]
        y = normalized_train_data[i:i + time_step, input_size, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    return batch_index, train_x, train_y


# 获取测试集
def get_test_data(time_step=20, test_begin=train_size):
    data_test = data[test_begin:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std  # 标准化
    size = (len(normalized_test_data) + time_step - 1) // time_step  # 有size个sample
    test_x, test_y = [], []
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :input_size]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, input_size]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i + 1) * time_step:, :input_size]).tolist())
    test_y.extend((normalized_test_data[(i + 1) * time_step:, input_size]).tolist())
    return mean, std, test_x, test_y


# ——————————————————定义神经网络变量——————————————————
# 输入层、输出层权重、偏置、dropout参数

weights = {
    'in': tf.Variable(tf.random_normal([input_size, rnn_unit])),
    'out': tf.Variable(tf.random_normal([rnn_unit, 1]))
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# ——————————————————定义神经网络变量——————————————————
def lstmCell():
    # basicLstm单元
    basicLstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    # dropout
    drop = tf.nn.rnn_cell.DropoutWrapper(basicLstm, output_keep_prob=keep_prob)
    return basicLstm


def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.MultiRNNCell([lstmCell() for i in range(lstm_layers)])
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states


# ————————————————训练模型————————————————————

def train_lstm(batch_size=60, time_step=20, train_begin=0, train_end=train_size):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(X)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):  # 这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                 Y: train_y[batch_index[step]:batch_index[step + 1]],
                                                                 keep_prob: 0.5})
            print("Number of iterations:", i, " loss:", loss_)
        print("model_save: ", saver.save(sess, 'model_save2\\modle.ckpt'))
        # I run the code on windows 10,so use  'model_save2\\modle.ckpt'
        # if you run it on Linux,please use  'model_save2/modle.ckpt'
        print("The train has finished")

train_lstm()


# ————————————————预测模型————————————————————
def prediction(time_step=20):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    mean, std, test_x, test_y = get_test_data(time_step)
    #set_trace()
    with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
        pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # 参数恢复
        module_file = tf.train.latest_checkpoint('model_save2')
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x) - 1):
            prob = sess.run(pred, feed_dict={X: [test_x[step]], keep_prob: 1})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y) * std[input_size] + mean[input_size]  #z-score归一化还原值
        test_predict = np.array(test_predict) * std[input_size] + mean[input_size]  #z-score归一化还原值

        #test_y = np.array(test_y) * (max[input_size]-min[input_size]) + min[input_size]  # z-score归一化还原值
        #test_predict = np.array(test_predict) * (max[input_size]-min[input_size]) + min[input_size]  # z-score归一化还原值

        mape = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 平均绝对百分比误差
        mae = np.average(np.abs(test_predict - test_y[:len(test_predict)]))  #平均绝对误差
        rmse = (1/len(test_predict)*np.sum(np.square(test_predict-test_y[:len(test_predict)])))**0.5

        print("The mape of this predict:", mape)
        print("The mae of this predict:", mae)
        print("The rmse of this predict:", rmse)
        wrexcel(test_predict, test_y)


def wrexcel(test_predict, test_y):
    data = {'pre': test_predict, 'true': test_y[0:560]}
    df = DataFrame(data)
    #set_trace()
    newname = filename.replace('.csv', '.xlsx')
    df.to_excel(newname)
    return

prediction()
