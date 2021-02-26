import tensorflow as tf
import numpy as np
import os
import time

from utils.load_training_data import load_train
train_file_dir = './training_data/'
train_mat_file = 'train_block_size_64.mat'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
q = 0.5
layer_num = 9

block_size = 64
n_input,n_output = 102,4096
n_output_of_A = 1024
m_rate = 10
batch_size = 64
learning_rate = 0.0001
epoch_num = 20

X_output = tf.placeholder(tf.float32, [None, n_output])

def add_con2d_weight_bias(w_shape, b_shape, order_no):
    Weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='Weights_%d' % order_no)
    biases = tf.Variable(tf.random_normal(b_shape, stddev=0.05), name='biases_%d' % order_no)
    return [Weights, biases]

def add_fc(shape1, order_no):
    AA = tf.get_variable(shape=shape1, initializer=tf.contrib.layers.xavier_initializer(), name='FC_%d' % order_no)
    return AA

def ista_block(input_layers, X_output, layer_no, QY1,QY2,QY3,QY4,ATA):
    step_size = tf.Variable(1e-1, dtype=tf.float32)
    alpha = tf.Variable(1e-5, dtype=tf.float32)
    beta = tf.Variable(1.0, dtype=tf.float32)
    
    XXX = tf.reshape(input_layers[-1],shape=[-1,64,64])
    Im11 = XXX[:,0:32,0:32]
    vec11 = tf.reshape(Im11,shape=[-1,n_output_of_A])
    Im22 = XXX[:,0:32,32:64]
    vec22 = tf.reshape(Im22,shape=[-1,n_output_of_A])
    Im33 = XXX[:,32:64,0:32]
    vec33 = tf.reshape(Im33,shape=[-1,n_output_of_A])
    Im44 = XXX[:,32:64,32:64]
    vec44 = tf.reshape(Im44,shape=[-1,n_output_of_A])
    
    x1_ista_1 = tf.add(vec11 - tf.scalar_mul(step_size, tf.matmul(vec11, ATA)), tf.scalar_mul(step_size, QY1))    
    x1_ista_2 = tf.add(vec22 - tf.scalar_mul(step_size, tf.matmul(vec22, ATA)), tf.scalar_mul(step_size, QY2))    
    x1_ista_3 = tf.add(vec33 - tf.scalar_mul(step_size, tf.matmul(vec33, ATA)), tf.scalar_mul(step_size, QY3))    
    x1_ista_4 = tf.add(vec44 - tf.scalar_mul(step_size, tf.matmul(vec44, ATA)), tf.scalar_mul(step_size, QY4))    
    
    CC1 = tf.reshape(x1_ista_1,shape=[-1,32,32])
    CC2 = tf.reshape(x1_ista_2,shape=[-1,32,32])
    CC3 = tf.reshape(x1_ista_3,shape=[-1,32,32])
    CC4 = tf.reshape(x1_ista_4,shape=[-1,32,32])
    SS11 = tf.concat((CC1,CC2),axis=2)
    SS22 = tf.concat((CC3,CC4),axis=2)
    SSS = tf.concat((SS11,SS22),axis=1)
    x1_ista = tf.reshape(SSS,shape=[-1,n_output])
    
    x2_ista = tf.reshape(x1_ista, shape=[-1, 64, 64, 1])

    [Weights0, bias0] = add_con2d_weight_bias([3, 3, 1, 32], [32], 0)
    [Weights1, bias1] = add_con2d_weight_bias([3, 3, 32, 32], [32], 1)
    [Weights2, bias2] = add_con2d_weight_bias([3, 3, 32, 32], [32], 2)
    [Weights3, bias3] = add_con2d_weight_bias([3, 3, 32, 32], [32], 3)
    [Weights4, bias4] = add_con2d_weight_bias([3, 3, 32, 32], [32], 4)
    [Weights5, bias5] = add_con2d_weight_bias([3, 3, 32, 32], [32], 5)
    [Weights6, bias6] = add_con2d_weight_bias([3, 3, 32, 32], [32], 6)
    [Weights7, bias7] = add_con2d_weight_bias([3, 3, 32, 1], [1], 7)
    
    x3_ista = tf.nn.conv2d(x2_ista, Weights0, strides=[1, 1, 1, 1], padding='SAME')
    x4_ista = tf.nn.relu(tf.nn.conv2d(x3_ista, Weights1, strides=[1, 1, 1, 1], padding='SAME'))
    x40_ista = tf.nn.relu(tf.nn.conv2d(x4_ista, Weights2, strides=[1, 1, 1, 1], padding='SAME'))
    x44_ista = tf.nn.conv2d(x40_ista, Weights3, strides=[1, 1, 1, 1], padding='SAME')

    trun_param = alpha / ((0.1 + tf.abs(x44_ista))**(1-q))
    x50_ista = tf.multiply(tf.sign(x44_ista), tf.nn.relu(tf.abs(x44_ista) - trun_param))
    x5_ista = x50_ista - x44_ista
    
    x6_ista = tf.nn.relu(tf.nn.conv2d(x5_ista, Weights4, strides=[1, 1, 1, 1], padding='SAME'))
    x60_ista = tf.nn.relu(tf.nn.conv2d(x6_ista, Weights5, strides=[1, 1, 1, 1], padding='SAME'))
    x66_ista = tf.nn.conv2d(x60_ista, Weights6, strides=[1, 1, 1, 1], padding='SAME')
    x7_ista = tf.nn.conv2d(x66_ista, Weights7, strides=[1, 1, 1, 1], padding='SAME')

    x7_ista = x2_ista + beta * x7_ista
    x8_ista = tf.reshape(x7_ista, shape=[-1, n_output])

    x3_ista_sym = tf.nn.relu(tf.nn.conv2d(x3_ista, Weights1, strides=[1, 1, 1, 1], padding='SAME'))
    x30_ista_sym = tf.nn.relu(tf.nn.conv2d(x3_ista_sym, Weights2, strides=[1, 1, 1, 1], padding='SAME'))
    x4_ista_sym = tf.nn.conv2d(x30_ista_sym, Weights3, strides=[1, 1, 1, 1], padding='SAME')
    x60_ista_sym = tf.nn.relu(tf.nn.conv2d(x4_ista_sym, Weights4, strides=[1, 1, 1, 1], padding='SAME'))
    x6_ista_sym = tf.nn.relu(tf.nn.conv2d(x60_ista_sym, Weights5, strides=[1, 1, 1, 1], padding='SAME'))
    x7_ista_sym = tf.nn.conv2d(x6_ista_sym, Weights6, strides=[1, 1, 1, 1], padding='SAME')

    x11_ista = x7_ista_sym - x3_ista

    return [x8_ista, x11_ista]

def inference_ista(n, X_output, reuse):
    XX = X_output
    AT = add_fc([n_output_of_A,n_input], 0)
    ATT = add_fc([n_input,n_output_of_A], 1)

    XX2 = tf.reshape(XX,shape=[-1,64,64])
    Im1 = XX2[:,0:32,0:32]
    vec1 = tf.reshape(Im1,shape=[-1,n_output_of_A])
    Im2 = XX2[:,0:32,32:64]
    vec2 = tf.reshape(Im2,shape=[-1,n_output_of_A])
    Im3 = XX2[:,32:64,0:32]
    vec3 = tf.reshape(Im3,shape=[-1,n_output_of_A])
    Im4 = XX2[:,32:64,32:64]
    vec4 = tf.reshape(Im4,shape=[-1,n_output_of_A])
    YT1 = tf.matmul(vec1, AT)
    YT2 = tf.matmul(vec2, AT)
    YT3 = tf.matmul(vec3, AT)
    YT4 = tf.matmul(vec4, AT)
    QY1 = tf.matmul(YT1, ATT)
    QY2 = tf.matmul(YT2, ATT)
    QY3 = tf.matmul(YT3, ATT)
    QY4 = tf.matmul(YT4, ATT)
    CD1 = tf.reshape(QY1,shape=[-1,32,32])
    CD2 = tf.reshape(QY2,shape=[-1,32,32])
    CD3 = tf.reshape(QY3,shape=[-1,32,32])
    CD4 = tf.reshape(QY4,shape=[-1,32,32])
    SS1 = tf.concat((CD1,CD2),axis=2)
    SS2 = tf.concat((CD3,CD4),axis=2)
    SS = tf.concat((SS1,SS2),axis=1)
    QY = tf.reshape(SS,shape=[-1,n_output])
    ATA = tf.matmul(AT,ATT)
    
    layers = []
    layers_symetric = []
    layers.append(QY)
    for i in range(n):
        with tf.variable_scope('conv_%d' %i, reuse=reuse):
            [conv1, conv1_sym] = ista_block(layers, X_output, i, QY1,QY2,QY3,QY4,ATA)
            layers.append(conv1)
            layers_symetric.append(conv1_sym)
    return [layers, layers_symetric]

[Prediction, Pre_symetric] = inference_ista(layer_num, X_output, reuse=False)

def compute_cost(Prediction, X_output, layer_num):
    cost = tf.reduce_mean(tf.square(Prediction[-1] - X_output))
    cost_sym = 0
    for k in range(layer_num):
        cost_sym += tf.reduce_mean(tf.square(Pre_symetric[k]))

    return [cost, cost_sym]

[cost, cost_sym] = compute_cost(Prediction, X_output, layer_num)

cost_all = cost + 0.01*cost_sym
optm_all = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_all)

init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

saver = tf.train.Saver(tf.global_variables(), max_to_keep=200)

sess = tf.Session(config=config)
sess.run(init)

print("\n Layer no. %d, m-rate is %d%%" % (layer_num, m_rate))
print("Start Training...")

model_dir = 'Layer_%d_mrate_0_%d_Model' % (layer_num, m_rate)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
output_file_name_log = "Log_output_%s.txt" % (model_dir)
PSNR_record = 0.0

nrtrain, Training_labels = load_train(train_file_dir,train_mat_file)

for epoch_i in range(0, epoch_num+1):
    epoch_using_time_begin = time.time()
    randidx_all = np.random.permutation(nrtrain)
#    for batch_i in range(nrtrain // batch_size):
    for batch_i in range(10):
        print('\rtraining epoch {0}, batch {1}/{2}'.format(epoch_i,batch_i+1,nrtrain // batch_size),end='')
        randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]

        batch_ys = Training_labels[randidx, :]

        feed_dict = {X_output: batch_ys}
        sess.run(optm_all, feed_dict=feed_dict)

    epoch_using_time = time.time() - epoch_using_time_begin
    print('')
    print('epoch {0} cost {1:<.3f} sec'.format(epoch_i,epoch_using_time))
    
    saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)
        
    output_data_train = "[%02d/%02d] cost: %.5f, cost_sym: %.5f, using time: %f sec\n" % (epoch_i, epoch_num, sess.run(cost, feed_dict=feed_dict), sess.run(cost_sym, feed_dict=feed_dict), epoch_using_time)
    print(output_data_train)

    output_file = open(output_file_name_log, 'a')
    output_file.write(output_data_train)
    output_file.close()
    
print("Training Finished")
sess.close()
