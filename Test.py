import tensorflow as tf
import numpy as np
import os
import time
from PIL import Image
import glob

from utils.imread import imread
from utils.img2col import img2col
from utils.col2img import col2img
from utils.psnr import psnr

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
q = 0.5
layer_num = 9

block_size = 64
n_input,n_output = 102,4096
n_output_of_A = 1024
m_rate = 10
batch_size = 64
cpkt_model_no = 20

begin_time = time.time()

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

init = tf.global_variables_initializer()
config = tf.ConfigProto()
config = tf.ConfigProto(device_count = {'GPU': 0})
config.gpu_options.allow_growth = True

saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

sess = tf.Session(config=config)

model_dir = 'Layer_%d_mrate_0_%d_Model' % (layer_num, m_rate)
output_file_name_PSNR = "PSNR_Results_%s.txt" % (model_dir)
output_testing_file_name = 'testing_results.txt'
PSNR_record = 0.0

out0 = 'cpkt no. %d\n' %cpkt_model_no
output_file = open(output_testing_file_name, 'a')
output_file.write(out0)
output_file.close()

saver.restore(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, cpkt_model_no))

for test_im in range(5):
    if test_im == 0:
        Test_Img = './Test_Image_Set5'
        filepaths = glob.glob(Test_Img + '/*.bmp')
    elif test_im == 1:
        Test_Img = './Test_Image_Set11'
        filepaths = glob.glob(Test_Img + '/*.tif')
    elif test_im == 2:
        Test_Img = './Test_Image_Set14'
        filepaths = glob.glob(Test_Img + '/*.bmp')
    elif test_im == 3:
        Test_Img = './Test_Image_BSD68'
        filepaths = glob.glob(Test_Img + '/*.png')
    elif test_im == 4:
        Test_Img = './Test_Image_BSD100'
        filepaths = glob.glob(Test_Img + '/*.jpg')

    ImgNum = len(filepaths)
    print('img num=',ImgNum)
    PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
    SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)


    for img_no in range(ImgNum):
        imgName = filepaths[img_no]
        [Iorg, row, col, Ipad, row_new, col_new] = imread(imgName,block_size)
        Icol = img2col(Ipad, block_size).transpose()/255.0
        Img_output = Icol
        
        recon_start = time.time()
        Prediction_value = sess.run(Prediction[-1], feed_dict={X_output: Img_output})
        recon_using_time = time.time() - recon_start
        
        cost_value = sess.run(cost, feed_dict={X_output: Img_output})
        cost_sym_value = sess.run(cost_sym, feed_dict={X_output: Img_output})
        X_rec = col2img(Prediction_value.transpose(), row, col, row_new, col_new, block_size)
        X_rec = np.clip(X_rec * 255, 0, 255, out = X_rec)
        
        rec_PSNR = psnr(X_rec, Iorg)
        PSNR_All[0, img_no] = rec_PSNR
        
        temp = Iorg.shape
        rec_SSIM = sess.run(tf.image.ssim(tf.image.convert_image_dtype(tf.reshape(X_rec, shape=list(temp) + [1]), tf.float32), tf.reshape(Iorg, shape=list(temp) + [1]), max_val=255.0))
        SSIM_All[0, img_no] = rec_SSIM
        
        img_rec_name = "%s_rec_%s_PSNR_%.4f_SSIM_%.4f_time_%.5f.png" % (imgName, model_dir, rec_PSNR, rec_SSIM, recon_using_time)
        x_im_rec = Image.fromarray(X_rec.astype(np.uint8))
        x_im_rec.save(img_rec_name)
        
    mean_PSNR = np.mean(PSNR_All)
    mean_SSIM = np.mean(SSIM_All)
    print('PSNR = ',mean_PSNR)
    print('SSIM = ',mean_SSIM)
        
    total_using_time = time.time() - begin_time
    print('total using time = ',total_using_time)
    
    out1 = [Test_Img[13:] + '\n'][0]
    out2 = 'mean PSNR = %.4f\n'  % (mean_PSNR)
    out3 = 'mean SSIM = %.4f\n\n\n'  % (mean_SSIM)
    output_data_testing = [out1+out2+out3][0]
    
    output_file = open(output_testing_file_name, 'a')
    output_file.write(output_data_testing)
    output_file.close()
    
sess.close()
