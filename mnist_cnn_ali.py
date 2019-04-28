# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Import the deep learning library
import tensorflow as tf
import time
 
# Import the MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
 
# Network inputs and outputs
# The network's input is a 28×28 dimensional input
n = 28
m = 28
num_input = n * m # MNIST data input 
num_classes = 10 # MNIST total classes (0-9 digits)
 
# tf Graph input
# placeholder：插入占位符
# 函数定义：placeholder(dtype, shape=None, name=None)
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
 
# Storing the parameters of our LeNET-5 inspired Convolutional Neural Network
# 参数值怎么来的？
weights = {
    # 卷积核参数
   "W_ij": tf.Variable(tf.random_normal([5, 5, 1, 32])),
   "W_jk": tf.Variable(tf.random_normal([5, 5, 32, 64])),
   "W_kl": tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
   "W_lm": tf.Variable(tf.random_normal([1024, num_classes]))
    }

# 参数值怎么来的？ 
biases = {
   "b_ij": tf.Variable(tf.random_normal([32])),
   "b_jk": tf.Variable(tf.random_normal([64])),
   "b_kl": tf.Variable(tf.random_normal([1024])),
   "b_lm": tf.Variable(tf.random_normal([num_classes]))
    }
 
# The hyper-parameters of our Convolutional Neural Network

#学习率
learning_rate = 1e-3
num_steps = 500
batch_size = 128
display_step = 10

'''
函数名：tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
函数说明：除去name参数用以指定该操作的name，与方法有关的一共五个参数：

第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，
  具有[batch, in_height, in_width, in_channels]这样的shape，
  具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，
  要求类型为float32和float64其中之一

第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，
  具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
  具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维

第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4

第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）

第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true

结果返回一个Tensor，这个输出，就是我们常说的feature map，shape仍然是[batch, height, width, channels]这种形式。

函数名：tf.nn.bias_add( value,bias, name=None)
函数说明：将偏差项bias加到value上面

'''

def ConvolutionLayer(x, W, b, strides=1):
    # Convolution Layer
    # 卷积层
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return x

'''
函数名：tf.nn.relu(features, name = None)
函数说明：这个函数的作用是计算激活函数 relu，即 max(features, 0)。即将矩阵中每行的非最大值置0。
'''

def ReLU(x):
    # ReLU activation function 
    # 激活函数
    return tf.nn.relu(x)

'''
函数名：tf.nn.max_pool(value, ksize, strides, padding, name=None)
函数说明：
第一个参数value：需要池化的输入，一般池化层接在卷积层后面，
所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape

第二个参数ksize：池化窗口的大小，取一个四维向量，
一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1

第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]

第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'

返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
'''

def PoolingLayer(x, k=2, strides=2):
    # Max Pooling layer
    # 最大值池化层
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1],
                          padding='SAME')

'''
函数名：tf.nn.softmax(logits,axis=None,name=None,dim=None)
函数说明：计算softmax激活。
'''

def Softmax(x):
    # Softmax activation function for the CNN's final output
    return tf.nn.softmax(x)

'''
函数名：matmul(a,b,transpose_a=False,transpose_b=False,adjoint_a=False,
                    adjoint_b=False,a_is_sparse=False,b_is_sparse=False,name=None)
函数说明：将矩阵 a 乘以矩阵 b,生成a * b  

函数名：add(x, y, name=None)
函数说明：这个函数是使x,和y两个参数的元素相加，返回的tensor数据类型和x的数据类型相同
'''

# Create model
# 创建卷积神经网络模型
def ConvolutionalNeuralNetwork(x, weights, biases):
    # MNIST data input is a 1-D row vector of 784 features (28×28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    # 图片是4维的：[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
    # reshape将28x28一维数据重构为4维数据
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
 
    # Convolution Layer
    Conv1 = ConvolutionLayer(x, weights["W_ij"], biases["b_ij"])
    # Non-Linearity
    ReLU1 = ReLU(Conv1)
    # Max Pooling (down-sampling)
    Pool1 = PoolingLayer(ReLU1, k=2)
 
    # Convolution Layer
    Conv2 = ConvolutionLayer(Pool1, weights["W_jk"], biases["b_jk"])
    # Non-Linearity
    ReLU2 = ReLU(Conv2)
    # Max Pooling (down-sampling)
    Pool2 = PoolingLayer(ReLU2, k=2)
    
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    FC = tf.reshape(Pool2, [-1, weights["W_kl"].get_shape().as_list()[0]])
    FC = tf.add(tf.matmul(FC, weights["W_kl"]), biases["b_kl"])
    FC = ReLU(FC)
 
    # Output, class prediction
    output = tf.add(tf.matmul(FC, weights["W_lm"]), biases["b_lm"])
    
    return output
 
# Construct model
logits = ConvolutionalNeuralNetwork(X, weights, biases)
prediction = Softmax(logits)

'''
函数名：
tf.nn.softmax_cross_entropy_with_logits(_sentinel=None,labels=None,
    logits=None,dim=-1,name=None)
函数说明：计算logits和labels之间的softmax交叉熵

函数名：
reduce_mean(input_tensor,axis=None,keep_dims=False,name=None,reduction_indices=None)
函数说明：计算张量的各个维度上的元素的平均值.
'''

# Softamx cross entropy loss function
# 损失函数
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))

'''
函数名:
tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, 
    epsilon=1e-08, use_locking=False, name='Adam')
函数说明：
此函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。
相比于基础SGD算法，1.不容易陷于局部优点。2.速度更快
'''

# Optimization using the Adam Gradient Descent optimizer
# 优化器
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_process = optimizer.minimize(loss_function)

'''
函数名：tf.argmax(input,axis)
函数说明：tf.argmax(input,axis)根据axis取值的不同返回每行或者每列最大值的索引。
axis=0的时候，比较每一列的元素，将每一列最大元素所在的索引记录下来，最后输出每一列最大元素所在的索引数组。
axis=1的时候，将每一行最大元素所在的索引记录下来，最后返回每一行最大元素所在的索引数组。

函数名：tf.equal(A, B)
函数说明：tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，
如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的

函数名：tf.cast(x,dtype,name=None)
函数说明：将x的数据格式转化成dtype
'''

# Evaluate model
# 评估模型
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

'''
函数名：tf.summary.scalar(tags, values, collections=None, name=None)
函数说明：用来显示标量信息

函数名：tf.summary.merge(inputs,collections=None,name=None)
函数说明:合并标量信息
'''

# recording how the loss functio varies over time during training
cost = tf.summary.scalar("cost", loss_function)
training_accuracy = tf.summary.scalar("accuracy", accuracy)
train_summary_op = tf.summary.merge([cost,training_accuracy])

'''
函数名：tf.summary.FileWriter(self,logdir,graph=None,max_queue=10,flush_secs=120,graph_def=None)
函数说明：将汇总结果写入事件
'''

train_writer = tf.summary.FileWriter("./Desktop/logs",
                                        graph=tf.get_default_graph())

'''
函数名：tf.global_variables_initializer()
函数说明：初始化模型参数
'''

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
 
# Start training
with tf.Session() as sess:
 
    # Run the initializer
    sess.run(init)
    
    start_time = time.time()
    
    
    writer = tf.summary.FileWriter("./tensorbord_data", sess.graph)

    for step in range(1, num_steps+1):
        
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(training_process, feed_dict={X: batch_x, Y: batch_y})
        
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc, summary = sess.run([loss_function, accuracy, train_summary_op], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            train_writer.add_summary(summary, step)
            
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
            
    end_time = time.time() 
    
    print("Time duration: " + str(int(end_time-start_time)) + " seconds")
    print("Optimization Finished!")
            
    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                      Y: mnist.test.labels[:256]}))