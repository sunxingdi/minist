# refer:https://www.w3cschool.cn/tensorflow_python/?

# Import the deep learning library
import tensorflow as tf
import time
 
#import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# placeholder：插入占位符
# 函数定义：placeholder(dtype, shape=None, name=None)
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

# tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差。
# 这个函数产生正太分布，均值和标准差自己设定。
def weight_variable(shape):
  with tf.variable_scope("var_weight"):
    initial = tf.truncated_normal(shape, stddev=0.1)
    var_weight = tf.Variable(initial)
    tf.summary.histogram('var_weight',var_weight)
    return var_weight

# tf.constant 创建一个常数张量,传入list或者数值来填充
def bias_variable(shape):
  with tf.variable_scope("var_bias"):
    initial = tf.constant(0.1, shape=shape)
    var_bias = tf.Variable(initial) 
    tf.summary.histogram('var_bias',var_bias)    
    return var_bias

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
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

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
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

'''
函数名：tf.nn.relu(features, name = None)
函数说明：这个函数的作用是计算激活函数 relu，即 max(features, 0)。即将矩阵中每行的非最大值置0。
'''
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
'''
函数名：matmul(a,b,transpose_a=False,transpose_b=False,adjoint_a=False,
                    adjoint_b=False,a_is_sparse=False,b_is_sparse=False,name=None)
函数说明：将矩阵 a 乘以矩阵 b,生成a * b  

函数名：add(x, y, name=None)
函数说明：这个函数是使x,和y两个参数的元素相加，返回的tensor数据类型和x的数据类型相同
'''
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# tf.nn.dropout()是tensorflow里面为了防止或减轻过拟合而使用的函数，它一般用在全连接层
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

'''
函数名：tf.nn.softmax(logits,axis=None,name=None,dim=None)
函数说明：计算softmax激活。
'''
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

'''
函数名：
tf.nn.softmax_cross_entropy_with_logits(_sentinel=None,labels=None,
    logits=None,dim=-1,name=None)
函数说明：计算logits和labels之间的softmax交叉熵

函数名：
reduce_mean(input_tensor,axis=None,keep_dims=False,name=None,reduction_indices=None)
函数说明：计算张量的各个维度上的元素的平均值.
'''

# 损失函数是目标类别和预测类别之间的交叉熵.
# 注意,tf.reduce_sum把minibatch里的每张图片的交叉熵值都加起来了.我们计算的交叉熵是指整个minibatch的.
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

'''
函数名:
tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, 
    epsilon=1e-08, use_locking=False, name='Adam')
函数说明：
此函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。
相比于基础SGD算法，1.不容易陷于局部优点。2.速度更快
'''

# 用梯度下降算法(gradient descent algorithm)以0.01的学习速率最小化交叉熵
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

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
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
# accuracy准确度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

'''
函数名：tf.summary.scalar(tags, values, collections=None, name=None)
函数说明：用来显示标量信息

函数名：tf.summary.merge(inputs,collections=None,name=None)
函数说明:合并标量信息
'''
cost = tf.summary.scalar("cost", cross_entropy)
training_accuracy = tf.summary.scalar("accuracy", accuracy)
#train_summary_merge = tf.summary.merge([cost,training_accuracy])
train_summary_merge = tf.summary.merge_all()

'''
batch是一个二维向量,包含50个训练数据信息
batch[0] 训练数据图片，sharp ：(50,784)
batch[1] 训练数据答案，sharp : (50,10)
'''

'''
tensor.eval()的意义和sess.run()一样，t.eval()等效于sess.run(t).
但是二者也有些微区别:run可以同时运行多个tensor;t.eval() 运行必须包含在默认的sess之内
'''

with tf.Session() as sess:
  '''
  函数名：tf.global_variables_initializer()
  函数说明：初始化模型参数
  '''
  sess.run(tf.global_variables_initializer())

  '''
  函数名：tf.summary.FileWriter(self,logdir,graph=None,max_queue=10,flush_secs=120,graph_def=None)
  函数说明：将汇总结果写入事件
  '''
  train_writer = tf.summary.FileWriter("./my_tensorbord_data", sess.graph)

  for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
      # 在feed_dict中加入额外的参数keep_prob来控制 dropout 比例
      train_accuracy = accuracy.eval(feed_dict={
          x:batch[0], y_: batch[1], keep_prob: 1.0})
      print ("step %d, training accuracy %g " % (i,train_accuracy))
    summary, result = sess.run([train_summary_merge,train_step],feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    train_writer.add_summary(summary, i)

  print ("test accuracy %g"%accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
