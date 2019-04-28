# Import the deep learning library
import tensorflow as tf

# Define our compuational graph 
W1 = tf.constant(5.0, name = "x")
W2 = tf.constant(3.0, name = "y")
W3 = tf.cos(W1, name = "cos")
W4 = tf.sin(W2, name = "sin")
W5 = tf.multiply(W3, W4, name = "mult")
W6 = tf.divide(W1, W2, name = "div")
W7 = tf.add(W5, W6, name = "add")

# Open the session
with tf.Session() as sess:

    cos = sess.run(W3)
    sin = sess.run(W4)
    mult = sess.run(W5)
    div = sess.run(W6)
    add = sess.run(W7)
    
    # Before running TensorBoard, make sure you have generated summary data in a log directory by creating a summary writer
    writer = tf.summary.FileWriter("./Desktop/ComputationGraph", sess.graph)
    
    # Once you have event files, run TensorBoard and provide the log directory
    # Command: tensorboard --logdir="path/to/logs"
    # tensorboard --logdir="E:\90.work\python\tensorflow-master\tensorflow\examples\tutorials\mnist\Desktop\ComputationGraph"