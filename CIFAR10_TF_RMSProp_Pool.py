# This uses the hdf5 data to for training and testing
# main change: 
# 1. the batch update is now written by hand
# 2. uses now the sparse_softmax_cross_entropy_error: for which the label is only a single number
import time
import numpy as np
import os
filename = os.path.basename(__file__)
filename = filename.split(".")[0] + ".log"


# Import data
import h5py
CIFAR10_data = h5py.File('CIFAR10.hdf5', 'r')
x_train = np.float32(CIFAR10_data['X_train'][:] )
y_train = np.int32(np.array(CIFAR10_data['Y_train'][:]))
x_test = np.float32(CIFAR10_data['X_test'][:] )
y_test = np.int32( np.array(CIFAR10_data['Y_test'][:]  ) )

CIFAR10_data.close()

x_train = np.swapaxes(x_train, 1, 3)
x_test = np.swapaxes(x_test, 1, 3)



import tensorflow as tf
sess = tf.InteractiveSession()

# define weight initialization functions
def weight_variable(shape, stddev):
  initial = tf.truncated_normal(shape=shape, stddev=stddev)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)




# Define convolution and pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='SAME')



x = tf.placeholder(tf.float32, [None, 32,32,3])
y_ = tf.placeholder(tf.int64, [None])

# Define weights for 1st convolution layer
W_conv1 = weight_variable([5, 5, 3, 64], 5e-2)
b_conv1 = bias_variable([64])


# Convolve x with kernel, apply ReLU
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
# Apply max pooling
h_pool1 = max_pool_2x2(h_conv1)

norm1 = tf.nn.lrn(h_pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75)

# Define weights for 2nd convolution layer
W_conv2 = weight_variable([5, 5, 64, 64], 5e-2)
b_conv2 = bias_variable([64])

# Convolve output from first layer with the 2nd kernel
h_conv2 = tf.nn.relu(conv2d(norm1, W_conv2) + b_conv2)
norm2 = tf.nn.lrn(h_conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75)

h_pool2 = max_pool_2x2(norm2)

# Apply a single feed forward layer
"""
1. How to determine the size of the output from conv and pooling layers
2. How does the reshape work, the first dimension (-1) stands for?

"""
W_fc1 = weight_variable([8 * 8 * 64, 384], 0.04)
b_fc1 = bias_variable([384])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


W_fc2 = weight_variable([384, 192], 0.04)
b_fc2 = bias_variable([192])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)


# define the drop out rate
keep_prob = tf.placeholder(tf.float32)
# Apply drop out
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# After the dropout layer, apply a single feedfoward layer
# This can be considered as the softmax layer without normalization
# The softmax is inherently applied in the softmax_cross_entropy
W_fc3 = weight_variable([192, 10], 1/192)
b_fc3 = bias_variable([10])

y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3


# Define loss and optimizer


cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv,y_))
train_step = tf.train.RMSPropOptimizer(0.01, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp').minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

batch_size = 64
num_epochs = 30000
L_Y_train = len(y_train)

target = open(filename, 'w')
line="Configurations:\nDropout:\nCNN_Layer:\nPooling_Layer:\nFeedForwardLayer:\nBatchNorm:\nOptimizer: RMSProp\n"
target.write(line)
target.write("\n")


time1 = time.time()
for epoch in range(num_epochs):
  I_permutation = np.random.permutation(L_Y_train)
  x_train = x_train[I_permutation]
  y_train = y_train[I_permutation]
  x_batch = x_train[0:batch_size,:]
  y_batch = y_train[0:batch_size]
  train_accuracy = accuracy.eval(feed_dict={x:x_batch, y_:y_batch, keep_prob:1.0})
  if epoch %100 ==0:
    line = "step %d, training accuracy %g"%(epoch, train_accuracy)
    target.write(line)
    target.write("\n")
    print(line) 
  train_step.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})
  if epoch %500 ==0:
    test_accuracy=accuracy.eval(feed_dict={x: x_test[1:2000,:], y_: y_test[1:2000], keep_prob: 1.0})
    line = "step %d, test accuracy %g"%(epoch, test_accuracy)
    target.write(line)
    target.write("\n")
    print(line)


test_accuracy = accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
line = "Final test accuracy is %g"%test_accuracy
target.write(line)
target.write("\n")
print(line)

time2 = time.time()
training_time = time2 - time1  
line = "Training time is: %f" % training_time
target.write(line)
target.write("\n")
print(line)




