import math
import random
import time
import os,sys
import numpy as np
import tensorflow as tf
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Hyperparameters
learning_rate = 0.001
num_steps = 2000
batch_size = 32

im_size = 400
num_input = im_size*im_size 
im_size_flat = im_size * im_size * 3
num_classes = 10 # 10 classes of prices ranges
dropout_rate = 0.75
classes = [[0,100],[100,250],[250,400],[400,600],[600,1000],[1000,1500],[1500,2500],[2500,5000],[5000,12000],[12000,60000000]]

def load_data(start,end):
  cwd = os.getcwd()
  im_path = cwd + "/images/"
  X = []
  labels = []
  image_dir = os.listdir( im_path )
  num_item = 0
  count = 0
  for item in image_dir:
      if os.path.isfile(im_path+item):
        price_label = int(item.split('_')[0])
        im = np.array(cv2.imread(im_path+item,-1))
        if im is not None and im.shape == (im_size,im_size,3):
          if end == count:
            break
          if start <= count:
            X.append(im)
            print(item)
            print(num_item)
            num_item += 1
            for i in range(len(classes)):
              if price_label >= classes[i][0] and price_label < classes[i][1]:
                labels.append(i)
          count += 1
    
  
  X = np.vstack(X)
  num_images  = int(X.shape[0]/im_size)
  X = X.reshape((num_images,im_size_flat))
  test_indices = np.random.choice(num_images,int(num_images*.1),replace=False)
  train_indices = np.setdiff1d(range(num_images),test_indices)

  labels = np.array(labels)
  xtrain = X[train_indices,:]
  labelstrain = labels[train_indices]
  xtest = X[test_indices,:]
  labelstest = labels[test_indices]
  return xtrain,labelstrain,xtest,labelstest


# Helper Functions
def generate_batch(data, batch_size = batch_size):
  x, y = data
  indices = np.random.choice(len(x), batch_size, replace=False)
  return x[indices], y[indices]

def onehot_conv(array):
  onehot = np.zeros((10,len(array)))
  for i,el in enumerate(array):
    onehot[el,i] = 1
  return onehot.transpose()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# file to write results
f = open('cnn_results.txt', 'w')

#Load our data
xtrain,labelstrain,xtest,labelstest = load_data(0,30000)
data_train = (xtrain,onehot_conv(labelstrain))
data_test = (xtest,onehot_conv(labelstest))

x = tf.placeholder(tf.float32, shape=[None, im_size_flat])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, im_size, im_size, 3])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 64, 16])
b_conv3 = bias_variable([16])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([50 * 50 * 16, 1024])
b_fc1 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool3, [-1, 50*50*16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver({"W_conv1": W_conv1,"b_conv1": b_conv1,"W_conv2": W_conv2,"b_conv2": b_conv2,"W_conv3": W_conv3,"b_conv3": b_conv3,"W_fc1": W_fc1,"b_fc1": b_fc1,"W_fc2": W_fc2,"b_fc2": b_fc2})

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
    print(i)
    x_batch, y_batch = generate_batch(data_train)
    if i % 10 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: x_batch, y_: y_batch, keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
      f.write(str((i, train_accuracy)))
    if i % 100 == 0:
      saver.save(sess, os.path.join(os.getcwd(), 'trained_variables2.ckpt'))
    train_step.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: xtest, y_: labelstest, keep_prob: 1.0}))
  f.write(str(accuracy.eval(feed_dict={
      x: xtest, y_: labelstest, keep_prob: 1.0})))
f.close()


