import numpy as np
import tensorflow as tf

# Load our data
xall = np.load('xdata.npy')
yall = np.load('ydata.npy')

rnd_idx = np.random.permutation(xall.shape[0])
xall = xall[rnd_idx,:]
yall = yall[rnd_idx,:]


xtrain = xall[:600,:]
ytrain = yall[:600,:]
xdev = xall[600:700,:]
ydev = yall[600:700,:]
xtest = xall[700:,:]
ytest = yall[700:,:]

# Parameters
learning_rate = 1e-2
training_epochs = 30
display_step = 1
beta = 0.01


x = tf.placeholder(tf.float32, [None, 268]) 
y = tf.placeholder(tf.float32, [None, 10]) 

# Set model weights
W = tf.Variable(tf.zeros([268, 10]))
b = tf.Variable(tf.zeros([10]))

regularizer = tf.nn.l2_loss(W)

# Construct model
z = tf.matmul(x, W) + b

# Minimize error using cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z,labels=y)+beta*regularizer)

# Gradient Descent
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Hold costs for each epoch
    epoch_cost = np.zeros((training_epochs,))

    # Training cycle
    for ind,epoch in enumerate(range(training_epochs)):
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={x: xtrain,
                                                      y: ytrain})
        # Compute average loss
        epoch_cost[ind] = c

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
            # Test model
            correct_prediction = tf.equal(tf.argmax(z, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Train Accuracy:", accuracy.eval({x: xtrain, y: ytrain}))
            correct_prediction = tf.equal(tf.argmax(z, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Validation Accuracy:", accuracy.eval({x: xdev, y: ydev}))


    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(z, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Test Accuracy:", accuracy.eval({x: xtest, y: ytest}))