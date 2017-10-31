import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={inputs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={inputs: v_xs, outputs: v_ys})
    return result

# parameters definition
in_size = 784        # input size
out_size = 10        # output size
learning_rate = 0.1  # learning rate
max_epochs = 3000    # maximum train steps

inputs = tf.placeholder(tf.float32, [None, in_size]) # 28x28
outputs = tf.placeholder(tf.float32, [None, out_size])# 10

                         
Weights = tf.Variable(tf.random_normal([in_size, out_size]))
biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)

logits = tf.matmul(inputs, Weights) + biases
prediction = tf.nn.softmax(logits)
#################################################################
# complete expressions below
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=outputs))

train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

#################################################################
sess = tf.Session()
init = tf.global_variables_initializer() # tf_version > 0.12
sess.run(init)
train_acc = []
train_loss = []
test_acc = []
test_loss = []
for i in range(max_epochs):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={inputs: batch_xs, outputs: batch_ys})
    #train_loss.append(loss)
    #train_acc.append(compute_accuracy(batch_xs, batch_ys))
    if i % 500 == 0:
        acc = compute_accuracy(mnist.test.images, mnist.test.labels)
        #test_acc.append(acc)
        if i % 500 == 0:
            print(acc)
        #test_loss.append(sess.run(cross_entropy, feed_dict={inputs: batch_xs, outputs: batch_ys}))