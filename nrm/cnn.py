import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
tfe.enable_eager_execution()


source = np.array([[5,4,2,5,5,7,9,7,8,0],[7,2,7,7,0,5,4,2,4,7]]) # TX = 10
embedding_matrix = tf.get_variable(name='embedding',shape=[10,8],dtype=tf.float32)

max_time = 10
batch_size = 2
num_units = 8
character_emb_inp = tf.nn.embedding_lookup(embedding_matrix, source)
conv_inputs = tf.reshape(character_emb_inp, [max_time, batch_size, num_units, 1])

# [batch, height = num_units, width = max_time, channels = 1]
conv_inputs = tf.transpose(conv_inputs, perm=[1, 2, 0, 3])

# CNN layer, windows width from 1 to 5
conv_outputs = []
filter_nums = 5
for width in range(1, 6):
  filter = tf.get_variable("filter_%d" % (width), shape=[num_units, width, 1, 1])
  strides = [1, num_units, 1, 1]
  # [batch, height = 1, width = max_time, channels = 1]
  conv_out = tf.nn.relu(tf.nn.conv2d(conv_inputs,filter,strides, padding='SAME'))
  conv_outputs.append(conv_out)

# max_pooling with strides=5
pool_outputs = []
width_strides = 3
strides = [1, 1, width_strides, 1]
segment_len = np.ceil(max_time / 3)
for conv_output in conv_outputs:
  pool_out = tf.nn.max_pool(conv_output, [1, 1, width_strides, 1], strides, padding='SAME')
  pool_out = tf.reshape(pool_out, [batch_size, segment_len])
  pool_outputs.append(pool_out)

# segment embedding
# [batch, height = 1, width = max_time /3 , channels = 1]


def highway(x, size, activation, carry_bias=-1.0):
  W_T = tfe.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight_transform")
  b_T = tfe.Variable(tf.constant(carry_bias, shape=[size]), name="bias_transform")
  W = tfe.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight")
  b = tfe.Variable(tf.constant(0.1, shape=[size]), name="bias")
  T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name="transform_gate")
  H = activation(tf.matmul(x, W) + b, name="activation")
  C = tf.subtract(1.0, T, name="carry_gate")
  y = tf.add(tf.multiply(H, T), tf.multiply(x, C), "y")
  return y
# [batch, width, height]
stacked_results = tf.stack(pool_outputs, axis=2)

highway_outputs = highway(tf.reshape(stacked_results,[-1,filter_nums]), filter_nums, tf.nn.relu)

highway_outputs = tf.reshape(highway_outputs, [batch_size, segment_len, filter_nums])
# [time_width, batch, height]

time_major_embedding = tf.transpose(highway_outputs, perm=[1, 0, 2])

i = 1


