from .utils import misc_utils as utils
import collections
from . import model_helper
import tensorflow as tf

class EncoderParam(
    collections.namedtuple("EncoderParam", ("encoder_type", "num_layers", "num_residual_layers",
                                            "unit_type","forget_bias","dropout","num_gpus","mode",
                                            "enocder_seq_input","encoder_seq_len","dtype",
                                          "single_cell_fn","num_units","name"))):
  pass

class CNNEncoderParam(
    collections.namedtuple("CNNEncoderParam", ("max_time", "batch_size", "embed_dim","relu_type",
                                            "min_windows","max_windows","flexible_configs","filters_per_windows","width_strides",
                                            "high_way_type","high_way_layers","name","max_k"))):
  pass



def _build_encoder_cell(encoder_param, num_layers, num_residual_layers,
                      base_gpu=0):
    """Build a multi-layer RNN cell that can be used by encoder."""
    return model_helper.create_rnn_cell(
        unit_type=encoder_param.unit_type,
        num_units=encoder_param.num_units,
        num_layers=num_layers,
        num_residual_layers=num_residual_layers,
        forget_bias=encoder_param.forget_bias,
        dropout=encoder_param.dropout,
        num_gpus=encoder_param.num_gpus,
        mode=encoder_param.mode,
        base_gpu=base_gpu,
        single_cell_fn=encoder_param.single_cell_fn)

def _build_bidirectional_rnn(inputs, sequence_length,
                               dtype, encoder_param,
                               num_bi_layers,
                               num_bi_residual_layers,
                               base_gpu=0):
    """Create and call biddirectional RNN cells.

    Args:
      num_residual_layers: Number of residual layers from top to bottom. For
        example, if `num_bi_layers=4` and `num_residual_layers=2`, the last 2 RNN
        layers in each RNN cell will be wrapped with `ResidualWrapper`.
      base_gpu: The gpu device id to use for the first forward RNN layer. The
        i-th forward RNN layer will use `(base_gpu + i) % num_gpus` as its
        device id. The `base_gpu` for backward RNN cell is `(base_gpu +
        num_bi_layers)`.

    Returns:
      The concatenated bidirectional output and the bidirectional RNN cell"s
      state.
    """
    # Construct forward and backward cells
    fw_cell = _build_encoder_cell(encoder_param,
                                       num_bi_layers,
                                       num_bi_residual_layers,
                                       base_gpu=base_gpu)
    bw_cell = _build_encoder_cell(encoder_param,
                                       num_bi_layers,
                                       num_bi_residual_layers,
                                       base_gpu=(base_gpu + num_bi_layers))

    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
        fw_cell,
        bw_cell,
        inputs,
        dtype=dtype,
        sequence_length=sequence_length,
        time_major=True,
        swap_memory=True)

    return tf.concat(bi_outputs, -1), bi_state



"""

My functions
"""


def highway(x, size, activation, carry_bias=-1.0, name='highway'):
    with tf.variable_scope(name):
        W_T = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight_transform")
        b_T = tf.Variable(tf.constant(carry_bias, shape=[size]), name="bias_transform")
        W = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight")
        b = tf.Variable(tf.constant(0.1, shape=[size]), name="bias")
        T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name="transform_gate")
        H = activation(tf.matmul(x, W) + b, name="activation")
        C = tf.subtract(1.0, T, name="carry_gate")
        y = tf.add(tf.multiply(H, T), tf.multiply(x, C), "y")
        return y


def build_cnn_encoder(embedding_emb_inp, cnn_encoder_param):

    # TODO 检查输入的格式，以及原来attention encoder里的格式，时间
    # Parameters
    max_time = cnn_encoder_param.max_time
    batch_size = cnn_encoder_param.batch_size
    embed_dim = cnn_encoder_param.embed_dim

    min_windows = cnn_encoder_param.min_windows
    max_windows = cnn_encoder_param.max_windows
    filters_per_windows = cnn_encoder_param.filters_per_windows
    # 增加设置
    flexible_configs = cnn_encoder_param.flexible_configs

    relu_type = cnn_encoder_param.relu_type



    high_way_type = cnn_encoder_param.high_way_type
    high_way_layers = cnn_encoder_param.high_way_layers
    max_k = cnn_encoder_param.max_k







    #input
    embedding_inputs = embedding_emb_inp # [max_time, batch_size, embed_dim]
    conv_inputs = tf.reshape(embedding_inputs, [max_time, batch_size, embed_dim, 1])
    conv_inputs = tf.transpose(conv_inputs, perm=[1, 2, 0, 3])

    # CNN layer
    conv_outputs = []
    conv_heights = []
    if flexible_configs == 'none':
        filter_nums = (max_windows - min_windows + 1) * filters_per_windows
        for width in range(min_windows, max_windows + 1):
            # filter: [filter_height, filter_width, in_channels, out_channels]
            filter = tf.get_variable("filter_%d" % ( width ), shape=[embed_dim, width, 1, filters_per_windows])
            conv_heights.append(filters_per_windows)
            strides = [1, embed_dim, 1, 1]
            # [batch, height = 1, width = max_time, channels = filters_per_windows]
            if relu_type == 'relu':
                
                conv_out = tf.nn.relu(tf.nn.conv2d(conv_inputs, filter, strides, padding='SAME'))
            elif relu_type == 'leaky':
                conv_out = tf.nn.leaky_relu(tf.nn.conv2d(conv_inputs, filter, strides, padding='SAME'))
            print(conv_out)
            conv_outputs.append(conv_out)
    else:
        print('Flexible CharCNN Configurations :  %s' % flexible_configs)
        params = flexible_configs.split('-')
        filter_nums = 0
        for para in params:
            items = para.split('/')
            width = int(items[0])
            height = int(items[1])
            filter_nums += height
            conv_heights.append(height)

            filter = tf.get_variable("flexible_filter_%d_%d" % (width,height), shape=[embed_dim, width, 1, height])
            strides = [1, embed_dim, 1, 1]
            # [batch, height = 1, width = max_time, channels = filters_per_windows]
            if relu_type == 'relu':
                conv_out = tf.nn.relu(tf.nn.conv2d(conv_inputs, filter, strides, padding='SAME'))
            elif relu_type == 'leaky':
                conv_out = tf.nn.leaky_relu(tf.nn.conv2d(conv_inputs, filter, strides, padding='SAME'))
            conv_outputs.append(conv_out)

            
    def k_max_pooling_without_order(conv_out, k=2):
        """
        conv_out:[batch,1,seg_len,filter_num]
        """
        tmp = tf.transpose(conv_out, perm=[0,1,3,2])
        tmp,_ = tf.nn.top_k(tmp, k, sorted=False)
        res = tf.transpose(tmp, perm=[0,1,3,2])
        return res
        
        

    # max_pooling over time
    pool_outputs = []
    width_strides = max_time
    strides = [1, 1, width_strides, 1]
    filter_nums *= max_k
    segment_len = tf.cast(tf.ceil(max_time / width_strides), tf.int32)
    for conv_output,conv_height in zip(conv_outputs,conv_heights):
        #
        # [batch, height = 1, width = segment_len, channels = filters_per_windows]
        if max_k > 1:
            print('K-Max pooling')
            pool_out = k_max_pooling_without_order(conv_output,max_k)
        else:
            pool_out = tf.nn.max_pool(conv_output, [1, 1, width_strides, 1], strides, padding='SAME')
        pool_out = tf.reshape(pool_out, [batch_size, max_k, conv_height])
        pool_outputs.append(pool_out)

    # Highway network
    if high_way_layers > 0:

        if relu_type == 'relu':
            relu_func = tf.nn.relu
        elif relu_type == 'leaky':
            relu_func = tf.nn.leaky_relu
        if high_way_type == 'uniform':
            stacked_results = tf.concat(pool_outputs, axis=-1)
            high_way_tmp = tf.reshape(stacked_results, [-1, filter_nums])
            for i in range(high_way_layers):
                high_way_tmp = highway(high_way_tmp, filter_nums, relu_func, name='highway_%d' % i)
            highway_outputs = tf.reshape(high_way_tmp, [batch_size, filter_nums])
        elif high_way_type == 'per_filter':
            highway_results = []
            w = 0
            for height, pool_result in zip(conv_heights, pool_outputs):
                pool_highway_tmp = tf.reshape(pool_result, [-1, height])
                for i in range(high_way_layers):
                    pool_highway_tmp = highway(pool_highway_tmp, height, relu_func,
                                               name='highway_w%d_%d' % (w + min_windows, i))
                highway_results.append(pool_highway_tmp)
                w += 1
            stacked_results = tf.concat(highway_results, axis=-1)
            highway_outputs = tf.reshape(stacked_results, [batch_size, filter_nums])
    else:
        highway_outputs = tf.concat(pool_outputs, axis=-1)



    return highway_outputs,filter_nums



def build_attention_sum_layer(num_units, embeddings, max_time):
    """

    :param embeddings: [batch_size, max_time, ...].
    :param max_time:
    :return:
    """
    logits = tf.reshape(tf.layers.dense(embeddings, 1, activation=tf.nn.relu),[-1, max_time])
    probs = tf.reshape(tf.nn.softmax(logits=logits, dim=-1),[-1, max_time, 1])
    weighted = tf.multiply(embeddings, probs)
    sumed = tf.reshape(tf.reduce_sum(weighted, axis=1),[-1, num_units])
    return sumed

def build_attention2_sum_layer(num_units, memory, embeddings, max_time):
    """

    :param embeddings: [batch_size, max_time, ...].
    :param max_time:
    :return:
    """
    logits = tf.reshape(tf.layers.dense(memory, 1, activation=tf.nn.relu),[-1, max_time])
    probs = tf.reshape(tf.nn.softmax(logits=logits, dim=-1),[-1, max_time, 1])
    weighted = tf.multiply(embeddings, probs)
    sumed = tf.reshape(tf.reduce_sum(weighted, axis=1),[-1, num_units])
    return sumed








def build_rnn_encoder(encoder_param):
    """
    RNN Encoder
    Time major
    :return:
    """
    num_layers = encoder_param.num_layers
    num_residual_layers = encoder_param.num_residual_layers
    if encoder_param.encoder_type == "uni":
        utils.print_out("build a uni-encoder: num_layers = %d, num_residual_layers=%d" %
                        (num_layers, num_residual_layers))
        cell = _build_encoder_cell(
            encoder_param, num_layers, num_residual_layers)
        '''
        encoder_outputs: [max_time, batch_size, cell.output_size].
        '''
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            cell,
            encoder_param.enocder_seq_input,
            dtype=encoder_param.dtype,
            sequence_length=encoder_param.encoder_seq_len,
            time_major=True,
            swap_memory=True)
    elif encoder_param.encoder_type == "bi":
        num_bi_layers = int(num_layers / 2)
        num_bi_residual_layers = int(num_residual_layers / 2)
        utils.print_out("build a num_bi_layers = %d, num_bi_residual_layers=%d" %
                        (num_bi_layers, num_bi_residual_layers))

        encoder_outputs, bi_encoder_state = (
                _build_bidirectional_rnn(
                inputs=encoder_param.enocder_seq_input,
                sequence_length=encoder_param.encoder_seq_len,
                dtype=encoder_param.dtype,
                encoder_param=encoder_param,
                num_bi_layers=num_bi_layers,
                num_bi_residual_layers=num_bi_residual_layers))

        if num_bi_layers == 1:
            encoder_state = bi_encoder_state
        else:
            # alternatively concat forward and backward states
            encoder_state = []
            for layer_id in range(num_bi_layers):
                encoder_state.append(bi_encoder_state[0][layer_id])  # forward
                encoder_state.append(bi_encoder_state[1][layer_id])  # backward
            encoder_state = tuple(encoder_state)
    else:
        raise ValueError("Unknown encoder_type %s" % encoder_param.encoder_type)
    return encoder_outputs, encoder_state

def projection(input, input_dim, output_dim, activation=None):
    """
    Projection function
    :param input:
    :param input_dim:
    :param output_dim:
    :param activation:
    :return:
    """
    W = tf.get_variable(name='embedding_projection_w', shape=[input_dim, output_dim])
    b = tf.get_variable(name='embedding_bias_w', shape=[output_dim])
    tmp = tf.matmul(input,W) + b
    if activation is not None:
        tmp = activation(tmp)
    return tmp

def simple_3D_concat_gate_function(input_a, input_b, dimension):
    shapes = tf.unstack(tf.shape(input_a))
    d1 = shapes[0]
    d2 = shapes[1]
    W = tf.get_variable(name='simple_concat_gate_w', shape=[dimension*2, dimension])
    b = tf.get_variable(name='simple_concat_gate_b', shape=[dimension])
    concatenation = tf.reshape(tf.concat([input_a,input_b], axis=-1), [-1,2*dimension])
    gate = tf.sigmoid(tf.matmul(concatenation, W) + b)
    gate = tf.reshape(gate,[d1, d2, dimension])
    output = input_a * gate + input_b * (1.0 - gate)
    return output

def simple_3D_concat_weighted_function(input_a, input_b, dimension):
    shapes = tf.unstack(tf.shape(input_a))
    d1 = shapes[0]
    d2 = shapes[1]
    W = tf.get_variable(name='simple_concat_weight_w', shape=[dimension*2, 1])
    b = tf.get_variable(name='simple_concat_weight_b', shape=[1])
    concatenation = tf.reshape(tf.concat([input_a,input_b], axis=-1), [-1,2*dimension])
    weight = tf.sigmoid(tf.matmul(concatenation, W) + b)
    weight = tf.reshape(weight,[d1, d2, 1])
    output = input_a * weight + input_b * (1.0 - weight)
    return output

def simple_3D_concat_mask_weighted_function(input_a, input_b, unknown_mask, dimension,activation_for_source):
    """

    :param input_a:  charCNN [_seq_len, _batch_size, embed_size]
    :param input_b:  WordEmbedding [_seq_len, _batch_size, embed_size]
    :param unknown_mask: [_seq_len, _batch_size, 1]
    :param dimension:
    :return:
    """
    shapes = tf.unstack(tf.shape(input_a))
    d1 = shapes[0]
    d2 = shapes[1]
    W = tf.get_variable(name='simple_concat_weight_w', shape=[dimension*2+1, 1])
    b = tf.get_variable(name='simple_concat_weight_b', shape=[1])
    if activation_for_source is not None:
        utils.print_out('word_embedding+ RELU')
        input_b = activation_for_source(input_b)
    concatenation = tf.reshape(tf.concat([input_a,input_b,unknown_mask], axis=-1), [-1,2*dimension+1])
    weight = tf.sigmoid(tf.matmul(concatenation, W) + b)
    weight = tf.reshape(weight, [d1, d2, 1])
    output = input_a * weight + input_b * (1.0 - weight)

    return output
