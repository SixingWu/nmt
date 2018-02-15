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
