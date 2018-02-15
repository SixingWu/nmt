# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Attention-based sequence-to-sequence model with dynamic RNN support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from . import embedding_helper
from .embedding_helper import EncoderParam

import tensorflow as tf

from .utils import misc_utils as utils

from . import attention_model
from . import model_helper
import numpy as np

__all__ = ["AttentionCharModel"]


class AttentionCharModel(attention_model.AttentionModel):


  def __init__(self,
               hparams,
               mode,
               iterator,
               source_vocab_table,
               target_vocab_table,
               reverse_target_vocab_table=None,
               scope=None,
               extra_args=None):
    # Set attention_mechanism_fn
    if extra_args and extra_args.attention_mechanism_fn:
      self.attention_mechanism_fn = extra_args.attention_mechanism_fn
    else:
      self.attention_mechanism_fn = create_attention_mechanism

    super(AttentionCharModel, self).__init__(
        hparams=hparams,
        mode=mode,
        iterator=iterator,
        source_vocab_table=source_vocab_table,
        target_vocab_table=target_vocab_table,
        reverse_target_vocab_table=reverse_target_vocab_table,
        scope=scope,
        extra_args=extra_args)

    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_summary = self._get_infer_summary(hparams)

  def _build_encoder(self, hparams):
      """Build an encoder."""
      utils.print_out("Build a CNN_Encoder")
      num_layers = hparams.num_layers
      num_residual_layers = hparams.num_residual_layers

      iterator = self.iterator

      source = iterator.source
      if self.time_major:
          source = tf.transpose(source)

      with tf.variable_scope("encoder") as scope:
          dtype = scope.dtype
          # Look up embedding, emp_inp: [max_time, batch_size, num_units]
          dtype = scope.dtype
          utils.print_out('source embedding type=%s' % hparams.src_embed_type)
          if hparams.src_embed_type == 'raw':
              # Look up embedding, emp_inp: [max_time, batch_size, embedding]
              encoder_emb_inp = tf.nn.embedding_lookup(
                  self.embedding_encoder, source)
          elif hparams.src_embed_type == 'avg_segment':
              seg_source = iterator.seg_source
              encoder_emb_inp = tf.nn.embedding_lookup(self.embedding_encoder, seg_source)

          elif hparams.src_embed_type == 'rnn_segment':
              with tf.variable_scope('rnn_segment_embedding'):
                  # [batch, seq_len, seg_len]
                  seg_source = iterator.seg_source
                  # [batch, seq_len]
                  seg_len_source = iterator.seg_src_lens
                  # [batch, seq_len, seg_len, embedding]
                  encoder_emb_inp = tf.nn.embedding_lookup(self.seg_embedding_encoder, seg_source)


                  _batch_size = tf.shape(encoder_emb_inp)[0]
                  _seq_len = tf.shape(encoder_emb_inp)[1]
                  # flattern to [batch_size*seq_len,seg_len,embed]
                  encoder_emb_inp = tf.reshape(encoder_emb_inp,
                                               [_batch_size * _seq_len, hparams.seg_len, hparams.seg_embed_dim])
                  encoder_emb_inp = tf.transpose(encoder_emb_inp, perm=[1, 0, 2])
                  seg_len_source = tf.reshape(seg_len_source, [_batch_size, _seq_len])
                  flattern_sequence_length = tf.reshape(seg_len_source, [-1])
                  with tf.variable_scope('word_embedding_encoder'):
                      word_encoder = EncoderParam(encoder_type="uni",
                                                  num_layers=1,
                                                  num_residual_layers=0,
                                                  unit_type="gru",  # In order to lightweight
                                                  forget_bias=hparams.forget_bias,
                                                  dropout=hparams.dropout,
                                                  num_gpus=hparams.num_gpus,
                                                  mode=self.mode,
                                                  enocder_seq_input=encoder_emb_inp,
                                                  encoder_seq_len=flattern_sequence_length,
                                                  dtype=dtype,
                                                  single_cell_fn=self.single_cell_fn,
                                                  # LSTM 2 Tuple
                                                  num_units=(hparams.seg_embed_dim),
                                                  name=None)
                      word_encoder_outputs, word_encoder_state = embedding_helper.build_rnn_encoder(word_encoder)
                      print("debug" + str(word_encoder_state))
                      # [batch_size * seq_len, embed]
                      encoder_emb_inp = word_encoder_state
                      encoder_emb_inp = embedding_helper.projection(encoder_emb_inp, hparams.seg_embed_dim,
                                                                    hparams.embed_dim)
                      encoder_emb_inp = tf.reshape(encoder_emb_inp, [_batch_size, _seq_len, hparams.embed_dim])
                      # [max_time, batch_size, embedding]
                      encoder_emb_inp = tf.transpose(encoder_emb_inp, perm=[1, 0, 2])
                      encoder_emb_inp += tf.nn.embedding_lookup(
                          self.embedding_encoder, source)

                  # Look up embedding, emp_inp: [max_time, batch_size, num_units]

          else:
              raise Exception('Unknown src_embed_type  %s' % hparams.src_embed_type)

          original_encoder_emb_inp = encoder_emb_inp
          dims = tf.unstack(tf.shape(encoder_emb_inp))
          max_time = dims[0]
          batch_size = dims[1]
          num_units = hparams.num_units
          embed_dim = hparams.embed_dim

          min_windows = hparams.cnn_min_window_size
          max_windows = hparams.cnn_max_window_size
          high_way_layers = hparams.high_way_layer
          filters_per_windows = hparams.filters_per_windows

          utils.print_out('debug:')
          print(tf.shape(source))
          print(tf.shape(encoder_emb_inp))
          conv_inputs = tf.reshape(encoder_emb_inp, [max_time, batch_size, embed_dim, 1])
          # [batch, height = num_units, width = max_time, channels = 1]
          conv_inputs = tf.transpose(conv_inputs, perm=[1, 2, 0, 3])

          # CNN layer, windows width from 1 to 5
          conv_outputs = []
          filter_nums = (max_windows - min_windows + 1) * filters_per_windows
          for width in range(min_windows, max_windows + 1):
              filter = tf.get_variable("filter_%d" % (width), shape=[embed_dim, width, 1, filters_per_windows])
              strides = [1, embed_dim, 1, 1]
              # [batch, height = 1, width = max_time, channels = filters_per_windows]
              conv_out = tf.nn.relu(tf.nn.conv2d(conv_inputs, filter, strides, padding='SAME'))
              conv_outputs.append(conv_out)

          # max_pooling with strides=3
          pool_outputs = []
          width_strides = hparams.width_strides
          strides = [1, 1, width_strides, 1]
          segment_len = tf.cast(tf.ceil(max_time / width_strides), tf.int32)
          for conv_output in conv_outputs:
              pool_out = tf.nn.max_pool(conv_output, [1, 1, width_strides, 1], strides, padding='SAME')
              # [batch, height = 1, width = segment_len, channels = filters_per_windows]
              pool_out = tf.reshape(pool_out, [batch_size, segment_len, filters_per_windows])
              pool_outputs.append(pool_out)

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

          # [batch_size, segment_len, filter_nums=num_windows*filters_per_windows]

          if hparams.high_way_type == 'uniform':
              stacked_results = tf.concat(pool_outputs, axis=-1)
              high_way_tmp = tf.reshape(stacked_results, [-1, filter_nums])
              for i in range(high_way_layers):
                  high_way_tmp = highway(high_way_tmp, filter_nums, tf.nn.relu, name='highway_%d' % i)
              highway_outputs = tf.reshape(high_way_tmp, [batch_size, segment_len, filter_nums])
          elif hparams.high_way_type == 'per_filter':
              highway_results = []
              for w,pool_result in enumerate(pool_outputs):
                  pool_highway_tmp = tf.reshape(pool_result,[-1,filters_per_windows])
                  for i in range(high_way_layers):
                      pool_highway_tmp = highway(pool_highway_tmp, filters_per_windows, tf.nn.relu, name='highway_w%d_%d' % (w+min_windows,i))
                  highway_results.append(pool_highway_tmp)
              stacked_results = tf.concat(highway_results, axis=-1)
              highway_outputs = tf.reshape(stacked_results, [batch_size, segment_len, filter_nums])


          # [time_width, batch, height]
          encoder_emb_inp = tf.transpose(highway_outputs, perm=[1, 0, 2])

          # segment_lens
          segment_length = tf.cast(tf.ceil(self.iterator.source_sequence_length / width_strides), tf.int64)

          if hparams.residual_cnn_layer:
              if hparams.residual_cnn_layer_type == 'concat':
                  assert int(width_strides) == 1,'concat resudual_cnn_layer asks width_strides == 1'
                  utils.print_out("Residual CNN is enabled")
                  encoder_emb_inp = tf.concat([encoder_emb_inp,original_encoder_emb_inp],axis=-1)
              elif hparams.residual_cnn_layer_type == 'transform':
                  assert int(width_strides) == 1, 'transformed resudual_cnn_layer asks width_strides == 1'
                  W_transform = tf.Variable(tf.truncated_normal([filter_nums, embed_dim], stddev=0.1), name="res_transform_w")
                  b_transform = tf.Variable(tf.truncated_normal([embed_dim], stddev=0.1),name="res_transform_b")
                  encoder_emb_inp = tf.matmul(tf.reshape(encoder_emb_inp,[-1,filter_nums]), W_transform) + b_transform
                  encoder_emb_inp = tf.reshape(encoder_emb_inp,[max_time,batch_size,embed_dim])
                  encoder_emb_inp = encoder_emb_inp + original_encoder_emb_inp
              elif hparams.residual_cnn_layer_type == 'transformGate':
                  assert int(width_strides) == 1, 'transformed resudual_cnn_layer asks width_strides == 1'
                  W_transform = tf.Variable(tf.truncated_normal([filter_nums, hparams.embed_dim], stddev=0.1), name="res_transform_w")
                  b_transform = tf.Variable(tf.truncated_normal([hparams.embed_dim], stddev=0.1),name="res_transform_b")
                  #res [max_time*batch_size, num_units]
                  encoder_emb_inp = tf.matmul(tf.reshape(encoder_emb_inp,[-1,filter_nums]), W_transform) + b_transform

                  gate_W = tf.Variable(tf.truncated_normal([hparams.embed_dim, hparams.embed_dim], stddev=0.1), name="gate_transform_b")
                  gate_b = tf.Variable(tf.truncated_normal([hparams.embed_dim], stddev=0.1), name="gate_transform_b")

                  gate_val = tf.sigmoid(tf.matmul(encoder_emb_inp, gate_W) + gate_b)
                  encoder_emb_inp = tf.multiply(encoder_emb_inp, gate_val)
                  original_encoder_emb_inp = tf.reshape(original_encoder_emb_inp,[-1, embed_dim])
                  original_encoder_emb_inp = tf.multiply(original_encoder_emb_inp, 1.0 - gate_val)
                  encoder_emb_inp = tf.reshape(encoder_emb_inp, [max_time, batch_size, embed_dim])
                  original_encoder_emb_inp = tf.reshape(original_encoder_emb_inp, [max_time, batch_size, embed_dim])
                  encoder_emb_inp = encoder_emb_inp + original_encoder_emb_inp
              elif hparams.residual_cnn_layer_type == 'transformRe':
                  assert int(width_strides) == 1, 'transformed resudual_cnn_layer asks width_strides == 1'
                  W_transform = tf.Variable(tf.truncated_normal([filter_nums, hparams.embed_dim], stddev=0.1),
                                            name="res_transform_w")
                  b_transform = tf.Variable(tf.truncated_normal([hparams.embed_dim], stddev=0.1), name="res_transform_b")
                  encoder_emb_inp = tf.nn.relu(tf.matmul(tf.reshape(encoder_emb_inp, [-1, filter_nums]), W_transform) + b_transform)
                  encoder_emb_inp = tf.reshape(encoder_emb_inp, [max_time, batch_size, embed_dim])
                  encoder_emb_inp = encoder_emb_inp + original_encoder_emb_inp
          # Encoder_outpus: [max_time, batch_size, num_units]
          utils.print_out("encoder_emb_inp : %s" % str(encoder_emb_inp))
          # build the top level Encoder
          top_level_encoder = EncoderParam(encoder_type=hparams.encoder_type,
                                           num_layers=hparams.num_layers,
                                           num_residual_layers=hparams.num_residual_layers,
                                           unit_type=hparams.unit_type,
                                           forget_bias=hparams.forget_bias,
                                           dropout=hparams.dropout,
                                           num_gpus=hparams.num_gpus,
                                           mode=self.mode,
                                           enocder_seq_input=encoder_emb_inp,
                                           encoder_seq_len=iterator.source_sequence_length,
                                           dtype=dtype,
                                           single_cell_fn=self.single_cell_fn,
                                           num_units=hparams.num_units,
                                           name=None)
      encoder_outputs, encoder_state = embedding_helper.build_rnn_encoder(top_level_encoder)
      return encoder_outputs, encoder_state

  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Build a RNN cell with attention mechanism that can be used by decoder."""
    attention_option = hparams.attention
    attention_architecture = hparams.attention_architecture

    if attention_architecture != "char_standard":
      raise ValueError(
          "Unknown(Unmatched!) attention architecture %s" % attention_architecture)

    num_units = hparams.num_units
    num_layers = hparams.num_layers
    num_residual_layers = hparams.num_residual_layers
    num_gpus = hparams.num_gpus
    beam_width = hparams.beam_width

    width_strides = hparams.width_strides
    source_sequence_length = tf.cast(tf.ceil(self.iterator.source_sequence_length / width_strides), tf.int64)


    dtype = tf.float32

    # Ensure memory is batch-major
    if self.time_major:
      memory = tf.transpose(encoder_outputs, [1, 0, 2])
    else:
      memory = encoder_outputs

    if self.mode == tf.contrib.learn.ModeKeys.INFER and beam_width > 0:
      memory = tf.contrib.seq2seq.tile_batch(
          memory, multiplier=beam_width)
      source_sequence_length = tf.contrib.seq2seq.tile_batch(
          source_sequence_length, multiplier=beam_width)
      encoder_state = tf.contrib.seq2seq.tile_batch(
          encoder_state, multiplier=beam_width)
      batch_size = self.batch_size * beam_width
    else:
      batch_size = self.batch_size

    attention_mechanism = self.attention_mechanism_fn(
        attention_option, num_units, memory, source_sequence_length, self.mode)

    cell = model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=num_units,
        num_layers=num_layers,
        num_residual_layers=num_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=num_gpus,
        mode=self.mode,
        single_cell_fn=self.single_cell_fn)

    # Only generate alignment in greedy INFER mode.
    alignment_history = (self.mode == tf.contrib.learn.ModeKeys.INFER and
                         beam_width == 0)
    cell = tf.contrib.seq2seq.AttentionWrapper(
        cell,
        attention_mechanism,
        attention_layer_size=num_units,
        alignment_history=alignment_history,
        output_attention=hparams.output_attention,
        name="attention")

    # TODO(thangluong): do we need num_layers, num_gpus?
    cell = tf.contrib.rnn.DeviceWrapper(cell,
                                        model_helper.get_device_str(
                                            num_layers - 1, num_gpus))

    if hparams.pass_hidden_state:
      decoder_initial_state = cell.zero_state(batch_size, dtype).clone(
          cell_state=encoder_state)
    else:
      decoder_initial_state = cell.zero_state(batch_size, dtype)

    return cell, decoder_initial_state

  def _get_infer_summary(self, hparams):
    if hparams.beam_width > 0:
      return tf.no_op()
    return _create_attention_images_summary(self.final_context_state)


def create_attention_mechanism(attention_option, num_units, memory,
                               source_sequence_length):
  """Create attention mechanism based on the attention_option."""

  # Mechanism
  if attention_option == "luong":
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units, memory, memory_sequence_length=source_sequence_length)
  elif attention_option == "scaled_luong":
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units,
        memory,
        memory_sequence_length=source_sequence_length,
        scale=True)
  elif attention_option == "bahdanau":
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units, memory, memory_sequence_length=source_sequence_length)
  elif attention_option == "normed_bahdanau":
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units,
        memory,
        memory_sequence_length=source_sequence_length,
        normalize=True)
  else:
    raise ValueError("Unknown attention option %s" % attention_option)

  return attention_mechanism


def _create_attention_images_summary(final_context_state):
  """create attention image and attention summary."""
  attention_images = (final_context_state.alignment_history.stack())
  # Reshape to (batch, src_seq_len, tgt_seq_len,1)
  attention_images = tf.expand_dims(
      tf.transpose(attention_images, [1, 2, 0]), -1)
  # Scale to range [0, 255]
  attention_images *= 255
  attention_summary = tf.summary.image("attention_images", attention_images)
  return attention_summary
