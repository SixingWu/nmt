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
          encoder_emb_inp = tf.nn.embedding_lookup(
              self.embedding_encoder, source)
          max_time = hparams.src_max_len
          batch_size = hparams.batch_size
          num_units = hparams.num_units


          conv_inputs = tf.reshape(encoder_emb_inp, [max_time, batch_size, num_units, 1])
          # [batch, height = num_units, width = max_time, channels = 1]
          conv_inputs = tf.transpose(conv_inputs, perm=[1, 2, 0, 3])

          # CNN layer, windows width from 1 to 5
          conv_outputs = []
          filter_nums = 5
          for width in range(1, 6):
              filter = tf.get_variable("filter_%d" % (width), shape=[num_units, width, 1, 1])
              strides = [1, num_units, 1, 1]
              # [batch, height = 1, width = max_time, channels = 1]
              conv_out = tf.nn.relu(tf.nn.conv2d(conv_inputs, filter, strides, padding='SAME'))
              conv_outputs.append(conv_out)

          # max_pooling with strides=3
          pool_outputs = []
          width_strides = 3
          strides = [1, 1, width_strides, 1]
          segment_len = int(np.ceil(max_time / 3))
          for conv_output in conv_outputs:
              pool_out = tf.nn.max_pool(conv_output, [1, 1, width_strides, 1], strides, padding='SAME')
              pool_out = tf.reshape(pool_out, [batch_size, segment_len])
              pool_outputs.append(pool_out)

          def highway(x, size, activation, carry_bias=-1.0):
              W_T = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight_transform")
              b_T = tf.Variable(tf.constant(carry_bias, shape=[size]), name="bias_transform")
              W = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name="weight")
              b = tf.Variable(tf.constant(0.1, shape=[size]), name="bias")
              T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name="transform_gate")
              H = activation(tf.matmul(x, W) + b, name="activation")
              C = tf.subtract(1.0, T, name="carry_gate")
              y = tf.add(tf.multiply(H, T), tf.multiply(x, C), "y")
              return y

          # [batch, width, height]
          stacked_results = tf.stack(pool_outputs, axis=2)

          highway_outputs = highway(tf.reshape(stacked_results, [-1, filter_nums]), filter_nums, tf.nn.relu)

          highway_outputs = tf.reshape(highway_outputs, [batch_size, segment_len, filter_nums])
          # [time_width, batch, height]

          encoder_emb_inp = tf.transpose(highway_outputs, perm=[1, 0, 2])


          # Encoder_outpus: [max_time, batch_size, num_units]
          if hparams.encoder_type == "uni":
              utils.print_out("  num_layers = %d, num_residual_layers=%d" %
                              (num_layers, num_residual_layers))
              cell = self._build_encoder_cell(
                  hparams, num_layers, num_residual_layers)

              encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                  cell,
                  encoder_emb_inp,
                  dtype=dtype,
                  sequence_length=segment_len,
                  time_major=self.time_major,
                  swap_memory=True)
          elif hparams.encoder_type == "bi":
              num_bi_layers = int(num_layers / 2)
              num_bi_residual_layers = int(num_residual_layers / 2)
              utils.print_out("  num_bi_layers = %d, num_bi_residual_layers=%d" %
                              (num_bi_layers, num_bi_residual_layers))

              encoder_outputs, bi_encoder_state = (
                  self._build_bidirectional_rnn(
                      inputs=encoder_emb_inp,
                      sequence_length=segment_len,
                      dtype=dtype,
                      hparams=hparams,
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
              raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)
      return encoder_outputs, encoder_state

  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Build a RNN cell with attention mechanism that can be used by decoder."""
    attention_option = hparams.attention
    attention_architecture = hparams.attention_architecture

    if attention_architecture != "standard":
      raise ValueError(
          "Unknown attention architecture %s" % attention_architecture)

    num_units = hparams.num_units
    num_layers = hparams.num_layers
    num_residual_layers = hparams.num_residual_layers
    num_gpus = hparams.num_gpus
    beam_width = hparams.beam_width

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
                               source_sequence_length, mode):
  """Create attention mechanism based on the attention_option."""
  del mode  # unused

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
