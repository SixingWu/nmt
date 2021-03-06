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

"""Basic sequence-to-sequence model with dynamic RNN support."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np


import tensorflow as tf

from tensorflow.python.layers import core as layers_core

from . import model_helper
from .utils import iterator_utils
from .utils import misc_utils as utils
from . import embedding_helper
from .embedding_helper import EncoderParam
from .embedding_helper import CNNEncoderParam

utils.check_tensorflow_version()

__all__ = ["BaseModel", "Model"]


class BaseModel(object):
  """Sequence-to-sequence base class.
  """

  def __init__(self,
               hparams,
               mode,
               iterator,
               source_vocab_table,
               target_vocab_table,
               reverse_target_vocab_table=None,
               scope=None,
               extra_args=None):
    """Create the model.

    Args:
      hparams: Hyperparameter configurations.
      mode: TRAIN | EVAL | INFER
      iterator: Dataset Iterator that feeds data.
      source_vocab_table: Lookup table mapping source words to ids.
      target_vocab_table: Lookup table mapping target words to ids.
      reverse_target_vocab_table: Lookup table mapping ids to target words. Only
        required in INFER mode. Defaults to None.
      scope: scope of the model.
      extra_args: model_helper.ExtraArgs, for passing customizable functions.

    """
    assert isinstance(iterator, iterator_utils.BatchedInput) or isinstance(iterator, iterator_utils.SegBatchedInput)
    self.iterator = iterator
    self.mode = mode
    self.src_vocab_table = source_vocab_table
    self.tgt_vocab_table = target_vocab_table

    self.src_vocab_size = hparams.src_vocab_size
    self.tgt_vocab_size = hparams.tgt_vocab_size
    self.num_layers = hparams.num_layers
    self.num_gpus = hparams.num_gpus
    self.time_major = hparams.time_major
    self.loss_items = []

    # extra_args: to make it flexible for adding external customizable code
    self.single_cell_fn = None
    if extra_args:
      self.single_cell_fn = extra_args.single_cell_fn

    # Initializer
    initializer = model_helper.get_initializer(
        hparams.init_op, hparams.random_seed, hparams.init_weight)
    tf.get_variable_scope().set_initializer(initializer)

    # Embeddings
    self.init_embeddings(hparams, scope)
    self.batch_size = tf.size(self.iterator.source_sequence_length)

    # Projection
    with tf.variable_scope(scope or "build_network"):
      with tf.variable_scope("decoder/output_projection"):
        self.output_layer = layers_core.Dense(
            hparams.tgt_vocab_size, use_bias=False, name="output_projection")

    ## Train graph
    res = self.build_graph(hparams, scope=scope)

    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.train_loss = res[1]
      self.word_count = tf.reduce_sum(
          self.iterator.source_sequence_length) + tf.reduce_sum(
              self.iterator.target_sequence_length)
    elif self.mode == tf.contrib.learn.ModeKeys.EVAL:
      self.eval_loss = res[1]
    elif self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_logits, _, self.final_context_state, self.sample_id = res
      self.sample_words = reverse_target_vocab_table.lookup(
          tf.to_int64(self.sample_id))

    if self.mode != tf.contrib.learn.ModeKeys.INFER:
      ## Count the number of predicted words for compute ppl.
      self.predict_count = tf.reduce_sum(
          self.iterator.target_sequence_length)

    self.global_step = tf.Variable(0, trainable=False)
    params = tf.trainable_variables()

    # Gradients and SGD update operation for training the model.
    # Arrage for the embedding vars to appear at the beginning.
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.learning_rate = tf.constant(hparams.learning_rate)
      # warm-up
      self.learning_rate = self._get_learning_rate_warmup(hparams)
      # decay
      self.learning_rate = self._get_learning_rate_decay(hparams)

      # Optimizer
      if hparams.optimizer == "sgd":
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        tf.summary.scalar("lr", self.learning_rate)
      elif hparams.optimizer == "adam":
        opt = tf.train.AdamOptimizer(self.learning_rate)

      # Gradients
      gradients = tf.gradients(
          self.train_loss,
          params,
          colocate_gradients_with_ops=hparams.colocate_gradients_with_ops)

      clipped_grads, grad_norm_summary, grad_norm = model_helper.gradient_clip(
          gradients, max_gradient_norm=hparams.max_gradient_norm)
      self.grad_norm = grad_norm

      self.update = opt.apply_gradients(
          zip(clipped_grads, params), global_step=self.global_step)

      # Summary
      self.train_summary = tf.summary.merge([
          tf.summary.scalar("lr", self.learning_rate),
          tf.summary.scalar("train_loss", self.train_loss),
      ] + grad_norm_summary)

    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      self.infer_summary = self._get_infer_summary(hparams)

    # Saver
    self.saver = tf.train.Saver(
        tf.global_variables(), max_to_keep=hparams.num_keep_ckpts)

    # Print trainable variables
    utils.print_out("# Trainable variables")
    for param in params:
      utils.print_out("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                        param.op.device))

  def _get_learning_rate_warmup(self, hparams):
    """Get learning rate warmup."""
    warmup_steps = hparams.warmup_steps
    warmup_scheme = hparams.warmup_scheme
    utils.print_out("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
                    (hparams.learning_rate, warmup_steps, warmup_scheme))

    # Apply inverse decay if global steps less than warmup steps.
    # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
    # When step < warmup_steps,
    #   learing_rate *= warmup_factor ** (warmup_steps - step)
    if warmup_scheme == "t2t":
      # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
      warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
      inv_decay = warmup_factor**(
          tf.to_float(warmup_steps - self.global_step))
    else:
      raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

    return tf.cond(
        self.global_step < hparams.warmup_steps,
        lambda: inv_decay * self.learning_rate,
        lambda: self.learning_rate,
        name="learning_rate_warump_cond")

  def _get_learning_rate_decay(self, hparams):
    """Get learning rate decay."""
    if hparams.decay_scheme == "luong10":
      start_decay_step = int(hparams.num_train_steps / 2)
      remain_steps = hparams.num_train_steps - start_decay_step
      decay_steps = int(remain_steps / 10)  # decay 10 times
      decay_factor = 0.5
    elif hparams.decay_scheme == "luong234":
      start_decay_step = int(hparams.num_train_steps * 2 / 3)
      remain_steps = hparams.num_train_steps - start_decay_step
      decay_steps = int(remain_steps / 4)  # decay 4 times
      decay_factor = 0.5
    elif not hparams.decay_scheme:  # no decay
      start_decay_step = hparams.num_train_steps
      decay_steps = 0
      decay_factor = 1.0
    elif hparams.decay_scheme:
      raise ValueError("Unknown decay scheme %s" % hparams.decay_scheme)
    utils.print_out("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
                    "decay_factor %g" % (hparams.decay_scheme,
                                         start_decay_step,
                                         decay_steps,
                                         decay_factor))

    return tf.cond(
        self.global_step < start_decay_step,
        lambda: self.learning_rate,
        lambda: tf.train.exponential_decay(
            self.learning_rate,
            (self.global_step - start_decay_step),
            decay_steps, decay_factor, staircase=True),
        name="learning_rate_decay_cond")

  def init_embeddings(self, hparams, scope):
    """
        CONSTRUCT_EMBEDDING
        Init embeddings.
    """
    self.embedding_encoder, self.embedding_decoder = (
        model_helper.create_emb_for_encoder_and_decoder(
            share_vocab=hparams.share_vocab,
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            src_embed_size=hparams.embed_dim,
            tgt_embed_size=hparams.embed_dim,
            num_partitions=hparams.num_embeddings_partitions,
            src_vocab_file=hparams.src_vocab_file,
            tgt_vocab_file=hparams.tgt_vocab_file,
            src_embed_file=hparams.src_embed_file,
            tgt_embed_file=hparams.tgt_embed_file,
            scope=scope,))

    if 'segment' in hparams.src_embed_type:
        with tf.variable_scope('segment_embedding'):
            if hparams.seg_embed_mode == 'share':
                self.seg_embedding_encoder=self.embedding_encoder
                self.seg_embedding_decoder=self.embedding_decoder
            elif hparams.seg_embed_mode == 'separate':
                self.seg_embedding_encoder, self.seg_embedding_decoder = (
                    model_helper.create_emb_for_encoder_and_decoder(
                        share_vocab=hparams.share_vocab,
                        src_vocab_size=hparams.seg_src_vocab_size,
                        tgt_vocab_size=hparams.seg_tgt_vocab_size,
                        src_embed_size=hparams.seg_embed_dim,
                        tgt_embed_size=hparams.seg_embed_dim,
                        num_partitions=hparams.num_embeddings_partitions,
                        src_vocab_file=hparams.src_vocab_file+'_seg',
                        tgt_vocab_file=hparams.tgt_vocab_file+'_seg',
                        src_embed_file=hparams.seg_src_embed_file,
                        tgt_embed_file=hparams.seg_tgt_embed_file,
                        scope=scope, ))
            else:
                raise Exception('Unkown seg_embed_mode')

  def train(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
    return sess.run([self.update,
                     self.train_loss,
                     self.predict_count,
                     self.train_summary,
                     self.global_step,
                     self.word_count,
                     self.batch_size,
                     self.grad_norm,
                     self.learning_rate])

  def eval(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.EVAL
    return sess.run([self.eval_loss,
                     self.predict_count,
                     self.batch_size])

  def build_graph(self, hparams, scope=None):
    """Subclass must implement this method.

    Creates a sequence-to-sequence model with dynamic RNN decoder API.
    Args:
      hparams: Hyperparameter configurations.
      scope: VariableScope for the created subgraph; default "dynamic_seq2seq".

    Returns:
      A tuple of the form (logits, loss, final_context_state),
      where:
        logits: float32 Tensor [batch_size x num_decoder_symbols].
        loss: the total loss / batch_size.
        final_context_state: The final state of decoder RNN.

    Raises:
      ValueError: if encoder_type differs from mono and bi, or
        attention_option is not (luong | scaled_luong |
        bahdanau | normed_bahdanau).
    """
    utils.print_out("# creating %s graph ..." % self.mode)
    dtype = tf.float32
    num_layers = hparams.num_layers
    num_gpus = hparams.num_gpus

    with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):

      encoder_outputs, encoder_state = self._build_encoder(hparams)

      ## Decoder
      logits, sample_id, final_context_state = self._build_decoder(
          encoder_outputs, encoder_state, hparams)

      ## Loss
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        with tf.device(model_helper.get_device_str(num_layers - 1, num_gpus)):
          loss = self._compute_loss(logits)
      else:
        loss = None

      return logits, loss, final_context_state, sample_id

  @abc.abstractmethod
  def _build_encoder(self, hparams):
    """Subclass must implement this.

    Build and run an RNN encoder.

    Args:
      hparams: Hyperparameters configurations.

    Returns:
      A tuple of encoder_outputs and encoder_state.
    """
    pass

  def _build_encoder_cell(self, hparams, num_layers, num_residual_layers,
                          base_gpu=0):
    """Build a multi-layer RNN cell that can be used by encoder."""

    return model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=hparams.num_units,
        num_layers=num_layers,
        num_residual_layers=num_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=hparams.num_gpus,
        mode=self.mode,
        base_gpu=base_gpu,
        single_cell_fn=self.single_cell_fn)

  def _get_infer_maximum_iterations(self, hparams, source_sequence_length):
    """Maximum decoding steps at inference time."""
    if hparams.tgt_max_len_infer:
      maximum_iterations = hparams.tgt_max_len_infer
      utils.print_out("  decoding maximum_iterations %d" % maximum_iterations)
    else:
      # TODO(thangluong): add decoding_length_factor flag
      decoding_length_factor = 2.0
      max_encoder_length = tf.reduce_max(source_sequence_length)
      maximum_iterations = tf.to_int32(tf.round(
          tf.to_float(max_encoder_length) * decoding_length_factor))
    return maximum_iterations

  def _build_decoder(self, encoder_outputs, encoder_state, hparams):
    """Build and run a RNN decoder with a final projection layer.

    Args:
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.
      hparams: The Hyperparameters configurations.

    Returns:
      A tuple of final logits and final decoder state:
        logits: size [time, batch_size, vocab_size] when time_major=True.
    """
    tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.sos)),
                         tf.int32)
    tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.eos)),
                         tf.int32)

    num_layers = hparams.num_layers
    num_gpus = hparams.num_gpus

    iterator = self.iterator

    # maximum_iteration: The maximum decoding steps.
    maximum_iterations = self._get_infer_maximum_iterations(
        hparams, iterator.source_sequence_length)

    ## Decoder.
    with tf.variable_scope("decoder") as decoder_scope:
      cell, decoder_initial_state = self._build_decoder_cell(
          hparams, encoder_outputs, encoder_state,
          iterator.source_sequence_length)

      ## Train or eval
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        # decoder_emp_inp: [max_time, batch_size, num_units]
        target_input = iterator.target_input
        if self.time_major:
          target_input = tf.transpose(target_input)
        decoder_emb_inp = tf.nn.embedding_lookup(
            self.embedding_decoder, target_input)

        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_emb_inp, iterator.target_sequence_length,
            time_major=self.time_major)

        # Decoder
        my_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell,
            helper,
            decoder_initial_state,)

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

        sample_id = outputs.sample_id

        # Note: there's a subtle difference here between train and inference.
        # We could have set output_layer when create my_decoder
        #   and shared more code between train and inference.
        # We chose to apply the output_layer to all timesteps for speed:
        #   10% improvements for small models & 20% for larger ones.
        # If memory is a concern, we should apply output_layer per timestep.
        device_id = num_layers if num_layers < num_gpus else (num_layers - 1)
        with tf.device(model_helper.get_device_str(device_id, num_gpus)):
          logits = self.output_layer(outputs.rnn_output) # TODO Debug

      ## Inference
      else:
        beam_width = hparams.beam_width
        length_penalty_weight = hparams.length_penalty_weight
        start_tokens = tf.fill([self.batch_size], tgt_sos_id)
        end_token = tgt_eos_id

        if beam_width > 0:
          my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
              cell=cell,
              embedding=self.embedding_decoder,
              start_tokens=start_tokens,
              end_token=end_token,
              initial_state=decoder_initial_state,
              beam_width=beam_width,
              output_layer=self.output_layer,
              length_penalty_weight=length_penalty_weight)
        else:
          # Helper
          helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
              self.embedding_decoder, start_tokens, end_token)

          # Decoder
          my_decoder = tf.contrib.seq2seq.BasicDecoder(
              cell,
              helper,
              decoder_initial_state,
              output_layer=self.output_layer  # applied per timestep
          )

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            maximum_iterations=maximum_iterations,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

        if beam_width > 0:
          logits = tf.no_op()
          sample_id = outputs.predicted_ids
        else:
          logits = outputs.rnn_output
          sample_id = outputs.sample_id

    return logits, sample_id, final_context_state

  def get_max_time(self, tensor):
    time_axis = 0 if self.time_major else 1
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

  @abc.abstractmethod
  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Subclass must implement this.

    Args:
      hparams: Hyperparameters configurations.
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.
      source_sequence_length: sequence length of encoder_outputs.

    Returns:
      A tuple of a multi-layer RNN cell used by decoder
        and the intial state of the decoder RNN.
    """
    pass

  def _compute_loss(self, logits):

    """Compute optimization loss."""
    target_output = self.iterator.target_output
    if self.time_major:
      target_output = tf.transpose(target_output)
    max_time = self.get_max_time(target_output)
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_output, logits=logits)
    target_weights = tf.sequence_mask(
        self.iterator.target_sequence_length, max_time, dtype=logits.dtype)
    if self.time_major:
      target_weights = tf.transpose(target_weights)

    loss = tf.reduce_sum(
        crossent * target_weights) / tf.to_float(self.batch_size)

    for item in self.loss_items:
        loss += tf.reduce_sum(item)
    print(self.loss_items)
    return loss

  def _get_infer_summary(self, hparams):
    return tf.no_op()

  def infer(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.INFER
    return sess.run([
        self.infer_logits, self.infer_summary, self.sample_id, self.sample_words
    ])

  def decode(self, sess):
    """Decode a batch.

    Args:
      sess: tensorflow session to use.

    Returns:
      A tuple consiting of outputs, infer_summary.
        outputs: of size [batch_size, time]
    """
    _, infer_summary, _, sample_words = self.infer(sess)

    # make sure outputs is of shape [batch_size, time] or [beam_width,
    # batch_size, time] when using beam search.
    if self.time_major:
      sample_words = sample_words.transpose()
    elif sample_words.ndim == 3:  # beam search output in [batch_size,
                                  # time, beam_width] shape.
      sample_words = sample_words.transpose([2, 0, 1])
    return sample_words, infer_summary


class Model(BaseModel):
  """Sequence-to-sequence dynamic model.

  This class implements a multi-layer recurrent neural network as encoder,
  and a multi-layer recurrent neural network decoder.
  """



  def _build_encoder(self, hparams):
    """Build an encoder.   TODO bug CNN Encoder会导致全部正值"""
    utils.print_out("utilizing the basic model to build encoder")
    #TODO 增加对输入Embedding的处理方法 [1] raw_embedding, [2] cnn_per_word_embedding
    num_layers = hparams.num_layers
    num_residual_layers = hparams.num_residual_layers
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
        dropout = self.mode
        charcnn_dropout = hparams.charcnn_dropout
    else:
        dropout = 0.0
        charcnn_dropout=0.0
    iterator = self.iterator
    source = iterator.source
    # 默认是true，也就是iterator过来的是batch，*，、*
    utils.print_out('Time major: %s' % self.time_major)
    if self.time_major:
      source = tf.transpose(source)

    with tf.variable_scope("encoder") as scope:
      dtype = scope.dtype
      utils.print_out('source embedding type=%s' % hparams.src_embed_type)
      if hparams.src_embed_type == 'raw':
          # Look up embedding, emp_inp: [max_time, batch_size, embedding]
          encoder_emb_inp = tf.nn.embedding_lookup(
              self.embedding_encoder, source)
      elif 'cnn_segment' in hparams.src_embed_type:
          with tf.variable_scope('cnn_segment_embedding'):
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
              encoder_emb_inp=tf.transpose(encoder_emb_inp,perm=[1,0,2] )
              flattern_sequence_length = tf.reshape(seg_len_source, [-1])
              with tf.variable_scope('cnn_word_embedding_encoder'):
                  word_encoder = CNNEncoderParam(
                      dropout=charcnn_dropout,
                      max_time=hparams.seg_len,
                      batch_size= _batch_size * _seq_len,
                      embed_dim=hparams.seg_embed_dim,
                      min_windows=hparams.charcnn_min_window_size,
                      max_windows=hparams.charcnn_max_window_size,
                      flexible_configs=hparams.flexible_charcnn_windows,
                      filters_per_windows=hparams.charcnn_filters_per_windows,
                      width_strides=1,
                      high_way_type=hparams.charcnn_high_way_type,
                      high_way_layers=hparams.charcnn_high_way_layer,
                      max_k=hparams.charcnn_max_k,
                      name='cnn_encoder',
                      relu_type=hparams.charcnn_relu,
                  )
                  cnn_output,filter_num = embedding_helper.build_cnn_encoder(encoder_emb_inp, word_encoder)
                  # [batch_size seq_len, embed]
                  encoder_emb_inp = cnn_output
                  encoder_emb_inp = embedding_helper.projection(encoder_emb_inp, filter_num,
                                                                hparams.embed_dim)
                  encoder_emb_inp = tf.reshape(encoder_emb_inp, [_batch_size, _seq_len, hparams.embed_dim])
                  # [max_time, batch_size, embedding]
                  encoder_emb_inp = tf.transpose(encoder_emb_inp, perm=[1, 0, 2])

                  # merge 都只取一半
                  if hparams.src_embed_type[0:3] == 'cnn':
                      # Simply add two embeddings
                      encoder_emb_inp += tf.nn.embedding_lookup(self.embedding_encoder, source)
                  elif hparams.src_embed_type[0:3] == 'rl2':
                      word_level = tf.nn.embedding_lookup(self.embedding_encoder, source)
                      self.loss_items.append(tf.nn.l2_loss(word_level - encoder_emb_inp))
                      encoder_emb_inp += word_level
                  elif hparams.src_embed_type[0:3] == 'rl1':
                      word_level = tf.nn.embedding_lookup(self.embedding_encoder, source)
                      self.loss_items.append(tf.reduce_mean(tf.abs(word_level - encoder_emb_inp) / 2))
                      encoder_emb_inp += word_level
                  else:
                      raise Exception('Unknown hparams.src_embed_type : %s' % hparams.src_embed_type)


      elif 'rnn_segment' in hparams.src_embed_type:


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
              encoder_emb_inp = tf.reshape(encoder_emb_inp,[_batch_size * _seq_len, hparams.seg_len, hparams.seg_embed_dim])
              encoder_emb_inp = tf.transpose(encoder_emb_inp, perm=[1,0,2])
              seg_len_source = tf.reshape(seg_len_source, [_batch_size,_seq_len])
              flattern_sequence_length = tf.reshape(seg_len_source,[-1])
              with tf.variable_scope('word_embedding_encoder'):
                  word_encoder = EncoderParam(encoder_type="uni",
                                                   num_layers=1,
                                                   num_residual_layers=0,
                                                   unit_type="gru", # In order to lightweight
                                                   forget_bias=hparams.forget_bias,
                                                   dropout=hparams.dropout,
                                                   num_gpus=hparams.num_gpus,
                                                   mode=dropout,
                                                   enocder_seq_input=encoder_emb_inp,
                                                   encoder_seq_len=flattern_sequence_length,
                                                   dtype=dtype,
                                                   single_cell_fn=self.single_cell_fn,
                                                   #LSTM 2 Tuple
                                                   num_units=(hparams.seg_embed_dim),
                                                   name=None)

                  if 'attention' not in hparams.src_embed_type:
                      word_encoder_outputs, word_encoder_state = embedding_helper.build_rnn_encoder(word_encoder)
                      #[batch_size * seq_len, embed]
                      encoder_emb_inp = word_encoder_state
                      encoder_emb_inp = embedding_helper.projection(encoder_emb_inp,hparams.seg_embed_dim,hparams.embed_dim)
                      encoder_emb_inp = tf.reshape(encoder_emb_inp, [_batch_size, _seq_len, hparams.embed_dim])
                      #[max_time, batch_size, embedding]
                      encoder_emb_inp = tf.transpose(encoder_emb_inp, perm=[1,0,2])
                  else:
                      print('Attentive Word RNN Encoder')
                      '''
                      encoder_outputs: [max_time, batch_size, cell.output_size].
                      '''
                      if 'attentionv2' in hparams.src_embed_type:
                          with tf.variable_scope("AttentionV2"):
                              print('AttentionV2')
                              char_emb_inp = tf.transpose(encoder_emb_inp, perm=[1, 0, 2])
                              char_emb_inp = tf.reshape(char_emb_inp, [_batch_size * _seq_len, hparams.seg_len,
                                                                             hparams.seg_embed_dim])
                              word_encoder_outputs, _ = embedding_helper.build_rnn_encoder(word_encoder)
                              #  [batch_size * seq_len, max_time, cell.output_size].
                              subunits_embedding = tf.transpose(word_encoder_outputs, [1, 0, 2])
                              memory = tf.concat([subunits_embedding, char_emb_inp], axis=-1, name='memory')
                              encoder_emb_inp = embedding_helper.build_attention2_sum_layer(hparams.seg_embed_dim,
                                                                                            memory,
                                                                                            subunits_embedding,
                                                                                            hparams.seg_len)
                              if 'highway' in hparams.src_embed_type:
                                  print('highway_network')
                                  for i in range(hparams.al_highway_layers):
                                      encoder_emb_inp = embedding_helper.highway(encoder_emb_inp, hparams.seg_embed_dim,
                                                                                 tf.nn.relu, name='highway_%d' % i)

                              encoder_emb_inp = embedding_helper.projection(encoder_emb_inp, hparams.seg_embed_dim,
                                                                            hparams.embed_dim)
                              encoder_emb_inp = tf.reshape(encoder_emb_inp, [_batch_size, _seq_len, hparams.embed_dim])
                              # [max_time, batch_size, embedding]
                              encoder_emb_inp = tf.transpose(encoder_emb_inp, perm=[1, 0, 2])
                      else:
                          with tf.variable_scope("Attention"):
                              word_encoder_outputs, _ = embedding_helper.build_rnn_encoder(word_encoder)
                              #  [batch_size * seq_len, max_time, cell.output_size].
                              subunits_embedding = tf.transpose(word_encoder_outputs,[1,0,2])
                              encoder_emb_inp = embedding_helper.build_attention_sum_layer(hparams.seg_embed_dim, subunits_embedding, hparams.seg_len)
                              encoder_emb_inp = embedding_helper.projection(encoder_emb_inp, hparams.seg_embed_dim,
                                                                            hparams.embed_dim)
                              encoder_emb_inp = tf.reshape(encoder_emb_inp, [_batch_size, _seq_len, hparams.embed_dim])
                              # [max_time, batch_size, embedding]
                              encoder_emb_inp = tf.transpose(encoder_emb_inp, perm=[1, 0, 2])





                  #merge
                  if hparams.src_embed_type[0:3] == 'rnn':
                    # Simply add two embeddings
                    encoder_emb_inp += tf.nn.embedding_lookup(self.embedding_encoder, source)
                  elif hparams.src_embed_type[0:3] == 'rl2':
                      word_level = tf.nn.embedding_lookup(self.embedding_encoder, source)
                      self.loss_items.append(tf.nn.l2_loss(word_level - encoder_emb_inp))
                      encoder_emb_inp += word_level
                  elif hparams.src_embed_type[0:3] == 'rl1':
                      word_level = tf.nn.embedding_lookup(self.embedding_encoder, source)
                      self.loss_items.append(tf.reduce_mean(tf.abs(word_level - encoder_emb_inp) / 2))
                      encoder_emb_inp += word_level
                  elif hparams.src_embed_type[0:3] == 'lfw':
                    encoder_emb_inp = encoder_emb_inp * iterator.unknown_src + tf.nn.embedding_lookup(
                        self.embedding_encoder, source) * (1.0 - iterator.unknown_src)

                  elif hparams.src_embed_type[0:3] == 'gtf':
                      #gtf : Gate function
                    encoder_emb_inp = embedding_helper.simple_3D_concat_gate_function(encoder_emb_inp,
                                                                                   tf.nn.embedding_lookup(self.embedding_encoder, source),
                                                                                   hparams.embed_dim)
                  elif hparams.src_embed_type[0:3] == 'wtf':
                      #gtf : Gate function
                    encoder_emb_inp = embedding_helper.simple_3D_concat_weighted_function(encoder_emb_inp,
                                                                                   tf.nn.embedding_lookup(self.embedding_encoder, source),
                                                                                   hparams.embed_dim)
                  else:
                      raise  Exception('Unknown hparams.src_embed_type : %s' % hparams.src_embed_type)






              # Look up embedding, emp_inp: [max_time, batch_size, num_units]

      else:
          raise Exception('Unknown src_embed_type  %s' % hparams.src_embed_type )


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
                                       num_units= hparams.num_units,
                                       name=None)
    encoder_outputs, encoder_state = embedding_helper.build_rnn_encoder(top_level_encoder)
    return encoder_outputs, encoder_state

  def _build_bidirectional_rnn(self, inputs, sequence_length,
                               dtype, hparams,
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
    fw_cell = self._build_encoder_cell(hparams,
                                       num_bi_layers,
                                       num_bi_residual_layers,
                                       base_gpu=base_gpu)
    bw_cell = self._build_encoder_cell(hparams,
                                       num_bi_layers,
                                       num_bi_residual_layers,
                                       base_gpu=(base_gpu + num_bi_layers))

    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
        fw_cell,
        bw_cell,
        inputs,
        dtype=dtype,
        sequence_length=sequence_length,
        time_major=self.time_major,
        swap_memory=True)

    return tf.concat(bi_outputs, -1), bi_state

  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Build an RNN cell that can be used by decoder."""
    # We only make use of encoder_outputs in attention-based models
    if hparams.attention:
      raise ValueError("BasicModel doesn't support attention.")

    num_layers = hparams.num_layers
    num_residual_layers = hparams.num_residual_layers

    cell = model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=hparams.num_units,
        num_layers=num_layers,
        num_residual_layers=num_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=hparams.num_gpus,
        mode=self.mode,
        single_cell_fn=self.single_cell_fn)

    # For beam search, we need to replicate encoder infos beam_width times
    if self.mode == tf.contrib.learn.ModeKeys.INFER and hparams.beam_width > 0:
      decoder_initial_state = tf.contrib.seq2seq.tile_batch(
          encoder_state, multiplier=hparams.beam_width)
    else:
      decoder_initial_state = encoder_state

    return cell, decoder_initial_state
