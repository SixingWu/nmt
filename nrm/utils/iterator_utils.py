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
"""For loading data into NMT models."""
from __future__ import print_function

import collections

import tensorflow as tf

__all__ = ["BatchedInput", "get_iterator", "get_infer_iterator"]


# NOTE(ebrevdo): When we subclass this, instances' __dict__ becomes empty.
class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source", "target_input",
                            "target_output", "source_sequence_length",
                            "target_sequence_length"))):
    pass


class SegBatchedInput(
        collections.namedtuple("BatchedInput",
                               ("initializer", "source", "target_input",
                                "target_output", "source_sequence_length",
                                "target_sequence_length","seg_source", "seg_target_input",
                                "seg_target_output","seg_src_lens","seg_tgt_lens",))):
  pass


def get_infer_iterator(src_dataset,
                       src_vocab_table,
                       batch_size,
                       eos,
                       src_max_len=None):
  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)

  if src_max_len:
    src_dataset = src_dataset.map(lambda src: src[:src_max_len])
  # Convert the word strings to ids
  src_dataset = src_dataset.map(
      lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))
  # Add in the word counts.
  src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The entry is the source line rows;
        # this has unknown-length vectors.  The last entry is
        # the source row size; this is a scalar.
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([])),  # src_len
        # Pad the source sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src
            0))  # src_len -- unused

  batched_dataset = batching_func(src_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, src_seq_len) = batched_iter.get_next()
  return BatchedInput(
      initializer=batched_iter.initializer,
      source=src_ids,
      target_input=None,
      target_output=None,
      source_sequence_length=src_seq_len,
      target_sequence_length=None)


def get_iterator(src_dataset,
                 tgt_dataset,
                 src_vocab_table,
                 tgt_vocab_table,
                 batch_size,
                 sos,
                 eos,
                 random_seed,
                 num_buckets,
                 seg_src_dataset=None,
                 seg_tgt_dataset=None,
                 seg_len_src_dataset=None,
                 seg_len_tgt_dataset=None,
                 seg_len=None,
                 seg_inter_separator=None,
                 seg_separator=None,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0):
  if not output_buffer_size:
    output_buffer_size = batch_size * 1000
  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
  tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

  if seg_src_dataset is not None:
      src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset, seg_src_dataset, seg_tgt_dataset, seg_len_src_dataset, seg_len_tgt_dataset))

      src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
      if skip_count is not None:
          src_tgt_dataset = src_tgt_dataset.skip(skip_count)

      src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size, random_seed)

      # First To chars
      src_tgt_dataset = src_tgt_dataset.map(
          lambda src, tgt, seg_src,seg_tgt, len_src, len_tgt: (
              tf.string_split([src]).values, tf.string_split([tgt]).values, tf.string_split([seg_src],seg_separator).values, tf.string_split([seg_tgt],seg_separator).values,tf.string_split([len_src]).values, tf.string_split([len_tgt]).values),
          num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

      # Filter zero length input sequences.
      src_tgt_dataset = src_tgt_dataset.filter(
          lambda src, tgt, seg_src, seg_tgt, len_src, len_tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

      if src_max_len:
          src_tgt_dataset = src_tgt_dataset.map(
              lambda src, tgt, seg_src, seg_tgt, len_src, len_tgt: (src[:src_max_len], tgt,seg_src[:src_max_len],seg_tgt, len_src[:src_max_len], len_tgt),
              num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
      if tgt_max_len:
          src_tgt_dataset = src_tgt_dataset.map(
              lambda src, tgt, seg_src, seg_tgt, len_src, len_tgt: (src, tgt[:tgt_max_len], seg_src, seg_tgt[:tgt_max_len], len_src, len_tgt[:tgt_max_len]),
              num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

      # Convert the seg_word to seg_chars
      def my_func(src):
          new_src = tf.string_split(tf.reshape(src, [-1]), seg_inter_separator).values
          new_src = tf.reshape(new_src, [-1, seg_len])
          return new_src

      src_tgt_dataset = src_tgt_dataset.map(lambda src, tgt, seg_src, seg_tgt, len_src, len_tgt: (src, tgt, my_func(seg_src), my_func(seg_tgt), len_src, len_tgt),
              num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

      # Convert the word strings to ids.  Word strings that are not in the
      # vocab get the lookup table's default_value integer.
      src_tgt_dataset = src_tgt_dataset.map(
          lambda src, tgt, seg_src, seg_tgt, len_src, len_tgt : (tf.cast(src_vocab_table.lookup(src), tf.int32),
                                               tf.cast(tgt_vocab_table.lookup(tgt), tf.int32),
                                               tf.cast(tgt_vocab_table.lookup(seg_src), tf.int32),
                                               tf.cast(tgt_vocab_table.lookup(seg_tgt), tf.int32),
                                               tf.string_to_number(len_src, out_type=tf.int32),
                                               tf.string_to_number(len_tgt, out_type=tf.int32)),
          num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
      # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
      src_tgt_dataset = src_tgt_dataset.map(
          lambda src, tgt, seg_src, seg_tgt, len_src, len_tgt: (src,
                            tf.concat(([tgt_sos_id], tgt), 0),
                            tf.concat((tgt, [tgt_eos_id]), 0),
                            seg_src,
                            # TODO 这里的开始结束符号是否需要修改
                            tf.concat(([[tgt_sos_id] * seg_len], seg_tgt), 0),
                            tf.concat((seg_tgt, [[tgt_eos_id] * seg_len]), 0),
                            tf.concat(([0], len_src), 0),
                            tf.concat((len_tgt, [0]), 0),
                                                                ),
          num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
      # Add in sequence lengths.
      src_tgt_dataset = src_tgt_dataset.map(
          lambda src, tgt_in, tgt_out, seg_src, seg_tgt_in, seg_tgt_out, len_src,len_tgt: (
              src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in), seg_src, seg_tgt_in, seg_tgt_out,len_src,len_tgt),
          num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

      # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
      def batching_func(x):
          return x.padded_batch(
              batch_size,
              # The first three entries are the source and target line rows;
              # these have unknown-length vectors.  The last two entries are
              # the source and target row sizes; these are scalars.
              padded_shapes=(
                  tf.TensorShape([None]),  # src
                  tf.TensorShape([None]),  # tgt_input
                  tf.TensorShape([None]),  # tgt_output
                  tf.TensorShape([]),  # src_len
                  tf.TensorShape([]),  # tgt_len
                  tf.TensorShape([None, seg_len] ),  # seg_src
                  tf.TensorShape([None, seg_len] ),  # seg_tgt_input
                  tf.TensorShape([None, seg_len] ),
                  tf.TensorShape([None]),  # seg_src_len
                  tf.TensorShape([None]),  # seg_tgt_len
              ) , # seg_tgt_output
              # Pad the source and target sequences with eos tokens.
              # (Though notice we don't generally need to do this since
              # later on we will be masking out calculations past the true sequence.
              padding_values=(
                  src_eos_id,  # src
                  tgt_eos_id,  # tgt_input
                  tgt_eos_id,  # tgt_output
                  0,  # src_len -- unused
                  0,
                  src_eos_id,  # seg_src
                  tgt_eos_id,  # seg_tgt_input
                  tgt_eos_id,  # seg_tgt_output
                  0,
                  0,
              ))  # tgt_len -- unused
  else:
      src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

      src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
      if skip_count is not None:
        src_tgt_dataset = src_tgt_dataset.skip(skip_count)

      src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size, random_seed)

      src_tgt_dataset = src_tgt_dataset.map(
          lambda src, tgt: (
              tf.string_split([src]).values, tf.string_split([tgt]).values),
          num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

      # Filter zero length input sequences.
      src_tgt_dataset = src_tgt_dataset.filter(
          lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

      if src_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src[:src_max_len], tgt),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
      if tgt_max_len:
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src, tgt[:tgt_max_len]),
            num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
      # Convert the word strings to ids.  Word strings that are not in the
      # vocab get the lookup table's default_value integer.
      src_tgt_dataset = src_tgt_dataset.map(
          lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                            tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
          num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
      # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
      src_tgt_dataset = src_tgt_dataset.map(
          lambda src, tgt: (src,
                            tf.concat(([tgt_sos_id], tgt), 0),
                            tf.concat((tgt, [tgt_eos_id]), 0)),
          num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
      # Add in sequence lengths.
      src_tgt_dataset = src_tgt_dataset.map(
          lambda src, tgt_in, tgt_out: (
              src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
          num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

      # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
      def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(
                tf.TensorShape([None]),  # src
                tf.TensorShape([None]),  # tgt_input
                tf.TensorShape([None]),  # tgt_output
                tf.TensorShape([]),  # src_len
                tf.TensorShape([])),  # tgt_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(
                src_eos_id,  # src
                tgt_eos_id,  # tgt_input
                tgt_eos_id,  # tgt_output
                0,  # src_len -- unused
                0))  # tgt_len -- unused

  if num_buckets > 1:

    def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
      # Calculate bucket_width by maximum source sequence length.
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 10

      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def seg_key_func(unused_1, unused_2, unused_3, src_len, tgt_len,unused_4, unused_5, unused_6, seg_src_len, seg_tgt_len):
      # Calculate bucket_width by maximum source sequence length.
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 10

      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    if seg_src_dataset is None:
        batched_dataset = src_tgt_dataset.apply(
            tf.contrib.data.group_by_window(
                key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
    else:
        batched_dataset = src_tgt_dataset.apply(
            tf.contrib.data.group_by_window(
                key_func=seg_key_func, reduce_func=reduce_func, window_size=batch_size))

  else:
    batched_dataset = batching_func(src_tgt_dataset)

  if seg_src_dataset is None:
      batched_iter = batched_dataset.make_initializable_iterator()
      (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len,
       tgt_seq_len) = (batched_iter.get_next())
      return BatchedInput(
          initializer=batched_iter.initializer,
          source=src_ids,
          target_input=tgt_input_ids,
          target_output=tgt_output_ids,
          source_sequence_length=src_seq_len,
          target_sequence_length=tgt_seq_len)
  else:
      batched_iter = batched_dataset.make_initializable_iterator()
      (src_ids, tgt_input_ids, tgt_output_ids, src_seq_len,
       tgt_seq_len,seg_src_ids, seg_tgt_input_ids, seg_tgt_output_ids,seg_src_lens,seg_tgt_lens) = (batched_iter.get_next())
      return SegBatchedInput(
          initializer=batched_iter.initializer,
          source=src_ids,
          target_input=tgt_input_ids,
          target_output=tgt_output_ids,
          source_sequence_length=src_seq_len,
          target_sequence_length=tgt_seq_len,
          seg_source=seg_src_ids,
          seg_target_input=seg_tgt_input_ids,
          seg_target_output=seg_tgt_output_ids,
          seg_src_lens=seg_src_lens,
          seg_tgt_lens=seg_tgt_lens,
      )