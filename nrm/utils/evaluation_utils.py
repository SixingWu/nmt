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

"""Utility for evaluating various tasks, e.g., translation & summarization."""
import codecs
import os
import re
import subprocess
import sys

import tensorflow as tf

from ..scripts import bleu
from ..scripts import rouge


__all__ = ["evaluate"]






def evaluate(ref_file, trans_file, metric, subword_option=None):
  """Pick a metric and evaluate depending on task."""
  # BLEU scores for translation task
  if '@' in metric.lower():
    pos = metric.lower().index('@')
    if subword_option is None:
      subword_option = 'None'
    subword_option += metric[pos:]
    metric = metric[0:pos]

  if metric.lower() == "bleu":
    evaluation_score = _bleu(ref_file, trans_file,
                             subword_option=subword_option)
  elif len(metric.lower()) > 4 and metric.lower()[0:4]=='bleu':
    max_order = int(metric.lower()[5:])
    evaluation_score = _bleu(ref_file, trans_file,max_order=max_order,
                             subword_option=subword_option)
  # ROUGE scores for summarization tasks
  elif metric.lower() == "rouge":
    evaluation_score = _rouge(ref_file, trans_file,
                              subword_option=subword_option)
  elif metric.lower() == "accuracy":
    evaluation_score = _accuracy(ref_file, trans_file,
                             subword_option=subword_option)
  elif metric.lower() == "word_accuracy":
    evaluation_score = _word_accuracy(ref_file, trans_file,
                             subword_option=subword_option)
  elif metric.lower()[0:len('distinct')] == 'distinct':
    max_order = int(metric.lower()[len('distinct')+1:])
    evaluation_score = _distinct(trans_file,max_order,subword_option=subword_option)
  else:
    raise ValueError("Unknown metric %s" % metric)

  return evaluation_score


def _clean(sentence, subword_option):
  """Clean and handle BPE or SPM outputs."""
  sentence = sentence.strip()
  if subword_option is not None and '@' in subword_option:
    subword_option_0 = subword_option.split('@')[0]
    subword_option_1 = subword_option.split('@')[1]
  else:
    subword_option_0 = None
    subword_option_1 = None
  # BPE
  if subword_option_0 == "bpe":
    sentence = re.sub("@@ ", "", sentence)

  # SPM
  elif subword_option_0 == "spm":
    sentence = u"".join(sentence.split()).replace(u"\u2581", u" ").lstrip()

  # speical for chinese
  if subword_option_1 == 'space':
    sentence = sentence.replace(" ", "")
    sentence = sentence.replace("<SPACE>"," ")
  if subword_option_1 == 'char':
    sentence = sentence.replace("<SPACE>", "")
    sentence = sentence.replace("@@", "")
    sentence = sentence.replace(" ","")
    sentence = " ".join(sentence)
  elif subword_option_1 == 'char2char':
    sentence = sentence.replace(" ", "")
    sentence = sentence.replace("@@", "")
    sentence = " ".join(sentence)
  elif subword_option_1 == 'char2word':
    sentence = sentence.replace(" ", "")
    sentence = sentence.replace("@@", " ")
    # sentence = " ".join(sentence)
  elif subword_option_1 == 'hybrid':
    sentence = sentence.replace(" @@ ", "")
    sentence = sentence.replace("@@ ", "")
    sentence = sentence.replace(" @@", "")
  elif subword_option_1 == 'hybrid2':
    sentence = sentence.replace(" ", "")
    sentence = sentence.replace("@@", " ")
  return sentence


def _distinct(trans_file,max_order=1, subword_option=None):
  """Compute Distinct Score"""

  translations = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
    for line in fh:
      line = _clean(line, subword_option=subword_option)
      translations.append(line.split(" "))

  num_tokens = 0
  unique_tokens = set()
  for items in translations:

      #print(items)
      for i in range(0, len(items) - max_order + 1):
        tmp = ' '.join(items[i:i+max_order])
        unique_tokens.add(tmp)
        num_tokens += 1
  ratio = len(unique_tokens) / num_tokens
  return 100 * ratio


# Follow //transconsole/localization/machine_translation/metrics/bleu_calc.py
def _bleu(ref_file, trans_file,max_order=4, subword_option=None):
  """Compute BLEU scores and handling BPE."""
  smooth = False

  ref_files = [ref_file]
  reference_text = []
  for reference_filename in ref_files:
    with codecs.getreader("utf-8")(
        tf.gfile.GFile(reference_filename, "rb")) as fh:
      reference_text.append(fh.readlines())

  per_segment_references = []
  for references in zip(*reference_text):
    reference_list = []
    for reference in references:
      reference = _clean(reference, subword_option)
      reference_list.append(reference.split(" "))
    per_segment_references.append(reference_list)

  #print(per_segment_references[0:15])

  translations = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
    for line in fh:
      line = _clean(line, subword_option=subword_option)
      translations.append(line.split(" "))
  #print(translations[0:15])
  # bleu_score, precisions, bp, ratio, translation_length, reference_length
  bleu_score, _, _, _, _, _ = bleu.compute_bleu(
      per_segment_references, translations, max_order, smooth)
  return 100 * bleu_score


def _rouge(ref_file, summarization_file, subword_option=None):
  """Compute ROUGE scores and handling BPE."""

  references = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(ref_file, "rb")) as fh:
    for line in fh:
      references.append(_clean(line, subword_option))

  hypotheses = []
  with codecs.getreader("utf-8")(
      tf.gfile.GFile(summarization_file, "rb")) as fh:
    for line in fh:
      hypotheses.append(_clean(line, subword_option=subword_option))

  rouge_score_map = rouge.rouge(hypotheses, references)
  return 100 * rouge_score_map["rouge_l/f_score"]


def _accuracy(label_file, pred_file,subword_option=None):
  """Compute accuracy, each line contains a label."""

  with open(label_file, "r", encoding='utf-8') as label_fh:
    with open(pred_file, "r", encoding='utf-8') as pred_fh:
      count = 0.0
      match = 0.0
      for label in label_fh:
        label = label.strip()
        label = " ".join(_clean(label,subword_option))
        pred = pred_fh.readline().strip()
        pred = " ".join(_clean(pred,subword_option))
        if label == pred:
          match += 1
        count += 1
  return 100 * match / count


def _word_accuracy(label_file, pred_file,subword_option=None):
  """Compute accuracy on per word basis."""

  with open(label_file, "r", encoding='utf-8') as label_fh:
    with open(pred_file, "r", encoding='utf-8') as pred_fh:
      total_acc, total_count = 0., 0.
      for sentence in label_fh:
        sentence = " ".join(_clean(sentence, subword_option))
        labels = sentence.strip().split(" ")
        preds = " ".join(_clean(pred_fh.readline(), subword_option))
        preds = preds.strip().split(" ")
        match = 0.0
        for pos in range(min(len(labels), len(preds))):
          label = labels[pos]
          pred = preds[pos]
          if label == pred:
            match += 1
        total_acc += 100 * match / max(len(labels), len(preds))
        total_count += 1
  return total_acc / total_count


def _moses_bleu(multi_bleu_script, tgt_test, trans_file, subword_option=None):
  """Compute BLEU scores using Moses multi-bleu.perl script."""

  # TODO(thangluong): perform rewrite using python
  # BPE
  if subword_option == "bpe":
    debpe_tgt_test = tgt_test + ".debpe"
    if not os.path.exists(debpe_tgt_test):
      # TODO(thangluong): not use shell=True, can be a security hazard
      subprocess.call("cp %s %s" % (tgt_test, debpe_tgt_test), shell=True)
      subprocess.call("sed s/@@ //g %s" % (debpe_tgt_test),
                      shell=True)
    tgt_test = debpe_tgt_test
  elif subword_option == "spm":
    despm_tgt_test = tgt_test + ".despm"
    if not os.path.exists(despm_tgt_test):
      subprocess.call("cp %s %s" % (tgt_test, despm_tgt_test))
      subprocess.call("sed s/ //g %s" % (despm_tgt_test))
      subprocess.call(u"sed s/^\u2581/g %s" % (despm_tgt_test))
      subprocess.call(u"sed s/\u2581/ /g %s" % (despm_tgt_test))
    tgt_test = despm_tgt_test
  cmd = "%s %s < %s" % (multi_bleu_script, tgt_test, trans_file)

  # subprocess
  # TODO(thangluong): not use shell=True, can be a security hazard
  bleu_output = subprocess.check_output(cmd, shell=True)

  # extract BLEU score
  m = re.search("BLEU = (.+?),", bleu_output)
  bleu_score = float(m.group(1))

  return bleu_score

if __name__ == "__main__":
  model_id = sys.argv[1]
  ref_file = sys.argv[2] # r"D:\nmt\ref\dev.20000.response"
  #ref_file = r"D:\nmt\ref\char2_dev.response"
  trans_file = sys.argv[3]  # r"D:\nmt\beam_search\word_4W_10_dev_f.inf.response"
  out_path = sys.argv[4]
  metrics = sys.argv[5].split(',')
  subword = None

  with open(out_path,'w+',encoding='utf-8') as fout:
    for metric in metrics:
        score = evaluate(ref_file, trans_file, metric, subword_option=subword)
        fout.write(('%s\t%f\n') % (metric, score))
  # print('res file: %s' % ref_file)
  # print('trans_file:%s' % trans_file)
  # scores = []
  # for metric in metrics:
  #   score = evaluate(ref_file,trans_file,metric+'@hybrid',subword_option=subword)
  #   scores.append(str(score))
  # print('\t'.join(scores))
  # scores = []
  # for metric in ['rouge', 'bleu-1', 'bleu-2', 'bleu-3', 'bleu-4']:
  #   score = evaluate(ref_file, trans_file, metric + '@char', subword_option=subword)
  #   scores.append(str(score))
  # print('\t'.join(scores))