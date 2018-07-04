python3 -m nrm.utils.evaluation_utils enword_lstm /ldev/tensorflow/nmt2/nmt/data/enwordlevel/test.response infer_test/enword_lstm.test.txt infer_test/scores/enword_lstm.test.txt rouge,bleu-1,bleu-2,bleu-3,bleu-4,distinct-1,distinct-2
python3 -m nrm.utils.evaluation_utils enbpe_lstm /ldev/tensorflow/nmt2/nmt/data/enbpelevel/test.response infer_test/enbpe_lstm.test.txt infer_test/scores/enbpe_lstm.test.txt rouge@bpe,bleu-1@bpe,bleu-2@bpe,bleu-3@bpe,bleu-4@bpe,distinct-1@bpe,distinct-2@bpe
python3 -m nrm.utils.evaluation_utils en_hl /ldev/tensorflow/nmt2/nmt/data/enhllevel/test.40000.response infer_test/en_hl.test.txt infer_test/scores/en_hl.test.txt rouge@hybrid,bleu-1@hybrid,bleu-2@hybrid,bleu-3@hybrid,bleu-4@hybrid,distinct-1@hybrid,distinct-2@hybrid
python3 -m nrm.utils.evaluation_utils en_hl /ldev/tensorflow/nmt2/nmt/data/enhllevel/test.40000.response infer_test/en_hlV2.test.txt infer_test/scores/en_hlV2.test.txt rouge@hybrid,bleu-1@hybrid,bleu-2@hybrid,bleu-3@hybrid,bleu-4@hybrid,distinct-1@hybrid,distinct-2@hybrid

