export CUDA_VISIBLE_DEVICES=2
python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/NRL2_charCNN_hybrid_4W --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/hybrid2/vocab.40000 --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/hybrid2/test.40000.message --inference_output_file=infer_test/NRL2.test.txt >> infer_test/log/NRL2.test.txt
  



python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/NRL2_charCNN_hybrid_4W --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/hybrid2/vocab.40000 --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/hybrid2/dev.40000.message --inference_output_file=infer_test/NRL2.dev.txt >> infer_test/log/NRL2.dev.txt

python3 -m nrm.utils.evaluation_utils en_rnn_hl /ldev/tensorflow/nmt2/nmt/data/hybrid2/test.40000.response infer_test/NRL2.test.txt infer_test/scores/NRL2.test.txt rouge@hybrid,bleu-1@hybrid,bleu-2@hybrid,bleu-3@hybrid,bleu-4@hybrid,distinct-1@hybrid,distinct-2@hybrid
  
python3 -m nrm.utils.evaluation_utils en_rnn_hl /ldev/tensorflow/nmt2/nmt/data/hybrid2/dev.40000.response infer_test/NRL2.dev.txt infer_test/scores/NRL2.dev.txt rouge@hybrid,bleu-1@hybrid,bleu-2@hybrid,bleu-3@hybrid,bleu-4@hybrid,distinct-1@hybrid,distinct-2@hybrid



cat infer_test/scores/NRL2.test.txt
cat infer_test/scores/NRL2.dev.txt


python3 -m nrm.utils.evaluation_utils en_rnn_hl /ldev/tensorflow/nmt2/nmt/data/hybrid2/test.40000.response infer_test/NRL2.test.txt infer_test/scores/NRL2C.test.txt rouge@char2char,bleu-1@char2char,bleu-2@char2char,bleu-3@char2char,bleu-4@char2char,distinct-1@char2char,distinct-2@char2char
python3 -m nrm.utils.evaluation_utils en_rnn_hl /ldev/tensorflow/nmt2/nmt/data/hybrid2/dev.40000.response infer_test/NRL2.dev.txt infer_test/scores/NRL2C.dev.txt rouge@char2char,bleu-1@char2char,bleu-2@char2char,bleu-3@char2char,bleu-4@char2char,distinct-1@char2char,distinct-2@char2char

cat infer_test/scores/NRL2C.test.txt
cat infer_test/scores/NRL2C.dev.txt
    