export CUDA_VISIBLE_DEVICES=1

#to_evaluate=("NRL1V2_charCNN_hybrid_4W" "NRL0_charCNN_hybrid_4W" "NRL1_charCNN_hybrid_4W" "NRL2_charCNN_hybrid_4W" "NRL3_charCNN_hybrid_4W")
to_evaluate=("NRL1V3_charCNN_hybrid_4W" "NRL1V4_charCNN_hybrid_4W")


for item in ${to_evaluate[@]};
do
python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/${item} --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/hybrid2/vocab.40000 --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/hybrid2/test.40000.message --inference_output_file=infer_test/${item}.test.txt >> infer_test/log/${item}.test.txt
python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/${item} --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/hybrid2/vocab.40000 --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/hybrid2/dev.40000.message --inference_output_file=infer_test/${item}.dev.txt >> infer_test/log/${item}.dev.txt
python3 -m nrm.utils.evaluation_utils en_rnn_hl /ldev/tensorflow/nmt2/nmt/data/hybrid2/test.40000.response infer_test/${item}.test.txt infer_test/scores/${item}.test.txt rouge@hybrid,bleu-1@hybrid,bleu-2@hybrid,bleu-3@hybrid,bleu-4@hybrid,distinct-1@hybrid,distinct-2@hybrid
python3 -m nrm.utils.evaluation_utils en_rnn_hl /ldev/tensorflow/nmt2/nmt/data/hybrid2/dev.40000.response infer_test/${item}.dev.txt infer_test/scores/${item}.dev.txt rouge@hybrid,bleu-1@hybrid,bleu-2@hybrid,bleu-3@hybrid,bleu-4@hybrid,distinct-1@hybrid,distinct-2@hybrid
python3 -m nrm.utils.evaluation_utils en_rnn_hl /ldev/tensorflow/nmt2/nmt/data/hybrid2/test.40000.response infer_test/${item}.test.txt infer_test/scores/${item}.char.test.txt rouge@char2char,bleu-1@char2char,bleu-2@char2char,bleu-3@char2char,bleu-4@char2char,distinct-1@char2char,distinct-2@char2char
python3 -m nrm.utils.evaluation_utils en_rnn_hl /ldev/tensorflow/nmt2/nmt/data/hybrid2/dev.40000.response infer_test/${item}.dev.txt infer_test/scores/${item}.char.dev.txt rouge@char2char,bleu-1@char2char,bleu-2@char2char,bleu-3@char2char,bleu-4@char2char,distinct-1@char2char,distinct-2@char2char
done



    