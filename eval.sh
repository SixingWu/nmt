export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --infer_beam_width=1 --out_dir=models/word_4W --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/wordlevel/vocab.40000  --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/wordlevel/dev.40000.message  --inference_output_file=beam_search/word_4W_1_dev_f.inf.response >> beam_search/log/word_4W_1_dev_f.txt
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --infer_beam_width=1 --out_dir=models/word_4W --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/wordlevel/vocab.40000  --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/wordlevel/test.40000.message  --inference_output_file=beam_search/word_4W_1_test_f.inf.response >> beam_search/log/word_4W_1_test_f.txt
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --infer_beam_width=1 --out_dir=models/word_2W --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/wordlevel/vocab.20000  --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/wordlevel/dev.20000.message  --inference_output_file=beam_search/word_2W_1_dev_f.inf.response >> beam_search/log/word_2W_1_dev_f.txt
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --infer_beam_width=1 --out_dir=models/word_2W --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/wordlevel/vocab.20000  --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/wordlevel/test.20000.message  --inference_output_file=beam_search/word_2W_1_test_f.inf.response >> beam_search/log/word_2W_1_test_f.txt
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/word_4W --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/wordlevel/vocab.40000  --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/wordlevel/dev.40000.message  --inference_output_file=beam_search/word_4W_10_dev_f.inf.response >> beam_search/log/word_4W_10_dev_f.txt
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/word_4W --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/wordlevel/vocab.40000  --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/wordlevel/test.40000.message  --inference_output_file=beam_search/word_4W_10_test_f.inf.response >> beam_search/log/word_4W_10_test_f.txt
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/word_2W --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/wordlevel/vocab.20000  --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/wordlevel/dev.20000.message  --inference_output_file=beam_search/word_2W_10_dev_f.inf.response >> beam_search/log/word_2W_10_dev_f.txt
export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/word_2W --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/wordlevel/vocab.20000  --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/wordlevel/test.20000.message  --inference_output_file=beam_search/word_2W_10_test_f.inf.response >> beam_search/log/word_2W_10_test_f.txt