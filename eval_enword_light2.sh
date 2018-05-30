export CUDA_VISIBLE_DEVICES=2

python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/enword_light2 --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/enword_light/vocab.15000.separate --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/enword_light/test.message --inference_output_file=infer_test/enword_light2.test.10.txt >> infer_test/log/enword_light2.test.txt

python3 -m nrm.nrm  --infer_beam_width=5 --out_dir=models/enword_light2 --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/enword_light/vocab.15000.separate --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/enword_light/test.message --inference_output_file=infer_test/enword_light2.test.5.txt >> infer_test/log/enword_light2.test.txt

python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/enword_light --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/enword_light/vocab.15000.separate --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/enword_light/test.message --inference_output_file=infer_test/enword_light.test.10.txt >> infer_test/log/enword_light.test.txt

python3 -m nrm.nrm  --infer_beam_width=5 --out_dir=models/enword_light --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/enword_light/vocab.15000.separate --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/enword_light/test.message --inference_output_file=infer_test/enword_light.test.5.txt >> infer_test/log/enword_light.test.txt
    