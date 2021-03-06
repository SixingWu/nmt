export CUDA_VISIBLE_DEVICES=3
python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/enal_light --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/enal_light/vocab.15000 --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/enal_light/test.15000.message --inference_output_file=infer_test/enal_light.test.10.txt >> infer_test/log/enal_light.test.txt

python3 -m nrm.nrm  --infer_beam_width=5 --out_dir=models/enal_light --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/enal_light/vocab.15000 --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/enal_light/test.15000.message --inference_output_file=infer_test/enal_light.test.5.txt >> infer_test/log/enal_light.test.txt

python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/enal_light2 --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/enal_light/vocab.15000 --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/enal_light/test.15000.message --inference_output_file=infer_test/enal_light2.test.10.txt >> infer_test/log/enal_light.test.txt

python3 -m nrm.nrm  --infer_beam_width=5 --out_dir=models/enal_light2 --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/enal_light/vocab.15000 --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/enal_light/test.15000.message --inference_output_file=infer_test/enal_light2.test.5.txt >> infer_test/log/enal_light.test.txt
    
    