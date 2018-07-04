export CUDA_VISIBLE_DEVICES=0
python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/en_hl_light --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/enhl_light/vocab.15000 --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/enhl_light/test.15000.message --inference_output_file=infer_test/en_hl_light.test.10.txt >> infer_test/log/en_hl_light.test.txt &
python3 -m nrm.nrm  --infer_beam_width=5 --out_dir=models/en_hl_light --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/enhl_light/vocab.15000 --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/enhl_light/test.15000.message --inference_output_file=infer_test/en_hl_light.test.5.txt >> infer_test/log/en_hl_light.test.txt &
    
python3 -m nrm.utils.evaluation_utils en_hl_light /ldev/tensorflow/nmt2/nmt/data/enhl_light/test.15000.response infer_test/en_hl_light.test.10.txt infer_test/scores/en_hl_light.test.10.txt rouge@hybrid,bleu-1@hybrid,bleu-2@hybrid,bleu-3@hybrid,bleu-4@hybrid,distinct-1@hybrid,distinct-2@hybrid
python3 -m nrm.utils.evaluation_utils en_hl_light /ldev/tensorflow/nmt2/nmt/data/enhl_light/test.15000.response infer_test/en_hl_light.test.5.txt infer_test/scores/en_hl_light.test.5.txt rouge@hybrid,bleu-1@hybrid,bleu-2@hybrid,bleu-3@hybrid,bleu-4@hybrid,distinct-1@hybrid,distinct-2@hybrid


