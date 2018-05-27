python3 -m nrm.nrm      --num_units=160      --out_dir=models/enword_lstm      --infer_batch_size=10      --num_layers=4      --test_prefix=/ldev/tensorflow/nmt2/nmt/data/enwordlevel/test      --metrics=rouge      --src_max_len=30      --tgt=response      --unit_type=lstm      --train_prefix=/ldev/tensorflow/nmt2/nmt/data/enwordlevel/train      --dev_prefix=/ldev/tensorflow/nmt2/nmt/data/enwordlevel/dev      --batch_size=256      --share_vocab=False      --encoder_type=bi      --num_train_steps=1000000      --attention=luong      --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/enwordlevel/vocab.40000.separate      --steps_per_stats=100      --tgt_max_len=30      --src=message      --embed_dim=512     >> logs/enword_lstm.txt 

python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/enword_lstm --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/enwordlevel/vocab.40000.separate --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/enwordlevel/test.message --inference_output_file=infer_test/enword_lstm.test.txt >> infer_test/log/enword_lstm.test.txt
    
python3 -m nrm.utils.evaluation_utils enword_lstm /ldev/tensorflow/nmt2/nmt/data/enwordlevel/test.response infer_test/enword_lstm.test.txt infer_test/scores/enword_lstm.test.txt rouge
    
