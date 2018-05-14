python3 -m nrm.nrm      --steps_per_stats=100      --attention=luong      --metrics=rouge,bleu-1,bleu-2,bleu-3,bleu-4,distinct-1,distinct-2      --test_prefix=/home/mebiuw/nmt/data/enwordlevel/test      --vocab_prefix=/home/mebiuw/nmt/data/enwordlevel/vocab.40000.separate      --unit_type=lstm      --tgt_max_len=30      --share_vocab=False      --num_units=512      --num_layers=2      --dev_prefix=/home/mebiuw/nmt/data/enwordlevel/dev      --infer_batch_size=10      --src_max_len=30      --train_prefix=/home/mebiuw/nmt/data/enwordlevel/train      --embed_dim=512      --out_dir=models/enword_lstm      --batch_size=256      --num_train_steps=1000000      --tgt=response      --encoder_type=bi      --src=message     >> logs/enword_lstm.txt 

python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/enword_lstm --vocab_prefix=/home/mebiuw/nmt/data/enwordlevel/vocab.40000.separate --inference_input_file=/home/mebiuw/nmt/data/enwordlevel/test.message --inference_output_file=infer_test/enword_lstm.test.txt >> infer_test/log/enword_lstm.test.txt
    
python3 -m nrm.utils.evaluation_utils enword_lstm /home/mebiuw/nmt/data/enwordlevel/test.response infer_test/enword_lstm.test.txt infer_test/scores/enword_lstm.test.txt rouge,bleu-1,bleu-2,bleu-3,bleu-4,distinct-1,distinct-2
    
