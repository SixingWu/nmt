python3 -m nrm.nrm      --num_units=160      --out_dir=models/english_char_lstm      --infer_batch_size=10      --num_layers=4      --test_prefix=/home/mebiuw/nmt/data/encharspace/test      --metrics=rouge@space,bleu-1@space,bleu-2@space,bleu-3@space,bleu-4@space,distinct-1@space,distinct-2@space      --src_max_len=140      --tgt=response      --unit_type=lstm      --train_prefix=/home/mebiuw/nmt/data/encharspace/train      --dev_prefix=/home/mebiuw/nmt/data/encharspace/dev      --batch_size=256      --share_vocab=False      --encoder_type=bi      --num_train_steps=1000000      --attention=luong      --vocab_prefix=/home/mebiuw/nmt/data/encharspace/vocab.40000.separate      --steps_per_stats=100      --tgt_max_len=140      --src=message      --embed_dim=512     >> logs/english_char_lstm.txt 

python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/english_char_lstm --vocab_prefix=/home/mebiuw/nmt/data/encharspace/vocab.40000.separate --inference_input_file=/home/mebiuw/nmt/data/encharspace/test.message --inference_output_file=infer_test/english_char_lstm.test.txt >> infer_test/log/english_char_lstm.test.txt
    
python3 -m nrm.utils.evaluation_utils english_char_lstm /home/mebiuw/nmt/data/encharspace/test.response infer_test/english_char_lstm.test.txt infer_test/scores/english_char_lstm.test.txt rouge@space,bleu-1@space,bleu-2@space,bleu-3@space,bleu-4@space,distinct-1@space,distinct-2@space
    
