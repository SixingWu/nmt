python3 -m nrm.nrm      --out_dir=models/cnnencdec      --infer_batch_size=10      --num_layers=4      --test_prefix=/ldev/tensorflow/nmt2/nmt/data/charspace/test      --attention=luong      --dev_prefix=/ldev/tensorflow/nmt2/nmt/data/charspace/dev      --high_way_type=uniform      --batch_size=256      --encoder_type=bi      --cnn_min_window_size=1      --tgt_max_len=50      --src=message      --metrics=rouge@space,bleu-1@space,bleu-2@space,bleu-3@space,bleu-4@space,distinct-1@space,distinct-2@space      --num_units=160      --unit_type=lstm      --cnn_max_window_size=4      --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/charspace/vocab.40000.separate      --src_max_len=50      --tgt=response      --train_prefix=/ldev/tensorflow/nmt2/nmt/data/charspace/train      --share_vocab=False      --attention_architecture=char_standard      --high_way_layer=4      --num_train_steps=1000000      --steps_per_stats=100      --embed_dim=512     >> logs/cnnencdec.txt 

python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/cnnencdec --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/charspace/vocab.40000.separate --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/charspace/test.message --inference_output_file=infer_test/cnnencdec.test.txt >> infer_test/log/cnnencdec.test.txt
    
python3 -m nrm.utils.evaluation_utils cnnencdec /ldev/tensorflow/nmt2/nmt/data/charspace/test.response infer_test/cnnencdec.test.txt infer_test/scores/cnnencdec.test.txt rouge@space,bleu-1@space,bleu-2@space,bleu-3@space,bleu-4@space,distinct-1@space,distinct-2@space
    
