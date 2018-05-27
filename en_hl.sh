python3 -m nrm.nrm      --charcnn_relu=relu      --out_dir=models/en_hl      --infer_batch_size=10      --seg_len=10      --num_layers=4      --charcnn_filters_per_windows=100      --test_prefix=/ldev/tensorflow/nmt2/nmt/data/enhllevel/test.40000      --attention=luong      --dev_prefix=/ldev/tensorflow/nmt2/nmt/data/enhllevel/dev.40000      --seg_embed_dim=160      --batch_size=256      --encoder_type=bi      --charcnn_min_window_size=1      --charcnn_max_window_size=8      --flexible_charcnn_windows=1/50-2/100-3/150-4/200-5/200-7/200-8/200      --tgt_max_len=50      --charcnn_high_way_layer=4      --src=message      --metrics=rouge@hybrid      --num_units=160      --unit_type=lstm      --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/enhllevel/vocab.40000      --src_max_len=35      --tgt=response      --train_prefix=/ldev/tensorflow/nmt2/nmt/data/enhllevel/train.40000      --src_embed_type=cnn_segment      --share_vocab=False      --charcnn_high_way_type=uniform      --high_way_layer=4      --num_train_steps=1000000      --steps_per_stats=100      --seg_embed_mode=separate      --embed_dim=512     >> logs/en_hl.txt 

python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/en_hl --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/enhllevel/vocab.40000 --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/enhllevel/test.40000.message --inference_output_file=infer_test/en_hl.test.txt >> infer_test/log/en_hl.test.txt
    
python3 -m nrm.utils.evaluation_utils en_hl /ldev/tensorflow/nmt2/nmt/data/enhllevel/test.40000.response infer_test/en_hl.test.txt infer_test/scores/en_hl.test.txt rouge@hybrid
    
