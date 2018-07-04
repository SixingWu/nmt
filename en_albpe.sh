python3 -m nrm.nrm      --charcnn_relu=relu      --out_dir=models/en_albpe      --infer_batch_size=10      --seg_len=8      --num_layers=2      --charcnn_filters_per_windows=100      --test_prefix=/ldev/tensorflow/nmt2/nmt/data/enallevel/test.40000      --min_steps=0      --attention=luong      --dev_prefix=/ldev/tensorflow/nmt2/nmt/data/enallevel/dev.40000      --seg_embed_dim=160      --batch_size=64      --encoder_type=bi      --charcnn_min_window_size=1      --charcnn_max_window_size=6      --flexible_charcnn_windows=1/30-2/60-3/100-4/100-5/100-6/100      --tgt_max_len=45      --charcnn_high_way_layer=4      --src=message      --metrics=rouge@hybrid,bleu-1@hybrid,bleu-2@hybrid,bleu-3@hybrid,bleu-4@hybrid,distinct-1@hybrid,distinct-2@hybrid      --num_units=240      --unit_type=lstm      --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/enallevel/vocab.40000      --src_max_len=20      --tgt=response      --train_prefix=/ldev/tensorflow/nmt2/nmt/data/enallevel/train.40000      --src_embed_type=rl1_rnn_segment_attention      --share_vocab=False      --charcnn_high_way_type=uniform      --high_way_layer=4      --num_train_steps=1000000      --steps_per_stats=100      --seg_embed_mode=separate      --embed_dim=320     >> logs/en_albpe.txt 

python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/en_albpe --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/enallevel/vocab.40000 --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/enallevel/test.40000.message --inference_output_file=infer_test/en_albpe.test.txt >> infer_test/log/en_albpe.test.txt
    
python3 -m nrm.utils.evaluation_utils en_albpe /ldev/tensorflow/nmt2/nmt/data/enallevel/test.40000.response infer_test/en_albpe.test.txt infer_test/scores/en_albpe.test.txt rouge@hybrid,bleu-1@hybrid,bleu-2@hybrid,bleu-3@hybrid,bleu-4@hybrid,distinct-1@hybrid,distinct-2@hybrid
    
