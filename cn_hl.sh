python3 -m nrm.nrm      --embed_dim=512      --charcnn_relu=relu      --num_units=512      --steps_per_stats=100      --unit_type=lstm      --src=message      --out_dir=models/cn_hl      --seg_embed_mode=separate      --charcnn_max_window_size=3      --tgt_max_len=30      --seg_len=5      --high_way_layer=4      --infer_batch_size=10      --num_layers=2      --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/hllevel/vocab.40000      --num_train_steps=1000000      --encoder_type=bi      --charcnn_filters_per_windows=200      --seg_embed_dim=160      --metrics=rouge@hybrid,bleu-1@hybrid,bleu-2@hybrid,bleu-3@hybrid,bleu-4@hybrid,distinct-1@hybrid,distinct-2@hybrid      --charcnn_min_window_size=1      --dev_prefix=/ldev/tensorflow/nmt2/nmt/data/hllevel/dev.40000      --src_max_len=30      --batch_size=256      --share_vocab=False      --src_embed_type=cnn_segment      --test_prefix=/ldev/tensorflow/nmt2/nmt/data/hllevel/test.40000      --attention=luong      --charcnn_high_way_layer=4      --tgt=response      --charcnn_high_way_type=uniform      --train_prefix=/ldev/tensorflow/nmt2/nmt/data/hllevel/train.40000     >> logs/cn_hl.txt 
