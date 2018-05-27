python3 -m nrm.nrm      --charcnn_relu=relu      --out_dir=models/cn_rnn_hl      --infer_batch_size=10      --seg_len=5      --num_layers=4      --charcnn_filters_per_windows=200      --test_prefix=/ldev/tensorflow/nmt2/nmt/data/hllevel/test.40000      --attention=luong      --dev_prefix=/ldev/tensorflow/nmt2/nmt/data/hllevel/dev.40000      --seg_embed_dim=160      --batch_size=256      --encoder_type=bi      --charcnn_min_window_size=1      --charcnn_max_window_size=3      --tgt_max_len=30      --charcnn_high_way_layer=4      --src=message      --metrics=rouge@hybrid,bleu-1@hybrid,bleu-2@hybrid,bleu-3@hybrid,bleu-4@hybrid,distinct-1@hybrid,distinct-2@hybrid      --num_units=160      --unit_type=lstm      --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/hllevel/vocab.40000      --src_max_len=30      --tgt=response      --train_prefix=/ldev/tensorflow/nmt2/nmt/data/hllevel/train.40000      --src_embed_type=rnn_segment      --share_vocab=False      --charcnn_high_way_type=uniform      --high_way_layer=4      --num_train_steps=1000000      --steps_per_stats=100      --seg_embed_mode=separate      --embed_dim=512     >> logs/cn_rnn_hl.txt 

python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/cn_rnn_hl --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/hllevel/vocab.40000 --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/hllevel/test.40000.message --inference_output_file=infer_test/cn_rnn_hl.test.txt >> infer_test/log/cn_rnn_hl.test.txt
    
python3 -m nrm.utils.evaluation_utils cn_rnn_hl /ldev/tensorflow/nmt2/nmt/data/hllevel/test.40000.response infer_test/cn_rnn_hl.test.txt infer_test/scores/cn_rnn_hl.test.txt rouge@hybrid,bleu-1@hybrid,bleu-2@hybrid,bleu-3@hybrid,bleu-4@hybrid,distinct-1@hybrid,distinct-2@hybrid
    
