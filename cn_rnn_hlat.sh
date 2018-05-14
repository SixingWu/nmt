python3 -m nrm.nrm      --steps_per_stats=100      --attention=luong      --metrics=rouge@hybrid,bleu-1@hybrid,bleu-2@hybrid,bleu-3@hybrid,bleu-4@hybrid,distinct-1@hybrid,distinct-2@hybrid      --embed_dim=512      --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/hllevel/vocab.40000      --charcnn_high_way_layer=4      --charcnn_max_window_size=3      --charcnn_high_way_type=uniform      --tgt_max_len=30      --charcnn_filters_per_windows=200      --src_embed_type=rl1_rnn_segment_attention      --unit_type=lstm      --seg_len=5      --src=message      --seg_embed_dim=160      --out_dir=models/cn_rnn_hlat      --infer_batch_size=10      --train_prefix=/ldev/tensorflow/nmt2/nmt/data/hllevel/train.40000      --batch_size=256      --num_train_steps=1000000      --tgt=response      --encoder_type=bi      --test_prefix=/ldev/tensorflow/nmt2/nmt/data/hllevel/test.40000      --seg_embed_mode=separate      --dev_prefix=/ldev/tensorflow/nmt2/nmt/data/hllevel/dev.40000      --charcnn_min_window_size=1      --num_units=512      --num_layers=2      --src_max_len=30      --high_way_layer=4      --charcnn_relu=relu      --share_vocab=False     >> logs/cn_rnn_hlat.txt 

python3 -m nrm.nrm  --infer_beam_width=10 --out_dir=models/cn_rnn_hlat --vocab_prefix=/ldev/tensorflow/nmt2/nmt/data/hllevel/vocab.40000 --inference_input_file=/ldev/tensorflow/nmt2/nmt/data/hllevel/test.40000.message --inference_output_file=infer_test/cn_rnn_hlat.test.txt >> infer_test/log/cn_rnn_hlat.test.txt
    
python3 -m nrm.utils.evaluation_utils cn_rnn_hlat /ldev/tensorflow/nmt2/nmt/data/hllevel/test.40000.response infer_test/cn_rnn_hlat.test.txt infer_test/scores/cn_rnn_hlat.test.txt rouge@hybrid,bleu-1@hybrid,bleu-2@hybrid,bleu-3@hybrid,bleu-4@hybrid,distinct-1@hybrid,distinct-2@hybrid
    
