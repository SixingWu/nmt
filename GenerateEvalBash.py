
beam_sizes = [1, 5, 10]

model_types = {
    'char_cnn_concat':'models/cnn_separate_nopoolingres_char_level',
    'char_cnn_strides1':'models/cnn_separate_nopooling_char_level',
    'char_cnn_strides2':'models/cnn_separate_original3_beam_char_level',
    'char_cnn_strides3':'models/cnn_separate_original_beam_char_level',
    'char_cnn_strides5_original':'models/cnn_separate_original_beam_char_level',
    'char_cnn_strides3_nobeam':'models/cnn_separate_original_char_level',
    'char_cnn_strides5_max8':'models/cnn_separate_original2_char_level ',
    'char_rnn_separate':'models/separate_char_level',
    'char_rnn_shared':'models/shared_char_level',
    'word_rnn_separate_8w':'models/shared_word_80000_level',
    'word_rnn_shared_4W':'models/separate_word_40000_level',
    'word_rnn_separate_4w':'models/real_separate_word_40000_level',
    'sub1_rnn_separate_4w':'models/separate_sub_80000_level',
    'sub2_rnn_separate_8w':'models/separate_sub2_80000_level',
}

source = {
    'char':'/ldev/tensorflow/nmt/data/charlevel/',
    'word':'/ldev/tensorflow/nmt/data/wordlevel/',
    'sub1':'/ldev/tensorflow/nmt/data/sublevel/',
    'sub2':'/ldev/tensorflow/nmt/data/sublevel2/'
}

for beam_size in beam_sizes:
    for task in model_types.keys():
        for folder in ['','/best_rouge','/best_bleu']:
            for test_file in ['dev','test']:
                model_path = model_types[task] + folder
                input_file = source[task[0:4]] + test_file
                output_file = 'decoded/%s_%d_%s_f%s' % (task,beam_size,test_file,folder.replace('/',''))
                bash = 'python3 -m nrm.nrm '
                bash += ' --beam_width=%d' % beam_size
                bash += ' --out_dir=%s ' % model_path
                bash += ' --inference_input_file=%s.message ' % input_file
                bash += ' --inference_output_file=%s' % output_file
                bash += ' >> decoded/log/%s.txt' % task
            print('export CUDA_VISIBLE_DEVICES=0')
            print(bash)








