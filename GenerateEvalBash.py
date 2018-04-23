
beam_sizes = [1, 5, 10]

model_types = {
    'hybrid2W':'models/hybrid_2W',
    'hybrid4W':'models/hybrid_4W',
    'word_2W':'models/word_2W',
    'word_2W':'models/word_4W',
    'char_raw':'char_raw',
}

source = {
    'hybrid2W':'/ldev/tensorflow/nmt2/nmt/data/hybrid2/TYPE1.20000.TYPE2',
    'hybrid4W':'/ldev/tensorflow/nmt2/nmt/data/hybrid2/TYPE1.40000.TYPE2',
    'word_2W':'/ldev/tensorflow/nmt2/nmt/data/wordlevel/TYPE1.20000.TYPE2,
    'word_2W':'/ldev/tensorflow/nmt2/nmt/data/wordlevel/TYPE1.40000.TYPE2,
    'char_raw':'/ldev/tensorflow/nmt2/nmt/data/charlevel/TYPE1.TYPE2',
}

vocab_prefixs={
        'hybrid2W':'/ldev/tensorflow/nmt2/nmt/data/hybrid2/TYPE1.20000.TYPE2',
    'hybrid4W':'/ldev/tensorflow/nmt2/nmt/data/hybrid2/TYPE1.40000.TYPE2',
    'word_2W':'/ldev/tensorflow/nmt2/nmt/data/wordlevel/TYPE1.20000.TYPE2,
    'word_2W':'/ldev/tensorflow/nmt2/nmt/data/wordlevel/TYPE1.40000.TYPE2,
    'char_raw':'/ldev/tensorflow/nmt2/nmt/data/charlevel/TYPE1.TYPE2',
}


for beam_size in beam_sizes:
    for task in model_types.keys():
        for folder in ['']:
            for test_file in ['dev','test']:
                model_path = model_types[task]
                input_file = source[task].replace('TYPE1',test_file).replace('TYPE2','message')
                output_file = 'decoded/%s_%d_%s_f%s' % (task,beam_size,test_file,folder.replace('/',''))
                bash = 'python3 -m nrm.nrm '
                bash += ' --beam_width=%d' % beam_size
                bash += ' --out_dir=%s ' % model_path
                bash += ' --vocab_prefix=%s ' % vocab_prefixs[task]
                bash += ' --inference_input_file=%s.message ' % input_file
                bash += ' --inference_output_file=%s.inf.response' % output_file
                bash += ' >> decoded/log/%s_%d_%s_f%s.txt' % (task,beam_size,test_file,folder.replace('/',''))

                print('export CUDA_VISIBLE_DEVICES=1')
                print(bash)








