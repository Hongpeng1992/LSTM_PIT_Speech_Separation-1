class NNET_PARAM:
  '''
  @decode
  Flag indicating decoding or training.
  Show (mixed_wav, mixed_wav_spec, cleaned_wav, cleaned_wav_pic, cleaned_wav_spec).
  wav_pic is oscillograph.
  wav_spec is spectrum
  '''
  decode = 0
  '''
  @decode_show_more
  Flag indicating show  (label_wav, label_wav_spec, label_wav_pic) or not.
  wav_pic is oscillograph.
  wav_spec is spectrum
  '''
  decode_show_more = 1
  resume_training = 'false' # Flag indicating whether to resume training from cptk.
  input_size = 257  # The dimension of input.
  output_size = 257  # The dimension of output per speaker.
  rnn_size = 496  # Number of rnn units to use.
  rnn_num_layers = 2  # Number of layer of rnn model.
  batch_size = 256
  learning_rate = 0.001  # Initial learning rate.
  min_epochs = 10  # Min number of epochs to run trainer without halving.
  max_epochs = 50  # Max number of epochs to run trainer totally.
  halving_factor = 0.7  # Factor for halving.
  start_halving_impr = 0.003 # Halving when ralative loss is lower than start_halving_impr.
  end_halving_impr = 0.001  # Stop when relative loss is lower than end_halving_impr.
  num_threads_processing_data = 64  # The num of threads to read tfrecords files.
  save_dir = 'exp/lstm_pit'  # Directory to put the train result.
  keep_prob = 0.8  # Keep probability for training dropout.
  max_grad_norm = 5.0  # The max gradient normalization.
  model_type = 'BLSTM'  # BLSTM or LSTM
  GPU_RAM_ALLOW_GROWTH=True

  minibatch_size=400  # batch num to show
  time_line = False # generate timeline file
  timeline_type = 'minibatch'  # timeline write method. 'epoch' ro 'minibatch'


class MIXED_AISHELL_PARAM:
  RAW_DATA = '/home/student/work/pit_test/data'  # rawdata, dirs by speakerid, like "....data_aishell/wav/train".
  DATA_DICT_DIR = '_data/mixed_aishell'
  GENERATE_TFRECORD = False
  PROCESS_NUM_GENERATE_TFERCORD = 64

  '''
  TFRECORDS_DIR='/big-data/tmplhf/pit-data/feature_tfrecords_utt03s'
  TFRECORDS_FILE_TYPE='small' # 'large' or 'small'.if 'small', one file per record.
  LEN_WAWE_PAD_TO = 16000*3 # Mixed wave length (16000*3 is 3 seconds)
  UTT_SEG_FOR_MIX = [260, 290]  # Separate utt to [0:260],[260,290],[290:end]
  DATASET_NAMES = ['train', 'validation', 'test_cc']
  DATASET_SIZES = [1400000, 18000, 180000]
  '''

  TFRECORDS_DIR='/ntfs/tmplhf/pit-data/feature_tfrecords_utt03s_big'
  TFRECORDS_FILE_TYPE='large' # 'large' or 'small'.if 'small', one file per record.
  LEN_WAWE_PAD_TO = 16000*3 # Mixed wave length (16000*3 is 3 seconds)
  UTT_SEG_FOR_MIX = [260, 290]  # Separate utt to [0:260],[260,290],[290:end]
  DATASET_NAMES = ['train', 'validation', 'test_cc']
  DATASET_SIZES = [1400000, 18000, 180000]

  '''
  TFRECORDS_DIR = '/big-data/tmplhf/pit-data/feature_tfrecords_utt10s'
  TFRECORDS_FILE_TYPE='small' # 'large' or 'small'.if 'small', one file per record.
  LEN_WAWE_PAD_TO = 16000*10 # Mixed wave length (16000*3 is 3 seconds)
  UTT_SEG_FOR_MIX = [260, 290]  # Separate utt to [0:260],[260,290],[290:end]
  DATASET_NAMES = ['train', 'validation', 'test_cc']
  DATASET_SIZES = [405600, 5400, 20000]
  '''
  '''
  TFRECORDS_DIR='/ntfs/tmplhf/pit-data/feature_tfrecords_utt10s_big'
  TFRECORDS_FILE_TYPE='large' # 'large' or 'small'.if 'small', one file per record.
  LEN_WAWE_PAD_TO = 16000*10 # Mixed wave length (16000*3 is 3 seconds)
  UTT_SEG_FOR_MIX = [260, 290]  # Separate utt to [0:260],[260,290],[290:end]
  DATASET_NAMES = ['train', 'validation', 'test_cc']
  DATASET_SIZES = [405600, 5400, 20000]
  '''



  # WAVE_NORM=True
  WAVE_NORM = False
  LOG_NORM_MAX = 5
  LOG_NORM_MIN = -3
  NFFT = 512
  OVERLAP = 256
  FS = 16000
