import tensorflow as tf
import numpy as np
import librosa
import os
import shutil
import time
import multiprocessing
import copy
import scipy.io
import datetime
import wave
from utils import spectrum_tool
from FLAGS import NNET_PARAM
from FLAGS import MIXED_AISHELL_PARAM

FILE_NAME = __file__[max(__file__.rfind('/')+1, 0):__file__.rfind('.')]
# region define
DATA_DICT_DIR = MIXED_AISHELL_PARAM.DATA_DICT_DIR
RAW_DATA = MIXED_AISHELL_PARAM.RAW_DATA
TFRECORD_DIR = MIXED_AISHELL_PARAM.TFRECORDS_DIR
PROCESS_NUM_GENERATE_TFERCORD = MIXED_AISHELL_PARAM.PROCESS_NUM_GENERATE_TFERCORD
LOG_NORM_MAX = MIXED_AISHELL_PARAM.LOG_NORM_MAX
LOG_NORM_MIN = MIXED_AISHELL_PARAM.LOG_NORM_MIN
NFFT = MIXED_AISHELL_PARAM.NFFT
OVERLAP = MIXED_AISHELL_PARAM.OVERLAP
FS = MIXED_AISHELL_PARAM.FS
LEN_WAWE_PAD_TO = MIXED_AISHELL_PARAM.LEN_WAWE_PAD_TO
UTT_SEG_FOR_MIX = MIXED_AISHELL_PARAM.UTT_SEG_FOR_MIX
DATASET_NAMES = MIXED_AISHELL_PARAM.DATASET_NAMES
DATASET_SIZES = MIXED_AISHELL_PARAM.DATASET_SIZES
WAVE_NORM = MIXED_AISHELL_PARAM.WAVE_NORM
# endregion


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _ini_data(speaker_dir, out_dir):
  data_dict_dir = out_dir
  if os.path.exists(data_dict_dir):
    shutil.rmtree(data_dict_dir)
  os.makedirs(data_dict_dir)
  clean_wav_speaker_set_dir = speaker_dir
  os.makedirs(data_dict_dir+'/train')
  os.makedirs(data_dict_dir+'/validation')
  os.makedirs(data_dict_dir+'/test_cc')
  cwl_train_file = open(data_dict_dir+'/train/clean_wav_dir.list', 'a+')
  cwl_validation_file = open(
      data_dict_dir+'/validation/clean_wav_dir.list', 'a+')
  cwl_test_cc_file = open(data_dict_dir+'/test_cc/clean_wav_dir.list', 'a+')
  clean_wav_list_train = []
  clean_wav_list_validation = []
  clean_wav_list_test_cc = []
  speaker_list = os.listdir(clean_wav_speaker_set_dir)
  speaker_list.sort()
  for speaker_name in speaker_list:
    speaker_dir = clean_wav_speaker_set_dir+'/'+speaker_name
    if os.path.isdir(speaker_dir):
      speaker_wav_list = os.listdir(speaker_dir)
      speaker_wav_list.sort()
      for wav in speaker_wav_list[:UTT_SEG_FOR_MIX[0]]:
        if wav[-4:] == ".wav":
          cwl_train_file.write(speaker_dir+'/'+wav+'\n')
          clean_wav_list_train.append(speaker_dir+'/'+wav)
      for wav in speaker_wav_list[UTT_SEG_FOR_MIX[0]:UTT_SEG_FOR_MIX[1]]:
        if wav[-4:] == ".wav":
          cwl_validation_file.write(speaker_dir+'/'+wav+'\n')
          clean_wav_list_validation.append(speaker_dir+'/'+wav)
      for wav in speaker_wav_list[UTT_SEG_FOR_MIX[1]:]:
        if wav[-4:] == ".wav":
          cwl_test_cc_file.write(speaker_dir+'/'+wav+'\n')
          clean_wav_list_test_cc.append(speaker_dir+'/'+wav)

  cwl_train_file.close()
  cwl_validation_file.close()
  cwl_test_cc_file.close()
  print('train clean: '+str(len(clean_wav_list_train)))
  print('validation clean: '+str(len(clean_wav_list_validation)))
  print('test_cc clean: '+str(len(clean_wav_list_test_cc)))

  dataset_names = DATASET_NAMES
  dataset_mixedutt_num = DATASET_SIZES
  all_mixed = 0
  all_stime = time.time()
  for (clean_wav_list, j) in zip((clean_wav_list_train, clean_wav_list_validation, clean_wav_list_test_cc), range(3)):
    print('\n'+dataset_names[j]+" data preparing...")
    s_time = time.time()
    mixed_wav_list_file = open(
        data_dict_dir+'/'+dataset_names[j]+'/mixed_wav_dir.list', 'a+')
    mixed_wave_list = []
    len_wav_list = len(clean_wav_list)
    generated_num = 0
    while generated_num < dataset_mixedutt_num[j]:
      uttid = np.random.randint(len_wav_list, size=2)
      uttid1 = uttid[0]
      uttid2 = uttid[1]
      utt1_dir = clean_wav_list[uttid1]
      utt2_dir = clean_wav_list[uttid2]
      speaker1 = utt1_dir.split('/')[-2]
      speaker2 = utt2_dir.split('/')[-2]
      if speaker1 == speaker2:
        continue
      generated_num += 1
      mixed_wav_list_file.write(utt1_dir+' '+utt2_dir+'\n')
      mixed_wave_list.append([utt1_dir, utt2_dir])
    # for i_utt in range(len_wav_list): # n^2混合，数据量巨大
    #   for j_utt in range(i_utt,len_wav_list):
    #     utt1_dir=clean_wav_list[i_utt]
    #     utt2_dir=clean_wav_list[j_utt]
    #     speaker1 = utt1_dir.split('/')[-2]
    #     speaker2 = utt2_dir.split('/')[-2]
    #     if speaker1 == speaker2:
    #       continue
    #     mixed_wav_list_file.write(utt1_dir+' '+utt2_dir+'\n')
    #     mixed_wave_list.append([utt1_dir, utt2_dir])
    mixed_wav_list_file.close()
    scipy.io.savemat(
        data_dict_dir+'/'+dataset_names[j]+'/mixed_wav_dir.mat', {"mixed_wav_dir": mixed_wave_list})
    all_mixed += len(mixed_wave_list)
    print(dataset_names[j]+' data preparation over, Mixed num: ' +
          str(len(mixed_wave_list))+(', Cost time %dS.') % (time.time()-s_time))
  print('\nData preparation over, all mixed num: %d,cost time: %dS' %
        (all_mixed, time.time()-all_stime))


def _get_waveData1_waveData2(file1, file2):
  f1 = wave.open(file1, 'rb')
  f2 = wave.open(file2, 'rb')
  waveData1 = np.fromstring(f1.readframes(f1.getnframes()),
                            dtype=np.int16)
  waveData2 = np.fromstring(f2.readframes(f2.getnframes()),
                            dtype=np.int16)
  f1.close()
  f2.close()
  #!!!!! aishell dataset have zero length wave
  if len(waveData1) == 0:
    waveData1 = np.array([0], dtype=np.int16)
  if len(waveData2) == 0:
    waveData2 = np.array([0], dtype=np.int16)
  while len(waveData1) < LEN_WAWE_PAD_TO:
    waveData1 = np.tile(waveData1, 2)
  while len(waveData2) < LEN_WAWE_PAD_TO:
    waveData2 = np.tile(waveData2, 2)

  if WAVE_NORM:
    waveData1 = waveData1/np.max(np.abs(waveData1)) * 32767
    waveData2 = waveData2/np.max(np.abs(waveData2)) * 32767
  # if len(waveData1) < len(waveData2):
  #   waveData1, waveData2 = waveData2, waveData1
  # # print(np.shape(waveData1))
  # gap = len(waveData1)-len(waveData2)
  # waveData2 = np.concatenate(
  #     (waveData2, np.random.randint(-400, 400, size=(gap,))))
  return waveData1[:LEN_WAWE_PAD_TO], waveData2[:LEN_WAWE_PAD_TO]


def _mix_wav(waveData1, waveData2):
  # 混合语音
  mixedData = (waveData1+waveData2)/2
  mixedData = np.array(mixedData, dtype=np.int16)  # 必须指定是16位，因为写入音频时写入的是二进制数据
  return mixedData


def rmNormalization(tmp):
  tmp = (10**(tmp*(LOG_NORM_MAX-LOG_NORM_MIN)+LOG_NORM_MIN))-0.5
  ans = np.where(tmp > 0, tmp, 0)  # 防止计算误差导致的反归一化结果为负数
  return ans


def _extract_norm_log_mag_spec(data):
  # 归一化的幅度谱对数
  mag_spec = spectrum_tool.magnitude_spectrum_librosa_stft(
      data, NFFT, OVERLAP)
  # Normalization
  log_mag_spec = np.log10(mag_spec+0.5)
  log_mag_spec[log_mag_spec > LOG_NORM_MAX] = LOG_NORM_MAX
  log_mag_spec[log_mag_spec < LOG_NORM_MIN] = LOG_NORM_MIN
  log_mag_spec += np.abs(LOG_NORM_MIN)
  log_mag_spec /= (np.abs(LOG_NORM_MIN)+LOG_NORM_MAX)
  # mean=np.mean(log_mag_spec)
  # var=np.var(log_mag_spec)
  # log_mag_spec=(log_mag_spec-mean)/var
  return log_mag_spec


def _extract_phase(data):
  theta = spectrum_tool.phase_spectrum_librosa_stft(data, NFFT, OVERLAP)
  return theta


def _extract_feature_x(utt_dir1, utt_dir2):
  waveData1, waveData2 = _get_waveData1_waveData2(
      utt_dir1, utt_dir2)
  mixedData = _mix_wav(waveData1, waveData2)
  return _extract_norm_log_mag_spec(mixedData)


def _extract_feature_y(utt_dir1, utt_dir2):
  waveData1, waveData2 = _get_waveData1_waveData2(
      utt_dir1, utt_dir2)
  clean1_log_mag_spec = _extract_norm_log_mag_spec(waveData1)
  clean2_log_mag_spec = _extract_norm_log_mag_spec(waveData2)
  return np.concatenate([clean1_log_mag_spec,
                         clean2_log_mag_spec],
                        axis=1)


def _extract_feature_x_y(utt_dir1, utt_dir2):
  waveData1, waveData2 = _get_waveData1_waveData2(
      utt_dir1, utt_dir2)
  mixedData = _mix_wav(waveData1, waveData2)
  clean1_log_mag_spec = _extract_norm_log_mag_spec(waveData1)
  clean2_log_mag_spec = _extract_norm_log_mag_spec(waveData2)
  X = _extract_norm_log_mag_spec(mixedData)
  Y = np.concatenate([clean1_log_mag_spec,
                      clean2_log_mag_spec],
                     axis=1)
  return [X, Y]


def _extract_x_theta(utt_dir1, utt_dir2):
  waveData1, waveData2 = _get_waveData1_waveData2(
      utt_dir1, utt_dir2)
  mixedData = _mix_wav(waveData1, waveData2)
  return _extract_phase(mixedData)


def parse_func(example_proto):
  sequence_features = {
      'inputs': tf.FixedLenSequenceFeature(shape=[NNET_PARAM.input_size],
                                           dtype=tf.float32),
      'labels1': tf.FixedLenSequenceFeature(shape=[NNET_PARAM.output_size],
                                            dtype=tf.float32),
      'labels2': tf.FixedLenSequenceFeature(shape=[NNET_PARAM.output_size],
                                            dtype=tf.float32), }
  _, sequence = tf.parse_single_sequence_example(
      example_proto, sequence_features=sequence_features)
  length = tf.shape(sequence['inputs'])[0]
  return sequence['inputs'], sequence['labels1'], sequence['labels2'], length


def _gen_tfrecord_minprocess_smallfile(dataset_index_list, s_site, e_site, dataset_dir):
  for i in range(s_site, e_site):
    tfrecord_savedir = os.path.join(dataset_dir, ('%08d.tfrecords' % i))
    with tf.python_io.TFRecordWriter(tfrecord_savedir) as writer:
      index_ = dataset_index_list[i]
      X_Y = _extract_feature_x_y(index_[0], index_[1])
      X = np.reshape(np.array(X_Y[0], dtype=np.float32),
                     newshape=[-1, NNET_PARAM.input_size])
      Y = np.reshape(np.array(X_Y[1], dtype=np.float32),
                     newshape=[-1, NNET_PARAM.output_size*2])
      Y1 = Y[:, :NNET_PARAM.output_size]
      Y2 = Y[:, NNET_PARAM.output_size:]
      input_features = [
          tf.train.Feature(float_list=tf.train.FloatList(value=input_))
          for input_ in X]
      label_features1 = [
          tf.train.Feature(float_list=tf.train.FloatList(value=label))
          for label in Y1]
      label_features2 = [
          tf.train.Feature(float_list=tf.train.FloatList(value=label))
          for label in Y2]
      feature_list = {
          'inputs': tf.train.FeatureList(feature=input_features),
          'labels1': tf.train.FeatureList(feature=label_features1),
          'labels2': tf.train.FeatureList(feature=label_features2),
      }
      feature_lists = tf.train.FeatureLists(feature_list=feature_list)
      record = tf.train.SequenceExample(feature_lists=feature_lists)
      writer.write(record.SerializeToString())
    # print(dataset_dir + ('/%08d.tfrecords' % i), 'write done')


def _gen_tfrecord_minprocess_largefile(
        dataset_index_list, s_site, e_site, dataset_dir, i_process):
  tfrecord_savedir = os.path.join(dataset_dir, ('%08d.tfrecords' % i_process))
  with tf.python_io.TFRecordWriter(tfrecord_savedir) as writer:
    for i in range(s_site, e_site):
      index_ = dataset_index_list[i]
      X_Y = _extract_feature_x_y(index_[0], index_[1])
      X = np.reshape(np.array(X_Y[0], dtype=np.float32),
                     newshape=[-1, NNET_PARAM.input_size])
      Y = np.reshape(np.array(X_Y[1], dtype=np.float32),
                     newshape=[-1, NNET_PARAM.output_size*2])
      Y1 = Y[:, :NNET_PARAM.output_size]
      Y2 = Y[:, NNET_PARAM.output_size:]
      input_features = [
          tf.train.Feature(float_list=tf.train.FloatList(value=input_))
          for input_ in X]
      label_features1 = [
          tf.train.Feature(float_list=tf.train.FloatList(value=label))
          for label in Y1]
      label_features2 = [
          tf.train.Feature(float_list=tf.train.FloatList(value=label))
          for label in Y2]
      feature_list = {
          'inputs': tf.train.FeatureList(feature=input_features),
          'labels1': tf.train.FeatureList(feature=label_features1),
          'labels2': tf.train.FeatureList(feature=label_features2),
      }
      feature_lists = tf.train.FeatureLists(feature_list=feature_list)
      record = tf.train.SequenceExample(feature_lists=feature_lists)
      writer.write(record.SerializeToString())
    writer.flush()
    # print(dataset_dir + ('/%08d.tfrecords' % i), 'write done')


def generate_tfrecord(gen=True):
  tfrecords_dir = TFRECORD_DIR
  train_tfrecords_dir = os.path.join(tfrecords_dir, 'train')
  val_tfrecords_dir = os.path.join(tfrecords_dir, 'validation')
  testcc_tfrecords_dir = os.path.join(tfrecords_dir, 'test_cc')
  dataset_dir_list = [train_tfrecords_dir,
                      val_tfrecords_dir, testcc_tfrecords_dir]

  if gen:
    _ini_data(RAW_DATA, DATA_DICT_DIR)
    if os.path.exists(train_tfrecords_dir):
      shutil.rmtree(train_tfrecords_dir)
    if os.path.exists(val_tfrecords_dir):
      shutil.rmtree(val_tfrecords_dir)
    if os.path.exists(testcc_tfrecords_dir):
      shutil.rmtree(testcc_tfrecords_dir)
    os.makedirs(train_tfrecords_dir)
    os.makedirs(val_tfrecords_dir)
    os.makedirs(testcc_tfrecords_dir)

    gen_start_time = time.time()
    for dataset_dir in dataset_dir_list:
      start_time = time.time()
      dataset_index_list = None
      if dataset_dir[-2:] == 'in':
        dataset_index_list = scipy.io.loadmat(
            '_data/mixed_aishell/train/mixed_wav_dir.mat')["mixed_wav_dir"]
      elif dataset_dir[-2:] == 'on':
        dataset_index_list = scipy.io.loadmat(
            '_data/mixed_aishell/validation/mixed_wav_dir.mat')["mixed_wav_dir"]
      elif dataset_dir[-2:] == 'cc':
        dataset_index_list = scipy.io.loadmat(
            '_data/mixed_aishell/test_cc/mixed_wav_dir.mat')["mixed_wav_dir"]

      len_dataset = len(dataset_index_list)
      minprocess_utt_num = int(
          len_dataset/PROCESS_NUM_GENERATE_TFERCORD)
      pool = multiprocessing.Pool(PROCESS_NUM_GENERATE_TFERCORD)
      for i_process in range(PROCESS_NUM_GENERATE_TFERCORD):
        s_site = i_process*minprocess_utt_num
        e_site = s_site+minprocess_utt_num
        if i_process == (PROCESS_NUM_GENERATE_TFERCORD-1):
          e_site = len_dataset
        # print(s_site,e_site)
        if MIXED_AISHELL_PARAM.TFRECORDS_FILE_TYPE == 'small':
          pool.apply_async(_gen_tfrecord_minprocess_smallfile,
                           (dataset_index_list,
                            s_site,
                            e_site,
                            dataset_dir))
        elif MIXED_AISHELL_PARAM.TFRECORDS_FILE_TYPE == 'large':
          pool.apply_async(_gen_tfrecord_minprocess_largefile,
                           (dataset_index_list,
                            s_site,
                            e_site,
                            dataset_dir,
                            i_process))
        # _gen_tfrecord_minprocess(dataset_index_list,
        #                          s_site,
        #                          e_site,
        #                          dataset_dir)
      pool.close()
      pool.join()

      print(dataset_dir+' set extraction over. cost time %06dS' %
            (time.time()-start_time))
    print('Generate TFRecord over. cost time %06dS' %
          (time.time()-gen_start_time))

  train_set = os.path.join(train_tfrecords_dir, '*.tfrecords')
  val_set = os.path.join(val_tfrecords_dir, '*.tfrecords')
  testcc_set = os.path.join(testcc_tfrecords_dir, '*.tfrecords')
  return train_set, val_set, testcc_set


def get_batch_use_tfdata(tfrecords_list):
  files = tf.data.Dataset.list_files(tfrecords_list)
  # dataset = tf.data.TFRecordDataset(files)
  dataset = files.interleave(tf.data.TFRecordDataset,
                             cycle_length=128,
                             #  block_length=128,
                             #  num_parallel_calls=32,
                             )
  # OOM???
  dataset = dataset.map(
      map_func=parse_func,
      num_parallel_calls=64)
  dataset = dataset.padded_batch(
      NNET_PARAM.batch_size,
      padded_shapes=([None, NNET_PARAM.input_size],
                     [None, NNET_PARAM.output_size],
                     [None, NNET_PARAM.output_size],
                     []))
  # dataset = dataset.apply(tf.data.experimental.map_and_batch(
  #     map_func=parse_func,
  #     batch_size=NNET_PARAM.batch_size,
  #     # num_parallel_calls=32,
  #     num_parallel_batches=64,
  # ))
  dataset = dataset.prefetch(buffer_size=NNET_PARAM.batch_size)
  dataset_iter = dataset.make_initializable_iterator()
  x_batch_tr, y1_batch_tr, y2_batch_tr, lengths_batch_tr = dataset_iter.get_next()
  return x_batch_tr, y1_batch_tr, y2_batch_tr, lengths_batch_tr, dataset_iter


def get_batch_use_queue(tfrecords_list):
  num_enqueuing_threads = 64
  file_list = list(os.listdir(tfrecords_list[:-11]))
  file_queue = tf.train.string_input_producer(
      file_list, num_epochs=NNET_PARAM.max_epochs, shuffle=False)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(file_queue)

  sequence_features = {
      'inputs': tf.FixedLenSequenceFeature(shape=[NNET_PARAM.input_size],
                                           dtype=tf.float32),
      'labels1': tf.FixedLenSequenceFeature(shape=[NNET_PARAM.output_size],
                                            dtype=tf.float32),
      'labels2': tf.FixedLenSequenceFeature(shape=[NNET_PARAM.output_size],
                                            dtype=tf.float32), }
  _, sequence = tf.parse_single_sequence_example(
      serialized_example, sequence_features=sequence_features)

  length = tf.shape(sequence['inputs'])[0]

  capacity = 1000 + (num_enqueuing_threads + 1) * NNET_PARAM.batch_size
  queue = tf.PaddingFIFOQueue(
      capacity=capacity,
      dtypes=[tf.float32, tf.float32, tf.float32, tf.int32],
      shapes=[(None, NNET_PARAM.input_size), (None, NNET_PARAM.output_size), (1, 2), ()])

  enqueue_ops = [queue.enqueue([sequence['inputs'],
                                sequence['labels'],
                                sequence['genders'],
                                length])] * num_enqueuing_threads

  tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))
  return queue.dequeue_many(NNET_PARAM.batch_size)
