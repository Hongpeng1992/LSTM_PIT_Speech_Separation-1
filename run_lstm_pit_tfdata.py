# run without TFRecord, more efficient.
import argparse
import os
import sys
# import sys.stdout
import time
import numpy as np
import tensorflow as tf
import utils
import utils.tf_tool
import wave
import shutil
import traceback
from dataManager.mixed_aishell_tfrecord_io import get_batch_use_tfdata, generate_tfrecord, rmNormalization
from dataManager import mixed_aishell_tfrecord_io as wav_tool
from tensorflow.python.client import timeline
from FLAGS import NNET_PARAM
from FLAGS import MIXED_AISHELL_PARAM


os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
if NNET_PARAM.USE_MULTIGPU is True:
  from models.lstm_pit_multiGPU import LSTM
else:
  from models.lstm_pit import LSTM


def show_onewave(decode_ans_dir, name, x_spec, x_angle, cleaned1, cleaned2, y_spec1, y_spec2):
  # show the 5 data.(wav,spec,sound etc.)
  x_spec = np.array(rmNormalization(x_spec))
  cleaned1 = np.array(rmNormalization(cleaned1))
  cleaned2 = np.array(rmNormalization(cleaned2))
  # 去噪阈值 #TODO
  # cleaned1 = np.where(cleaned1 > 100, cleaned1, 0)
  # cleaned2 = np.where(cleaned2 > 100, cleaned2, 0)
  y_spec1 = np.array(rmNormalization(y_spec1))
  y_spec2 = np.array(rmNormalization(y_spec2))

  # wav_spec(spectrum)
  cleaned = np.concatenate([cleaned1, cleaned2], axis=-1)
  y_spec = np.concatenate([y_spec1, y_spec2], axis=-1)
  utils.spectrum_tool.picture_spec(np.log10(cleaned+0.001),
                                   decode_ans_dir+'/restore_spec_'+name)
  utils.spectrum_tool.picture_spec(np.log10(x_spec+0.001),
                                   decode_ans_dir+'/mixed_spec_'+name)
  if NNET_PARAM.decode_show_more:
    utils.spectrum_tool.picture_spec(np.log10(y_spec+0.001),
                                     decode_ans_dir+'/raw_spec_'+name)

  cleaned_spec1 = cleaned1 * np.exp(x_angle*1j)
  cleaned_spec2 = cleaned2 * np.exp(x_angle*1j)
  y_spec1 = y_spec1 * np.exp(x_angle*1j)
  y_spec2 = y_spec2 * np.exp(x_angle*1j)
  x_spec = x_spec * np.exp(x_angle*1j)

  # for i in range(speech_num):
  # write restore wave
  reY1 = utils.spectrum_tool.librosa_istft(
      cleaned_spec1.T, (NNET_PARAM.input_size-1)*2, NNET_PARAM.input_size-1)
  reY2 = utils.spectrum_tool.librosa_istft(
      cleaned_spec2.T, (NNET_PARAM.input_size-1)*2, NNET_PARAM.input_size-1)
  # norm resotred wave
  reY1 = reY1/np.max(np.abs(reY1)) * 32767
  reY2 = reY2/np.max(np.abs(reY2)) * 32767
  reCONY = np.concatenate([reY1, reY2])
  wavefile = wave.open(
      decode_ans_dir+'/restore_audio_'+name+'.wav', 'wb')
  nchannels = 1
  sampwidth = 2  # 采样位宽，2表示16位
  framerate = 16000
  nframes = len(reCONY)
  comptype = "NONE"
  compname = "not compressed"
  wavefile.setparams((nchannels, sampwidth, framerate, nframes,
                      comptype, compname))
  wavefile.writeframes(
      np.array(reCONY, dtype=np.int16))

  # write raw wave
  if NNET_PARAM.decode_show_more:
    rawY1 = utils.spectrum_tool.librosa_istft(
        y_spec1.T, (NNET_PARAM.input_size-1)*2, NNET_PARAM.input_size-1)
    rawY2 = utils.spectrum_tool.librosa_istft(
        y_spec2.T, (NNET_PARAM.input_size-1)*2, NNET_PARAM.input_size-1)
    rawCONY = np.concatenate([rawY1, rawY2])
    wavefile = wave.open(
        decode_ans_dir+'/raw_audio_'+name+'.wav', 'wb')
    nframes = len(rawCONY)
    wavefile.setparams((nchannels, sampwidth, framerate, nframes,
                        comptype, compname))
    wavefile.writeframes(
        np.array(rawCONY, dtype=np.int16))

  # write mixed wave
  mixedWave = utils.spectrum_tool.librosa_istft(
      x_spec.T, (NNET_PARAM.input_size-1)*2, NNET_PARAM.input_size-1)
  wavefile = wave.open(
      decode_ans_dir+'/mixed_audio_'+name+'.wav', 'wb')
  nframes = len(mixedWave)
  wavefile.setparams((nchannels, sampwidth, framerate, nframes,
                      comptype, compname))
  wavefile.writeframes(
      np.array(mixedWave, dtype=np.int16))

  # wav_pic(oscillograph)
  utils.spectrum_tool.picture_wave(reCONY,
                                   decode_ans_dir +
                                   '/restore_wav_'+name,
                                   16000)
  if NNET_PARAM.decode_show_more:
    utils.spectrum_tool.picture_wave(rawCONY,
                                     decode_ans_dir +
                                     '/raw_wav_' + name,
                                     16000)


def decode_oneset(setname, set_index_list_dir, ckpt_dir='nnet'):
  dataset_index_file = open(set_index_list_dir, 'r')
  dataset_index_strlist = dataset_index_file.readlines()
  if len(dataset_index_strlist) <= 0:
    print('Set %s have no element.' % setname)
    return
  mixed_wave = []
  x_spec = []
  y_spec1 = []
  y_spec2 = []
  lengths = []
  x_theta = []
  for i, index_str in enumerate(dataset_index_strlist):
    uttdir1, uttdir2 = index_str.replace('\n', '').split(' ')
    # print(uttdir1,uttdir2)
    uttwave1, uttwave2 = wav_tool._get_waveData1_waveData2(uttdir1, uttdir2)
    mixed_wave_t = wav_tool._mix_wav(uttwave1, uttwave2)
    x_spec_t = wav_tool._extract_norm_log_mag_spec(mixed_wave_t)
    y_spec1_t = wav_tool._extract_norm_log_mag_spec(uttwave1)
    y_spec2_t = wav_tool._extract_norm_log_mag_spec(uttwave2)
    x_theta_t = wav_tool._extract_phase(mixed_wave_t)
    mixed_wave.append(mixed_wave_t)
    x_spec.append(x_spec_t)
    y_spec1.append(y_spec1_t)
    y_spec2.append(y_spec2_t)
    x_theta.append(x_theta_t)
    lengths.append(np.shape(x_spec_t)[0])
  mixed_wave = np.array(mixed_wave, dtype=np.float32)
  x_spec = np.array(x_spec, dtype=np.float32)
  y_spec1 = np.array(y_spec1, dtype=np.float32)
  y_spec2 = np.array(y_spec2, dtype=np.float32)
  lengths = np.array(lengths, dtype=np.int32)
  x_theta = np.array(x_theta, dtype=np.float32)

  g = tf.Graph()
  with g.as_default():
    with tf.device('/cpu:0'):
      with tf.name_scope('input'):
        dataset = tf.data.Dataset.from_tensor_slices(
            (x_spec, y_spec1, y_spec2, lengths))
        dataset = dataset.batch(64)
        dataset_iter = dataset.make_one_shot_iterator()
        # dataset_iter = dataset.make_initializable_iterator()
        x_batch_tr, y1_batch_tr, y2_batch_tr, lengths_batch_tr = dataset_iter.get_next()

    with tf.name_scope('model'):
      model = LSTM(x_batch_tr, y1_batch_tr, y2_batch_tr,
                   lengths_batch_tr, infer=True)

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(init)

    ckpt = tf.train.get_checkpoint_state(
        os.path.join(NNET_PARAM.save_dir, ckpt_dir))
    if ckpt and ckpt.model_checkpoint_path:
      tf.logging.info("Restore from " + ckpt.model_checkpoint_path)
      model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      tf.logging.fatal("checkpoint not found.")
      sys.exit(-1)
  g.finalize()

  cleaned1, cleaned2 = sess.run([model.cleaned1, model.cleaned2])
  decode_num = np.shape(x_spec)[0]
  decode_ans_dir = os.path.join(
      NNET_PARAM.save_dir, 'decode_ans', setname)
  if os.path.exists(decode_ans_dir):
    shutil.rmtree(decode_ans_dir)
  os.makedirs(decode_ans_dir)
  for i in range(decode_num):
    show_onewave(decode_ans_dir, str(i), x_spec[i], x_theta[i],
                 cleaned1[i], cleaned2[i], y_spec1[i], y_spec2[i])
  sess.close()
  tf.logging.info("Decoding done.")


def decode():
  set_list = os.listdir('_decode_index')
  for list_file in set_list:
    if list_file[-4:] == 'list':
      # print(list_file)
      decode_oneset(
          list_file[:-5], os.path.join('_decode_index', list_file), ckpt_dir='nnet_C06')


def train_one_epoch(sess, tr_model, i_epoch, run_metadata):
  """Runs the model one epoch on given data."""
  tr_loss, i = 0, 0
  stime = time.time()
  while True:
    try:
      if NNET_PARAM.time_line:
        _, loss, current_batchsize = sess.run(
            # [tr_model.train_op, tr_model.loss, tf.shape(tr_model.lengths)[0]], # memery leak
            [tr_model.train_op, tr_model.loss, tr_model.batch_size],
            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
            run_metadata=run_metadata)
      else:
        _, loss, current_batchsize = sess.run(
            # [tr_model.train_op, tr_model.loss, tf.shape(tr_model.lengths)[0]])
            [tr_model.train_op, tr_model.loss, tr_model.batch_size])
      tr_loss += loss
      if (i+1) % NNET_PARAM.minibatch_size == 0:
        if NNET_PARAM.time_line and NNET_PARAM.timeline_type == 'minibatch':
          tl = timeline.Timeline(run_metadata.step_stats)
          ctf = tl.generate_chrome_trace_format()
          with open('_timeline/%03dtimeline%04d.json' % (i_epoch, i+1), 'w') as f:
            f.write(ctf)
        lr = sess.run(tr_model.lr)
        costtime = time.time()-stime
        stime = time.time()
        print("MINIBATCH %05d: TRAIN AVG.LOSS %04.6f, "
              "(learning rate %02.6f)" % (
                  i + 1, tr_loss / (i*NNET_PARAM.batch_size+current_batchsize), lr), 'DURATION: %06dS' % costtime)
        sys.stdout.flush()
      i += 1
    except tf.errors.OutOfRangeError:
      break
  tr_loss /= ((i-1)*NNET_PARAM.batch_size+current_batchsize)
  return tr_loss


def eval_one_epoch(sess, val_model, run_metadata):
  """Cross validate the model on given data."""
  val_loss = 0
  data_len = 0
  while True:
    try:
      if NNET_PARAM.time_line:
        loss, current_batchsize = sess.run(
            [val_model.loss, val_model.batch_size],
            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
            run_metadata=run_metadata)
      else:
        loss, current_batchsize = sess.run(
            [val_model.loss, val_model.batch_size])
      val_loss += loss
      data_len += current_batchsize
    except tf.errors.OutOfRangeError:
      break
  val_loss /= data_len
  return val_loss


def train():

  g = tf.Graph()
  with g.as_default():
    # region TFRecord+DataSet
    # tf.data with cpu is faster, but padded_batch may not surpport.
    with tf.device('/cpu:0'):
      with tf.name_scope('input'):
        train_tfrecords, val_tfrecords, testcc_tfrecords = generate_tfrecord(
            gen=MIXED_AISHELL_PARAM.GENERATE_TFRECORD)
        if MIXED_AISHELL_PARAM.GENERATE_TFRECORD:
          exit(0)  # set gen=True and exit to generate tfrecords
        x_batch_tr, y1_batch_tr, y2_batch_tr, lengths_batch_tr, iter_train = get_batch_use_tfdata(
            train_tfrecords)
        x_batch_val, y1_batch_val, y2_batch_val, lengths_batch_val, iter_val = get_batch_use_tfdata(
            val_tfrecords)
    # endregion

    # build model
    with tf.name_scope('model'):
      tr_model = LSTM(x_batch_tr,
                      y1_batch_tr,
                      y2_batch_tr,
                      lengths_batch_tr)
      tf.get_variable_scope().reuse_variables()
      val_model = LSTM(x_batch_val,
                       y1_batch_val,
                       y2_batch_val,
                       lengths_batch_val)

    utils.tf_tool.show_all_variables()
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = NNET_PARAM.GPU_RAM_ALLOW_GROWTH
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(init)

    # resume training
    if NNET_PARAM.resume_training.lower() == 'true':
      ckpt = tf.train.get_checkpoint_state(NNET_PARAM.save_dir + '/nnet')
      if ckpt and ckpt.model_checkpoint_path:
        tf.logging.info("restore from" + ckpt.model_checkpoint_path)
        tr_model.saver.restore(sess, ckpt.model_checkpoint_path)
        best_path = ckpt.model_checkpoint_path
      else:
        tf.logging.fatal("checkpoint not found")
      with open(os.path.join(NNET_PARAM.save_dir, 'train.log'), 'a+') as f:
        f.writelines('Training resumed.\n')
    else:
      if os.path.exists(os.path.join(NNET_PARAM.save_dir, 'train.log')):
        os.remove(os.path.join(NNET_PARAM.save_dir, 'train.log'))

    # prepare run_metadata for timeline
    run_metadata = None
    if NNET_PARAM.time_line:
      run_metadata = tf.RunMetadata()
      if os.path.exists('_timeline'):
        shutil.rmtree('_timeline')
      os.mkdir('_timeline')

    # validation before training.
    valstart_time = time.time()
    sess.run(iter_val.initializer)
    loss_prev = eval_one_epoch(sess,
                               val_model,
                               run_metadata)
    tf.logging.info("CROSSVAL PRERUN AVG.LOSS %.4FS  costime %d" %
                    (loss_prev, time.time()-valstart_time))

    tr_model.assign_lr(sess, NNET_PARAM.learning_rate)
    g.finalize()

    # epochs training
    reject_num = 0
    for epoch in range(NNET_PARAM.max_epochs):
      sess.run([iter_train.initializer, iter_val.initializer])
      start_time = time.time()

      # train one epoch
      tr_loss = train_one_epoch(sess,
                                tr_model,
                                epoch,
                                run_metadata)

      # Validation
      val_loss = eval_one_epoch(sess,
                                val_model,
                                run_metadata)

      end_time = time.time()

      # Determine checkpoint path
      ckpt_name = "nnet_iter%d_lrate%e_trloss%.4f_cvloss%.4f_duration%ds" % (
          epoch + 1, NNET_PARAM.learning_rate, tr_loss, val_loss, end_time - start_time)
      ckpt_dir = NNET_PARAM.save_dir + '/nnet'
      if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
      ckpt_path = os.path.join(ckpt_dir, ckpt_name)

      # Relative loss between previous and current val_loss
      rel_impr = np.abs(loss_prev - val_loss) / loss_prev
      # Accept or reject new parameters
      msg = ""
      if val_loss < loss_prev:
        reject_num = 0
        tr_model.saver.save(sess, ckpt_path)
        # Logging train loss along with validation loss
        loss_prev = val_loss
        best_path = ckpt_path
        msg = ("Iteration %03d: TRAIN AVG.LOSS %.4f, lrate%e, VAL AVG.LOSS %.4f,\n"
               "%s, ckpt(%s) saved,\nEPOCH DURATION: %.2fs") % (
            epoch + 1, tr_loss, NNET_PARAM.learning_rate, val_loss,
            "NNET Accepted", ckpt_name, end_time - start_time)
        tf.logging.info(msg)
      else:
        reject_num += 1
        tr_model.saver.restore(sess, best_path)
        msg = ("ITERATION %03d: TRAIN AVG.LOSS %.4f, (lrate%e) VAL AVG.LOSS %.4f,\n"
               "%s, ckpt(%s) abandoned,\nEPOCH DURATION: %.2fs") % (
            epoch + 1, tr_loss, NNET_PARAM.learning_rate, val_loss,
            "NNET Rejected", ckpt_name, end_time - start_time)
        tf.logging.info(msg)
      with open(os.path.join(NNET_PARAM.save_dir, 'train.log'), 'a+') as f:
        f.writelines(msg+'\n')

      # Start halving when improvement is lower than start_halving_impr
      if (rel_impr < NNET_PARAM.start_halving_impr) or (reject_num >= 3):
        reject_num = 0
        NNET_PARAM.learning_rate *= NNET_PARAM.halving_factor
        tr_model.assign_lr(sess, NNET_PARAM.learning_rate)

      # Stopping criterion
      if rel_impr < NNET_PARAM.end_halving_impr:
        if epoch < NNET_PARAM.min_epochs:
          tf.logging.info(
              "we were supposed to finish, but we continue as "
              "min_epochs : %s" % NNET_PARAM.min_epochs)
          continue
        else:
          tf.logging.info(
              "finished, too small rel. improvement %g" % rel_impr)
          break

      # save timeline
      if NNET_PARAM.time_line and NNET_PARAM.timeline_type == 'epoch':
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('_timeline/%03dtimeline.json' % (epoch,), 'w') as f:
          f.write(ctf)

    sess.close()
    tf.logging.info("Done training")


def main(_):
  if not os.path.exists(NNET_PARAM.save_dir):
    os.makedirs(NNET_PARAM.save_dir)
  if NNET_PARAM.decode:
    decode()
  else:
    train()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
