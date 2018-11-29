# run without TFRecord, more efficient.
import argparse
import os
import sys
import time
import numpy as np
import tensorflow as tf
from models.lstm_pit import LSTM
import utils
import wave
import shutil
import traceback
from dataManager.mixed_aishell_tfrecord_io import get_batch, generate_tfrecord, rmNormalization
from dataManager import mixed_aishell_tfrecord_io as wav_tool
from FLAGS import NNET_PARAM


os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]


def decode_one(sess, model, name, uttwave1, uttwave2=None):
  if uttwave2 is None:
    uttwave2 = np.array([0]*np.shape(uttwave2)[0])
  mixed_wave = wav_tool._mix_wav(uttwave1, uttwave2)
  x_spec = wav_tool._extract_norm_log_mag_spec(mixed_wave)
  y_spec1 = wav_tool._extract_norm_log_mag_spec(uttwave1)
  y_spec2 = wav_tool._extract_norm_log_mag_spec(uttwave2)

  model.inputs = np.reshape(x_spec, [1, -1, np.shape(x_spec)[-1]])
  cleaned1, cleaned2 = sess.run([model.cleaned1, model.cleaned2])
  cleanedshape = np.shape(cleaned1)
  cleaned1 = np.reshape(cleaned1, [cleanedshape[-2], cleanedshape[-1]])
  cleaned2 = np.reshape(cleaned2, [cleanedshape[-2], cleanedshape[-1]])

  # show the 5 data.(wav,spec,sound etc.)
  x_spec = np.array(rmNormalization(x_spec))
  cleaned1 = np.array(rmNormalization(cleaned1))
  cleaned2 = np.array(rmNormalization(cleaned2))
  y_spec1 = np.array(rmNormalization(y_spec1))
  y_spec2 = np.array(rmNormalization(y_spec2))

  decode_ans_dir = os.path.join(NNET_PARAM.save_dir, 'decode_ans')
  if os.path.exists(decode_ans_dir):
    shutil.rmtree(decode_ans_dir)
  os.makedirs(decode_ans_dir)

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

  x_angle = wav_tool._extract_phase(mixed_wave)
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


def decode():
  with tf.Graph().as_default():
    with tf.name_scope('model'):
      model = LSTM(None, None, None, None, infer=True)

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session()
    sess.run(init)

    ckpt = tf.train.get_checkpoint_state(NNET_PARAM.save_dir+'/nnet')
    if ckpt and ckpt.model_checkpoint_path:
      tf.logging.info("Restore from " + ckpt.model_checkpoint_path)
      model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      tf.logging.fatal("checkpoint not found.")
      sys.exit(-1)

  dataset_index_strlist = []
  for i, index_str in enumerate(dataset_index_strlist):
    uttdir1, uttdir2 = index_str.replace('\n', '').split(' ')
    uttwave1, uttwave2 = wav_tool._get_waveData1_waveData2(uttdir1, uttdir2)
    decode_one(sess, model, str(i), uttwave1, uttwave2)
  sess.close()
  tf.logging.info("Decoding done.")


def train_one_epoch(sess, tr_model):
  """Runs the model one epoch on given data."""
  tr_loss = 0
  i = 0
  while True:
    try:
      stime = time.time()
      _, loss, current_batchsize = sess.run(
          [tr_model.train_op, tr_model.loss, tf.shape(tr_model.lengths)[0]])
      tr_loss += loss
      if (i+1) % int(100*256/NNET_PARAM.batch_size) == 0:
        lr = sess.run(tr_model.lr)
        costtime = time.time()-stime
        print("MINIBATCH %d: TRAIN AVG.LOSS %f, "
              "(learning rate %e)" % (
                  i + 1, tr_loss / (i*NNET_PARAM.batch_size+current_batchsize), lr), 'cost time: %f' % costtime)
        sys.stdout.flush()
      i += 1
    except tf.errors.OutOfRangeError:
      break
  tr_loss /= ((i-1)*NNET_PARAM.batch_size+current_batchsize)
  return tr_loss


def eval_one_epoch(sess, val_model):
  """Cross validate the model on given data."""
  val_loss = 0
  data_len = 0
  while True:
    try:
      loss, current_batchsize = sess.run(
          [val_model.loss, tf.shape(val_model.lengths)[0]])
      val_loss += loss
      data_len += current_batchsize
    except tf.errors.OutOfRangeError:
      break
  val_loss /= data_len
  return val_loss


def train():

  g = tf.Graph()
  with g.as_default():
    with tf.name_scope('model'):
      # region TFRecord+DataSet
      train_tfrecords, val_tfrecords, testcc_tfrecords = generate_tfrecord(
          gen=True)
      exit(0)

      x_batch_tr, y1_batch_tr, y2_batch_tr, lengths_batch_tr, iter_train = get_batch(
          train_tfrecords)
      x_batch_val, y1_batch_val, y2_batch_val, lengths_batch_val, iter_val = get_batch(
          val_tfrecords)
      # endregion
      tr_model = LSTM(x_batch_tr,
                      y1_batch_tr,
                      y2_batch_tr,
                      lengths_batch_tr)
      tf.get_variable_scope().reuse_variables()
      val_model = LSTM(x_batch_val,
                       y1_batch_val,
                       y2_batch_val,
                       lengths_batch_val)

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(init)
    if NNET_PARAM.resume_training.lower() == 'true':
      ckpt = tf.train.get_checkpoint_state(NNET_PARAM.save_dir + '/nnet')
      if ckpt and ckpt.model_checkpoint_path:
        tf.logging.info("restore from" + ckpt.model_checkpoint_path)
        tr_model.saver.restore(sess, ckpt.model_checkpoint_path)
        best_path = ckpt.model_checkpoint_path
      else:
        tf.logging.fatal("checkpoint not found")
    # g.finalize()

    try:
      # validation before training.
      sess.run(iter_val.initializer)
      loss_prev = eval_one_epoch(sess,
                                 val_model)
      tf.logging.info("CROSSVAL PRERUN AVG.LOSS %.4F" % loss_prev)

      sess.run(tf.assign(tr_model.lr, NNET_PARAM.learning_rate))
      for epoch in range(NNET_PARAM.max_epochs):
        sess.run([iter_train.initializer, iter_val.initializer])
        start_time = time.time()

        # Training
        # print('shape')
        # print(sess.run([tf.shape(x_batch_tr), tf.shape(y1_batch_tr),
        #                 tf.shape(y2_batch_val), tf.shape(lengths_batch_tr)]))
        # print('time prepare data :', time.time()-start_time)
        tr_loss = train_one_epoch(sess,
                                  tr_model)
        # exit(0)

        # Validation
        val_loss = eval_one_epoch(sess,
                                  val_model)

        end_time = time.time()
        # Determine checkpoint path
        ckpt_name = "nnet_iter%d_lrate%e_trloss%.4f_cvloss%.4f" % (
            epoch + 1, NNET_PARAM.learning_rate, tr_loss, val_loss)
        ckpt_dir = NNET_PARAM.save_dir + '/nnet'
        if not os.path.exists(ckpt_dir):
          os.makedirs(ckpt_dir)
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        # Relative loss between previous and current val_loss
        rel_impr = np.abs(loss_prev - val_loss) / loss_prev
        # Accept or reject new parameters
        if val_loss < loss_prev:
          tr_model.saver.save(sess, ckpt_path)
          # Logging train loss along with validation loss
          loss_prev = val_loss
          best_path = ckpt_path
          tf.logging.info(
              "ITERATION %03d: TRAIN AVG.LOSS %.4f, (lrate%e) CROSSVAL"
              " AVG.LOSS %.4f, %s (%s), TIME USED: %.2fs" % (
                  epoch + 1, tr_loss, NNET_PARAM.learning_rate, val_loss,
                  "nnet accepted", ckpt_name,
                  (end_time - start_time) / 1))
        else:
          tr_model.saver.restore(sess, best_path)
          tf.logging.info(
              "ITERATION %03d: TRAIN AVG.LOSS %.4f, (lrate%e) CROSSVAL"
              " AVG.LOSS %.4f, %s, (%s), TIME USED: %.2fs" % (
                  epoch + 1, tr_loss, NNET_PARAM.learning_rate, val_loss,
                  "nnet rejected", ckpt_name,
                  (end_time - start_time) / 1))

        # Start halving when improvement is low
        if rel_impr < NNET_PARAM.start_halving_impr:
          NNET_PARAM.learning_rate *= NNET_PARAM.halving_factor
          sess.run(tf.assign(tr_model.lr, NNET_PARAM.learning_rate))

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
    except Exception as e:
      print(e)

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
