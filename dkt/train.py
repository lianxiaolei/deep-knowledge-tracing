import time
import datetime
import os

import numpy as np
import tensorflow as tf

from model import DKT
from data_helper import DataGenerator
from config import Config
from base_op import gen_metrics, mean


class DKTTraining(object):

  def __init__(self):
    self.config = Config()
    self.global_step = 0

  def add_gradient_noise(self, grad, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807
    """
    with tf.op_scope([grad, stddev], name, "add_gradient_noise") as name:
      grad = tf.convert_to_tensor(grad, name="grad")
      gn = tf.random_normal(tf.shape(grad), stddev=stddev)
      return tf.add(grad, gn, name=name)

  def train_step(self, params, train_op, train_summary_op, train_summary_writer):
    """
    A single training step
    """
    dkt = self.train_dkt
    sess = self.sess
    global_step = self.global_step

    feed_dict = {dkt.input_data: params['input_x'],
                 dkt.target_id: params['target_id'],
                 dkt.target_correctness: params['target_correctness'],
                 dkt.max_steps: params['max_len'],
                 dkt.sequence_len: params['seq_len'],
                 dkt.keep_prob: self.config.dropout_keep_prob,
                 dkt.batch_size: self.config.batch_size}

    _, step, summaries, loss, binary_pred, pred, target_correctness = sess.run(
      [train_op, global_step, train_summary_op, dkt.loss, dkt.binary_pred, dkt.pred, dkt.target_correctness],
      feed_dict)

    auc, accuracy = gen_metrics(params['seq_len'], binary_pred, pred, target_correctness)

    time_str = datetime.datetime.now().isoformat()
    print("train: {}: step {}, loss {}, acc {}, auc: {}".format(time_str, step, loss, accuracy, auc))
    train_summary_writer.add_summary(summaries, step)

  def dev_step(self, params, dev_summary_op, writer=None):
    """
    Evaluates model on a dev set
    """
    dkt = self.test_dkt
    sess = self.sess
    global_step = self.global_step

    feed_dict = {dkt.input_data: params['input_x'],
                 dkt.target_id: params['target_id'],
                 dkt.target_correctness: params['target_correctness'],
                 dkt.max_steps: params['max_len'],
                 dkt.sequence_len: params['seq_len'],
                 dkt.keep_prob: 1.0,
                 dkt.batch_size: len(params["seq_len"])}
    step, summaries, loss, pred, binary_pred, target_correctness = sess.run(
      [global_step, dev_summary_op, dkt.loss, dkt.pred, dkt.binary_pred, dkt.target_correctness],
      feed_dict)

    auc, accuracy = gen_metrics(params['seq_len'], binary_pred, pred, target_correctness)
    # precision, recall, f_score = precision_recall_fscore_support(target_correctness, binary_pred)

    if writer:
      writer.add_summary(summaries, step)

    return loss, accuracy, auc

  def run_epoch(self, fname):
    """

    Args:
      fname:

    Returns:

    """

    config = Config()

    dg = DataGenerator(fname, config)
    dg.gen_attr()

    train_seqs = dg.train_seqs
    test_seqs = dg.test_seqs

    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False
    )
    sess = tf.Session(config=session_conf)
    self.sess = sess

    with sess.as_default():
      with tf.name_scope("train"):
        with tf.variable_scope("dkt", reuse=None):
          train_dkt = DKT(categories=config.categories,
                          hidden_units=config.hidden_units)

      with tf.name_scope("test"):
        with tf.variable_scope("dkt", reuse=True):
          test_dkt = DKT(categories=config.categories,
                         hidden_units=config.hidden_units)

      self.train_dkt = train_dkt
      self.test_dkt = test_dkt

      global_step = tf.Variable(0, name="global_step", trainable=False)
      self.global_step = global_step

      optimizer = tf.train.AdamOptimizer(config.learning_rate)
      grads_and_vars = optimizer.compute_gradients(train_dkt.loss)

      grads_and_vars = [(tf.clip_by_norm(g, config.max_grad_norm), v)
                        for g, v in grads_and_vars if g is not None]

      train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step, name="train_op")

      grad_summaries = []
      for g, v in grads_and_vars:
        if g is not None:
          grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
          sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
          grad_summaries.append(grad_hist_summary)
          grad_summaries.append(sparsity_summary)
      grad_summaries_merged = tf.summary.merge(grad_summaries)

      timestamp = str(int(time.time()))
      out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
      print("writing to {}".format(out_dir))

      train_loss_summary = tf.summary.scalar("loss", train_dkt.loss)
      train_summary_op = tf.summary.merge([train_loss_summary, grad_summaries_merged])
      train_summary_dir = os.path.join(out_dir, "summaries", "train")
      train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

      test_loss_summary = tf.summary.scalar("loss", test_dkt.loss)
      dev_summary_op = tf.summary.merge([test_loss_summary])
      dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
      dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

      saver = tf.train.Saver(tf.global_variables())

      sess.run(tf.global_variables_initializer())

      batch_size = config.batch_size
      for i in range(config.epochs):
        np.random.shuffle(train_seqs)
        for params in dg.next_batch(train_seqs, batch_size, "train"):
          self.train_step(params, train_op, train_summary_op, train_summary_writer)

          current_step = tf.train.global_step(sess, global_step)
          if current_step % config.evaluate_every == 0:
            print("\nEvaluation:")
            # 获得测试数据

            losses = []
            accuracys = []
            aucs = []
            for params in dg.next_batch(test_seqs, batch_size, "test"):
              loss, accuracy, auc = self.dev_step(params, dev_summary_op, writer=None)
              losses.append(loss)
              accuracys.append(accuracy)
              aucs.append(auc)

            time_str = datetime.datetime.now().isoformat()
            print("dev: {}, step: {}, loss: {}, acc: {}, auc: {}".
                  format(time_str, current_step, mean(losses), mean(accuracys), mean(aucs)))

          if current_step % config.checkpoint_every == 0:
            path = saver.save(sess, "model/my-model", global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))

      builder = tf.saved_model.builder.SavedModelBuilder("./sevenSkillModel")
      inputs = {"input_x": tf.saved_model.utils.build_tensor_info(self.train_dkt.input_data),
                "target_id": tf.saved_model.utils.build_tensor_info(self.train_dkt.target_id),
                "max_steps": tf.saved_model.utils.build_tensor_info(self.train_dkt.max_steps),
                "sequence_len": tf.saved_model.utils.build_tensor_info(self.train_dkt.sequence_len),
                "keep_prob": tf.saved_model.utils.build_tensor_info(self.train_dkt.keep_prob),
                "batch_size": tf.saved_model.utils.build_tensor_info(self.train_dkt.batch_size)}

      outputs = {"pred_all": tf.saved_model.utils.build_tensor_info(self.train_dkt.pred_all)}

      prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs,
                                                                                    outputs=outputs,
                                                                                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
      legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
      builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                           signature_def_map={"predict": prediction_signature},
                                           legacy_init_op=legacy_init_op)

      builder.save()


if __name__ == "__main__":
  fname = "../data/assistments.txt"
  dktt = DKTTraining()
  dktt.run_epoch(fname)
