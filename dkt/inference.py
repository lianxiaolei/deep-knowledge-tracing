import tensorflow as tf

from data_helper import DataGenerator
from config import Config
from base_op import gen_metrics, mean


def inference(fname):
  config = Config()

  dg = DataGenerator(fname, config.categories)
  dg.gen_attr(is_inference=True, need2gen=False)  # 生成训练集和测试集

  infer_seqs = dg.inference_seqs

  session_conf = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False
  )
  with tf.Session(config=session_conf) as sess:
    checkpoint_file = tf.train.latest_checkpoint("model/")
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)

    graph = tf.get_default_graph()

    accuracys = []
    aucs = []
    # step = 1

    batch_size = config.batch_size

    for params in dg.next_batch(infer_seqs, batch_size, "infer"):
      # print("step: {}".format(step))

      # 获得需要喂给模型的参数，输出的结果依赖的输入值
      input_x = graph.get_operation_by_name("test/dkt/input_x").outputs[0]
      target_id = graph.get_operation_by_name("test/dkt/target_id").outputs[0]
      keep_prob = graph.get_operation_by_name("test/dkt/keep_prob").outputs[0]
      max_steps = graph.get_operation_by_name("test/dkt/max_steps").outputs[0]
      sequence_len = graph.get_operation_by_name("test/dkt/sequence_len").outputs[0]
      batch_size = graph.get_operation_by_name("test/dkt/batch_size").outputs[0]

      # 获得输出的结果
      pred_all = graph.get_tensor_by_name("test/dkt/pred_all:0")
      pred = graph.get_tensor_by_name("test/dkt/pred:0")
      binary_pred = graph.get_tensor_by_name("test/dkt/binary_pred:0")

      target_correctness = params['target_correctness']
      pred_all, pred, binary_pred = sess.run([pred_all, pred, binary_pred],
                                             feed_dict={input_x: params["input_x"],
                                                        target_id: params["target_id"],
                                                        keep_prob: 1.0,
                                                        max_steps: params["max_len"],
                                                        sequence_len: params["seq_len"],
                                                        batch_size: len(params["seq_len"])})

      auc, acc = gen_metrics(params["seq_len"], binary_pred, pred, target_correctness)
      # print('pred_all', pred_all.shape, 'binary_pred', binary_pred.shape)
      # print('target_id', params["target_id"].shape)

      # import numpy as np
      # np.savetxt('pred.txt', pred, fmt='%.2f')
      # np.savetxt('pred_bin.txt', binary_pred, fmt='%d')
      # np.savetxt('target_id.txt', params["target_id"], fmt='%d')
      # np.savetxt('target_correctness.txt', params['target_correctness'], fmt='%d')

      accuracys.append(acc)
      aucs.append(auc)
      # step += 1

    auc_mean = mean(aucs)
    acc_mean = mean(accuracys)

    print("inference  auc: {}  acc: {}".format(auc_mean, acc_mean))


if __name__ == "__main__":
  # fname = "../data/xbdata.csv"
  fname = "../data/khan_test.txt"
  inference(fname)
