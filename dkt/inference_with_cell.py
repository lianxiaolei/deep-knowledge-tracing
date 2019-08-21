import tensorflow as tf

from data_helper import DataGenerator
from config import Config
from base_op import gen_metrics, mean


def inference(fname, global_step):
  config = Config()

  dg = DataGenerator(fname, config.categories)
  dg.gen_attr(is_inference=True)  # 生成训练集和测试集

  infer_seqs = dg.inference_seqs

  with tf.Session() as sess:
    checkpoint_file = tf.train.latest_checkpoint("model/")
    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
    saver.restore(sess, checkpoint_file)

    graph = tf.get_default_graph()

    tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    for tensor_name in tensor_name_list:
      print(tensor_name)

    accuracys = []
    aucs = []
    step = 1

    batch_size = config.batch_size

    def loop_body(state, input, step):
      """

      Args:
        state: The last step hidden state of encoder RNN.
        trg_ids: A TensorArray var, init array which used by sentence saving.
        step: Time step.

      Returns:

      """
      current_output, next_state = lstm_cell.call(inputs=input, state=state)
      logits = tf.matmul(current_output, output_w) + output_b
      pred = tf.nn.sigmoid(logits)
      input = input.write(step + 1, pred[0])

      return next_state, input, step + 1

    def continue_loop_condition(state, input, step):
      return tf.logical_and(tf.less(step, global_step))

    step = 0
    for params in dg.next_batch(infer_seqs, batch_size, "infer"):
      print("step: {}".format(step))

      # 获得需要喂给模型的参数，输出的结果依赖的输入值
      input_x = graph.get_operation_by_name("test/dkt/input_x").outputs[0]
      target_id = graph.get_operation_by_name("test/dkt/target_id").outputs[0]
      keep_prob = graph.get_operation_by_name("test/dkt/keep_prob").outputs[0]
      max_steps = graph.get_operation_by_name("test/dkt/max_steps").outputs[0]
      sequence_len = graph.get_operation_by_name("test/dkt/sequence_len").outputs[0]
      batch_size = graph.get_operation_by_name("test/dkt/batch_size").outputs[0]

      output_w = graph.get_tensor_by_name('dkt/W:0')
      output_b = graph.get_tensor_by_name('dkt/b:0')

      print(type(tf.nn.sigmoid(output_w)))
      print(graph.get_operation_by_name('dkt/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel') + 1)

      lstm_cell = graph.get_operation_by_name("test/dkt/rnn_cell")

      # Initialize a dynamic tensorArray to cache the generating sentence.
      init_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)

      # Initialize sentence with a start signal.
      # init_array = init_array.write(0, params['input_x'])

      output, state = tf.nn.dynamic_rnn(lstm_cell, input_x, step + 1, dtype=tf.float32)

      # Construct the loop status, contains encoder hidden state, tensorArray, step number.
      init_loop_var = (state, init_array, 0)

      state, input, step = tf.while_loop(continue_loop_condition, loop_body, init_loop_var)


if __name__ == "__main__":
  fname = "../data/xbdata.csv"
  inference(fname, 1)
