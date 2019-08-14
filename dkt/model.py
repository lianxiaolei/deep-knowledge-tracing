import tensorflow as tf
import numpy as np


class DKT(object):
  def __init__(self, categories, hidden_units):
    self.hidden_units = hidden_units
    self.categories = categories
    self.input_units = self.categories * 2

    self.max_steps = tf.placeholder(tf.int32, name="max_steps")

    self.input_data = tf.placeholder(tf.float32, [None, None, categories], name="input_x")

    self.sequence_len = tf.placeholder(tf.int32, [None], name="sequence_len")

    self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    self.target_id = tf.placeholder(tf.int32, [None, None], name="target_id")

    self.target_correctness = tf.placeholder(tf.float32, [None, None], name="target_correctness")

    self.batch_size = tf.placeholder(tf.int32, name="batch_size")

  def head_rnn(self):
    """
    Make multi-rnn layer.
    Returns:

    """
    hidden_layers = []
    for idx, hidden_size in enumerate(self.hidden_units):
      lstm_layer = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
      hidden_layer = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_layer,
                                                   output_keep_prob=self.keep_prob)
      hidden_layers.append(hidden_layer)
    self.multi_rnn_cenn = tf.nn.rnn_cell.MultiRNNCell(cells=hidden_layers, state_is_tuple=True)

    outputs, self.current_state = tf.nn.dynamic_rnn(cell=self.multi_rnn_cenn,
                                                    inputs=self.input_data,
                                                    sequence_length=self.sequence_len,
                                                    dtype=tf.float32)
    return outputs

  def tail_affine(self, inputs):
    """
    Make affine layer.
    Args:
      inputs: A tensor

    Returns:

    """
    output_w = tf.get_variable("W", [self.hidden_units[-1], self.categories])
    output_b = tf.get_variable("b", [self.categories])

    self.output = tf.reshape(inputs, [-1, self.hidden_units[-1]])

    self.logits = tf.matmul(self.output, output_w) + output_b

    self.mat_logits = tf.reshape(self.logits, [-1, self.max_steps, self.categories])

    self.pred_all = tf.sigmoid(self.mat_logits, name="pred_all")

    flat_logits = tf.reshape(self.logits, [-1])

    self.flat_target_correctness = tf.reshape(self.target_correctness, [-1])

    # TODO: Define a range array to make concepts skips.
    #         eg. If the nubmer of concept is 10, the range will be [0, 10, 20,...]
    flat_base_target_index = tf.range(self.batch_size * self.max_steps) * self.categories

    flat_base_target_id = tf.reshape(self.target_id, [-1])

    # TODO: Set target_id = base_trg_id + range, why to take suck operation?
    #       Because the logits has been flatten, the origin logits shape is [batch * seq_len, concepts]
    #         and it will be [batch * seq_len * concepts, 1] after flatten.
    #       If we wanna the pred for every question and every student,
    #         we should get value from logits with step concepts.
    flat_target_id = flat_base_target_id + flat_base_target_index
    self.flat_target_logits = tf.gather(flat_logits, flat_target_id)

    # Soft pred
    self.pred = tf.sigmoid(tf.reshape(self.flat_target_logits, [-1, self.max_steps]), name="pred")
    # Hard pred
    self.binary_pred = tf.cast(tf.greater_equal(self.pred, 0.5), tf.float32, name="binary_pred")

  def compile(self):
    """

    Returns:

    """
    # 定义损失函数
    with tf.name_scope("loss"):
      self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=self.flat_target_correctness,
        logits=self.flat_target_logits))

  def build_net(self):
    output = self.head_rnn()
    self.tail_affine(output)
    self.compile()
