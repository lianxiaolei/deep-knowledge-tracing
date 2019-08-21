# coding:utf8
class Config(object):
  # batch_size = 128
  batch_size = 16
  # categories = 267
  categories = 543

  epochs = 128
  learning_rate = 0.01
  evaluate_every = 100
  checkpoint_every = 100
  max_grad_norm = 3.0

  hidden_units = [200]
  dropout_keep_prob = 0.86
