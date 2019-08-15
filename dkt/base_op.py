# coding:utf8
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def mean(x):
  """

  Args:
    x:

  Returns:

  """
  return sum(x) / len(x)


def gen_metrics(sequence_len, binary_pred, pred, target_correctness):
  """

  Args:
    sequence_len:
    binary_pred:
    pred:
    target_correctness:

  Returns:

  """
  # 0, 1 pred
  binary_preds = []
  # 0~1 pred
  preds = []
  target_correctnesses = []
  for seq_idx, seq_len in enumerate(sequence_len):
    binary_preds.append(binary_pred[seq_idx, :seq_len])
    preds.append(pred[seq_idx, :seq_len])
    target_correctnesses.append(target_correctness[seq_idx, :seq_len])

  new_binary_pred = np.concatenate(binary_preds)
  new_pred = np.concatenate(preds)
  new_target_correctness = np.concatenate(target_correctnesses)

  try:
    auc = roc_auc_score(new_target_correctness, new_pred)
    accuracy = accuracy_score(new_target_correctness, new_binary_pred)
  except Exception as e:
    return None, None

  return auc, accuracy
