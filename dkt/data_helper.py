# coding:utf8

"""
1 call gen_attr to generate raw dataset.
2 call next_batch to get batch dataset.
3 call format_data to generate input dictionary.
"""

import numpy as np
import random
from pprint import pformat


class DataGenerator(object):
  def __init__(self, fname, num_concepts):
    self.fname = fname
    self.num_concepts = num_concepts

  def read_file(self):
    """
    Read data from given file,
    count the knowledge point and generate assessment dictionary just like:
      {stdent_id:[[kpid, result],[kpid, result]...], ...}
    Args:
      fname: A string value, file name that we'd read.

    Returns:
      A tuple contains assessment dictionary and concepts list.
        return[0] is {stdent_id:[[kpid, result],[kpid, result]...], ...}
        return[1] is [0,1,2,...]
    """
    assessment_seqs = {}
    concepts = []

    with open(self.fname, 'r', encoding='utf8') as fin:
      for line in fin:
        fields = line.strip().split(' ')  # a list like [sid, kid, result]
        student, concept, is_correct = [int(item) for item in fields]

        concepts.append(concept)

        assessment_seqs.setdefault(student, [])
        assessment_seqs[student].append([concept, is_correct])

    return assessment_seqs, list(set(concepts))

  def gen_dict(self, concepts):
    """
    Generate concept-index dict and index-concept dict.
    Args:
      concepts: assessment dictionary like
        {stdent_id:[[kpid, result],[kpid, result]...], ...}

    """
    self.idx_val_dict = {}
    self.val_idx_dict = {}
    for idx, v in enumerate(sorted(concepts)):
      self.idx_val_dict[idx] = v
      self.val_idx_dict[v] = idx

  def split_dataset(self, assessment_seqs, test_size=0.2, random_state=1.):
    """
    Divide the total dataset into train set and test set.
    Notice: The student id will be removed at the return dataset,
              but we can get student id with the return dataset's outer index.
    Args:
      assessment_seqs: A total dataset with type 'dict'.
      test_size: A float value, ratio of the number of test set to the total.
      random_state:

    Returns:
      A tuple contains train dataset and test dataset.
    """
    sorted_keys = sorted(assessment_seqs.keys())

    random.seed(random_state)

    test_keys = set(random.sample(sorted_keys,
                                  int(len(sorted_keys) * test_size)))

    assessment_seqs_test = [assessment_seqs[idx]
                            for idx in assessment_seqs if idx in test_keys]

    assessment_seqs_train = [assessment_seqs[idx]
                             for idx in assessment_seqs if idx not in test_keys]

    return assessment_seqs_train, assessment_seqs_test

  def gen_attr(self, is_inference=False):
    """

    Args:
      is_inference:

    Returns:

    """
    assessment_seqs, concepts = self.read_file()
    if is_inference:
      self.inference_seqs = [v for v in assessment_seqs.values()]
    else:
      self.train_seqs, self.test_seqs = self.split_dataset(assessment_seqs,
                                                           test_size=0.16)
    self.gen_dict(concepts)

  def pad_sequences(self, sequences, max_len=None, constant=0.):
    """
    Pad every assessment sequence to max length of sequences.
    Args:
      sequences: A assessment sequences list.
      max_len: A int value,
      if is None, max_len will be assigned by max length of given sequences.
      value: A numerical value, default is 0.0.

    Returns:
      A 2D ndarray with shape[batch_size, max_seq_lenngth]
    """
    if not max_len:
      max_len = max([len(seq) for seq in sequences])
    num_samples = len(sequences)

    sequence_arr = (np.ones([num_samples, max_len]) * constant).astype(np.int32)

    for idx, seq in enumerate(sequences):
      seq = np.asarray(seq, dtype=np.int32)
      sequence_arr[idx, :len(seq)] = seq
    return sequence_arr

  def num_to_onehot(self, que_corr_location, dim):
    mask = np.zeros(dim)
    if que_corr_location > 0:
      mask[que_corr_location] = 1.
    return mask

  def format_data(self, sequences):
    seq_len = np.array(list(map(lambda seq: len(seq) - 1, sequences)))
    max_len = np.max(seq_len)

    # The sequences like [[[10,0],[20,1],..],[[13,1],[15,0],..],...],
    #   outer list is assessments for students,
    #   medial list is assessment for a student,
    #   inner list is student'answers for questions.
    # ====
    # res[0] + self.num_concepts * res[1] means if the student's answer for a question is true,
    #   the value of the queid+len(concepts) position of the (q,ans) vector is 1.
    # ====
    # We use the seq[0:-1] as input and seq[1:] as target.
    x_sequences = np.array([[(self.val_idx_dict[res[0]] + self.num_concepts * res[1]) for res in seq[:-1]] for seq in sequences])

    x_padding = self.pad_sequences(x_sequences, max_len=max_len, constant=-1)

    x_input = np.array([[self.num_to_onehot(que, dim=self.num_concepts * 2) for que in stu] for stu in x_padding])

    # Generate question id sequence
    question_id_sequences = np.array(
      [[self.val_idx_dict[res[0]] for res in seq[1:]] for seq in sequences])
    question_id_padding = self.pad_sequences(question_id_sequences, max_len=max_len, constant=0)

    question_corr_sequences = np.array([[que[1] for que in seq[1:]] for seq in sequences])
    question_corr_padding = self.pad_sequences(question_corr_sequences, max_len=max_len, constant=0)

    return dict(input_x=x_input, target_id=question_id_padding, target_correctness=question_corr_padding,
                seq_len=seq_len, max_len=max_len)

  def next_batch(self, seqs, batch_size, mode):
    """

    Args:
      seqs:
      batch_size:
      mode:

    Returns:

    """

    length = len(seqs)
    num_batchs = length // batch_size
    if mode == "infer" or mode == "test":
      num_batchs += 1
    start = 0
    for i in range(num_batchs):
      batch_seqs = seqs[start: start + batch_size]
      start += batch_size
      params = self.format_data(batch_seqs)

      yield params


if __name__ == '__main__':
  dg = DataGenerator('../data/assistments.txt', 100)
  assessment_seqs, concepts = dg.read_file()
  print('assessment', pformat(assessment_seqs))
  train, test = dg.split_dataset(assessment_seqs, test_size=0.1)
  print(test)
