# coding:utf8

import numpy as np

pred = []
with open('pred.txt', 'r') as fin:
  lines = fin.readlines()
  for line in lines:
    pred.append(line.replace('\n', '').split(' '))

with open('idx_val_dict.txt', 'r') as fin:
  dic = eval(fin.readlines()[0])

kp_name = {}
with open('../data/kp_name.csv', 'r') as fin:
  lines = fin.readlines()
  for line in lines:
    k, v = line.replace('\n', '').split('	')
    kp_name[k] = v

origin = []
questions = []
flag = False
index = -1
with open('target_id.txt', 'r') as fin:
  lines = fin.readlines()
  for line in lines:
    line = line.replace('\n', '').split(' ')
    for idx, item in enumerate(line):
      if idx == 0: continue
      if item == '0' and line[idx - 1] == '0':
        if not flag:
          index = idx - 1
          flag = True
      else:
        index = idx
        flag = False
    origin.append(line[:index])
    questions.append([kp_name[str(dic[int(i)])] for i in line[:index]])
    print([dic[int(i)] for i in line[:index]])

result = {}
for idy, seq in enumerate(questions):
  for idx, kp in enumerate(seq):
    result.setdefault('%s-%s' % (idy, kp), [])
    result['%s-%s' % (idy, kp)].append(float(pred[idy][idx]))
print(result)

from matplotlib.font_manager import _rebuild
_rebuild()
import matplotlib.pyplot as plt

flag = 0
for name, seq in result.items():
  if not str(name.split('-')[0]) == flag:
    print(flag, str(name.split('-')[0]))
    plt.legend()
    plt.show()
    flag = str(name.split('-')[0])
  plt.plot(np.arange(0, len(seq), step=1), np.array(seq), label=name.split('-')[1])
plt.legend()
plt.show()
