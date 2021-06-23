"""
Evaluate pretrained language model on XNLI, finetuning for 3 epochs.
Takes as input:
- english vocab size
- other lang vocab size
- other lang code (el, ru, zh)
- pretraining epochs (for printing)
- model
- max sequence length
"""

import sys
import numpy as np
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')


epochs = 3
train_df = pd.read_csv('/mounts/data/proj/antmarakis/tasks/xnli/xnli_train_en.csv', names=['text_a', 'text_b', 'labels'], skiprows=1)
dev_df = pd.read_csv('/mounts/data/proj/antmarakis/tasks/xnli/xnli_dev_en.csv', names=['text_a', 'text_b', 'labels'], skiprows=1)
test_df = pd.read_csv('/mounts/data/proj/antmarakis/tasks/xnli/xnli_test_{}.csv'.format(sys.argv[3]), names=['text_a', 'text_b', 'labels'], skiprows=1)

train_df.labels = train_df.labels.astype(int)
dev_df = dev_df[dev_df.labels != '-']
dev_df.labels = dev_df.labels.astype(int)
test_df.labels = test_df.labels.astype(int)

max_len = 128
if len(sys.argv) == 7: max_len = int(sys.argv[6])

def run_test(lm):
    model = ClassificationModel('bert', lm, num_labels=3, args={'overwrite_output_dir': True, 'fp16': False, 'num_train_epochs': epochs, 'save_steps': -1,
                                                                'learning_rate': 2e-5, 'max_seq_length': max_len, 'eval_batch_size': 16, 'train_batch_size': 16,
                                                                })

    model.train_model(train_df)

    result_dev, model_outputs, wrong_predictions = model.eval_model(dev_df, f1=f1_multiclass, acc=accuracy_score)
    result_test, model_outputs, wrong_predictions = model.eval_model(test_df, f1=f1_multiclass, acc=accuracy_score)
    return result_dev, result_test


res_dev, res_test = run_test('/mounts/data/proj/antmarakis/wikipedia/lms/{}'.format(sys.argv[5]))
print(sys.argv[1], sys.argv[2])
print(res_dev)
print(res_test)

f = open('res_xnli_dev_{}_{}_{}_{}_{}.txt'.format(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], epochs), 'a+')
f.write(str(res_dev['acc']))
f.close()

f = open('res_xnli_test_{}_{}_{}_{}_{}.txt'.format(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], epochs), 'a+')
f.write(str(res_test['acc']))
f.close()