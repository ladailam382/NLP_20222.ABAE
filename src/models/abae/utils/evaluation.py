import argparse
import logging
import numpy as np
from time import time
import src.models.abae.utils.utils as U
from sklearn.metrics import classification_report, f1_score
import codecs
from tensorflow.keras.preprocessing import sequence
import src.models.abae.utils.reader as dataset
from src.models.abae.model import create_model
import keras.backend as K
from src.models.abae.utils.optimizers import get_optimizer
from src.models.abae.utils.preprocess import parseSentence


def evaluation(true, predict, domain):
    true_label = []
    predict_label = []

    if domain == 'restaurant':

        for line in predict:
            predict_label.append(line.strip())

        for line in true:
            true_label.append(line.strip())

        with open('test.txt', 'w') as f:
            f.write(str(predict_label))
            f.write('\n')
            f.write(str(true_label))
            f.close()

        print(classification_report(true_label, predict_label, labels=['Food', 'Staff', 'Ambience']))

    else:
        for line in predict:
            label = line.strip()
            if label == 'smell' or label == 'taste':
              label = 'taste+smell'
            predict_label.append(label)

        for line in true:
            label = line.strip()
            if label == 'smell' or label == 'taste':
              label = 'taste+smell'
            true_label.append(label)

        print(classification_report(true_label, predict_label, 
            ['feel', 'taste+smell', 'look', 'overall', 'None'], digits=3))


def prediction(test_labels, aspect_probs, cluster_map, domain):
    label_ids = np.argsort(aspect_probs, axis=1)[:,-1]
    predict_labels = [cluster_map[label_id] for label_id in label_ids]
    evaluation(open(test_labels), predict_labels, domain)



def max_margin_loss(y_true, y_pred):
    return K.mean(y_pred)


def save_attention_weights(test_x, att_weights, vocab_inv, out_dir, overall_maxlen):
    att_out = codecs.open(out_dir + '/att_weights', 'w', 'utf-8')
    print ('Saving attention weights on test sentences...')
    for c in range(len(test_x)):
        att_out.write('----------------------------------------\n')
        att_out.write(str(c) + '\n')

        word_inds = [i for i in test_x[c] if i!=0]
        line_len = len(word_inds)
        weights = att_weights[c]
        weights = weights[(overall_maxlen-line_len):]

        words = [vocab_inv[i] for i in word_inds]
        att_out.write(' '.join(words) + '\n')
        for j in range(len(words)):
            att_out.write(words[j] + ' '+str(round(weights[j], 3)) + '\n')


def preprocess_text(text, domain):
    tokens = parseSentence(text)

    if len(tokens) == 0:
        print("Input length error")

    else:
        with open(f'data/cache/test.txt', 'w') as f:
            f.write(' '.join(tokens))
            f.close()

