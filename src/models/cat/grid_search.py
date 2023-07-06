"""
This module is used to find the best parameters for our problem. 

"""
import json 
import numpy as np 
import pandas as pd

from tqdm import tqdm 
from collections import defaultdict, Counter
from itertools import product

from reach import Reach
from sklearn.metrics import precision_recall_fscore_support

from cat.dataset import citysearch_loader, semeval_loader
from cat.simple import rbf_attention, attention, get_scores

### SETTINGS
w2v_path    = 'embeddings/w2v_restaurant_300_ep_9.vec'
nouns_path  = 'data/nouns_restaurant_300_ep_9.json'

### List of gamma, [0.01, 0.02, ..., 0.1]
GAMMA       = np.arange(0.01, 0.1, 0.01)

### Default dataset SemEval-2014 test phase B 
data        = 'citysearch'      ## citysearch or semeval

if data == 'citysearch': 
    dataset = citysearch_loader()
elif data == 'semeval': 
    dataset = semeval_loader()

### Number of Nouns in use
noun_cands  = np.arange(100, 1000, 50)

func2name   = {
    attention: 'att', 
    rbf_attention: 'rbf', 
}

attentions  = [(-1, attention)]             ## gamma = -1 if use normal attention
attentions.extend(product(GAMMA, [rbf_attention]))


### GRID SEARCH

print('\tLoading word2vec and top nouns ... ')
w2v = Reach.load(w2v_path,
               unk_word="<UNK>")

all_nouns = json.load(open("data/nouns_restaurant.json"))
print("\tFinish loading. Start grid search ... ")

c = Counter()
for k, v in all_nouns.items():
    if k in w2v.items:
        c[k.lower()] += (v)

top_nouns, _ = zip(*c.most_common(1000))
top_nouns = [[x] for x in top_nouns]

pbar = tqdm(total=(len(attentions) * len(noun_cands)))

df = []
for g, att_func in attentions:
    # if att_func == rbf_attention:
    #     w2v.vectors[w2v.items["<UNK>"]] += 10e5
    # else:
    #     w2v.vectors[w2v.items["<UNK>"]] *= 0

    for n_noun in noun_cands:
        nouns = top_nouns[:n_noun]
        for idx, (inst, y, label_set) in enumerate(citysearch_loader()):
            s = get_scores(inst,
                           nouns,
                           w2v,
                           label_set,
                           gamma=g,
                           attention_func=att_func)

            y_pred = s.argmax(1)
            f1_macro = precision_recall_fscore_support(y,
                                                       y_pred,
                                                       average="weighted")[:-1]
            row = [g, func2name[att_func], n_noun, *f1_macro]
            df.append(row)
            # print(row)
            # print(df)
            # break
        pbar.update(1)
df = pd.DataFrame(df, columns=("gamma",
                                "function",
                                "n_noun",
                                "precision",
                                "recall",
                                "f1 macro"))

df.to_csv(f"grid_search/{data}_results_weighted.csv")

