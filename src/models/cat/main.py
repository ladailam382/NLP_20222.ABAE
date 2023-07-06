import json
from pprint import pprint

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score, precision_score, recall_score

from cat.simple import get_scores, rbf_attention, attention, get_nouns
from cat.dataset import semeval_loader, citysearch_loader
from collections import defaultdict, Counter
from reach import Reach

### SETTINGS

w2v_path = "embeddings/w2v_restaurant_200_ep_5.vec"
nouns_path = "data/nouns_restaurant_200_ep_5.json"

## Change between `rbf_attention` and `attention`
att = rbf_attention

## Change between `citysearch_loader()` and `semeval_loader()`
dataset = citysearch_loader()

if att == rbf_attention: 
    GAMMA = 0.04
    N_NOUNS = 200
else: 
    GAMMA = -1      # not in use if using normal attention head
    N_NOUNS = 950


print("\tLoading words embedding ...")
w2v = Reach.load(w2v_path,
                unk_word="<UNK>")
print("\tLoading top most frequent nouns ... ")
top_nouns = get_nouns(w2v, nouns_path, n_nouns=N_NOUNS)

for sentence, y, label_set in semeval_loader():

        s = get_scores(instances=sentence, 
                       nouns=top_nouns,
                       r=w2v,
                       labels=label_set,
                       gamma=GAMMA, 
                       attention_func=att,
                       )
        y_pred = s.argmax(1)
        f1 = precision_recall_fscore_support(y, y_pred, average=None)

        f1_macro = precision_recall_fscore_support(y, y_pred, average='macro')
        f1_weighted = precision_recall_fscore_support(y,
                                                    y_pred,
                                                    average="weighted")
        f1_micro = precision_recall_fscore_support(y,
                                                    y_pred,
                                                    average="micro")
        print('f1:')
        pprint(f1)
        print('-----' * 5)

        print('f1_weighted')
        f1 = f1_score(y, y_pred, average='weighted',)
        print(f1)