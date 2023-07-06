"""
To create word2vec.vec with training dataset of CitySearch (tokenized)
"""

import json
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from gensim.models import Word2Vec
from collections import defaultdict, Counter
from reach import Reach
from tqdm import tqdm 

vector_size = 200
epochs = 5
file_path = '../../../data/raw/train.txt'
w2v_path = f'embeddings/w2v_restaurant_{vector_size}_ep_{epochs}.vec'
nouns_path = f'data/nouns_restaurant_{vector_size}_ep_{epochs}.json'

def generate_nouns(file_path, 
                   word2vec="embeddings/restaurant_vecs_w2v.vec", 
                   out_path='data/nouns_restaurant.json'
                   ): 
    """
    This function is used to generate all nouns from the training file

    Parameter: 
    ------
    file_path   : <str>, path to the training file (tokenized) \\
    word2vec    : <str>, path to the words embedding file \\
    out_path    : <str>, path to the file storing the nouns with frequency

    Return: 
    ------
    None
    """

    print('\tStarting generate nouns ... ')
    
    with open(file_path, 'r') as f:
        text = f.readlines()
    nouns = []
    noun_counts = defaultdict(int)

    for sentence in tqdm(text):
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)
        nouns = [word for word, pos in tagged_words if pos[0] == 'N']
        for noun in nouns:
            noun_counts[noun] +=1

    w2v = Reach.load(word2vec, unk_word="<UNK>")
    nouns_dict = Counter()
    for k, v in tqdm(noun_counts.items()):
        if k.lower() in w2v.items:
            nouns_dict[k.lower()] += v
    
    json.dump(nouns_dict, open(out_path, 'w'))
    print(f"\tFinishing generate nouns, output path: {out_path}")
    return 

def word2vec(file_path, output="embeddings/restaurant_vecs_w2v.vec"): 
    """ Training words embedding with Word2Vec.
    Parameters:
    ------
    file_path   : path to training text file \\
    output      : path to store words embedding. 

    Return: 
    ------
    None
    """
    corpus = [x.lower().strip().split() for x in open(file_path)]
    
    print(f"\tTraining Word2Vec - vector_size = {vector_size} - epochs = {epochs} ... ")
    f = Word2Vec(corpus, 
                 negative=5,
                 window=10,
                 vector_size=vector_size,
                 min_count=1,
                 epochs=epochs,
                 workers=8)
    f.wv.save_word2vec_format(output)
    print(f"\tFinish training word embedding, output path: {w2v_path}")
if __name__ == "__main__": 
    word2vec(file_path=file_path, output=w2v_path)
    generate_nouns(file_path=file_path, word2vec=w2v_path, out_path=nouns_path)
    