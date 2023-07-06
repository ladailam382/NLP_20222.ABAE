from pprint import pprint
import numpy as np
import json
from sklearn.metrics.pairwise import rbf_kernel
from collections import Counter

def normalize(x):
    """Normalize a vector while controlling for zero vectors.
    Parameters: 
    ------
    `x`: np.ndarray.  a vector

    Return
    ------
    `new x`: np.ndarray. a normalized vector
    """
    x = np.copy(x)
    if np.ndim(x) == 1:
        norm = np.linalg.norm(x)
        print(norm)
        if norm == 0:
            return x
        return x / np.linalg.norm(x)
    norm = np.linalg.norm(x, axis=-1)
    mask = norm > 0
    x[mask] /= norm[mask][:, None]
    return x

def get_aspect(label_set:list, pred:list): 
    return [label_set[x] for x in pred]

def rbf_attention(vec, memory, gamma, **kwargs):
    """
    Single-head attention using RBF kernel.

    Parameters
    ----------
    vec : np.array
        an (N, D)-shaped array, representing the tokens of an instance.
    memory : np.array
        an (M, D)-shaped array, representing the memory items
    gamma : float
        the gamma of the RBF kernel.

    Returns
    -------
    attention : np.array
        A (1, N)-shaped array, representing a single-headed attention mechanism

    """
    # Compute rbf(x, y, gamma)
    z = rbf_kernel(vec, memory, gamma)

    s = z.sum()
    if s == 0:
        # If s happens to be 0, back off to uniform
        return np.ones((1, len(vec))) / len(vec)
    return (z.sum(1) / s)[None, :]

def softmax(x, axis=1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis, keepdims=True))
    s = e_x.sum(axis=axis, keepdims=True)
    return e_x / s

def attention(vec, memory, **kwargs):
    """
    Standard multi-head attention mechanism.

    Parameters
    ----------
    vec : np.array
        an (N, D)-shaped array, representing the tokens of an instance.
    memory : np.array
        an (M, D)-shaped array, representing the memory items

    Returns
    -------
    attention : np.array
        A (M, N)-shaped array, representing the attention over all memories.

    """
    z = memory.dot(vec.T)
    return softmax(z)

# def mean(vec, aspect_vecs, **kwargs):
#     """Just a mean weighting."""
#     return (np.ones(len(vec)) / len(vec))[None, :]
def get_nouns(w2v, noun_path, n_nouns=300): 
    all_nouns = json.load(open(noun_path))

    nouns = Counter()
    for k, v in all_nouns.items():
        if k.lower() in w2v.items:
            nouns[k.lower()] += v

    top_nouns, _ = zip(*nouns.most_common(n_nouns))
    top_nouns = [[x] for x in top_nouns]
    return top_nouns

def get_scores(instances,
               nouns,
               r,
               labels,
               remove_oov=False,
               attention_func=attention,
               **kwargs):
    """Scoring function.
    Parameters: 
    ------ 
    instances   : list of sentences [['food', 'sweet',], [...]]
    nouns       : list of nouns 
    r           : class <Reach> 
    labels      : list
    """

    assert all([x in r.items for x in labels])

    label_vecs = normalize(r.vectorize(labels))
    
    aspect_vecs = [x.mean(0)
                   for x in r.transform(nouns,
                                        remove_oov=False)]
    aspect_vecs = np.stack(aspect_vecs)

    t = r.transform(instances, remove_oov=remove_oov)
    out = []
    
    ### From attention head to score prediction
    for vec in t:
        att = attention_func(vec, aspect_vecs, **kwargs)
        # Att = (n_heads, n_words)
        
        z = att.dot(vec)
        # z = (n_heads, n_dim)
        
        x = normalize(z).dot(label_vecs.T)
        # x = (n_heads, n_labels)
        out.append(x.sum(0))
    return np.stack(out)

if __name__ == "__main__": 
    pass