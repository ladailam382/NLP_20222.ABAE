"""
Simple dataset loader for the semeval-2014 and citysearch test sets.
"""
from sklearn.preprocessing import LabelEncoder      
from functools import partial                       

def loader(instance_path,
           label_path,
           subset_labels,
           ):
    """
    instance_path   = path to review file? 
    label_path      = path to file contain set of labels (all set of label)
    subset_labels   = sub set of labels, paper uses subset of size 3

    Return
    ------ 
    instance    : list of sentences
    y           : list of encoded labels
    label_set   : label set, only in subset_labels ['food', 'staff', 'ambience']
    """
    subset_labels = set(subset_labels)
    labels = open(label_path)
    labels = [x.strip().lower().split() for x in labels]

    instances = []          # List of lists of words = list of sentences
    for line in open(instance_path):
        instances.append(line.strip().lower().split())


    instances, gold = zip(*[(x, y[0]) for x, y in zip(instances, labels)
                            if len(y) == 1 and y[0]
                            in subset_labels])

    le = LabelEncoder()
    y = le.fit_transform(gold)
    label_set = le.classes_.tolist()

    return instances, y, label_set

citysearch_test = partial(loader,
                          instance_path="data/citysearch/test.txt",
                          label_path="data/citysearch/test_label.txt",
                          subset_labels={"ambience", "staff", "food"}
                          )

semeval_14_test = partial(loader,
                       instance_path="data/semeval2014/test_se.txt",
                       label_path="data/semeval2014/test_label_se.txt",
                       subset_labels={"ambience", "staff", "food"},
                       )

def semeval_loader():
    yield semeval_14_test()

def citysearch_loader():
    yield citysearch_test()

if __name__ == "__main__": 
    instance, y, label_set = semeval_loader()
    print("Label set:", end=' ')
    print(label_set)
    
    print('Sample text:')
    print(instance[1])

    print('and the corresponding label:', end=' ')
    print(label_set[y[1]])
    