from keras.models import Model
import numpy as np
from tensorflow.keras.preprocessing import sequence
import src.models.abae.utils.reader as dataset
from src.models.abae.model import create_model
import keras.backend as K
from src.models.abae.utils.optimizers import get_optimizer
from src.models.abae.utils.evaluation import max_margin_loss, preprocess_text
from src.models.abae.utils.parsers import get_parsers
from src.models.acd.Model import UnsupervisedACD
from src.models.cat.cat.simple import get_nouns, get_scores, rbf_attention, attention, get_aspect, softmax
from reach import Reach
import torch
from argparse import ArgumentParser
from src.models.uce.utils.trainer import Trainer
from src.models.uce.utils.evaluation import predict_uce
from src.models.uce.utils.parsers import get_parsers_uce


w2v_path_cat = "src/models/cat/embeddings/w2v_restaurant_200_ep_5.vec"
w2v_cat = Reach.load(w2v_path_cat, unk_word="<UNK>")  

def predict_abae(input):

    args, _ = get_parsers().parse_known_args()
    out_dir = args.out_dir_path 

    cluster_map = {0: 'Food', 1: 'Miscellaneous', 2: 'Miscellaneous', 3: 'Food',
           4: 'Miscellaneous', 5: 'Food', 6:'Price',  7: 'Miscellaneous', 8: 'Staff', 
           9: 'Food', 10: 'Food', 11: 'Anecdotes', 
           12: 'Ambience', 13: 'Staff'}
    # print('Setup')
    preprocess_text(input, args.domain)

    ###### Get test data #############
    vocab, train_x, test_x, overall_maxlen = dataset.get_data(args.domain, vocab_size=args.vocab_size, maxlen=args.maxlen, train=False)
    test_x = sequence.pad_sequences(test_x, maxlen=overall_maxlen)

    ############# Build model architecture, same as the model used for training #########
    # print("Getting model")
    optimizer = get_optimizer(args)

    model = create_model(args, overall_maxlen, vocab)

    ## Load the save model parameters
    model.load_weights(out_dir+'/model_param1.h5')
    model.compile(optimizer=optimizer, loss=max_margin_loss, metrics=[max_margin_loss])
    # model.save_weights(out_dir+'/model_param1.h5', save_format='h5')
    # print("Finishing getting model")
    ## Create a dictionary that map word index to word 
    vocab_inv = {}
    for w, ind in vocab.items():
        vocab_inv[ind] = w

    test_model = Model(inputs=model.get_layer('sentence_input').input,
                       outputs=[model.get_layer('att_weights').output, model.get_layer('p_t').output])
    att_weights, aspect_probs = test_model.predict(test_x)
    temp_probs =  aspect_probs[0]
    softmax = dict()
    cluster_name = list(cluster_map.values())
    for i in range(len(cluster_map)):
        if cluster_name[i] in ['Staff', 'Food', 'Ambience']:
            try:
                softmax[cluster_name[i]] += float(round(temp_probs[i], 3))
            except KeyError:
                softmax[cluster_name[i]] = float(round(temp_probs[i], 3))
    # softmax 
    softmax = {i:j for i, j in zip(softmax.keys(),np.exp(list(softmax.values()))/np.sum(np.exp(list(softmax.values()))))}  
    label_ids = np.argmax(list(softmax.values()))
    return cluster_map[label_ids], softmax

def predict_acd(input):
    labels = ['Staff', 'Food', 'Ambience']
    save_path = "src/models/acd/save/model_sem_eval_train_11_city_search_train_sem_eval_train.pkl"
    model = UnsupervisedACD.load(save_path)
    pred = model.predict(input)

    res = {}
    for k, v in zip(labels, pred):
        res[k] = v
    return [labels[np.argmax(pred)], res]

def predict_cat(input):
    att         = rbf_attention # `rbf_attention` or `attention`
    GAMMA       = .03           # if attention then GAMMA is not in use
    N_NOUNS     = 200           # 200 for rbf_attention, 950 for attention
    w2v_path_cat = "src/models/cat/embeddings/w2v_restaurant_200_ep_5.vec"
    nouns_path = "src/models/cat/data/nouns_restaurant_200_ep_5.json"

    print("\tLoading words embedding ...")
    # w2v_cat = Reach.load(w2v_path_cat, unk_word="<UNK>")     
    
    print("\tLoading top most frequent nouns ... ")
    top_nouns = get_nouns(w2v_cat, nouns_path, N_NOUNS)
    print("\tFinish loading nouns")
    ### TESTING 

    # sentences = ["The seafood is so fresh, but the hotdog is much more better".split(), 
    #              "the waiter is friendly but he is too short".split(), 
    #              ]
    sentences = [input.split()]
    label_set = ['food', 'staff', 'ambience']

    s = get_scores(sentences,
                top_nouns,
                w2v_cat,
                label_set,
                gamma=GAMMA,
                attention_func=att)
    # print(s)
    logit = softmax(s)
    pred = logit.argmax(1)

    label_pred = get_aspect(label_set, pred)
    label = ['Food', 'Staff', 'Ambience']
    logit = {i: j for i, j in zip(label, logit.tolist()[0])}
    return label_pred[0].capitalize(), logit

def predict_uce(input):
    if torch.cuda.is_available():
    #   print(torch.cuda.get_device_name(0))
      device = torch.device("cuda")
    else:
      device = torch.device("cpu")

    torch.save([[input]], "data/cache/test_data.zip")

    args, _ = get_parsers_uce().parse_known_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(args, device)
    output = trainer.test_predict(args)
    label = ['Food', 'Staff', 'Ambience']
    return label[np.argmax(output)], {i:j for i, j in zip(label, output)}


# if __name__ == '__main__':

#     y = predict_uce('The bread is top notch as well.')

#     # y, probs = predict_abae('The bread is top notch as well.')
    
#     print("Answer:", y)
#     # print("Probs:", probs)

#     # print(predict_acd("The design and atmosphere is just as good."))