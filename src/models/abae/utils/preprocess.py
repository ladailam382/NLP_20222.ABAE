from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import codecs

def parseSentence(line):
    lmtzr = WordNetLemmatizer()    
    stop = stopwords.words('english')
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_rmstop = [i for i in text_token if i not in stop]
    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]
    return text_stem

def preprocess_train(domain):
    f = codecs.open('../../../../data/raw/train.txt', 'r', 'utf-8')
    out = codecs.open('../../../../data/preprocessed/train.txt', 'w', 'utf-8')

    for line in f:
        tokens = parseSentence(line)
        if len(tokens) > 0:
            out.write(' '.join(tokens)+'\n')

def preprocess_test(domain):
    # For restaurant domain, only keep sentences with single 
    # aspect label that in {Food, Staff, Ambience}

    f1 = codecs.open('../../../../data/raw/test.txt', 'r', 'utf-8')
    f2 = codecs.open('../../../../data/raw/test_label.txt', 'r', 'utf-8')
    out1 = codecs.open('../../../../data/preprocessed/test.txt', 'w', 'utf-8')
    out2 = codecs.open('../../../../data/preprocessed/test_label.txt', 'w', 'utf-8')

    for text, label in zip(f1, f2):
        label = label.strip()
        if domain == 'restaurant' and label not in ['Food', 'Staff', 'Ambience']:
            continue
        tokens = parseSentence(text)
        if len(tokens) > 0:
            out1.write(' '.join(tokens) + '\n')
            out2.write(label+'\n')

def preprocess(domain):
    print ('\t'+domain+' train set ...')
    preprocess_train(domain)
    print ('\t'+domain+' test set ...')
    preprocess_test(domain)

if __name__ == "__main__":
    print ('Preprocessing raw review sentences ...')
    preprocess('restaurant')
    print('Finish preprocessing')


