# working with nltk shakespeare corpus
import nltk
from nltk.corpus import shakespeare as sksp
import string
# you may need to download the corpus; follow the instructions in the python interpreter


# available plays in the corpus
sksp.fileids()
# check one play, e.g. a midsummer dream
play = sksp.xml('dream.xml')
txt = ''.join(play.itertext())
# separate the text into individual sentences
sents = nltk.sent_tokenize(txt)
# you may want to use all the plays in the data

# take a look at those sentences. do we need to do more cleanups?
for idx, s in enumerate(sents):
    sents[idx] = s.replace('\n', ' ')
# do we also want to remove punctuations and/or stopwords

from nltk.corpus import stopwords


# calculate ngrams
from nltk.util import ngrams
# lets take a look at different ngrams, say sentence 101
sents[101]
# first we need to break the sentence into words
nltk.word_tokenize(sents[101])
list(ngrams(nltk.word_tokenize(sents[101]), n=2))


def normalize_word(token):
    # remove punctuations and convert all the characters to lower case
    _token = remove_punctuation(token)  # remove any special chars
    return _token.strip().lower()


def remove_punctuation(text, punctuation_extended="'°`-"):
    return ''.join(c for c in text if c not in punctuation_extended)


def tokenize_sentence(sent):
    # tokenize
    tokens = [normalize_word(token) for token in nltk.word_tokenize(sent)]
    return tokens


list(ngrams(tokenize_sentence(sents[101]), n=2))
# any more clean up we need to do?
# how do we account for the start and end of sentence?
tk = tokenize_sentence(sents[101])
# nltk.lm.preprocessing.pad_both_ends does similar thing
tk = nltk.pad_sequence(tk, n=2, pad_left=True, left_pad_symbol='<s>',
                       pad_right=True, right_pad_symbol='</s>')
# the token '<s>' and '</s>' will be recognized when using nltk to build a language model
# tk is an iterator, to use it multiple time convert it into a list
tk = list(tk)
# calculate ngrams and corresponding dictionaries
unigram = nltk.ngrams(tk, 1)
uni_dict = nltk.FreqDist(unigram)
trigram = nltk.ngrams(tk, 3)
tri_dict = nltk.FreqDist(trigram)

# tokenize the text, sentence by sentence
tokenized_text = [tokenize_sentence(sent)
                  for sent in nltk.sent_tokenize(txt)]



