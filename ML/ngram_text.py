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
# you might need to download it as well
stopwords_eng = stopwords.words('english')


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
    tokens = [normalize_word(token) for token in nltk.word_tokenize(sent)
              if len(normalize_word(token)) > 0]
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
unigram_onesent = nltk.ngrams(tk, 1)
uni_dict_onesent = nltk.FreqDist(unigram_onesent)
trigram_onesent = nltk.ngrams(tk, 3)
tri_dict_onesent = nltk.FreqDist(trigram_onesent)

# tokenize the text, sentence by sentence
tokenized_text = [tokenize_sentence(sent)
                  for sent in nltk.sent_tokenize(txt)]

# build ngram dicts for all the sentences
unigram = nltk.ngrams(sum(tokenized_text, []), 1)
uni_dict = nltk.FreqDist(unigram)
bigram = nltk.ngrams(sum(tokenized_text, []), 2)
bi_dict = nltk.FreqDist(bigram)
trigram = nltk.ngrams(sum(tokenized_text, []), 3)
tri_dict = nltk.FreqDist(trigram)
# or, just use a pipeline in nltk: nltk.lm.preprocessing.padded_everygram_pipeline
# the everygram is basically all the 'gram' models up to and including n
gram_order = 3
train, vocab = nltk.lm.preprocessing.padded_everygram_pipeline(gram_order, tokenized_text)

# now we can use those ngram vocabs to train a linear model
lm = nltk.lm.MLE(gram_order)
lm.fit(train, vocab)

# you can check how many times or probability a word or a phrase occurred in the training set
# try lm.counts and lm.score
# it can also be used to generate new text
lm.generate(20, random_seed=3)
# or generate from a given word/phrase
lm.generate(20, text_seed=['helen'])
# still lots of details we do not cover, e.g. if some words not known in the vocab and so on
# in this case you need to do some sort of smoothing. see
# http://mlwiki.org/index.php/Smoothing_for_Language_Models

# what do you think are the problems/limitations of these type of models?


# you can also try of (not so) state of art NLP models. specifically, here is a post about the GPT-2 model:
# https://towardsdatascience.com/text-generation-with-python-and-gpt-2-1fecbff1635b
# follow it and see how far you can get!
# for the description of the model, see https://huggingface.co/gpt2?text=A+long+time+ago%2C
# for an introduction of the transformers network, see
# https://builtin.com/artificial-intelligence/transformer-neural-network

