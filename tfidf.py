import numpy as np
import pandas as pd
from nltk import word_tokenize

class TF_IDF:
    # corpus must be list of sentences -> ['hello world', 'I am a data scientist']
    def __init__(self, corpus, stopwords):
        self.corpus = corpus
        self.stopwords = stopwords

    def preprocess_text(self):
        clean_corpus = []
        for i in range(len(self.corpus)):
            clean_sent = [word.lower() for word in word_tokenize(self.corpus[i]) if
                          word.isalpha() and word not in self.stopwords]
            clean_corpus.append(clean_sent)
        return clean_corpus

    def get_vocab(self):
        words_set = set()
        clean_corpus = self.preprocess_text()
        for i in range(len(clean_corpus)):
            words = clean_corpus[i]
            for word in words:
                words_set.add(word)
        return words_set

    def tf(self, doc, word_i):
        word_i_occurence = 0
        doc_len = len(doc)
        for word in doc:
            if word == word_i:
                word_i_occurence += 1
        return word_i_occurence / doc_len

    def idf(self, corpus, word_i):
        doc_nums = len(corpus)
        doc_word_freq = {}
        for i in range(len(corpus)):
            visited = set()
            words = corpus[i]
            for word in words:
                if word in visited:
                    continue
                if word not in doc_word_freq:
                    doc_word_freq[word] = 1
                else:
                    doc_word_freq[word] += 1
                visited.add(word)
        return np.log(doc_nums / doc_word_freq[word_i])

    def tf_idf(self):
        clean_corpus = self.preprocess_text()
        word_set = self.get_vocab()
        tf_idf = pd.DataFrame(columns=[word for word in word_set],
                              index=['sent_{}'.format(i) for i in range(len(clean_corpus))])
        sent_num = -1
        for sent in clean_corpus:
            sent_num += 1
            for word in sent:
                tf_idf.iloc[sent_num][word] = self.tf(sent, word) * self.idf(clean_corpus, word)
        tf_idf = tf_idf.fillna(-1)
        return tf_idf

